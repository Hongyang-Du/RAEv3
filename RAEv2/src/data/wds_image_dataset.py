"""Generic WebDataset wrapper for image-only training (stage-1 RAE decoder).

Used for sources where captions/labels are not consumed by the loss path
(e.g., rendertext-256, flux-synthetic-256). Modeled on BLIP3OWebDataset;
the only real differences are: dataset-agnostic subset discovery, and caption
sidecars are returned as empty strings (the trainer drops them via
`for images, _ in dataloader`).
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import webdataset as wds
from torchvision import transforms


# Per-dataset subset metadata used for epoch-length estimation.
# Numbers are nominal (paper / repo-card values); a small mis-count only
# shifts virtual epoch length, not correctness.
GENERIC_WDS_METADATA: Dict[str, Dict[str, Dict[str, int]]] = {
    # Preprocessed flat 256 pool: 384 PNG -> 256 JPEG, all 5 folders pooled.
    # 2439 shards x 10000 samples/shard = 24.39M samples (~459 GB on disk).
    "flux-synthetic-256": {
        "root": {"num_samples": 24_390_000, "num_shards": 2439},
    },
    # Preprocessed flat 256 pool: 1024 PNG -> 256 JPEG, root + remaining/ pooled,
    # 10 input shards merged per output shard. 1204 shards x 10000 samples/shard
    # = 12.04M samples (~217 GB on disk). Regenerated 2026-05-13 with more samples.
    "rendertext-256": {
        "root": {"num_samples": 12_040_000, "num_shards": 1204},
    },
}


def _filter_valid_samples(sample):
    return sample[0] is not None


_SHARD_INDEX_JSON = Path(__file__).resolve().parents[2] / "data" / "raev2_hf_shards.json"


def hf_tar_urls(hf_repo: str, hf_paths: List[str], revision: str = "main") -> List[str]:
    """Resolve `*.tar` shard URLs under the given HF dataset path prefixes.

    Streams shards straight from the Hub (no full download); each prefix is a folder
    like 'rendertext-256', 'scale-rae-256', or 'blip3o-256/short-caption'.

    Prefers a pre-baked static index (data/raev2_hf_shards.json) so that hundreds of
    DDP ranks don't each hammer the HF tree API at startup (which triggers HTTP 429
    rate-limiting). Falls back to a single live API listing only if the index is
    missing a requested prefix.
    """
    import json

    index = {}
    if _SHARD_INDEX_JSON.exists():
        try:
            data = json.loads(_SHARD_INDEX_JSON.read_text())
            if data.get("repo") == hf_repo:
                index = data.get("paths", {})
        except Exception:
            index = {}

    urls: List[str] = []
    missing: List[str] = []
    for prefix in hf_paths:
        got = index.get(prefix.rstrip("/"))
        if got:
            urls.extend(got)
        else:
            missing.append(prefix)

    if missing:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(hf_repo, repo_type="dataset", revision=revision)
        base = f"https://huggingface.co/datasets/{hf_repo}/resolve/{revision}"
        for prefix in missing:
            pfx = prefix.rstrip("/") + "/"
            urls.extend(f"{base}/{f}" for f in files if f.startswith(pfx) and f.endswith(".tar"))

    if not urls:
        raise ValueError(f"No .tar shards found under {hf_paths} in {hf_repo}")
    return sorted(urls)


class GenericWebDataset:
    """WebDataset wrapper that globs `*.tar` from one or more subset directories.

    Returns (image_tensor, "") for parity with BLIP3OWebDataset's (image, caption)
    interface. Stage-1 reconstruction discards the second element.
    """

    def __init__(
        self,
        data_dir: str,
        subsets: Union[str, List[str]],
        dataset_name: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 256,
        shuffle_buffer: int = 20000,
        seed: int = 42,
        hf_repo: Optional[str] = None,
        hf_paths: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        cache_size: int = 0,
    ):
        """
        Args:
            data_dir: Base path containing tar shards or one folder per subset
                     (e.g., 'data/rendertext-256', 'data/scale-rae-flux-synthetic-256').
            subsets: Single subset name or list of subset folder names to combine.
                     Use ['root'] for flat pools (tars at data_dir itself).
            dataset_name: Optional key into GENERIC_WDS_METADATA for sample
                     count lookup. If None, falls back to a per-shard estimate.
            transform: Optional torchvision transform. Defaults to resize-to-image_size + ToTensor.
            image_size: Target square resolution.
            shuffle_buffer: WebDataset sample-level shuffle buffer.
            seed: Base RNG seed; per-epoch seed = seed + epoch.
        """
        self.data_dir = Path(data_dir)
        self.subsets = [subsets] if isinstance(subsets, str) else list(subsets)
        self.dataset_name = dataset_name
        self.transform = transform
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.hf_repo = hf_repo
        self.cache_dir = cache_dir
        self.cache_size = cache_size

        meta = GENERIC_WDS_METADATA.get(dataset_name, {}) if dataset_name else {}

        self._total_samples = 0
        self._shard_urls: List[str] = []

        if hf_repo:
            # Stream tar shards straight from the HF Hub (cached to cache_dir on first
            # read; epoch 2+ reads local). hf_paths gives the HF folder prefix(es).
            paths = hf_paths or self.subsets
            self._shard_urls = hf_tar_urls(hf_repo, paths)
            for subset in self.subsets:
                if subset in meta:
                    self._total_samples += meta[subset]["num_samples"]
            if self._total_samples == 0:
                self._total_samples = len(self._shard_urls) * 10_000
        else:
            for subset in self.subsets:
                # 'root' is a sentinel for the data_dir itself (flat pool);
                # everything else is a real sub-folder.
                subset_dir = self.data_dir if subset == "root" else self.data_dir / subset
                if not subset_dir.exists():
                    raise ValueError(f"Subset directory not found: {subset_dir}")

                tar_files = sorted(subset_dir.glob("*.tar"))
                if not tar_files:
                    raise ValueError(f"No tar shards found in {subset_dir}")
                self._shard_urls.extend(str(f) for f in tar_files)

                if subset in meta:
                    self._total_samples += meta[subset]["num_samples"]
                else:
                    # Fallback: assume ~50k samples/shard (rough average for these sources).
                    self._total_samples += len(tar_files) * 50_000

        self._num_shards = len(self._shard_urls)

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    @property
    def estimated_size(self) -> int:
        return self._total_samples

    @property
    def num_shards(self) -> int:
        return self._num_shards

    def _decode_sample(self, sample):
        image = sample.get("jpg") or sample.get("png") or sample.get("jpeg") or sample.get("webp")
        if self.transform is not None and image is not None:
            image = self.transform(image)
        return image, ""

    def create_pipeline(self, epoch: int = 0) -> wds.WebDataset:
        return (
            wds.WebDataset(
                self._shard_urls,
                nodesplitter=wds.split_by_node,
                shardshuffle=1000,
                seed=self.seed + epoch,
                cache_dir=self.cache_dir,
                cache_size=self.cache_size,
            )
            .shuffle(self.shuffle_buffer, initial=self.shuffle_buffer // 2)
            .decode("pil", handler=wds.ignore_and_continue)
            .map(self._decode_sample, handler=wds.ignore_and_continue)
            .select(_filter_valid_samples)
        )
