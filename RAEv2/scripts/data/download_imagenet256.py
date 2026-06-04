"""
Download partial ImageNet-256 from HuggingFace (parallel, fast).
Uses hf_transfer for maximum speed if available.
"""
import argparse
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/home/colligo/data/imagenet-256/imagenet-256")
    parser.add_argument("--num-samples", type=int, default=93000)
    parser.add_argument("--num-shards", type=int, default=20)
    parser.add_argument("--hf-dataset", default="evanarlian/imagenet_1k_resized_256")
    args = parser.parse_args()

    arrow_dir = Path(args.out_dir) / "imagenet-latents-images"
    arrow_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.hf_dataset} (parallel, no streaming)...")
    from datasets import load_dataset
    ds = load_dataset(args.hf_dataset, split="train")  # parallel download

    print(f"Full dataset: {len(ds)} images. Taking first {args.num_samples}...")
    ds = ds.select(range(min(args.num_samples, len(ds))))

    print(f"Saving {len(ds)} images as {args.num_shards} Arrow shards to:\n  {arrow_dir}")
    ds.save_to_disk(str(arrow_dir), num_shards=args.num_shards)
    print(f"\nDone. {len(ds)} images saved.")


if __name__ == "__main__":
    main()
