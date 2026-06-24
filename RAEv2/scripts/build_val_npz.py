"""Build the ImageNet-256 val NPZ (data_eval/imagenet-256-val.npz) used by
eval_recon_subset_rfid.py, from the HF dataset's `test` split parquet shards.

Images are short-side-256 already; we center-crop to 256x256 and stack as a
single uint8 array of shape [N, 256, 256, 3].
"""
import glob
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

SNAP = "/mnt/localssd/.cache/huggingface/hub/datasets--evanarlian--imagenet_1k_resized_256/snapshots"
OUT = "data_eval/imagenet-256-val.npz"
SIZE = 256


def center_crop(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    # short side should already be `size`; crop the long side centrally.
    left = (w - size) // 2
    top = (h - size) // 2
    return im.crop((left, top, left + size, top + size))


def main():
    shards = sorted(glob.glob(os.path.join(SNAP, "*/data/test-*.parquet")))
    print(f"{len(shards)} test shards")
    imgs = []
    for s in shards:
        t = pq.read_table(s, columns=["image"])
        col = t.column("image").to_pylist()
        for rec in col:
            im = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
            if im.size != (SIZE, SIZE):
                # ensure short side >= SIZE then center-crop
                w, h = im.size
                if min(w, h) != SIZE:
                    scale = SIZE / min(w, h)
                    im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
                im = center_crop(im, SIZE)
            imgs.append(np.asarray(im, dtype=np.uint8))
        print(f"  {os.path.basename(s)}: {len(col)} imgs (total {len(imgs)})", flush=True)
    arr = np.stack(imgs, 0)  # [N, 256, 256, 3] uint8
    print(f"final array: {arr.shape} {arr.dtype}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # eval script does arr[arr.files[0]] -> single named array
    np.savez(OUT, images=arr)
    print(f"saved -> {OUT} ({os.path.getsize(OUT)//1024//1024} MB)")


if __name__ == "__main__":
    main()
