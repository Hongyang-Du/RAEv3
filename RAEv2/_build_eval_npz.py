import os, io, glob, numpy as np, pyarrow.parquet as pq
from PIL import Image
SRC = "/mnt/localssd/evanarlian_test/data"
OUT = "data_eval/imagenet-256-val-eval.npz"
def cc(im, s=256):
    w,h=im.size
    if min(w,h)!=s: sc=s/min(w,h); im=im.resize((round(w*sc),round(h*sc)),Image.BICUBIC)
    l,t=(im.size[0]-s)//2,(im.size[1]-s)//2; return im.crop((l,t,l+s,t+s))
imgs=[]
for sf in sorted(glob.glob(os.path.join(SRC,"test-*.parquet"))):
    for rec in pq.read_table(sf,columns=["image"]).column("image").to_pylist():
        im=Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        if im.size!=(256,256): im=cc(im)
        imgs.append(np.asarray(im,np.uint8))
    print("shard done, total", len(imgs), flush=True)
arr=np.stack(imgs,0); print("final", arr.shape)
np.savez(OUT, arr_0=arr); print("saved", OUT, os.path.getsize(OUT)//1024//1024, "MB")
