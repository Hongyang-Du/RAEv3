"""Measure how much SIGReg competes with reconstruction for the projector's
gradient, to set the right sigreg weight.

On one batch: build the BN projector + ViT-XL decoder, compute
  g_rec = ||d(L1+LPIPS)/d(projector)||
  g_sig = ||d(sigreg_unscaled)/d(projector)||
The projector-gradient contribution ratio for weight w (scale_by_n=False) is
  w * g_sig / g_rec
We want this ~0.1-0.3 (SIGReg a gentle nudge, not a gradient thief).
For scale_by_n=True the statistic is x N_total larger, so the equivalent weight
is divided by N_total.
"""
import sys
sys.path.insert(0, "src")
import torch
import torch.nn.functional as F
from torchvision import transforms

from stage1.combine import MLSCombine
from stage1.rae import _load_decoder
from encoders.vision_encoder import create_encoder
from stage1.disc.lpips import LPIPS as LPIPS_
from overfit_sigreg import sigreg_loss
from data.partial_imagenet import PartialImageNetDataset

device = torch.device("cuda")
B = 32
layers = list(range(1, 24))
enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, layers)) + "]",
                     device=device, resolution=256).eval()
for p in enc.parameters():
    p.requires_grad_(False)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

proj = MLSCombine(layers=layers, weighting="random_drop", projector="bn").to(device)
dec = _load_decoder("configs/decoder/ViTXL", 1024, 16, 256, None).to(device)
lpips = LPIPS_().to(device).eval()
for p in lpips.parameters():
    p.requires_grad_(False)

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
ds = PartialImageNetDataset("/datasets/imagenet-256", split="train", transform=tf)
imgs = torch.stack([ds[i][0] for i in range(B)]).to(device)

proj_params = list(proj.parameters())


def grad_norm(loss):
    g = torch.autograd.grad(loss, proj_params, retain_graph=True, allow_unused=True)
    return sum((x.detach() ** 2).sum() for x in g if x is not None).sqrt().item()


toks = list(enc.model.get_intermediate_layers((imgs - mean) / std, n=enc.layer_indices,
            reshape=False, return_class_token=False, norm=True))
proj.train()
z = proj(toks)
out = dec(z, drop_cls_token=False).logits
x_rec = (dec.unpatchify(out) * std + mean).clamp(0, 1)
loss_rec = F.l1_loss(x_rec, imgs) + lpips(x_rec * 2 - 1, imgs * 2 - 1).mean()
sig_unscaled = sigreg_loss(z.float().reshape(-1, 1024), distributed=False, scale_by_n=False)

g_rec = grad_norm(loss_rec)
g_sig = grad_norm(sig_unscaled)
N_local = z.shape[0] * z.shape[1]          # batch*256 on this 1 GPU
N_real = 8 * B * 256                        # the real 8-GPU run

print(f"batch={B}  N_local={N_local}  N_real(8gpu)={N_real}")
print(f"loss_rec={loss_rec.item():.4f}  sig_unscaled={sig_unscaled.item():.4f}")
print(f"||grad_rec||           = {g_rec:.4e}")
print(f"||grad_sig_unscaled||  = {g_sig:.4e}")
print(f"ratio (unscaled, w=1)  = {g_sig/g_rec:.3f}   <- sigreg/recon projector-grad ratio at w=1, scale_by_n=False")
print()
print("Target: sigreg projector-grad ~= 0.2 x recon  (gentle regularizer)")
w_unscaled = 0.2 * g_rec / g_sig
print(f"  scale_by_n=FALSE -> weight ~= {w_unscaled:.3f}")
w_scaled = w_unscaled / N_real
print(f"  scale_by_n=TRUE  -> weight ~= {w_scaled:.2e}   (current 0.02 is {0.02/w_scaled:.0f}x too strong)")
