"""RAEDecoderH1 — move the stage-2 diffusion boundary INTO the decoder.

Instead of diffusing the encoder combine latent z (the standard RAE target), the DiT
diffuses **h1**: the hidden state after the decoder's FIRST transformer block. The
remaining decoder blocks (1..N-1) reconstruct the image from h1.

  image --(frozen encoder + equal-mean combine)--> z  [B, N, 1024]   (raw, drop off)
        --decoder_embed + CLS + pos --> block[0] -->  h1 [B, N+1, Cdec]
  DiT target  = normalized h1 PATCH tokens (CLS dropped) [B, Cdec, H, W]
  decode(h1)  = [mean-CLS, h1 patches] -> block[1..N-1] -> norm -> pred -> unpatchify

The spatial DiT (PatchEmbed/unpatchify require T = H*W, a square) cannot carry the extra
CLS token, so the image-dependent block-0 CLS is replaced at decode time by the
dataset-mean CLS (precomputed by compute_h1_stats.py). No REPA: h1 is already a
decoder-internal, recon-aligned representation.

Built on RAECombine (loads our random-drop / no-sigreg k=23 decoder); combine runs in
deterministic full-mean mode (drop=False) so the h1 target is deterministic per image.
"""
import os
from math import sqrt

import torch

from .rae_variants import RAECombine


class RAEDecoderH1(RAECombine):
    def __init__(self, h1_stats_path: str = None, output_01: bool = False, **kwargs):
        kwargs.setdefault("drop", False)              # deterministic full-mean latent -> deterministic h1
        super().__init__(**kwargs)
        self._ckpt = None
        # output_01: the loaded decoder is a [0,1]-folded ckpt (de-norm baked into decoder_pred),
        # so neutralize the decode-time ImageNet de-norm (line ~90) to avoid double application.
        if output_01:
            self.img_mean.zero_(); self.img_std.fill_(1.0)
            print("RAEDecoderH1: output_01 -> de-norm neutralized (img_mean=0, img_std=1)")
        cdec = self.decoder.decoder_pred.in_features  # decoder_hidden_size (1152 for ViT-XL)
        has_stats = bool(h1_stats_path) and os.path.exists(h1_stats_path)
        st = torch.load(h1_stats_path, map_location="cpu") if has_stats else {}
        self.register_buffer("h1_mean", st["mean"] if "mean" in st else torch.zeros(1, cdec, 1, 1))
        self.register_buffer("h1_var", st["var"] if "var" in st else torch.ones(1, cdec, 1, 1))
        self.register_buffer("mean_cls", st["mean_cls"] if "mean_cls" in st else torch.zeros(1, 1, cdec))
        if not st:
            print("RAEDecoderH1: WARNING — no h1_stats_path; using identity stats / zero CLS "
                  "(fine for stats precompute, NOT for training).")
        else:
            print(f"RAEDecoderH1: loaded h1 stats (Cdec={cdec}) from {h1_stats_path}")

    @torch.no_grad()
    def _combine_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """images -> raw equal-mean combine latent [B, N, 1024] (the input the stage-1
        decoder was trained on; no stats normalization, drop off)."""
        x = self._imgs_to_norm(images)
        layer_tokens = list(self.encoder.model.get_intermediate_layers(
            x, n=self.encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))
        return self.combine(layer_tokens)

    @torch.no_grad()
    def _block0(self, images: torch.Tensor) -> torch.Tensor:
        """images -> h1 [B, N+1, Cdec] (decoder up to and including transformer block 0)."""
        z = self._combine_tokens(images)              # [B, N, 1024]
        d = self.decoder
        x = d.decoder_embed(z)                         # [B, N, Cdec]
        x = d.interpolate_latent(x)
        cls = d.trainable_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)                 # [B, N+1, Cdec]
        x = x + d.decoder_pos_embed
        return d.decoder_layers[0](x, head_mask=None)[0]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images -> normalized h1 patch tokens [B, Cdec, H, W] (the DiT diffusion target)."""
        h1 = self._block0(images)
        patches = h1[:, 1:, :]                         # drop CLS -> [B, N, Cdec]
        b, n, c = patches.shape
        hw = int(sqrt(n))
        z = patches.transpose(1, 2).view(b, c, hw, hw)
        return (z - self.h1_mean) / torch.sqrt(self.h1_var + self.eps)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """normalized h1 [B, Cdec, H, W] -> image [0,1] via decoder blocks 1..N-1.
        (pos-embed is already baked into h1; the CLS is the dataset-mean substitute.)"""
        z = z * torch.sqrt(self.h1_var + self.eps) + self.h1_mean
        b, c, h, w = z.shape
        patches = z.view(b, c, h * w).transpose(1, 2)  # [B, N, Cdec]
        cls = self.mean_cls.to(device=z.device, dtype=z.dtype).expand(b, -1, -1)
        hs = torch.cat([cls, patches], dim=1)          # [B, N+1, Cdec]
        d = self.decoder
        for layer in d.decoder_layers[1:]:
            hs = layer(hs, head_mask=None)[0]
        hs = d.decoder_norm(hs)
        logits = d.decoder_pred(hs)[:, 1:, :]          # drop CLS -> [B, N, p*p*3]
        return d.unpatchify(logits) * self.img_std + self.img_mean
