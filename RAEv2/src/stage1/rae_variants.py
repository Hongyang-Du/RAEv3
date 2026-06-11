"""RAE wrappers exposing our locally-trained MLS stage-1 variants to stage-2 training.

Why the stock `stage1.RAE` cannot be used directly:
1) Our stage-1 decoders (train_decoder_mls*.py) were trained to output images in
   ImageNet-NORMALIZED space (the train loss compared `unpatchify(out)*std + mean`
   against [0,1] targets). The stock RAE.decode returns `unpatchify(out)` raw, and
   every consumer (eval/generation.py, array2grid, sample_and_decode) assumes that
   is already a [0,1] image -> decode() here must de-normalize.
2) For the SIGReg variants the diffusion latent is the PROJECTOR output (the space
   SIGReg shaped toward N(0, I)), not the raw encoder output -> encode() applies the
   frozen (EMA) projector.
3) The raev2-MLS baseline keeps the dinov3mls combine (mean over K layers + the
   final-layer global-mean CLS surrogate), exactly as in train_decoder_mls.py.

Both classes expose the RAE interface train.py / stage2.engine / eval rely on:
    encode(x in [0,1]) -> z [B, C, 16, 16]   (stats-normalized if stats provided)
    decode(z)          -> image [B, 3, H, W] (~[0,1], unclamped; callers clamp)
plus .latent_dim / .resolution / .base_patches attributes.
"""

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn

from encoders.vision_encoder import create_encoder
from .rae import _load_decoder, _load_normalization_stats

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class MLSProjector(nn.Module):
    """Per-token residual MLP over the plain mean of K layer tokens (eval-time form).

    Same parameters as the MLSProjector in train_decoder_mls_{nogate,dropmean}_sigreg.py
    (no gate parameter in either; dropmean's random layer dropout is train-time only,
    so at inference both variants reduce to skip(z0) + ffn(norm(z0)) on the full mean).
    Loaded from the stage-1 ckpt's `ema_proj` / `projector` state dict.
    """
    def __init__(self, dim: int = 1024, out_dim: int = 1024, mult: int = 4):
        super().__init__()
        self.skip = nn.Linear(dim, out_dim) if dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, out_dim),
        )

    def forward(self, layer_tokens) -> torch.Tensor:        # K x [B, N, dim] -> [B, N, out_dim]
        z0 = torch.stack(layer_tokens, dim=0).mean(0)
        return self.skip(z0) + self.ffn(self.norm(z0))


class _RAEVariantBase(nn.Module):
    """Shared plumbing: frozen DINOv3 encoder, decoder from a stage-1 train ckpt,
    optional latent stats normalization, ImageNet de-normalization on decode."""

    def __init__(self,
        encoder_name: str,
        resolution: int = 256,
        decoder_config_path: str = 'configs/decoder/ViTXL',
        decoder_patch_size: int = 16,
        stage1_ckpt_path: Optional[str] = None,
        use_ema: bool = True,
        latent_dim: int = 1024,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = create_encoder(encoder_name,
                                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                      resolution=resolution)
        self.resolution = resolution
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = latent_dim
        self.base_patches = (resolution // 16) ** 2
        self.eps = eps
        self.use_ema = use_ema

        self.decoder = _load_decoder(
            decoder_config_path, latent_dim, decoder_patch_size,
            self.base_patches, pretrained_path=None,
        )
        self.latent_mean, self.latent_var, self.do_normalization = \
            _load_normalization_stats(normalization_stat_path)

        self.register_buffer('img_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('img_std',  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self._ckpt = None
        if stage1_ckpt_path is not None:
            self._ckpt = torch.load(stage1_ckpt_path, map_location='cpu', weights_only=False)
            dec_key = 'ema_dec' if use_ema else 'decoder'
            self.decoder.load_state_dict(self._ckpt[dec_key])
            print(f"{type(self).__name__}: loaded decoder[{dec_key}] from {stage1_ckpt_path}"
                  f" (stage-1 epoch {self._ckpt.get('epoch')})")

        for p in self.parameters():
            p.requires_grad_(False)

    def _imgs_to_norm(self, x: torch.Tensor) -> torch.Tensor:
        """[0,1] (or [0,255]) images at any size -> ImageNet-normalized at self.resolution."""
        if x.max() > 1.0:
            x = x / 255.0
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(x, size=(self.resolution, self.resolution),
                                          mode='bicubic', align_corners=False)
        return (x - self.img_mean) / self.img_std

    def _tokens_to_latent(self, z: torch.Tensor) -> torch.Tensor:
        """[B, N, C] tokens -> [B, C, H, W] stats-normalized latent."""
        b, n, c = z.shape
        h = w = int(sqrt(n))
        z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        b, c, h, w = z.shape
        z = z.view(b, c, h * w).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False).logits
        # our stage-1 decoders predict in ImageNet-normalized space -> back to [0,1]
        return self.decoder.unpatchify(output) * self.img_std + self.img_mean

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        x_rec = self.decode(z)
        return (x_rec, z) if return_latent else x_rec


class RAEMLSBaseline(_RAEVariantBase):
    """raev2 MLS baseline: dinov3mls combine (mean over K layers + CLS surrogate),
    decoder retrained locally (train_decoder_mls.py ckpt). No projector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ckpt = None   # free the ckpt blob

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._imgs_to_norm(x)
        z = self.encoder.forward_features(x)["x_norm_patchtokens"]    # [B, N, C]
        return self._tokens_to_latent(z)


class RAEProjected(_RAEVariantBase):
    """SIGReg variants (nogate / dropmean): per-layer tokens -> frozen EMA projector.
    The diffusion latent is the projector output — the space SIGReg shaped toward
    N(0, I) — matching exactly what the paired decoder was trained on."""

    def __init__(self, projector_mult: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.projector = MLSProjector(dim=self.encoder.hidden_size,
                                      out_dim=self.latent_dim, mult=projector_mult)
        if self._ckpt is not None:
            proj_key = 'ema_proj' if self.use_ema else 'projector'
            self.projector.load_state_dict(self._ckpt[proj_key])
            print(f"RAEProjected: loaded projector[{proj_key}]")
        self.projector.eval()
        for p in self.projector.parameters():
            p.requires_grad_(False)
        self._ckpt = None

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._imgs_to_norm(x)
        layer_tokens = list(self.encoder.model.get_intermediate_layers(
            x, n=self.encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))
        z = self.projector(layer_tokens)                              # [B, N, latent]
        return self._tokens_to_latent(z)
