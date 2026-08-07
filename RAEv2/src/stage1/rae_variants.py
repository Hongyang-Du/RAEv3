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
            if dec_key in self._ckpt:
                self.decoder.load_state_dict(self._ckpt[dec_key])
                print(f"{type(self).__name__}: loaded decoder[{dec_key}] from {stage1_ckpt_path}"
                      f" (stage-1 epoch {self._ckpt.get('epoch')})")
            else:
                # Stage-0-only ckpt (JEPA fusion, no decoder). OK for a train-only DiT
                # with sample-viz/eval disabled: encode() never calls the decoder.
                print(f"{type(self).__name__}: no '{dec_key}' in {stage1_ckpt_path} -> "
                      f"decoder left at init (train-only; do NOT decode/sample/eval)")

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

    def _tokens_to_latent_train(self, z: torch.Tensor) -> torch.Tensor:
        """[B, N, C] tokens -> [B, C, H, W] per-batch unit-variance latent (grad-safe).
        Used by learned_gate: standardize per-channel over the batch so the gate cannot
        minimize the denoising loss by shrinking the target magnitude."""
        b, n, c = z.shape
        h = w = int(sqrt(n))
        z = z.transpose(1, 2).view(b, c, h, w)
        mu = z.mean(dim=(0, 2, 3), keepdim=True)
        var = z.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        return (z - mu) / torch.sqrt(var + self.eps)

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


class RAECombine(_RAEVariantBase):
    """Our NEW MLSCombine stage-1 (src/stage1/combine.py): random-drop / sigreg /
    plain. The diffusion latent is the combine output (ema_combine).

    With drop=True the per-step latent uses per-sample random LAYER dropout (combine
    kept in train mode so MLSCombine._mix fires its Bernoulli keep-mask) but the BN
    projector runs with FROZEN running stats (its submodule forced to eval). So the
    DiT trains on the dropped-latent DISTRIBUTION, exactly the augmentation the user
    asked for, without touching the (frozen) projector weights/stats."""

    def __init__(self, combine_config, drop: bool = True, **kwargs):
        super().__init__(**kwargs)
        from omegaconf import OmegaConf
        from utils.model_utils import get_obj_from_str
        cc = OmegaConf.to_container(combine_config, resolve=True) \
            if OmegaConf.is_config(combine_config) else dict(combine_config)
        self.combine = get_obj_from_str(cc["target"])(**cc["params"])
        if self._ckpt is not None:
            key = 'ema_combine' if self.use_ema else 'combine'
            # learned_gate adds a fresh gate_logits param absent from the pretrained
            # combine ckpt -> load non-strict so it keeps its uniform init.
            strict = cc["params"].get("weighting") != "learned_gate"
            missing, unexpected = self.combine.load_state_dict(self._ckpt[key], strict=strict)
            print(f"RAECombine: loaded combine[{key}]  (drop={drop}, strict={strict}, "
                  f"missing={list(missing)}, unexpected={list(unexpected)})")
        for p in self.combine.parameters():
            p.requires_grad_(False)
        # learned_gate: the K-dim softmax over layers stays TRAINABLE (learned by the
        # stage-2 DiT denoising loss); everything else in the combine stays frozen.
        self.has_learnable_gate = getattr(self.combine, "weighting", None) == "learned_gate"
        if self.has_learnable_gate:
            self.combine.gate_logits.requires_grad_(True)
            print(f"RAECombine: learned_gate ENABLED (K={self.combine.K}, trainable softmax)")
        self.drop = drop
        self._apply_combine_mode()
        self._ckpt = None

    def gate_parameters(self):
        return [self.combine.gate_logits] if self.has_learnable_gate else []

    @torch.no_grad()
    def gate_weights(self):
        """current softmax gate over the K layers [K], for logging."""
        return torch.softmax(self.combine.gate_logits / self.combine.tau, dim=0)

    def encode_train(self, x: torch.Tensor):
        """Grad-enabled encode for learned_gate: encoder runs frozen (no_grad), the
        gate-weighted combine carries gradient so the DiT loss can train the gate.
        Returns (z_norm, z_tok):
          z_norm : per-batch unit-variance standardized latent [B,C,H,W] -> DiT target
                   (the gate cannot cheat by shrinking magnitude = true layer selection).
          z_tok  : RAW gate-weighted combine tokens [B,N,C] -> frozen-decoder recon anchor."""
        x = self._imgs_to_norm(x)
        with torch.no_grad():
            layer_tokens = list(self.encoder.model.get_intermediate_layers(
                x, n=self.encoder.layer_indices, reshape=False,
                return_class_token=False, norm=True))
        z_tok = self.combine(layer_tokens)                # gate_logits has grad
        return self._tokens_to_latent_train(z_tok), z_tok

    def recon_from_tokens(self, z_tok: torch.Tensor) -> torch.Tensor:
        """Decode the RAW gate-weighted combine tokens [B,N,C] with the FROZEN decoder
        (gradient flows to the gate, not the decoder). Returns image in [0,1]. This is
        the recon anchor that pulls the gate away from the single most-diffusable layer
        toward layer mixes the decoder can actually reconstruct."""
        out = self.decoder(z_tok, drop_cls_token=False).logits
        return self.decoder.unpatchify(out) * self.img_std + self.img_mean

    def _apply_combine_mode(self):
        self.encoder.eval()                               # frozen DINOv3: ALWAYS deterministic
        self.decoder.eval()                               # (no encoder dropout/droppath)
        if self.drop:
            self.combine.train()                          # MLSCombine._mix random drop on
            for m in self.combine.modules():              # but keep BN on running stats
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        else:
            self.combine.eval()

    def train(self, mode: bool = True):
        # frozen stage-1: never propagate train mode to encoder/decoder. Only the
        # combine's random-drop sampling needs training=True (BN stays on running stats).
        self.training = mode
        self._apply_combine_mode()
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._imgs_to_norm(x)
        layer_tokens = list(self.encoder.model.get_intermediate_layers(
            x, n=self.encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))
        z = self.combine(layer_tokens)                    # per-sample random drop if self.drop
        return self._tokens_to_latent(z)

    @torch.no_grad()
    def encode_cond_target(self, x: torch.Tensor):
        """Decoupled-target encode (transport.decoupled_full_target). ONE frozen DINOv3
        forward, TWO combines over the SAME layer tokens:
          z_cond   : random-LAYER-drop latent (self.drop must be True) -> builds x_t (the
                     noised INPUT the DiT sees; identical distribution to the drop run).
          z_target : DETERMINISTIC full-mean latent over ALL K layers -> the FM regression
                     target (what eval-combine feeds the decoder at inference).
        Both are normalized by the SAME latent_stats so x_t and target share one space; the
        cls_surrogate term is added identically in both (unconditional in MLSCombine)."""
        assert self.drop, "encode_cond_target needs drop=True (z_cond is the random-drop latent)"
        x = self._imgs_to_norm(x)
        layer_tokens = list(self.encoder.model.get_intermediate_layers(
            x, n=self.encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))
        z_cond = self.combine(layer_tokens)               # combine in train() -> random drop
        self.combine.eval()                               # force deterministic full mean
        z_target = self.combine(layer_tokens)
        self._apply_combine_mode()                        # restore drop mode (+ BN eval)
        return self._tokens_to_latent(z_cond), self._tokens_to_latent(z_target)
