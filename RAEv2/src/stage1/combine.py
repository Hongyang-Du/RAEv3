"""MLSCombine: one configurable module that collapses every stage-1 MLS-decoder
combine variant into a single nn.Module.

The frozen DINOv3 encoder produces K per-layer patch-token tensors; MLSCombine
turns them into the latent z [B, N, out_dim] the ViT decoder consumes. All the
forked train_decoder_mls*.py scripts differ ONLY in this combine — here it is
parameterized by four knobs:

  weighting      mean | random_drop | softgate  how the K layers are mixed
  p_drop         float                          per-sample random LAYER-drop prob
  p_full         float                          per-sample prob of keeping ALL layers
                                                (no drop) -> mixes full + dropped passes
  cls_surrogate  bool                           add L_last token-mean (raev2 code)
  projector      none | ln | bn                 per-token residual MLP after mix

`random_drop` = "Random Drop Layer MLS": each sample keeps each layer with prob
(1 - p_drop) (>=1 kept) and z0 is the equal-weight mean over the kept subset.
Drops whole LAYERS (not units) -> structurally collapse-proof; eval = full mean.

Variant map (the three kept experiments + the legacy ones):
  raev2 K=23              weighting=mean,        projector=none, cls_surrogate=false
  Random Drop Layer MLS   weighting=random_drop, projector=none, cls_surrogate=false
  Random Drop Layer + MLP weighting=random_drop, projector=bn,   cls_surrogate=false
  nogate (legacy)         weighting=mean,        projector=ln
  softgate (legacy)       weighting=softgate,    projector=ln|none

forward(layer_tokens, idx=None):
  layer_tokens : list of K tensors, each [B, N, dim]
  idx          : optional list of positions into the FULL layer set; restricts
                 the combine to that subset (renormalized), powering the LOO/solo
                 probes from a single trained model with no retraining.
  returns      : z [B, N, out_dim]

DDP note: with weighting in {mean, random_drop} and projector=none the module has
ZERO parameters. DDP errors on a param-free module, so the trainer must call it
directly (not wrap it in DDP) when has_params is False.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_cond import sample_stratified_masks


class MLSCombine(nn.Module):
    def __init__(self,
                 layers,
                 weighting: str = "random_drop",
                 p_drop: float = 0.3,
                 p_full: float = 0.0,
                 cls_surrogate: bool = False,
                 projector: str = "none",
                 dim: int = 1024,
                 out_dim: int = 1024,
                 mult: int = 4,
                 tau: float = 1.0,
                 topk: int = 0,
                 noise_tau: float = 0.0):
        super().__init__()
        assert weighting in ("mean", "random_drop", "softgate", "learned_gate"), weighting
        assert projector in ("none", "ln", "bn"), projector
        assert 0.0 <= p_full <= 1.0, p_full
        self.layers = list(layers)
        self.K = len(self.layers)
        self.weighting = weighting
        self.p_drop = p_drop
        self.p_full = p_full           # per-sample prob of keeping ALL layers (no drop)
        self.cls_surrogate = cls_surrogate
        self.projector = projector
        # RAEv2 latent-noise augmentation (mirrors stage1.rae.RAE.noising): train-time
        # only, per-sample sigma ~ U[0, noise_tau] added to the final latent. 0 = off.
        self.noise_tau = noise_tau

        if weighting == "softgate":
            self.gate = nn.Parameter(torch.zeros(self.K))   # softmax logits, init uniform

        # learned_gate: a TRAINABLE softmax over the K layers, learned by the
        # stage-2 DiT denoising loss (the "DiT picks its own layers" experiment).
        # No random drop; eval == train (deterministic softmax-weighted mean).
        self.tau = tau
        self.topk = topk               # learned_gate: keep exactly top-k layers (0 = full softmax)
        if weighting == "learned_gate":
            self.gate_logits = nn.Parameter(torch.zeros(self.K))  # init uniform 1/K

        if projector == "none":
            self.skip = None
        else:
            self.skip = nn.Linear(dim, out_dim) if dim != out_dim else nn.Identity()
            if projector == "ln":
                self.norm = nn.LayerNorm(dim)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, out_dim))
            else:  # bn (LeWM recipe): BN on the hidden dim over B*N token samples
                self.fc1 = nn.Linear(dim, dim * mult)
                self.bn = nn.BatchNorm1d(dim * mult)
                self.fc2 = nn.Linear(dim * mult, out_dim)

    @property
    def has_params(self) -> bool:
        return any(True for _ in self.parameters())

    def _mix(self, stk, idx, mask=None):
        """stk [K, B, N, dim] -> z0 [B, N, dim] (weighted combine over layers)."""
        K, B = stk.shape[0], stk.shape[1]

        # external per-sample mask [B, K] (mask-conditioning trainer samples it once
        # so latent and decoder conditioning provably share the same mask). Same
        # renormalized masked mean as the internal random_drop path.
        if mask is not None:
            w = mask.to(stk.dtype).t()                       # [K, B]
            w = w / w.sum(0, keepdim=True).clamp_min(1e-6)
            return (w.view(K, B, 1, 1) * stk).sum(0)

        # learned_gate: deterministic softmax-weighted mean, grad flows to gate_logits.
        if self.weighting == "learned_gate":
            w = torch.softmax(self.gate_logits / self.tau, dim=0)   # [K]
            if idx is not None:                                     # probe subset (renorm)
                w = w[list(idx)]
                w = w / w.sum().clamp_min(1e-6)
                return (w.view(-1, 1, 1, 1) * stk[list(idx)]).sum(0)
            if self.topk and 0 < self.topk < self.K:
                # straight-through top-k: forward = EQUAL mean over the current top-k
                # layers (== what the fixed-k DiT will use); backward flows through the
                # soft softmax so the gate keeps learning WHICH k to keep. Guarantees the
                # DiT never trains on fewer than k layers (no collapse to 1).
                topi = torch.topk(w, self.topk).indices
                hard = torch.zeros_like(w)
                hard[topi] = 1.0 / self.topk
                w = w + (hard - w).detach()                        # forward=hard, grad=soft
            return (w.view(K, 1, 1, 1) * stk).sum(0)

        if self.weighting == "softgate":
            gate_w = torch.softmax(self.gate, dim=0)
            if idx is not None:
                gate_w = gate_w[list(idx)]
                gate_w = gate_w / gate_w.sum().clamp_min(1e-6)
        else:
            gate_w = None

        # probe subset (eval): plain/weighted mean over the given positions
        if idx is not None:
            sub = stk[list(idx)]                            # [|idx|, B, N, dim]
            if gate_w is not None:                          # already sliced+renormalized
                return (gate_w.view(-1, 1, 1, 1) * sub).sum(0)
            return sub.mean(0)

        if self.training and self.weighting in ("random_drop", "softgate") and self.p_drop > 0:
            keep = torch.rand(K, B, device=stk.device) > self.p_drop
            dead = ~keep.any(0)
            if dead.any():                                  # always keep >= 1 layer
                keep[torch.randint(K, (int(dead.sum()),), device=stk.device), dead] = True
            if self.p_full > 0:                             # some samples keep ALL layers
                full = torch.rand(B, device=stk.device) < self.p_full
                keep[:, full] = True
            w = keep.to(stk.dtype)
            if gate_w is not None:
                w = w * gate_w.view(K, 1)
            w = w / w.sum(0, keepdim=True).clamp_min(1e-6)
            return (w.view(K, B, 1, 1) * stk).sum(0)

        if gate_w is not None:
            return (gate_w.view(K, 1, 1, 1) * stk).sum(0)
        return stk.mean(0)                                   # mean / random_drop eval

    def _noise(self, z: torch.Tensor, idx) -> torch.Tensor:
        """RAEv2 latent-noise aug: train-time only, skip on probe/eval (idx given)."""
        if idx is not None or not self.training or self.noise_tau <= 0:
            return z
        sigma = self.noise_tau * torch.rand((z.size(0),) + (1,) * (z.dim() - 1), device=z.device)
        return z + sigma * torch.randn_like(z)

    def forward(self, layer_tokens, idx=None, mask=None) -> torch.Tensor:
        stk = torch.stack(layer_tokens, dim=0)              # [K, B, N, dim]
        z0 = self._mix(stk, idx, mask=mask)                 # [B, N, dim]
        if self.cls_surrogate:
            z0 = z0 + stk[-1].mean(dim=1, keepdim=True)     # raev2 L_last token-mean (fixed)

        if self.projector == "none":
            return self._noise(z0, idx)
        if self.projector == "ln":
            return self._noise(self.skip(z0) + self.ffn(self.norm(z0)), idx)
        # bn
        b, n, _ = z0.shape
        h = self.fc1(z0)
        h = self.bn(h.reshape(b * n, -1).float()).reshape(b, n, -1).to(h.dtype)
        return self._noise(self.skip(z0) + self.fc2(F.gelu(h)), idx)


# ---------------------------------------------------------------------------
# Variant B: per-position depth attention (learned token-level fusion)
# ---------------------------------------------------------------------------

class DepthAttnBlock(nn.Module):
    """One depth-attention refinement block: the fused query token cross-attends
    over the K per-layer tokens at its own spatial position. The attention
    out-projection AND the FFN output are zero-init -> the block is an exact
    no-op at init (residual around the query)."""

    def __init__(self, dim: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, mlp_mult * dim), nn.GELU(),
                                 nn.Linear(mlp_mult * dim, dim))
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, q, kv, pad):                   # q [S,1,d]  kv [S,K,d]  pad [S,K]
        kvn = self.norm_kv(kv)
        a, _ = self.attn(self.norm_q(q), kvn, kvn,
                         key_padding_mask=pad, need_weights=False)
        q = q + a
        return q + self.ffn(self.norm_ffn(q))


class DepthAttnFusion(nn.Module):
    """fused = masked_mean + AttnBlocks(query=masked_mean, kv=per-layer tokens).

    Pure reshape: (K, B, N, d) -> (B*N, K, d) sequences on the DEPTH axis; dropped
    layers are key_padding_mask'ed out per sample. A learnable depth embedding
    tells attention WHICH layer each kv token came from (it only reaches the
    output through the zero-init projections, so init identity is preserved)."""

    def __init__(self, dim: int, K: int, n_layers: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.depth_pos = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [DepthAttnBlock(dim, n_heads, mlp_mult) for _ in range(n_layers)])

    def forward(self, z0, stk, mask):                # z0 [B,N,d] stk [K,B,N,d] mask [B,K]
        K, B, N, d = stk.shape
        kv = stk.permute(1, 2, 0, 3).reshape(B * N, K, d) + self.depth_pos
        q = z0.reshape(B * N, 1, d)
        pad = (~mask.bool()).repeat_interleave(N, dim=0)          # [B*N, K], True = drop
        for blk in self.blocks:
            q = blk(q, kv, pad)
        return q.reshape(B, N, d)


class DepthAttnCombine(nn.Module):
    """Variant B combine: no lossy mean bottleneck -- the decoder-facing latent is
    the masked mean PLUS a learned per-position depth-attention correction over
    the kept layers' tokens. Mirrors the MLSCombine interface (forward(tokens,
    idx=, mask=), has_params, p_drop) so train_decoder.py / eval scripts work
    unchanged.

    Warm-start: init_from an UNconditional random-drop ckpt (param-free MLSCombine
    -> empty combine state_dict); the fresh `fusion.*` params are zero-init no-ops,
    so step 0 == the anchor's equal-weight masked mean (+ cls surrogate).

    Collapse note (Fig 2): a bare learned gate collapses to shallow layers, but
    only because they are always present; under random drop the missing layers
    cannot be leaned on -- dropout is what keeps this learned fusion honest. Still,
    start the GAN a bit late (disc_start >= 1): Fig 2 shows GAN accelerates
    collapse before the fusion settles.

    Training masks are sampled INTERNALLY with the same stratified sampler as
    Variant A (full / uniform-|S| / Bernoulli) unless an external `mask` is
    passed (the mask_cond trainer path, enabling an A+B combo)."""

    def __init__(self,
                 layers,
                 p_drop: float = 0.3,
                 full_frac: float = 1 / 3,
                 uniform_frac: float = 1 / 3,
                 cls_surrogate: bool = False,
                 dim: int = 1024,
                 out_dim: int = 1024,
                 n_layers: int = 2,
                 n_heads: int = 8,
                 mlp_mult: int = 2):
        super().__init__()
        assert dim == out_dim, "DepthAttnCombine is residual around the mean: dim must equal out_dim"
        self.layers = list(layers)
        self.K = len(self.layers)
        self.p_drop = p_drop                       # trainer's p_drop schedule hooks this
        self.full_frac = full_frac
        self.uniform_frac = uniform_frac
        self.cls_surrogate = cls_surrogate
        self.fusion = DepthAttnFusion(dim, self.K, n_layers, n_heads, mlp_mult)

    @property
    def has_params(self) -> bool:
        return True

    def forward(self, layer_tokens, idx=None, mask=None) -> torch.Tensor:
        stk = torch.stack(layer_tokens, dim=0)     # [K, B, N, dim]
        K, B = stk.shape[0], stk.shape[1]
        if mask is None:
            if idx is not None:                    # probe/eval subset -> k-hot
                mask = torch.zeros(B, K, dtype=torch.bool, device=stk.device)
                mask[:, list(idx)] = True
            elif self.training and self.p_drop > 0:
                mask = sample_stratified_masks(B, K, self.p_drop,
                                               full_frac=self.full_frac,
                                               uniform_frac=self.uniform_frac,
                                               device=stk.device)
            else:                                  # eval, full feed
                mask = torch.ones(B, K, dtype=torch.bool, device=stk.device)
        w = mask.to(stk.dtype).t()                 # [K, B]
        w = w / w.sum(0, keepdim=True).clamp_min(1e-6)
        z0 = (w.view(K, B, 1, 1) * stk).sum(0)     # masked equal-weight mean
        z = self.fusion(z0, stk, mask)
        if self.cls_surrogate:
            z = z + stk[-1].mean(dim=1, keepdim=True)   # raev2 L_last token-mean (fixed)
        return z
