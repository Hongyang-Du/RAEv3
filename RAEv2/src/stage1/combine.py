"""MLSCombine: one configurable module that collapses every stage-1 MLS-decoder
combine variant into a single nn.Module.

The frozen DINOv3 encoder produces K per-layer patch-token tensors; MLSCombine
turns them into the latent z [B, N, out_dim] the ViT decoder consumes. All the
forked train_decoder_mls*.py scripts differ ONLY in this combine — here it is
parameterized by four knobs:

  weighting      mean | random_drop | dirichlet_drop | softgate  how the K layers mix
  p_drop         float                          per-sample random LAYER-drop prob
                                                (dirichlet_drop: the MEAN drop rate)
  p_full         float                          per-sample prob of keeping ALL layers
                                                (no drop) -> mixes full + dropped passes
  cls_surrogate  bool                           add L_last token-mean (raev2 code)
  projector      none | ln | bn                 per-token residual MLP after mix

`random_drop` = "Random Drop Layer MLS": each sample keeps each layer with prob
(1 - p_drop) (>=1 kept) and z0 is the equal-weight mean over the kept subset.
Drops whole LAYERS (not units) -> structurally collapse-proof; eval = full mean.

`dirichlet_drop` = per-LAYER drop probabilities modeled with a Dirichlet. Every
step a per-sample simplex vector pi ~ Dirichlet(alpha) (alpha = softplus(log_alpha),
one CONCENTRATION per layer, learnable by default) allocates a fixed "drop budget":
the per-layer drop RATE is p_k = clamp(pi_k * K * p_drop, 0, p_max), so the mean
rate over layers stays p_drop (schedulable) while the Dirichlet redistributes it
UNEVENLY across layers and randomly each step. Layers with larger alpha_k are, on
average, dropped more; learnable alpha lets the model learn which layers to lean on.
The keep is still a HARD whole-layer Bernoulli (collapse-proof); a reparameterized
rsample + straight-through mask routes gradient back to alpha. Eval = full mean
(alpha unused at eval), so the probe / cross-decode pipeline is unchanged.

Variant map (the kept experiments + the legacy ones):
  raev2 K=23              weighting=mean,           projector=none, cls_surrogate=false
  Random Drop Layer MLS   weighting=random_drop,    projector=none, cls_surrogate=false
  Random Drop Layer + MLP weighting=random_drop,    projector=bn,   cls_surrogate=false
  Dirichlet Drop Layer    weighting=dirichlet_drop, projector=none, cls_surrogate=false
  nogate (legacy)         weighting=mean,           projector=ln
  softgate (legacy)       weighting=softgate,       projector=ln|none

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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    """Identity on the forward pass; NEGATES (and scales by lambd) the gradient on the
    backward pass. Wrapping log_alpha with this makes a single loss.backward() ASCEND the
    reconstruction loss w.r.t. alpha (adversarial drop) while still DESCENDING it w.r.t.
    the decoder -> no second optimizer / no trainer change needed."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


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
                 noise_tau: float = 0.0,
                 alpha_init: float = 1.0,
                 alpha_learnable: bool = True,
                 p_max: float = 0.95,
                 p_min: float = 0.0,
                 adv_lambda: float = 0.0,
                 alpha_eps: float = 1e-3):
        super().__init__()
        assert weighting in ("mean", "random_drop", "dirichlet_drop", "softgate", "learned_gate"), weighting
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

        # dirichlet_drop: per-layer Dirichlet concentration alpha = softplus(log_alpha)+eps.
        # log_alpha stores softplus^{-1}(alpha_init) so all layers start at alpha_init
        # (symmetric prior == same expected drop rate); training then breaks the symmetry.
        self.p_max = float(p_max)          # cap per-layer drop rate (keep every layer trainable)
        self.p_min = float(p_min)          # floor per-layer drop rate (no layer ever ~never dropped)
        self.adv_lambda = float(adv_lambda)  # >0: adversarial alpha (gradient-reversal ascent); 0: normal descent
        self.alpha_eps = float(alpha_eps)  # floor so alpha > 0 (Dirichlet needs positive conc.)
        if weighting == "dirichlet_drop":
            a0 = max(float(alpha_init) - self.alpha_eps, 1e-4)
            log_a0 = math.log(math.expm1(a0))          # softplus^{-1}(a0) = log(exp(a0)-1)
            init = torch.full((self.K,), log_a0)
            if alpha_learnable:
                self.log_alpha = nn.Parameter(init)    # learned per-layer reliance (has_params -> DDP)
            else:
                self.register_buffer("log_alpha", init)  # fixed per-layer alpha from config

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

    def _mix(self, stk, idx):
        """stk [K, B, N, dim] -> z0 [B, N, dim] (weighted combine over layers)."""
        K, B = stk.shape[0], stk.shape[1]

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

        # dirichlet_drop: per-layer drop RATES drawn from a Dirichlet each step.
        # Always taken while training (even if p_drop->0 by schedule) so log_alpha stays
        # in the autograd graph -> DDP never sees it as an unused parameter.
        if self.training and self.weighting == "dirichlet_drop":
            # adv_lambda>0: reverse the gradient into log_alpha so it ASCENDS the recon loss
            # (alpha learns to drop the layers the decoder leans on most -> forces robustness),
            # instead of descending it (which collapses to keep-shallow / drop-deep). 0 = descent.
            la = grad_reverse(self.log_alpha, self.adv_lambda) if self.adv_lambda > 0 else self.log_alpha
            alpha = (F.softplus(la) + self.alpha_eps).float()   # [K] > 0
            base_p = max(float(self.p_drop), 1e-6)          # mean drop rate (>0 keeps alpha in graph)
            # per-sample simplex allocation of the drop budget; rsample -> grad flows to alpha
            pi = torch.distributions.Dirichlet(alpha.expand(B, K)).rsample()  # [B, K], rows sum to 1
            # simplex -> per-layer drop rate; mean_k p ~= base_p (mean pi == 1/K). Bound to
            # [p_min, p_max]: prevents the extremes (some layer ~never dropped, some ~always) so
            # every layer stays both seen and dropped -> the adversary can't fully abandon a layer.
            p = (pi * K * base_p).clamp(self.p_min, self.p_max)    # [B, K]
            keep_prob = 1.0 - p
            keep_hard = (torch.rand_like(keep_prob) < keep_prob).float()      # HARD whole-layer keep
            dead = keep_hard.sum(dim=1) == 0                # always keep >= 1 layer per sample
            if dead.any():
                j = torch.randint(K, (int(dead.sum()),), device=stk.device)
                keep_hard[dead, j] = 1.0
            if self.p_full > 0:                             # some samples keep ALL layers
                full = torch.rand(B, device=stk.device) < self.p_full
                keep_hard[full, :] = 1.0
            # straight-through: forward = hard 0/1 mask, backward = grad through keep_prob -> alpha
            keep = (keep_prob + (keep_hard - keep_prob).detach()).transpose(0, 1)  # [K, B]
            w = keep / keep.sum(0, keepdim=True).clamp_min(1e-6)   # equal weight over kept subset
            return (w.view(K, B, 1, 1).to(stk.dtype) * stk).sum(0)

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

    def forward(self, layer_tokens, idx=None) -> torch.Tensor:
        stk = torch.stack(layer_tokens, dim=0)              # [K, B, N, dim]
        z0 = self._mix(stk, idx)                            # [B, N, dim]
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
# DepthAttnCombine: depth-attention fusion (re-added from the ckpt src snapshot).
#
# The latent = masked equal-weight mean over the kept DINOv3 layers PLUS a learned
# per-position depth-attention correction that cross-attends the fused query token
# over the K per-layer tokens at its own spatial position. Trained jointly with the
# ViT decoder (train_decoder.py) under the anti-collapse semantic rent. Mirrors the
# MLSCombine interface (forward(tokens, idx=, mask=), has_params, p_drop) so
# train_decoder.py / eval / stage-2 (RAECombine) work unchanged.
#
# self-contained: `sample_stratified_masks` (originally stage1.mask_cond) is inlined
# below so the module imports with no extra deps. Only used when training with
# drop=True; the stage-2 DiT runs drop=False (deterministic full-feed mean+fusion).
# ---------------------------------------------------------------------------


def sample_stratified_masks(B: int, K: int, p_drop: float,
                            full_frac: float = 1 / 3, uniform_frac: float = 1 / 3,
                            device=None) -> torch.Tensor:
    """Stratified per-sample layer masks [B, K] bool (True = layer kept).

    Per-sample group assignment (random, so every rank/micro-batch mixes):
      full_frac     m = all-ones (the sandwich full branch)
      uniform_frac  |S| ~ Uniform{1..K}, then a uniform subset of that size
                    -> extreme sizes (1, K) get real training mass
      remainder     i.i.d. Bernoulli(keep = 1-p_drop), >=1 kept
    """
    u = torch.rand(B, device=device)
    is_full = u < full_frac
    is_unif = (~is_full) & (u < full_frac + uniform_frac)

    keep = torch.rand(B, K, device=device) > p_drop
    dead = ~keep.any(1)
    if dead.any():                                   # always keep >= 1 layer
        keep[dead, torch.randint(K, (int(dead.sum()),), device=device)] = True

    if is_unif.any():                                # uniform-|S| group: keep top-s
        n = int(is_unif.sum())
        s = torch.randint(1, K + 1, (n,), device=device)              # [n] in {1..K}
        scores = torch.rand(n, K, device=device)
        rank = scores.argsort(dim=1).argsort(dim=1)                   # 0..K-1 per row
        keep[is_unif] = rank < s[:, None]                             # top-s kept

    keep[is_full] = True
    return keep


class MultiheadGateAttention(nn.Module):
    """Sigmoid 'gate' attention over the KEY axis: every key gets an INDEPENDENT gate
    sigmoid(q.k/sqrt(dh) + b) in [0,1] -- NO softmax normalization, so keys (= layers)
    do NOT compete. out = (gates @ V); key_padding_mask zeros dropped layers via a -inf
    logit. out_proj is zero-init so the enclosing DepthAttnBlock is an exact no-op at
    init. Returns the head-averaged gate [B,Lq,Lk] as the probe-facing 'weight'."""

    def __init__(self, dim: int, n_heads: int, bias_init: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, (dim, n_heads)
        self.h, self.dh = n_heads, dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gate_bias = nn.Parameter(torch.full((n_heads,), float(bias_init)))

    def forward(self, q, kv, key_padding_mask=None):   # q [B,Lq,d] kv [B,Lk,d] mask [B,Lk] True=drop
        B, Lq, d = q.shape
        Lk = kv.shape[1]
        Q = self.q_proj(q).view(B, Lq, self.h, self.dh).transpose(1, 2)    # [B,h,Lq,dh]
        Kk = self.k_proj(kv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        V = self.v_proj(kv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        logits = (Q @ Kk.transpose(-2, -1)) / (self.dh ** 0.5)            # [B,h,Lq,Lk]
        logits = logits + self.gate_bias.view(1, self.h, 1, 1)
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        gate = torch.sigmoid(logits)                                      # independent [0,1] per key
        out = (gate @ V).transpose(1, 2).reshape(B, Lq, d)
        return self.out_proj(out), gate.mean(1)                           # [B,Lq,d], gate_headavg [B,Lq,Lk]


class DepthAttnBlock(nn.Module):
    """One depth-attention refinement block: the fused query token cross-attends over
    the K per-layer tokens at its own spatial position. attn out-projection AND the FFN
    output are zero-init -> exact no-op at init (residual around the query).

    attn_kind='softmax' (default): competitive nn.MultiheadAttention (simplex weights).
    attn_kind='gate': independent per-layer sigmoid gates (MultiheadGateAttention)."""

    def __init__(self, dim: int, n_heads: int, mlp_mult: int,
                 attn_kind: str = "softmax", K: int = 23):
        super().__init__()
        assert attn_kind in ("softmax", "gate"), attn_kind
        self.attn_kind = attn_kind
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        if attn_kind == "gate":
            self.attn = MultiheadGateAttention(dim, n_heads, bias_init=-math.log(K))
        else:
            self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, mlp_mult * dim), nn.GELU(),
                                 nn.Linear(mlp_mult * dim, dim))
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, q, kv, pad, return_attn=False):   # q [S,1,d]  kv [S,K,d]  pad [S,K]
        kvn = self.norm_kv(kv)
        qn = self.norm_q(q)
        if self.attn_kind == "gate":
            a, w = self.attn(qn, kvn, key_padding_mask=pad)              # w = head-avg gate [S,1,K]
        else:
            a, w = self.attn(qn, kvn, kvn, key_padding_mask=pad,
                             need_weights=return_attn, average_attn_weights=True)
        q = q + a
        q = q + self.ffn(self.norm_ffn(q))
        return (q, w) if return_attn else q             # w [S,1,K] head-avg weight/gate (probe)


class DepthAttnFusion(nn.Module):
    """fused = masked_mean + AttnBlocks(query=masked_mean, kv=per-layer tokens).

    Pure reshape: (K, B, N, d) -> (B*N, K, d) sequences on the DEPTH axis; dropped
    layers are key_padding_mask'ed out per sample. A learnable depth embedding tells
    attention WHICH layer each kv token came from (reaches the output only through the
    zero-init projections, so init identity is preserved)."""

    def __init__(self, dim: int, K: int, n_layers: int, n_heads: int, mlp_mult: int,
                 attn_kind: str = "softmax"):
        super().__init__()
        self.depth_pos = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [DepthAttnBlock(dim, n_heads, mlp_mult, attn_kind=attn_kind, K=K)
             for _ in range(n_layers)])

    def forward(self, z0, stk, mask, return_attn=False):     # z0 [B,N,d] stk [K,B,N,d] mask [B,K]
        K, B, N, d = stk.shape
        kv = stk.permute(1, 2, 0, 3).reshape(B * N, K, d) + self.depth_pos
        q = z0.reshape(B * N, 1, d)
        pad = (~mask.bool()).repeat_interleave(N, dim=0)          # [B*N, K], True = drop
        attns = []
        for blk in self.blocks:
            if return_attn:
                q, w = blk(q, kv, pad, return_attn=True)
                attns.append(w.reshape(B, N, K))                  # per-block [B,N,K] weights
            else:
                q = blk(q, kv, pad)
        out = q.reshape(B, N, d)
        return (out, attns) if return_attn else out


class DepthAttnCombine(nn.Module):
    """Depth-attention combine: the decoder-facing latent is the masked equal-weight
    mean over the kept layers PLUS a learned per-position depth-attention correction
    over those layers' tokens. Mirrors the MLSCombine interface so train_decoder.py /
    eval / stage-2 RAECombine work unchanged. Training masks are sampled INTERNALLY
    (stratified full / uniform-|S| / Bernoulli) unless an external `mask` is passed;
    at eval (or drop=False) the mask is all-ones -> deterministic full-feed latent."""

    def __init__(self,
                 layers,
                 p_drop: float = 0.3,
                 full_frac: float = 0.15,
                 uniform_frac: float = 0.0,
                 cls_surrogate: bool = False,
                 dim: int = 1024,
                 out_dim: int = 1024,
                 n_layers: int = 2,
                 n_heads: int = 8,
                 mlp_mult: int = 2,
                 attn_kind: str = "softmax"):
        super().__init__()
        assert dim == out_dim, "DepthAttnCombine is residual around the mean: dim must equal out_dim"
        self.layers = list(layers)
        self.K = len(self.layers)
        self.p_drop = p_drop                       # trainer's p_drop schedule hooks this
        self.full_frac = full_frac
        self.uniform_frac = uniform_frac
        self.cls_surrogate = cls_surrogate
        self.fusion = DepthAttnFusion(dim, self.K, n_layers, n_heads, mlp_mult,
                                      attn_kind=attn_kind)

    @property
    def has_params(self) -> bool:
        return True

    def forward(self, layer_tokens, idx=None, mask=None, return_attn=False) -> torch.Tensor:
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
        if return_attn:
            z, attns = self.fusion(z0, stk, mask, return_attn=True)
        else:
            z = self.fusion(z0, stk, mask)
        if self.cls_surrogate:
            # mask-GATED L_last token-mean: samples that DROP the last layer must not
            # get its token-mean added back. At full feed (eval) gate==1 -> raev2 add.
            gate = mask[:, -1].to(z.dtype).view(B, 1, 1)
            z = z + gate * stk[-1].mean(dim=1, keepdim=True)
        return (z, attns) if return_attn else z
