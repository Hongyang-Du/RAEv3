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
            alpha = (F.softplus(self.log_alpha) + self.alpha_eps).float()   # [K] > 0
            base_p = max(float(self.p_drop), 1e-6)          # mean drop rate (>0 keeps alpha in graph)
            # per-sample simplex allocation of the drop budget; rsample -> grad flows to alpha
            pi = torch.distributions.Dirichlet(alpha.expand(B, K)).rsample()  # [B, K], rows sum to 1
            # simplex -> per-layer drop rate; mean_k p == base_p (mean pi == 1/K), clamp to keep valid
            p = (pi * K * base_p).clamp(0.0, self.p_max)    # [B, K]
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
