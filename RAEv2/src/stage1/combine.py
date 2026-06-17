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
                 topk: int = 0):
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

    def forward(self, layer_tokens, idx=None) -> torch.Tensor:
        stk = torch.stack(layer_tokens, dim=0)              # [K, B, N, dim]
        z0 = self._mix(stk, idx)                            # [B, N, dim]
        if self.cls_surrogate:
            z0 = z0 + stk[-1].mean(dim=1, keepdim=True)     # raev2 L_last token-mean (fixed)

        if self.projector == "none":
            return z0
        if self.projector == "ln":
            return self.skip(z0) + self.ffn(self.norm(z0))
        # bn
        b, n, _ = z0.shape
        h = self.fc1(z0)
        h = self.bn(h.reshape(b * n, -1).float()).reshape(b, n, -1).to(h.dtype)
        return self.skip(z0) + self.fc2(F.gelu(h))
