"""MLSCombine: one configurable module that collapses every stage-1 MLS-decoder
combine variant into a single nn.Module.

The frozen DINOv3 encoder produces K per-layer patch-token tensors; MLSCombine
turns them into the latent z [B, N, out_dim] the ViT decoder consumes. All the
forked train_decoder_mls*.py scripts differ ONLY in this combine — here it is
parameterized by four knobs:

  weighting      mean | dropmean | softgate    how the K layers are mixed
  p_drop         float                          per-sample layer dropout prob
  cls_surrogate  bool                           add L_last token-mean (raev2)
  projector      none | ln | bn                 per-token residual MLP after mix

Variant map (the three kept experiments + the legacy ones):
  raev2 K=23           weighting=mean,     projector=none, cls_surrogate=true
  Random Drop+Decoder  weighting=dropmean, projector=none, cls_surrogate=false
  Random Drop+MLP+SIG  weighting=dropmean, projector=bn,   cls_surrogate=false
  nogate (legacy)      weighting=mean,     projector=ln
  softgate (legacy)    weighting=softgate, projector=ln|none

forward(layer_tokens, idx=None):
  layer_tokens : list of K tensors, each [B, N, dim]
  idx          : optional list of positions into the FULL layer set; restricts
                 the combine to that subset (renormalized), powering the LOO/solo
                 probes from a single trained model with no retraining.
  returns      : z [B, N, out_dim]

DDP note: with weighting in {mean, dropmean} and projector=none the module has
ZERO parameters. DDP errors on a param-free module, so the trainer must call it
directly (not wrap it in DDP) when has_params is False.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLSCombine(nn.Module):
    def __init__(self,
                 layers,
                 weighting: str = "dropmean",
                 p_drop: float = 0.3,
                 cls_surrogate: bool = False,
                 projector: str = "none",
                 dim: int = 1024,
                 out_dim: int = 1024,
                 mult: int = 4):
        super().__init__()
        assert weighting in ("mean", "dropmean", "softgate"), weighting
        assert projector in ("none", "ln", "bn"), projector
        self.layers = list(layers)
        self.K = len(self.layers)
        self.weighting = weighting
        self.p_drop = p_drop
        self.cls_surrogate = cls_surrogate
        self.projector = projector

        if weighting == "softgate":
            self.gate = nn.Parameter(torch.zeros(self.K))   # softmax logits, init uniform

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

        if self.training and self.weighting in ("dropmean", "softgate") and self.p_drop > 0:
            keep = torch.rand(K, B, device=stk.device) > self.p_drop
            dead = ~keep.any(0)
            if dead.any():                                  # always keep >= 1 layer
                keep[torch.randint(K, (int(dead.sum()),), device=stk.device), dead] = True
            w = keep.to(stk.dtype)
            if gate_w is not None:
                w = w * gate_w.view(K, 1)
            w = w / w.sum(0, keepdim=True).clamp_min(1e-6)
            return (w.view(K, B, 1, 1) * stk).sum(0)

        if gate_w is not None:
            return (gate_w.view(K, 1, 1, 1) * stk).sum(0)
        return stk.mean(0)                                   # mean / dropmean eval

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
