"""Numerical sanity checks for eval.latent_metrics. Run on the GPU box:
    /opt/conda/envs/rae/bin/python tests/test_latent_metrics.py [cpu|cuda]
Analytic ground truth: for two Gaussians differing only in mean by delta,
FD = ||delta||^2; identical distributions give FD ~ 0 (up to estimation bias).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from eval.latent_metrics import (compute_all, fd_from_features, frechet_distance,
                                 per_position_fd, per_token_fd, pooled_fd,
                                 random_projection, rbf_mmd2)

device = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)
N, C, H, W = 2000, 64, 8, 8


def check(name, val, lo, hi):
    ok = lo <= val <= hi
    print(f"  {'OK ' if ok else 'FAIL'} {name} = {val:.5f} (expected [{lo}, {hi}])")
    return ok


all_ok = True

# 1. analytic FD: mean shift delta in every dim -> FD = C * delta^2
delta = 0.5
mu1 = torch.zeros(C, dtype=torch.float64)
mu2 = torch.full((C,), delta, dtype=torch.float64)
sig = torch.eye(C, dtype=torch.float64)
fd = frechet_distance(mu1, sig, mu2, sig)
all_ok &= check("analytic FD (mean shift)", fd, C * delta**2 - 1e-6, C * delta**2 + 1e-6)

# 2. analytic FD: variance scale s -> FD = C * (1 - sqrt(s))^2
s = 4.0
fd = frechet_distance(mu1, sig, mu1, s * sig)
all_ok &= check("analytic FD (var scale)", fd, C * (1 - s**0.5)**2 - 1e-6, C * (1 - s**0.5)**2 + 1e-6)

# 3. same-distribution latents -> everything near zero
z1 = rng.standard_normal((N, C, H, W)).astype(np.float16)
z2 = rng.standard_normal((N, C, H, W)).astype(np.float16)
all_ok &= check("per_token_fd null", per_token_fd(z1, z2, device), 0, 0.05)
all_ok &= check("pooled_fd null", pooled_fd(z1, z2, device), 0, 3.0)   # N=2000 vs dim=64: estimation bias dominates
hm = per_position_fd(z1, z2, device)
all_ok &= check("per_position_fd null (mean)", float(hm.mean()), 0, 5.0)
f1 = random_projection(z1, 256, device=device)
f2 = random_projection(z2, 256, device=device)
all_ok &= check("rp_fd null", fd_from_features(f1, f2, device), 0, 25.0)
all_ok &= check("rp_mmd2 null", rbf_mmd2(f1, f2, device=device), -1e-3, 1e-3)

# 4. shifted latents -> per-token FD ~ C * shift^2, MMD clearly positive
z3 = (rng.standard_normal((N, C, H, W)) + 0.3).astype(np.float16)
ptfd = per_token_fd(z1, z3, device)
all_ok &= check("per_token_fd shifted", ptfd, C * 0.09 * 0.8, C * 0.09 * 1.2)
f3 = random_projection(z3, 256, device=device)
all_ok &= check("rp_mmd2 shifted", rbf_mmd2(f1, f3, device=device), 1e-3, 10)

# 5. spatial-scramble blindness/sensitivity: permute positions consistently.
#    per-token FD stays ~0 (blind); rp_fd should rise above its null value.
perm = rng.permutation(H * W)
z4 = z2.reshape(N, C, H * W)[:, :, perm].reshape(N, C, H, W)
all_ok &= check("per_token_fd scrambled (blind)", per_token_fd(z1, z4, device), 0, 0.05)
f4 = random_projection(np.ascontiguousarray(z4), 256, device=device)
print(f"  info rp_fd scrambled = {fd_from_features(f1, f4, device):.3f} "
      f"(iid null has no spatial correlation, so expect ~same as rp_fd null; "
      f"real latents WOULD separate here)")

# 6. compute_all smoke test
m, hm = compute_all(z1[:500], z3[:500], device=device, proj_dim=128, mmd_max_samples=500)
print(f"  OK  compute_all keys: {sorted(m.keys())}, heatmap {hm.shape}")

print("ALL OK" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
