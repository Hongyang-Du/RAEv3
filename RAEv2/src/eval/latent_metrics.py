"""Latent-space distribution metrics for stage-2 evaluation.

All metrics compare two latent sets in the normalized latent space the DiT is
trained in (rae.encode output / ODE sampler output), shaped [N, C, H, W].
Inputs are numpy arrays (fp16 dumps are fine); statistics are accumulated on
`device` in float64.

What each metric can and cannot see:
  per_token_fd     marginal token distribution, all positions pooled into one
                   bag (N*H*W samples of dim C). Blind to spatial structure.
  per_position_fd  one FD per spatial location -> [H, W] heatmap (N samples of
                   dim C at each position). Localizes failures; still marginal.
  pooled_fd        spatial mean-pool -> one C-dim vector per image. Image-level
                   semantic view, blind to local detail.
  rp_fd / rbf_mmd2 fixed-seed Gaussian projection of the full flattened latent
                   (C*H*W -> proj_dim). The only ones that see cross-token
                   joint structure.

FD assumes Gaussianity (two moments only); report it together with the MMD so
a moments-only match cannot pass silently.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------- Frechet ---

def _sqrt_trace_of_product(sigma1: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """tr((S1^{1/2} S2 S1^{1/2})^{1/2}) via two symmetric eigendecompositions."""
    ev1, u1 = torch.linalg.eigh(sigma1)
    sq1 = (u1 * ev1.clamp(min=0).sqrt()) @ u1.mT
    m = sq1 @ sigma2 @ sq1
    ev = torch.linalg.eigvalsh((m + m.mT) / 2)
    return ev.clamp(min=0).sqrt().sum()


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    diff = mu1 - mu2
    fd = diff.dot(diff) + sigma1.trace() + sigma2.trace() \
        - 2 * _sqrt_trace_of_product(sigma1, sigma2)
    return float(fd.clamp(min=0))


def _stats_from_chunks(chunks, dim: int, device, eps: float = 1e-6):
    """Running mean/covariance in float64 over an iterable of [n, dim] tensors."""
    n = 0
    s = torch.zeros(dim, dtype=torch.float64, device=device)
    ss = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    for c in chunks:
        c = c.to(device=device, dtype=torch.float64)
        n += c.shape[0]
        s += c.sum(0)
        ss += c.mT @ c
    mu = s / n
    sigma = (ss - n * torch.outer(mu, mu)) / (n - 1)
    sigma += eps * torch.eye(dim, dtype=torch.float64, device=device)
    return mu, sigma


def fd_from_features(f1: torch.Tensor, f2: torch.Tensor, device="cuda", eps=1e-6) -> float:
    """FD between two [N, D] feature sets."""
    mu1, s1 = _stats_from_chunks([f1], f1.shape[1], device, eps)
    mu2, s2 = _stats_from_chunks([f2], f2.shape[1], device, eps)
    return frechet_distance(mu1, s1, mu2, s2)


# ------------------------------------------------------------ FD variants ---

def _token_chunks(z: np.ndarray, chunk: int, device):
    for i in range(0, z.shape[0], chunk):
        c = torch.from_numpy(z[i:i + chunk]).to(device=device, dtype=torch.float64)
        b, ch, h, w = c.shape
        yield c.permute(0, 2, 3, 1).reshape(b * h * w, ch)


def per_token_fd(z1: np.ndarray, z2: np.ndarray, device="cuda", chunk=256) -> float:
    """All spatial tokens pooled into one bag of C-dim samples."""
    c = z1.shape[1]
    mu1, s1 = _stats_from_chunks(_token_chunks(z1, chunk, device), c, device)
    mu2, s2 = _stats_from_chunks(_token_chunks(z2, chunk, device), c, device)
    return frechet_distance(mu1, s1, mu2, s2)


def per_position_fd(z1: np.ndarray, z2: np.ndarray, device="cuda") -> np.ndarray:
    """One FD per spatial location. Returns a [H, W] float array."""
    _, c, h, w = z1.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            x1 = torch.from_numpy(np.ascontiguousarray(z1[:, :, i, j]))
            x2 = torch.from_numpy(np.ascontiguousarray(z2[:, :, i, j]))
            mu1, s1 = _stats_from_chunks([x1], c, device)
            mu2, s2 = _stats_from_chunks([x2], c, device)
            out[i, j] = frechet_distance(mu1, s1, mu2, s2)
    return out


def pooled_fd(z1: np.ndarray, z2: np.ndarray, device="cuda", chunk=512) -> float:
    """Spatial mean-pool -> one C-dim vector per image, then FD."""
    def chunks(z):
        for i in range(0, z.shape[0], chunk):
            c = torch.from_numpy(z[i:i + chunk]).to(device=device, dtype=torch.float64)
            yield c.mean(dim=(2, 3))
    c = z1.shape[1]
    mu1, s1 = _stats_from_chunks(chunks(z1), c, device)
    mu2, s2 = _stats_from_chunks(chunks(z2), c, device)
    return frechet_distance(mu1, s1, mu2, s2)


# ------------------------------------------------- random projection + MMD ---

def random_projection(z: np.ndarray, proj_dim=2048, seed=1234, device="cuda",
                      dim_chunk=16384) -> torch.Tensor:
    """Project flattened latents [N, C*H*W] -> [N, proj_dim] with a fixed-seed
    Gaussian matrix (regenerated per dim-chunk, so real and gen sets get the
    identical matrix as long as seed and shapes match). Scaled by 1/sqrt(C*H*W)
    so projected features keep roughly the per-dim scale of the input latents.
    Like FID, the downstream FD/MMD are biased in N — only compare runs that
    use the same sample count, proj_dim and seed. Returns fp32 CPU tensor."""
    zf = z.reshape(z.shape[0], -1)
    out = torch.zeros(z.shape[0], proj_dim, dtype=torch.float32, device=device)
    for k, j in enumerate(range(0, zf.shape[1], dim_chunk)):
        blk = np.ascontiguousarray(zf[:, j:j + dim_chunk])
        x = torch.from_numpy(blk).to(device=device, dtype=torch.float32)
        g = torch.Generator(device=device).manual_seed(seed + 7919 * k)
        p = torch.randn(x.shape[1], proj_dim, generator=g, device=device,
                        dtype=torch.float32)
        out += x @ p
    return (out / math.sqrt(zf.shape[1])).cpu()


def rbf_mmd2(f1: torch.Tensor, f2: torch.Tensor, max_samples=10000, seed=0,
             device="cuda", block=4096) -> float:
    """Unbiased squared MMD with an RBF kernel, bandwidth by median heuristic
    (sigma^2 = median squared distance / 2, computed on a joint subsample)."""
    g = torch.Generator().manual_seed(seed)

    def subsample(f):
        if f.shape[0] > max_samples:
            idx = torch.randperm(f.shape[0], generator=g)[:max_samples]
            f = f[idx]
        return f.to(device=device, dtype=torch.float32)

    x, y = subsample(f1), subsample(f2)

    joint = torch.cat([x[:1024], y[:1024]], dim=0)
    med_sq = torch.cdist(joint, joint).square().flatten()
    med_sq = med_sq[med_sq > 0].median()

    def kernel_sum(a, b, skip_diag):
        total = torch.zeros((), dtype=torch.float64, device=device)
        for i in range(0, a.shape[0], block):
            ai = a[i:i + block]
            for j in range(0, b.shape[0], block):
                d2 = torch.cdist(ai, b[j:j + block]).square()
                k = torch.exp(-d2 / med_sq).double()
                if skip_diag and i == j:
                    k -= torch.eye(k.shape[0], k.shape[1], dtype=torch.float64,
                                   device=device)
                total += k.sum()
        return total

    m, n = x.shape[0], y.shape[0]
    mmd2 = kernel_sum(x, x, True) / (m * (m - 1)) \
        + kernel_sum(y, y, True) / (n * (n - 1)) \
        - 2 * kernel_sum(x, y, False) / (m * n)
    return float(mmd2)


# ------------------------------------------------------------------ driver ---

def compute_all(z_real: np.ndarray, z_gen: np.ndarray, device="cuda",
                proj_dim=2048, proj_seed=1234, mmd_max_samples=10000):
    """All latent metrics between real and generated latents [N, C, H, W].
    Returns (metrics dict, per-position FD heatmap [H, W])."""
    assert z_real.shape[1:] == z_gen.shape[1:], \
        f"latent shape mismatch: {z_real.shape} vs {z_gen.shape}"
    heatmap = per_position_fd(z_real, z_gen, device)
    f_real = random_projection(z_real, proj_dim, proj_seed, device)
    f_gen = random_projection(z_gen, proj_dim, proj_seed, device)
    metrics = {
        "per_token_fd": per_token_fd(z_real, z_gen, device),
        "per_position_fd_mean": float(heatmap.mean()),
        "per_position_fd_max": float(heatmap.max()),
        "pooled_fd": pooled_fd(z_real, z_gen, device),
        "rp_fd": fd_from_features(f_real, f_gen, device),
        "rp_rbf_mmd2": rbf_mmd2(f_real, f_gen, mmd_max_samples, device=device),
        "proj_dim": proj_dim,
        "num_real": int(z_real.shape[0]),
        "num_gen": int(z_gen.shape[0]),
    }
    return metrics, heatmap
