#!/usr/bin/env python3
"""
Compare RAEv1 vs RAEv2 single-image overfitting results.

Reads metrics.json from both output dirs and produces:
  - loss_curves_comparison.png  : side-by-side loss curves
  - final_loss_comparison.png   : grouped bar chart
  - latent_grid_comparison.png  : PCA latent grids stacked
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_results(results_dir):
    """Load metrics.json for each width from an overfit results directory."""
    results = {}
    img_dir = os.path.join(results_dir, "images")
    if not os.path.exists(img_dir):
        return results
    for d_name in sorted(os.listdir(img_dir)):
        mpath = os.path.join(img_dir, d_name, "metrics.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            r = json.load(f)
        hs = r["hidden_size"]
        results[hs] = r
    return results


def smooth(losses, w=30):
    return np.convolve(losses, np.ones(w) / w, mode='valid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir",  required=True, help="RAEv1 overfit_results dir")
    parser.add_argument("--v2-dir",  required=True, help="RAEv2 overfit_results_v2 dir")
    parser.add_argument("--out-dir", default="comparison_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    v1 = load_results(args.v1_dir)
    v2 = load_results(args.v2_dir)

    if not v1 and not v2:
        print("No results found. Run both sweeps first.")
        return

    all_widths = sorted(set(list(v1.keys()) + list(v2.keys())))
    C1, C2 = 768, 1024

    # ── Figure 1: loss curves side by side ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    cmap = plt.cm.RdYlGn

    for ax, results, C, label in [
        (axes[0], v1, C1, "RAEv1 (DINOv2-B, C=768, velocity)"),
        (axes[1], v2, C2, "RAEv2 (DINOv3-L, C=1024, x-pred)"),
    ]:
        if not results:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label); continue

        widths = sorted(results.keys())
        n = len(widths)
        colors = {hs: cmap(0.1 + 0.8 * i / max(n-1, 1)) for i, hs in enumerate(widths)}

        for hs in widths:
            losses = results[hs]["all_losses"]
            s = smooth(losses)
            lbl = f"d={hs} ({'≥' if hs >= C else '<'}C)"
            ax.semilogy(s, label=lbl, color=colors[hs], lw=1.8)

        ax.axvline(x=0, color='none')
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Flow Matching Loss (log)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("RAEv1 vs RAEv2: Single-Image Overfitting Loss Curves", fontsize=13)
    plt.tight_layout()
    out = os.path.join(args.out_dir, "loss_curves_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    # ── Figure 2: final loss grouped bar chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_widths))
    w = 0.35

    def bar_color(hs, C):
        return '#2ca02c' if hs >= C else '#d62728'

    v1_vals = [v1[hs]["final_loss"] if hs in v1 else np.nan for hs in all_widths]
    v2_vals = [v2[hs]["final_loss"] if hs in v2 else np.nan for hs in all_widths]
    v1_colors = [bar_color(hs, C1) for hs in all_widths]
    v2_colors = [bar_color(hs, C2) for hs in all_widths]

    bars1 = ax.bar(x - w/2, v1_vals, w, color=v1_colors, label="RAEv1 (C=768)",  alpha=0.85)
    bars2 = ax.bar(x + w/2, v2_vals, w, color=v2_colors, label="RAEv2 (C=1024)", alpha=0.85,
                   hatch='//')

    # threshold lines
    ax.axhline(0.1, color='gray', ls='--', lw=1, alpha=0.6, label="convergence threshold 0.1")

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([str(hs) for hs in all_widths])
    ax.set_xlabel("DiT hidden_size (d)")
    ax.set_ylabel("Final Loss (last 100 steps, log)")
    ax.set_title("Final Loss: RAEv1 (C=768) vs RAEv2 (C=1024)\n"
                 "Green = converged (d≥C), Red = not converged (d<C)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # annotations
    for bar, val in zip(bars1, v1_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
                    f'{val:.1e}', ha='center', va='bottom', fontsize=7, rotation=45)
    for bar, val in zip(bars2, v2_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
                    f'{val:.1e}', ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()
    out = os.path.join(args.out_dir, "final_loss_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    # ── Figure 3: per-width overlay (RAEv1 vs RAEv2 same width) ───────────────
    n_widths = len(all_widths)
    fig, axes = plt.subplots(2, n_widths, figsize=(3 * n_widths, 6),
                              gridspec_kw={'hspace': 0.05, 'wspace': 0.15})
    if n_widths == 1:
        axes = axes.reshape(2, 1)

    for col, hs in enumerate(all_widths):
        for row, (results, C, label, color) in enumerate([
            (v1, C1, "RAEv1", "#1f77b4"),
            (v2, C2, "RAEv2", "#ff7f0e"),
        ]):
            ax = axes[row][col]
            if hs in results:
                losses = results[hs]["all_losses"]
                s = smooth(losses)
                ax.semilogy(s, color=color, lw=1.5)
                fl = results[hs]["final_loss"]
                ax.set_title(f"d={hs}\n{label} {'✓' if fl<0.1 else '✗'}\n{fl:.1e}",
                             fontsize=8,
                             color='#2ca02c' if fl < 0.1 else '#d62728')
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 0:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Steps", fontsize=8)

    plt.suptitle("Per-Width Comparison: RAEv1 (C=768) vs RAEv2 (C=1024)", fontsize=12, y=1.02)
    out = os.path.join(args.out_dir, "per_width_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    # ── text summary ──────────────────────────────────────────────────────────
    print("\n── Comparison Summary ────────────────────────────────────────")
    print(f"{'width':>8}  {'RAEv1 (C=768)':>20}  {'RAEv2 (C=1024)':>20}")
    print("-" * 54)
    for hs in all_widths:
        v1s = f"{v1[hs]['final_loss']:.4e} {'✓' if v1[hs]['converged'] else '✗'}" if hs in v1 else "N/A"
        v2s = f"{v2[hs]['final_loss']:.4e} {'✓' if v2[hs]['converged'] else '✗'}" if hs in v2 else "N/A"
        print(f"  d={hs:5d}   {v1s:>20}   {v2s:>20}")


if __name__ == '__main__':
    main()
