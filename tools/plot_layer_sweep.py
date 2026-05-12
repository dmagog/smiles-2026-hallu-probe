"""
plot_layer_sweep.py — Re-run the single-layer probe sweep on the cached raw
features and save a publication-quality line plot to
``assets/layer_sweep.png``.

Requires matplotlib (install via `pip install matplotlib`).  The generated
PNG is checked into the repository so SOLUTION.md does not depend on it
being regenerated.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ablation import (  # noqa: E402
    agg_single_layer,
    kfold_metrics,
    load_cache_and_labels,
    probe_mlp,
)

OUT_PATH = os.path.join(ROOT, "assets", "layer_sweep.png")
LAYERS = list(range(0, 25))
HIGHLIGHT = (13, 23, 24)


def main() -> None:
    import matplotlib.pyplot as plt

    raw, y = load_cache_and_labels()
    accs, aurocs = [], []
    for L in LAYERS:
        m = kfold_metrics(agg_single_layer(raw, L), y, probe_mlp)
        accs.append(m["accuracy"])
        aurocs.append(m["auroc"])
        print(f"  layer {L:>2}  acc={m['accuracy']:.4f}  auroc={m['auroc']:.4f}")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(LAYERS, accs, marker="o", linewidth=1.7, color="#1f77b4",
            label="Test accuracy")
    ax.plot(LAYERS, aurocs, marker="s", linewidth=1.7, color="#d62728",
            label="Test AUROC")

    ymin = min(min(accs), min(aurocs)) - 0.012
    ymax = max(max(accs), max(aurocs)) + 0.025
    for L in HIGHLIGHT:
        ax.axvspan(L - 0.35, L + 0.35, color="#ffe699", alpha=0.55, zorder=0)

    ax.axhline(
        0.7010, color="#888888", linestyle="--", linewidth=1.0,
        label="Majority-class accuracy baseline (0.7010)",
    )

    # Annotate the highlighted layers above their AUROC marker.
    for L in HIGHLIGHT:
        ax.annotate(
            f"L{L}",
            xy=(L, aurocs[L]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#7a5b00",
            fontweight="bold",
        )

    ax.set_xlabel(
        "Transformer layer index (0 = token embeddings, 1..24 = transformer blocks)"
    )
    ax.set_ylabel("5-fold mean on held-out test partition")
    ax.set_title(
        "Per-layer single-MLP probe on Qwen2.5-0.5B last-token hidden state\n"
        "Shaded layers (13, 23, 24) make up the submitted ensemble"
    )
    ax.set_xticks(LAYERS)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
