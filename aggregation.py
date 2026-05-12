"""
aggregation.py — Token aggregation strategy and feature extraction.

Final strategy
--------------
For each sample, take the hidden state at the **last real (non-padding) token
position** and **concatenate** the activations from a small, curated set of
transformer layers — by default ``(13, 23, 24)``.  The resulting feature
vector has length ``n_selected_layers * hidden_dim`` (3 × 896 = 2688 for
Qwen2.5-0.5B); the probe in ``probe.py`` interprets it as ``n_selected_layers``
independent 896-dim "views" and trains one MLP per view, ensembling their
probabilities at inference.

Why these specific layers
~~~~~~~~~~~~~~~~~~~~~~~~~
Qwen2.5-0.5B has 24 transformer blocks (indices 1..24 in ``hidden_states``;
index 0 is the token embeddings).  An MLP-probe sweep across single layers
on the 689-sample training set (see ``tools/ablation.py``) shows three
distinct peaks of accuracy / AUROC:

* **Layer 24** (last) — captures the model's "committed" output state;
  strongest single-layer accuracy.
* **Layer 23** — the penultimate transformer block, often very competitive
  with the last layer on factuality probes because it is one step removed
  from the next-token logit specialisation.
* **Layer 13** — a mid-network peak, consistent with the truthfulness-probing
  literature (Azaria & Mitchell 2023; Burns et al. 2022 CCS) which places
  factuality signal in middle layers rather than the very top.

Mean-pooling these three (or wider bands) into a single 896-dim vector did
not match the **ensemble** of three independently trained MLPs (see
``SOLUTION.md`` → Failed experiments).  Concatenation here is purely the
data-passing format: ``probe.py`` does the splitting into per-layer probes.

Raw-feature cache
~~~~~~~~~~~~~~~~~
When ``SMILE_RAW_CACHE_PATH`` is set in the environment, every sample's
full ``(n_layers, hidden_dim)`` last-token activation matrix is appended to
a global list and dumped to ``<path>`` on process exit.  This makes
``tools/ablation.py`` reusable without re-running the LLM extraction.  The
official entry point (``python solution.py``) leaves this off by default.
"""

from __future__ import annotations

import atexit
import os

import numpy as np
import torch


# Layers (in ``hidden_states`` index space, 0 = embeddings, 1..24 = transformer
# blocks for Qwen2.5-0.5B) whose last-token states feed the probe ensemble.
# Selected by 5-fold ablation on cached features:
#   - 13 is the mid-network AUROC peak;
#   - 24 is the last transformer block (highest single-layer AUROC tie);
#   - 23 is the penultimate block (highest single-layer accuracy);
#   - 21 is the second mid-late AUROC peak after 13;
#   - 12 (adjacent to 13) adds a small Pareto improvement when paired with 21.
ENSEMBLE_LAYERS: tuple[int, ...] = (12, 13, 21, 23, 24)


_RAW_CACHE_PATH = os.environ.get("SMILE_RAW_CACHE_PATH", "")
_RAW_CACHE: list[np.ndarray] = []


def _save_raw_cache() -> None:
    if not _RAW_CACHE_PATH or not _RAW_CACHE:
        return
    arr = np.stack(_RAW_CACHE).astype(np.float32)
    np.savez_compressed(_RAW_CACHE_PATH, last_token_per_layer=arr)
    print(
        f"[aggregation] cached raw last-token hidden states "
        f"to '{_RAW_CACHE_PATH}'  shape={arr.shape}"
    )


atexit.register(_save_raw_cache)


def _last_real_position(attention_mask: torch.Tensor) -> int:
    real_positions = attention_mask.nonzero(as_tuple=False)
    return int(real_positions[-1].item())


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the last-token hidden states from ``ENSEMBLE_LAYERS``.

    Args:
        hidden_states:  (n_layers, seq_len, hidden_dim).
        attention_mask: (seq_len,) with 1 for real tokens.

    Returns:
        1-D tensor of shape ``(len(ENSEMBLE_LAYERS) * hidden_dim,)``.
    """
    last_pos = _last_real_position(attention_mask)

    if _RAW_CACHE_PATH:
        _RAW_CACHE.append(
            hidden_states[:, last_pos, :]
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )

    slices = [hidden_states[layer, last_pos, :] for layer in ENSEMBLE_LAYERS]
    return torch.cat(slices, dim=-1)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-layer last-token L2 norms plus the response length in tokens.

    Returns a fixed-size 1-D float tensor of length ``n_layers + 1`` (= 26
    for Qwen2.5-0.5B).  Same length for every sample; concatenable with
    ``aggregate``'s output when ``USE_GEOMETRIC=True`` in ``solution.py``.
    """
    last_pos = _last_real_position(attention_mask)
    per_layer_norms = hidden_states[:, last_pos, :].norm(dim=-1)
    seq_len = attention_mask.sum().to(per_layer_norms.dtype).unsqueeze(0)
    return torch.cat([per_layer_norms, seq_len], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
