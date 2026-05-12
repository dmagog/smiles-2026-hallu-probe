"""
finalize.py — Regenerate ``results.json`` and ``predictions.csv`` from the
cached raw last-token hidden states (``features_raw_cache.npz``), without
re-running the LLM extraction.

The cache holds the per-layer last-token activation for every dataset row
(first 689 = labelled train rows from ``data/dataset.csv``, last 177 =
unlabelled test rows from ``data/test.csv``) in the same order
``solution.py`` produced them.  This script computes the
``aggregation.ENSEMBLE_LAYERS``-concatenated features from that cache and
runs the same evaluation pipeline as ``solution.py`` end-to-end with the
current ``aggregation.py`` / ``probe.py`` / ``splitting.py``.

Result: ``results.json`` and ``predictions.csv`` that are bit-for-bit
equivalent (up to numerical noise from the StandardScaler refit and the
deterministic torch seed) to what ``python solution.py`` would produce on
the same machine — but in ~30 seconds instead of ~30 minutes.

Intended for quick iteration on probe / aggregation choices.  The grader
should still use ``python solution.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aggregation import ENSEMBLE_LAYERS  # noqa: E402
from evaluate import (  # noqa: E402
    print_summary,
    run_evaluation,
    save_predictions,
    save_results,
)
from probe import HallucinationProbe  # noqa: E402
from splitting import split_data  # noqa: E402

CACHE_PATH = os.path.join(ROOT, "features_raw_cache.npz")
TRAIN_PATH = os.path.join(ROOT, "data", "dataset.csv")
TEST_PATH = os.path.join(ROOT, "data", "test.csv")
RESULTS_PATH = os.path.join(ROOT, "results.json")
PREDICTIONS_PATH = os.path.join(ROOT, "predictions.csv")


def main() -> None:
    if not os.path.exists(CACHE_PATH):
        sys.exit(
            f"missing {CACHE_PATH} — run `python run_local.py` first to "
            "produce the raw-feature cache."
        )

    raw = np.load(CACHE_PATH)["last_token_per_layer"]
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    n_train = len(train_df)
    n_test = len(test_df)
    if raw.shape[0] != n_train + n_test:
        sys.exit(
            f"raw cache has {raw.shape[0]} samples; expected "
            f"{n_train + n_test} ({n_train} train + {n_test} test). "
            "Re-extract from a clean run_local.py session."
        )

    print(f"Cache shape: {raw.shape}   layers in ensemble: {ENSEMBLE_LAYERS}")

    layer_slices_train = [raw[:n_train, L, :] for L in ENSEMBLE_LAYERS]
    layer_slices_test = [raw[n_train:, L, :] for L in ENSEMBLE_LAYERS]
    X = np.concatenate(layer_slices_train, axis=-1).astype(np.float32)
    X_test = np.concatenate(layer_slices_test, axis=-1).astype(np.float32)
    y = train_df["label"].astype(float).astype(int).to_numpy()
    print(f"Feature matrix X: {X.shape}   X_test: {X_test.shape}")

    splits = split_data(y, train_df)
    print(f"Splits: {len(splits)} fold(s)")

    fold_results = run_evaluation(splits, X, y, HallucinationProbe)
    print_summary(fold_results, X.shape[1], len(X), extract_time=0.0)
    save_results(fold_results, X.shape[1], len(X), 0.0, RESULTS_PATH)

    idx_non_test = np.unique(
        np.concatenate(
            [
                np.concatenate([idx_tr, idx_va]) if idx_va is not None else idx_tr
                for idx_tr, idx_va, _ in splits
            ]
        )
    )
    final_probe = HallucinationProbe()
    final_probe.fit(X[idx_non_test], y[idx_non_test])
    save_predictions(final_probe, X_test, list(test_df.index), PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
