"""
splitting.py — Stratified k-fold train/val/test splits.

For each of ``N_FOLDS`` stratified folds, the dataset is split as follows:

* ``idx_test``  — one fold, ~1/N of the data, held out for unbiased evaluation;
* the remaining ~(N-1)/N is split (stratified) into:
  * ``idx_val``   — ``VAL_FRACTION`` of the non-test data, used for
                    threshold tuning by ``probe.fit_hyperparameters``;
  * ``idx_train`` — the rest, used for fitting the probe.

5-fold (default) gives us two practical benefits:

1. Per-fold test metrics averaged over 5 disjoint test splits are noticeably
   less variance-prone than a single 15% test split on 689 samples (which
   gives only ~103 test points — one swing in a handful of borderline
   examples moves accuracy by 1-2 pp).
2. ``solution.py`` re-fits the final probe (used for ``predictions.csv``) on
   the union of (train ∪ val) across all folds.  Under k-fold this union
   equals the entire labelled dataset, so the production probe sees every
   labelled example — no data is wasted on a held-out test split that the
   final submission would never benefit from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


N_FOLDS = 5
VAL_FRACTION = 0.15
RANDOM_STATE = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return a list of ``N_FOLDS`` stratified ``(train, val, test)`` splits.

    ``test_size`` and ``val_size`` are accepted for upstream-signature
    compatibility but are not used; the fold size is fixed by ``N_FOLDS``
    and the validation slice by ``VAL_FRACTION``.
    """
    del test_size, val_size

    y_int = y.astype(int)
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=random_state
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_train_val_idx, fold_test_idx in skf.split(np.zeros_like(y_int), y_int):
        idx_train, idx_val = train_test_split(
            fold_train_val_idx,
            test_size=VAL_FRACTION,
            random_state=random_state,
            stratify=y_int[fold_train_val_idx],
        )
        splits.append((idx_train, idx_val, fold_test_idx))

    return splits
