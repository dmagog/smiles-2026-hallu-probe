"""
probe.py — Hallucination probe: ensemble of MLPs, one per layer-view.

The feature vector handed in by ``aggregation.py`` is the concatenation of
``len(ENSEMBLE_LAYERS)`` per-layer last-token hidden states (3 × 896 = 2688
for Qwen2.5-0.5B with the default layers ``(13, 23, 24)``).  At fit time the
probe slices it back into the original ``HIDDEN_DIM``-wide chunks and trains
one small MLP per chunk; at inference time it averages their predicted
probabilities.

The MLP architecture, optimiser, and class-imbalance handling mirror the
upstream skeleton (one 256-unit hidden layer, full-batch Adam, BCE with
``pos_weight = n_neg / n_pos``).  We add two changes:

1. ``weight_decay = 5e-4`` on Adam — small L2 regularisation that nudges
   AUROC up by ~1 pp without hurting accuracy (see SOLUTION.md → ablation).
2. ``torch.manual_seed(42)`` before each MLP build — makes the run
   deterministic across re-executions of ``solution.py``.

The decision threshold ``self._threshold`` is tuned to maximise F1 on a
validation split via ``fit_hyperparameters`` (called by ``evaluate.py`` when
a val split is available).  Without a val split (e.g. the final probe in
``solution.py`` that produces ``predictions.csv``) the threshold stays at
0.5, matching the skeleton's convention.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from aggregation import ENSEMBLE_LAYERS


HIDDEN_DIM = 896                       # Qwen2.5-0.5B
N_SUBPROBES = len(ENSEMBLE_LAYERS)
MLP_HIDDEN = 256
MLP_EPOCHS = 200
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 5e-4
SEED = 42


class HallucinationProbe(nn.Module):
    """Average-probability ensemble of ``N_SUBPROBES`` skeleton-style MLPs."""

    def __init__(self) -> None:
        super().__init__()
        self._nets: list[nn.Sequential] = []
        self._scalers: list[StandardScaler] = []
        self._threshold: float = 0.5

    def _slices(self, X: np.ndarray) -> list[np.ndarray]:
        expected = N_SUBPROBES * HIDDEN_DIM
        if X.shape[1] != expected:
            raise ValueError(
                f"HallucinationProbe expects feature_dim={expected} "
                f"(= {N_SUBPROBES} layers x {HIDDEN_DIM}); got {X.shape[1]}. "
                "Check aggregation.ENSEMBLE_LAYERS matches probe.N_SUBPROBES."
            )
        return [X[:, i * HIDDEN_DIM : (i + 1) * HIDDEN_DIM] for i in range(N_SUBPROBES)]

    def _build_mlp(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(HIDDEN_DIM, MLP_HIDDEN),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        y_f = y.astype(np.float32)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32
        )

        self._nets, self._scalers = [], []
        for chunk in self._slices(X):
            scaler = StandardScaler().fit(chunk)
            X_t = torch.from_numpy(scaler.transform(chunk)).float()
            y_t = torch.from_numpy(y_f)

            torch.manual_seed(SEED)
            net = self._build_mlp()
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.Adam(
                net.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY
            )

            net.train()
            for _ in range(MLP_EPOCHS):
                optimizer.zero_grad()
                logits = net(X_t).squeeze(-1)
                loss = criterion(logits, y_t)
                loss.backward()
                optimizer.step()
            net.eval()

            self._nets.append(net)
            self._scalers.append(scaler)

        self._threshold = 0.5
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average raw logits of the sub-probes for upstream-API compatibility.

        ``evaluate.py`` never calls this directly; it goes through
        ``predict`` / ``predict_proba``.
        """
        if not self._nets:
            raise RuntimeError("Probe has not been fit yet.")
        chunks = [x[:, i * HIDDEN_DIM : (i + 1) * HIDDEN_DIM] for i in range(N_SUBPROBES)]
        logits = [net(chunk).squeeze(-1) for net, chunk in zip(self._nets, chunks)]
        return torch.stack(logits, dim=0).mean(dim=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []
        for net, scaler, chunk in zip(self._nets, self._scalers, self._slices(X)):
            X_t = torch.from_numpy(scaler.transform(chunk)).float()
            with torch.no_grad():
                p = torch.sigmoid(net(X_t).squeeze(-1)).numpy()
            probas.append(p)
        prob_pos = np.mean(probas, axis=0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 101)])
        )

        best_threshold, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self
