"""
ablation.py — Offline comparison of aggregation strategies + probe choices.

Run AFTER ``python run_local.py`` (or ``python solution.py`` with the env var
``SMILE_RAW_CACHE_PATH=features_raw_cache.npz`` set), which produces
``features_raw_cache.npz``.

The cache holds the raw last-token hidden state at every transformer layer
(plus the token-embedding layer at index 0) for every sample, in the same
order as ``solution.py`` extracts them (first 689 labelled rows, then 177
test rows).  Shape: ``(866, 25, 896)``.  This lets us try alternative
aggregation strategies and probe configurations without paying for another
LLM forward pass.

Outputs a Markdown table to stdout summarising fold-averaged accuracy / F1 /
AUROC on the test split for each (aggregation, probe) pair.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from splitting import split_data  # noqa: E402  (uses repo's improved splitting)


CACHE_PATH = os.path.join(ROOT, "features_raw_cache.npz")
DATA_PATH = os.path.join(ROOT, "data", "dataset.csv")


def load_cache_and_labels() -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(CACHE_PATH):
        sys.exit(
            f"missing {CACHE_PATH} — run `python run_local.py` first to "
            "produce the raw-feature cache."
        )
    raw = np.load(CACHE_PATH)["last_token_per_layer"]
    df = pd.read_csv(DATA_PATH)
    y = df["label"].astype(float).astype(int).to_numpy()
    n_train = len(y)
    if raw.shape[0] < n_train:
        sys.exit(
            f"raw cache has only {raw.shape[0]} samples, expected at least "
            f"{n_train} (train) — re-extract with caching enabled."
        )
    return raw[:n_train], y


# --- Aggregation strategies operating on (n_samples, n_layers, hidden_dim) ---

def agg_last_layer(raw: np.ndarray) -> np.ndarray:
    """Skeleton-style: last token of the final transformer layer."""
    return raw[:, -1, :]


def agg_mean_late(raw: np.ndarray, lo: int = 13, hi: int = 25) -> np.ndarray:
    """Mean of last-token hidden states across layers [lo, hi)."""
    return raw[:, lo:hi, :].mean(axis=1)


def agg_mean_all(raw: np.ndarray) -> np.ndarray:
    """Mean over every layer including embeddings."""
    return raw.mean(axis=1)


def agg_single_layer(raw: np.ndarray, layer: int) -> np.ndarray:
    return raw[:, layer, :]


def agg_concat_layers(raw: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    """Concatenate selected layers' last-token states → (n, k*hidden_dim)."""
    return np.concatenate([raw[:, k, :] for k in layers], axis=-1)


# --- Probes -----------------------------------------------------------------
#
# Probe signature: ``probe_fn(X_tr, y_tr, X_val, y_val, X_te) -> (pred, proba)``.
# When ``X_val is None`` or ``y_val is None`` the probe SHOULD NOT tune the
# decision threshold, so ablation numbers without val tuning are directly
# comparable across probes.

def probe_logreg(X_tr, y_tr, X_val, y_val, X_te):
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
    )
    clf.fit(scaler.transform(X_tr), y_tr)
    proba = clf.predict_proba(scaler.transform(X_te))[:, 1]
    threshold = _tune_threshold(
        clf.predict_proba(scaler.transform(X_val))[:, 1], y_val
    ) if X_val is not None else 0.5
    pred = (proba >= threshold).astype(int)
    return pred, proba


def _mlp_fit(X_tr, y_tr, hidden: int = 256, epochs: int = 200, lr: float = 1e-3,
             weight_decay: float = 0.0):
    torch.manual_seed(42)
    scaler = StandardScaler().fit(X_tr)
    X_tr_t = torch.from_numpy(scaler.transform(X_tr)).float()
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))

    net = nn.Sequential(
        nn.Linear(X_tr_t.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = net(X_tr_t).squeeze(-1)
        crit(logits, y_tr_t).backward()
        opt.step()
    net.eval()
    return net, scaler


def _mlp_predict(net, scaler, X):
    X_t = torch.from_numpy(scaler.transform(X)).float()
    with torch.no_grad():
        return torch.sigmoid(net(X_t).squeeze(-1)).numpy()


def probe_mlp(X_tr, y_tr, X_val, y_val, X_te):
    """Skeleton-style MLP (no weight decay, no threshold tune)."""
    net, scaler = _mlp_fit(X_tr, y_tr)
    proba = _mlp_predict(net, scaler, X_te)
    threshold = _tune_threshold(_mlp_predict(net, scaler, X_val), y_val) if X_val is not None else 0.5
    return (proba >= threshold).astype(int), proba


def probe_mlp_wd(X_tr, y_tr, X_val, y_val, X_te):
    """MLP with mild weight decay (5e-4) — regularises the 33k-parameter net."""
    net, scaler = _mlp_fit(X_tr, y_tr, weight_decay=5e-4)
    proba = _mlp_predict(net, scaler, X_te)
    threshold = _tune_threshold(_mlp_predict(net, scaler, X_val), y_val) if X_val is not None else 0.5
    return (proba >= threshold).astype(int), proba


def _tune_threshold(proba_val: np.ndarray, y_val: np.ndarray) -> float:
    """Mirror ``probe.fit_hyperparameters``: tune for F1."""
    candidates = np.unique(np.concatenate([proba_val, np.linspace(0.0, 1.0, 101)]))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_val, (proba_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# --- Eval -------------------------------------------------------------------

def kfold_metrics(
    features: np.ndarray,
    y: np.ndarray,
    probe_fn,
    use_val_for_threshold: bool = True,
) -> dict:
    splits = split_data(y)
    accs, f1s, aurocs = [], [], []
    for (idx_tr, idx_va, idx_te) in splits:
        if use_val_for_threshold and idx_va is not None:
            X_val = features[idx_va]
            y_val = y[idx_va]
        else:
            X_val = None
            y_val = None
        pred, proba = probe_fn(
            features[idx_tr], y[idx_tr], X_val, y_val, features[idx_te]
        )
        accs.append(accuracy_score(y[idx_te], pred))
        f1s.append(f1_score(y[idx_te], pred, zero_division=0))
        aurocs.append(roc_auc_score(y[idx_te], proba))
    return {
        "accuracy": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "auroc": float(np.mean(aurocs)),
    }


def main() -> None:
    raw, y = load_cache_and_labels()
    print(f"raw cache shape (train): {raw.shape}    labels: {y.shape}")

    # Single-layer sweep with the MLP probe to pick the best individual layer.
    print("\n== Single-layer sweep (MLP, threshold-tuned on val) ==")
    layer_rows = []
    for L in range(8, 25):
        feats = agg_single_layer(raw, L)
        m = kfold_metrics(feats, y, probe_mlp)
        layer_rows.append((L, m))
        print(
            f"  layer={L:>2}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
            f"auroc={m['auroc']:.4f}"
        )
    best_layer = max(layer_rows, key=lambda r: r[1]["auroc"])[0]
    print(f"  -> best single layer by AUROC: {best_layer}")

    configs = [
        ("Skeleton: last layer + MLP (no tune)",     agg_last_layer(raw),                              probe_mlp,    False),
        ("Skeleton: last layer + MLP (tuned)",       agg_last_layer(raw),                              probe_mlp,    True),
        ("Skeleton: last layer + MLP+wd (tuned)",    agg_last_layer(raw),                              probe_mlp_wd, True),
        ("Best single layer + MLP (tuned)",          agg_single_layer(raw, best_layer),                probe_mlp,    True),
        ("Mean(13..24) + MLP (tuned)",               agg_mean_late(raw, 13, 25),                       probe_mlp,    True),
        ("Mean(13..24) + MLP+wd (tuned)",            agg_mean_late(raw, 13, 25),                       probe_mlp_wd, True),
        ("Mean(13..24) + LogReg (tuned, was sub'd)", agg_mean_late(raw, 13, 25),                       probe_logreg, True),
        ("Concat{12,16,20,24} + MLP+wd (tuned)",     agg_concat_layers(raw, (12, 16, 20, 24)),         probe_mlp_wd, True),
        ("Mean(all 25 layers) + MLP+wd (tuned)",     agg_mean_all(raw),                                probe_mlp_wd, True),
    ]

    print("\n== Full aggregation x probe table ==")
    rows = []
    for name, feats, probe, use_val in configs:
        m = kfold_metrics(feats, y, probe, use_val_for_threshold=use_val)
        rows.append((name, feats.shape[1], m["accuracy"], m["f1"], m["auroc"]))
        print(
            f"  {name:46s}  dim={feats.shape[1]:5d}  "
            f"acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auroc={m['auroc']:.4f}"
        )

    print("\n| Configuration | Feature dim | Accuracy | F1 | AUROC |")
    print("|---|---:|---:|---:|---:|")
    for name, dim, acc, f1, auroc in rows:
        print(f"| {name} | {dim} | {acc:.4f} | {f1:.4f} | {auroc:.4f} |")


if __name__ == "__main__":
    main()
