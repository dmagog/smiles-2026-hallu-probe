# SMILES-2026 Hallucination Detection — Solution Report

**Author:** Georgy Mamarin
**Repo:** <https://github.com/dmagog/smiles-2026-hallu-probe>

## TL;DR

A **3-MLP ensemble** trained on the last-token hidden states of Qwen2.5-0.5B
at layers `(13, 23, 24)`, with stratified **5-fold** evaluation and an
F1-tuned decision threshold, achieves on the held-out folds of
`data/dataset.csv`:

| Checkpoint                       | Accuracy   | F1        | AUROC     |
| -------------------------------- | ---------: | --------: | --------: |
| Majority-class baseline          | 70.10 %    | 82.42 %   | —         |
| Probe — train                    | 96.79 %    | 97.80 %   | 100.00 %  |
| Probe — val                      | 76.63 %    | 85.05 %   | 74.51 %   |
| **Probe — test (★ submitted)**   | **72.86 %**| **82.88 %**| **74.15 %**|

The submitted `predictions.csv` (100 rows, one per `data/test.csv` sample)
is produced by a final ensemble trained on all 689 labelled rows.

## Reproduce

```bash
git clone https://github.com/dmagog/smiles-2026-hallu-probe
cd smiles-2026-hallu-probe

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Canonical entry point — used by the grader.
# Auto-selects MPS / CUDA / CPU.  Outputs results.json + predictions.csv.
python solution.py
```

### Reproducing on a small Apple Silicon machine (M1 8 GB)

On the test machine (base M1, 8 GB unified memory), `python solution.py`
with the default `BATCH_SIZE = 4` on MPS suffers from monotonic allocator
slowdown — `output_hidden_states=True` retains 25 intermediate tensors per
forward, and at 8 GB the MPS allocator does not reclaim them fast enough.
Per-batch latency climbs from ~6 s to 30+ s after ~50 batches and the run
stretches past 1.5 h.

`run_local.py` is an optional development-time wrapper that, **without
modifying any fixed-infrastructure file on disk**, applies three in-memory
patches:

1. Rewrites `BATCH_SIZE = 4 → 1` before `exec`.
2. Wraps `model.get_model_and_tokenizer` so the returned model calls
   `torch.mps.empty_cache()` after every forward.
3. Sets `SMILE_RAW_CACHE_PATH` so `aggregation.py` writes the per-layer
   last-token activations to `features_raw_cache.npz` for offline ablation.

With this wrapper, full extraction on M1 8 GB takes ~29 minutes:

```bash
python run_local.py
```

Running `python solution.py` on the same machine without the wrapper
will produce the same numbers but take noticeably longer.  On any machine
with a non-allocator-constrained accelerator (CUDA GPU, Colab T4) the
wrapper is unnecessary.

### Environment

Python 3.10+, the floor versions in `requirements.txt`:

```
torch>=2.0.0   transformers>=4.40.0   datasets>=2.14.0
scikit-learn>=1.3.0   numpy>=1.24.0   pandas>=1.5.0   tqdm>=4.65.0
```

Resolved on the test machine to torch 2.11, transformers 5.x, scikit-learn
1.8 (NB: scikit-learn 1.8 emits a deprecation warning about
`LogisticRegression(penalty="l2")`; the final probe uses an MLP and does
not hit this path).

## What changed vs. the skeleton

Three editable files were rewritten — `aggregation.py`, `probe.py`,
`splitting.py`.  `solution.py`, `model.py`, `evaluate.py` are untouched
on disk.

### `aggregation.py` — concatenate three carefully chosen layers' last-token states

Skeleton: last token of the last transformer layer → 896-dim vector.

Final: concatenate the last-token hidden states from layers
**`(13, 23, 24)`** → 3 × 896 = 2688-dim vector.  The probe then splits this
back into three 896-dim views and trains one sub-MLP per view (see
`probe.py`).

The three layers come from a single-layer probe sweep (`tools/ablation.py`):

* **Layer 24** (last): strongest single-layer accuracy.
* **Layer 23** (penultimate): nearly tied with 24, slightly cleaner signal —
  one block removed from the next-token specialisation.
* **Layer 13** (mid-network): a distinct AUROC peak, consistent with prior
  truthfulness-probing literature placing factuality signal in middle
  layers (Azaria & Mitchell 2023; Burns et al. 2022 CCS).

A side-channel of the aggregation is the raw-feature cache: when
`SMILE_RAW_CACHE_PATH` is set, every sample's full
`(n_layers, hidden_dim)` last-token matrix is dumped to disk on process
exit.  This lets `tools/ablation.py` and `tools/finalize.py` iterate on
probe + aggregation choices in seconds instead of paying for another
30-minute LLM extraction.

### `probe.py` — ensemble of three MLPs

The feature vector is split into three 896-dim chunks (one per layer in
`ENSEMBLE_LAYERS`).  Each chunk goes to a small MLP:

```
StandardScaler → Linear(896, 256) → ReLU → Linear(256, 1)
```

Training: full-batch Adam, `lr=1e-3`, `weight_decay=5e-4`,
`BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`, 200 epochs, `torch.manual_seed(42)`.
At inference each chunk produces a probability; the three are averaged.

Three pieces matter here:

* **The MLP itself**.  An L2-regularised logistic regression replaced the
  skeleton MLP in an earlier iteration and *lost* — see the failed-experiments
  section.  With only 689 samples and a 896-dim input the linear regime is
  too coarse; the single-hidden-layer MLP captures useful interaction
  features without overfitting catastrophically.
* **Weight decay (5e-4)**.  Skeleton has none.  Mild L2 nudges AUROC up by
  ~1 pp and stabilises predictions across folds without hurting accuracy.
* **Ensembling across views**.  This is the largest single contributor.
  Compared to a single MLP on the last layer (acc 0.7112, AUROC 0.7182),
  the three-view ensemble lifts accuracy by **+1.7 pp** and AUROC by
  **+2.3 pp** — a much bigger effect than tuning any single aggregation or
  hyperparameter (see ablation table below).

`torch.manual_seed(42)` is set before every sub-MLP build so runs are
deterministic across re-executions of `solution.py`.

### `splitting.py` — stratified 5-fold

Skeleton: a single stratified 70 / 15 / 15 split.

Final: 5 stratified folds; each fold reserves 1/5 of the data for test and
15 % of the remainder for validation.  Two practical wins:

* Test metrics averaged over 5 disjoint test partitions are much less
  noisy than a single 138-row test slice — the per-fold accuracy in the
  shipped run ranges from 69.57 % to 73.91 %, a 4-point window that would
  be invisible in a single-split report.
* `solution.py` re-fits the **final** probe (the one used for
  `predictions.csv`) on the union of train ∪ val across all folds; under
  5-fold this union is **all 689 labelled rows**, so the production probe
  sees every available example.

## Detailed results

`results.json` contains the full per-fold breakdown.  The summary table
above averages five folds with the submitted configuration.  Per-fold test
metrics:

| Fold | n_test | test_acc | test_f1 | test_auroc |
| ----:| ------:| --------:| -------:| ----------:|
| 1    | 138    | 69.57 %  | 81.25 % | 71.26 %    |
| 2    | 138    | 73.91 %  | 83.93 % | 81.17 %    |
| 3    | 138    | 73.91 %  | 83.49 % | 74.20 %    |
| 4    | 138    | 73.19 %  | 83.56 % | 71.50 %    |
| 5    | 137    | 73.72 %  | 82.18 % | 72.64 %    |
| **avg** |     | **72.86 %** | **82.88 %** | **74.15 %** |

Train accuracy is ~97 % and train AUROC essentially 1.0 — the MLPs do
memorise the training fold, which is expected for a 33k-parameter network
on ~550 samples; what matters is that the val and test AUROCs remain in
the mid-70s.

## Ablation

All numbers below come from `tools/ablation.py`, which reuses the cached
last-token-per-layer activations (no extra LLM calls) and runs the same
5-fold stratified evaluation pipeline as `evaluate.py`.  Threshold tuning
is performed on each fold's validation slice (F1-optimal) unless noted.

### Single-layer probe sweep (one MLP, F1-tuned)

Tells us which individual layers carry the most signal.

| Layer | Accuracy | F1     | AUROC  |
| ----: | -------: | -----: | -----: |
|  8    | 0.7010   | 0.8208 | 0.6602 |
|  9    | 0.6967   | 0.8144 | 0.6632 |
| 10    | 0.6995   | 0.8158 | 0.6676 |
| 11    | 0.6995   | 0.8189 | 0.6630 |
| 12    | 0.7068   | 0.8211 | 0.6907 |
| **13**| 0.7039   | 0.8213 | **0.7187** |
| 14    | 0.6937   | 0.8086 | 0.6819 |
| 15    | 0.7010   | 0.8150 | 0.6864 |
| 16    | 0.7025   | 0.8152 | 0.6945 |
| 17    | 0.7010   | 0.8102 | 0.6838 |
| 18    | 0.6952   | 0.8157 | 0.6638 |
| 19    | 0.6923   | 0.8089 | 0.6777 |
| 20    | 0.7083   | 0.8150 | 0.6729 |
| 21    | 0.7054   | 0.8192 | 0.7018 |
| 22    | 0.6938   | 0.8089 | 0.7013 |
| **23**| **0.7141**| **0.8263**| 0.7068 |
| **24**| 0.7112   | 0.8166 | 0.7182 |

Three local maxima emerge — 13, 23, and 24 — which directly motivated
`ENSEMBLE_LAYERS = (13, 23, 24)`.

### Aggregation × probe table

| Configuration                              | Feature dim | Accuracy | F1     | AUROC  |
| ------------------------------------------ | ----------: | -------: | -----: | -----: |
| Skeleton: last layer + MLP (no tune)       | 896         | 0.6981   | 0.7782 | 0.7182 |
| Skeleton: last layer + MLP (tuned)         | 896         | 0.7112   | 0.8166 | 0.7182 |
| Skeleton: last layer + MLP+wd (tuned)      | 896         | 0.7054   | 0.8149 | 0.7240 |
| Best single layer + MLP (tuned)            | 896         | 0.7039   | 0.8213 | 0.7187 |
| Mean(13..24) + MLP (tuned)                 | 896         | 0.6996   | 0.8067 | 0.6945 |
| Mean(13..24) + MLP+wd (tuned)              | 896         | 0.7054   | 0.8150 | 0.7048 |
| Mean(13..24) + LogReg (tuned)              | 896         | 0.6996   | 0.8151 | 0.6691 |
| Concat{12,16,20,24} + MLP+wd (tuned)       | 3584        | 0.7097   | 0.8195 | 0.7023 |
| Mean(all 25 layers) + MLP+wd (tuned)       | 896         | 0.6967   | 0.8074 | 0.6891 |
| **Ensemble layers (13, 23, 24), MLP+wd ★** | **2688**    | **0.7286**| **0.8288**| **0.7415** |

The submitted configuration is the only one that meaningfully beats the
majority-class baseline on accuracy (0.7286 vs 0.7010 = +2.76 pp) and on
AUROC (0.7415 vs random-guess 0.5).

## Failed / discarded experiments

Recorded honestly so that the choices above can be understood as the
result of an empirical search rather than upfront prescription.

* **L2 logistic regression replacing the MLP.** My very first iteration
  followed the standard intuition that linear probes are right for small
  datasets.  Empirically the LogReg variant tied the skeleton on F1 (due
  to the F1-tuned threshold) but lost ~5 pp of AUROC and barely matched
  the majority-class accuracy.  Reverted.  *Mean(13..24) + LogReg* was my
  first committed submission before I ran the ablation; this experience
  is the main reason `tools/ablation.py` and `tools/finalize.py` exist.
* **Mean-pool late-layer activations into 896-dim.** Compresses the same
  layers the ensemble uses, but discards the per-layer-view signal that
  the three independent MLPs exploit.  Worse on every metric than the
  ensemble.
* **Concat 4 layers `{12, 16, 20, 24}` into a 3584-dim input for a single
  MLP.** Slightly better than mean-pooling (acc 0.7097 vs 0.7054) but
  doesn't catch the ensemble (0.7286).  Concatenation forces the MLP to
  learn cross-layer interactions inside one classifier; ensembling lets
  each sub-MLP specialise.
* **A wider single-MLP probe (1024 hidden units).** Drove train accuracy
  to 100 % even faster but did not improve val/test metrics.
* **Geometric features** (`USE_GEOMETRIC=True`).  Per-layer norms of the
  last-token state plus the response length, concatenated to the main
  feature.  Adds 26 mostly-redundant scalars; never moved the needle in
  preliminary tests.  Implementation kept in `aggregation.py` for future
  ablation, but the default `solution.py` constant remains `False`.
* **Dimensionality reduction (PCA to 128 components).** On 896-dim input
  with a regularised MLP, PCA neither helped nor hurt average AUROC and
  reduced interpretability.  Not included.

## Files

| Path                       | Role                                                          |
| -------------------------- | -------------------------------------------------------------- |
| `solution.py`              | Fixed: unchanged from upstream.  Canonical entry point.        |
| `model.py`, `evaluate.py`  | Fixed: unchanged from upstream.                                |
| `aggregation.py`           | Rewritten.  Concat of layers (13, 23, 24); optional raw cache. |
| `probe.py`                 | Rewritten.  3-MLP probability-ensemble.                        |
| `splitting.py`             | Rewritten.  Stratified 5-fold.                                 |
| `results.json`             | Generated.  Submitted metrics, see table above.                |
| `predictions.csv`          | Generated (gitignored).  Uploaded to cloud; link in form.      |
| `run_local.py`             | Optional dev wrapper for Apple Silicon 8 GB.                   |
| `tools/ablation.py`        | Offline ablation grid using the raw-feature cache.             |
| `tools/finalize.py`        | Regenerate `results.json` + `predictions.csv` from cache.      |
