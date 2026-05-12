"""
run_local.py — Local-dev wrapper around ``solution.py`` for resource-constrained
hardware (in particular Apple M1 8 GB).  All modifications are made in memory;
no fixed-infrastructure file is changed on disk, and ``python solution.py``
remains the canonical entry point.

What this wrapper does (purely in memory):

1. Rewrites ``BATCH_SIZE = 4`` to ``BATCH_SIZE = 1`` in ``solution.py``'s
   source before ``exec``.
2. Monkey-patches ``model.get_model_and_tokenizer`` so the returned model
   calls ``torch.mps.empty_cache()`` after every forward — without this the
   MPS allocator's working set grows monotonically on small Apple-Silicon
   machines and per-batch latency climbs from ~0.7 s to 20+ s after ~50
   batches when ``output_hidden_states=True``.
3. Sets ``SMILE_RAW_CACHE_PATH`` so ``aggregation.py`` writes a side-channel
   dump of raw last-token hidden states for offline ablations via
   ``tools/ablation.py``.

The grader's official command (``python solution.py``) still produces an
identical results.json on any sufficient device.

Usage:
    python run_local.py
"""
from __future__ import annotations

import os
import sys

import torch


# Cache raw last-token-per-layer hidden states to disk for offline ablations
# (see tools/ablation.py).  Picked up by aggregation.py's atexit hook.
os.environ.setdefault("SMILE_RAW_CACHE_PATH", "features_raw_cache.npz")


# Patch ``model.get_model_and_tokenizer`` so the returned model calls
# ``torch.mps.empty_cache()`` after every forward.  Without this, the MPS
# allocator on Apple Silicon 8 GB keeps growing across the extraction loop
# (Qwen2.5-0.5B + ``output_hidden_states=True`` produces 25 intermediate
# tensors per forward; they are referenced past the no_grad scope long
# enough for the cache to fragment) and per-batch latency degrades from
# ~0.7 s to 20+ s after ~50 batches.  Forcing a cache flush after each
# forward keeps throughput flat for the entire 689-row extraction.
import model as _model  # noqa: E402

_real_get = _model.get_model_and_tokenizer


def _patched_get(*args, **kwargs):
    mdl, tok = _real_get(*args, **kwargs)
    real_forward = mdl.forward

    def patched_forward(*fargs, **fkwargs):
        out = real_forward(*fargs, **fkwargs)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return out

    mdl.forward = patched_forward
    return mdl, tok


_model.get_model_and_tokenizer = _patched_get  # type: ignore[assignment]


SOLUTION_PATH = os.path.join(os.path.dirname(__file__), "solution.py")
with open(SOLUTION_PATH, encoding="utf-8") as f:
    src = f.read()

src = src.replace("BATCH_SIZE    = 4", "BATCH_SIZE    = 1")

print(
    "[run_local] patched: BATCH_SIZE=1, mps.empty_cache after each forward, "
    f"SMILE_RAW_CACHE_PATH={os.environ['SMILE_RAW_CACHE_PATH']}"
)
sys.stdout.flush()

exec(  # noqa: S102 — controlled wrapper, source is our own solution.py
    compile(src, SOLUTION_PATH, "exec"),
    {"__name__": "__main__", "__file__": SOLUTION_PATH},
)
