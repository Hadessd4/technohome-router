"""
TechnoHome Customer Support Ticket Router — Model Utilities
Shared helpers for saving, loading, measuring size and latency of models.
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config

# ─── Label mapping constants (re-exported from config for convenience) ───────
DEPARTMENTS = config.DEPARTMENTS
NUM_DEPARTMENTS = config.NUM_DEPARTMENTS
DEPT_TO_IDX = config.DEPT_TO_IDX
IDX_TO_DEPT = config.IDX_TO_DEPT
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD


class Prediction(tuple):
    """Universal prediction object behaving as both a 2-tuple (dept, conf) and dict-like."""

    def __new__(cls, department, confidence):
        return super().__new__(cls, (str(department), float(confidence)))

    @property
    def department(self):
        return self[0]

    @property
    def confidence(self):
        return self[1]

    def get(self, key, default=None):
        if key == "department":
            return self.department
        if key == "confidence":
            return self.confidence
        return default

    def __getitem__(self, item):
        if isinstance(item, str):
            val = self.get(item)
            if val is not None:
                return val
            raise KeyError(item)
        return super().__getitem__(item)


# ─── Model size ──────────────────────────────────────────────────────────────

def get_model_size_mb(model_or_path):
    """Return the size of a model in megabytes.

    Accepts:
      - A file path (str / pathlib.Path) → measures the file on disk.
      - A sklearn Pipeline / estimator → serialises with joblib to a temp file.
      - A PyTorch nn.Module → serialises with torch.save to a temp file.
    """
    # If it is already a path that exists on disk
    if isinstance(model_or_path, (str, os.PathLike)) and os.path.isfile(str(model_or_path)):
        return os.path.getsize(str(model_or_path)) / (1024 * 1024)

    # Try PyTorch first (nn.Module)
    try:
        import torch
        if isinstance(model_or_path, torch.nn.Module):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmp.close()
            try:
                torch.save(model_or_path.state_dict(), tmp.name)
                size = os.path.getsize(tmp.name) / (1024 * 1024)
            finally:
                os.unlink(tmp.name)
            return size
    except ImportError:
        pass

    # Try sklearn / joblib
    try:
        import joblib
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
        tmp.close()
        try:
            joblib.dump(model_or_path, tmp.name)
            size = os.path.getsize(tmp.name) / (1024 * 1024)
        finally:
            os.unlink(tmp.name)
        return size
    except Exception:
        raise TypeError(
            f"Cannot determine size for object of type {type(model_or_path)}"
        )


# ─── Latency measurement ────────────────────────────────────────────────────

def measure_latency(predict_fn, test_inputs, n_runs=100):
    """Return the average prediction latency in **milliseconds**.

    Parameters
    ----------
    predict_fn : callable
        A function that takes a single input and returns a prediction.
    test_inputs : list
        Inputs to cycle through during measurement.
    n_runs : int
        Total number of prediction calls to average over.
    """
    if not test_inputs:
        raise ValueError("test_inputs must be a non-empty list")

    # Warm-up
    predict_fn(test_inputs[0])

    start = time.perf_counter()
    for i in range(n_runs):
        predict_fn(test_inputs[i % len(test_inputs)])
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_runs) * 1000
    return avg_ms


# ─── Generic save / load wrappers ───────────────────────────────────────────

def save_model(model, path):
    """Save *model* to *path*.

    • PyTorch ``nn.Module`` → ``torch.save`` (state_dict)
    • Everything else (sklearn, etc.) → ``joblib.dump``
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    try:
        import torch
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), path)
            return
    except ImportError:
        pass

    import joblib
    joblib.dump(model, path)


def load_model(path):
    """Load a model from *path*.

    Determines the format from the file extension:
      - ``.pt`` / ``.pth`` → ``torch.load``
      - anything else → ``joblib.load``
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth"):
        import torch
        return torch.load(path, map_location="cpu")
    else:
        import joblib
        return joblib.load(path)
