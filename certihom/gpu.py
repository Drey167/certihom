"""GPU support via the PRECISION SPLIT.

Certificates are float64 arithmetic. Most GPUs are terrible at float64 —
an NVIDIA T4 runs FP64 at 1:32 of its FP32 rate, so naively moving this
codebase to a T4 makes it SLOWER than a CPU. The resolution exploits the
project's central property:

    The bound holds for ANY admissible trial field. Training quality
    affects only the WIDTH of the interval, never its validity. So the
    fields may be trained in float32 on the GPU, then cast to float64 and
    certified exactly on the CPU. Reduced training precision can only
    widen the certificate; it cannot invalidate it.

Verified: float32-trained fields, certified in float64 by exact
integration, were Loewner-valid and bracketed EXACT laminate tensors in
9/9 runs across seeds and optimiser settings (RESULTS.md).

Usage
-----
    from certihom.gpu import pick_device, to_training, to_certification

    dev = pick_device()                       # 'cuda' if available
    model = AmortizedElasticity(dtype=torch.float32)
    model = to_training(model, dev)           # fp32 on GPU
    model, hist = fit_amortized(model, fam, ...)
    cert = to_certification(model)            # fp64 on CPU
    bounds = certified_upper_elasticity(nodal_values(...), Cf)

NEVER certify from the float32 model directly: `certify.py` and `bfs.py`
are float64 numpy by design and must stay that way.
"""

from __future__ import annotations

import copy

import torch


def pick_device(verbose: bool = True) -> torch.device:
    """Return cuda if available, else cpu, with an FP64 warning."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            print(f"device: {dev} ({name})")
            if "T4" in name:
                print("  NOTE: T4 FP64 is 1:32 of FP32 — train in float32 "
                      "and certify on CPU in float64 (see KAGGLE.md).")
    else:
        dev = torch.device("cpu")
        if verbose:
            print("device: cpu (no CUDA available)")
    return dev


def to_training(model: torch.nn.Module, device, dtype=torch.float32):
    """Move a model to the training device and precision (default fp32)."""
    return model.to(device=device, dtype=dtype)


def to_certification(model: torch.nn.Module):
    """Return an independent float64 CPU copy for exact certification.

    A copy rather than an in-place move, so training can continue on the
    GPU afterwards. Widening float32 -> float64 is exact, so the certified
    field is precisely the field that was trained.
    """
    return copy.deepcopy(model).to(device="cpu", dtype=torch.float64)


def report_precision(model: torch.nn.Module) -> str:
    p = next(model.parameters())
    return f"parameters: {p.dtype} on {p.device}"
