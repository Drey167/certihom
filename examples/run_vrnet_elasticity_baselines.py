"""Matched label-trained baselines for the DaRUS-5730 CertiHOM experiment.

This is a controlled, small-data comparison of three *constraint classes*:
an unconstrained image regressor, an SPD-by-Cholesky regressor, and a
Voigt--Reuss-constrained regressor.  It uses the exact 200-train / 100-
validation index protocol in ``run_vrnet_elasticity_training`` by default,
and never reuses a validation label during fitting.

The experiment is deliberately narrower than the full-data hyperparameter
sweep of Keshav and Fritzen: its purpose is a matched diagnostic alongside
CertiHOM, not a claim to reproduce their published neural-network numbers.
It reports point-prediction accuracy and structural rates only; none of the
three models supplies a two-sided certificate for an unseen reference tensor.

Example (Kaggle GPU, with the HDF5 attached as an input)::

    python examples/run_vrnet_elasticity_baselines.py \
        --h5 /kaggle/input/vrnet-mechanical/feature_engineering_mechanical_2D.h5 \
        --device cuda --epochs 1200 --batch 64 \
        --output-dir /kaggle/working/vrnet_baselines_200x100
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from certihom.vrnet_benchmark import load_vrnet_mechanical_samples, voigt_reuss_bounds
try:  # Script execution puts ``examples`` on sys.path.
    from run_vrnet_elasticity_training import MODE_GRIDS, SEED, selected_indices
except ModuleNotFoundError:  # Support importing this driver in a test module.
    from examples.run_vrnet_elasticity_training import MODE_GRIDS, SEED, selected_indices

SCALE = 1000.0


def unpack(x: torch.Tensor) -> torch.Tensor:
    C = torch.zeros((*x.shape[:-1], 3, 3), dtype=x.dtype, device=x.device)
    C[..., 0, 0], C[..., 1, 1], C[..., 2, 2] = x[..., 0], x[..., 1], x[..., 2]
    C[..., 1, 0] = C[..., 0, 1] = x[..., 3]
    C[..., 2, 0] = C[..., 0, 2] = x[..., 4]
    C[..., 2, 1] = C[..., 1, 2] = x[..., 5]
    return C


def symmetric_square_root_psd(matrix: torch.Tensor) -> torch.Tensor:
    """Return the principal square root of a symmetric PSD batch.

    The Voigt--Reuss difference may be singular for a nearly single-phase
    cell, where a Cholesky factorisation is not defined.  The spectral square
    root preserves the intended semidefinite algebra exactly (up to numerical
    round-off) and does not introduce an artificial ``jitter`` that could
    invalidate the analytic envelope.
    """
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    root_values = eigenvalues.clamp_min(0.0).sqrt()
    return (eigenvectors * root_values.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


class ImageRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 12, 5, padding=2, padding_mode="circular"), nn.GELU(), nn.AvgPool2d(2),
            nn.Conv2d(12, 24, 5, padding=2, padding_mode="circular"), nn.GELU(), nn.AvgPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1, padding_mode="circular"), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Linear(33, 64), nn.GELU(), nn.Linear(64, 6))

    def forward(self, image: torch.Tensor, solid_fraction: torch.Tensor) -> torch.Tensor:
        z = self.conv(image).flatten(1)
        return self.head(torch.cat([z, solid_fraction[:, None]], dim=1))


class Baseline(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.net = ImageRegressor()

    def forward(self, image: torch.Tensor, fraction: torch.Tensor,
                C_reuss: torch.Tensor, C_voigt: torch.Tensor) -> torch.Tensor:
        raw = self.net(image, fraction)
        if self.kind == "vanilla":
            return unpack(raw * SCALE)
        if self.kind == "cholesky":
            L = torch.zeros((raw.shape[0], 3, 3), dtype=raw.dtype, device=raw.device)
            L[..., 0, 0] = F.softplus(raw[..., 0])
            L[..., 1, 1] = F.softplus(raw[..., 1])
            L[..., 2, 2] = F.softplus(raw[..., 2])
            L[..., 1, 0], L[..., 2, 0], L[..., 2, 1] = raw[..., 3], raw[..., 4], raw[..., 5]
            L = L * np.sqrt(SCALE)
            return L @ L.transpose(-1, -2)
        if self.kind == "voigt_reuss":
            # C = V - L Q diag(a) Q^T L^T, 0<a<1.  Therefore R <= C <= V.
            # The envelope difference spans roughly three orders of magnitude
            # in this benchmark.  Perform the small 3x3 Loewner construction
            # in float64: float32 round-off otherwise creates spurious
            # O(1e-4) negative eigenvalues in the post-run envelope check.
            raw64 = raw.to(torch.float64)
            reuss64, voigt64 = C_reuss.to(torch.float64), C_voigt.to(torch.float64)
            L = symmetric_square_root_psd(voigt64 - reuss64)
            eig = torch.sigmoid(raw64[..., :3])
            skew = torch.zeros((raw.shape[0], 3, 3), dtype=torch.float64, device=raw.device)
            skew[..., 1, 0], skew[..., 0, 1] = raw64[..., 3], -raw64[..., 3]
            skew[..., 2, 0], skew[..., 0, 2] = raw64[..., 4], -raw64[..., 4]
            skew[..., 2, 1], skew[..., 1, 2] = raw64[..., 5], -raw64[..., 5]
            Q = torch.linalg.matrix_exp(skew)
            normalized = (Q * eig[:, None, :]) @ Q.transpose(-1, -2)
            return voigt64 - L @ normalized @ L.transpose(-1, -2)
        raise ValueError(self.kind)


def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target).square().sum(dim=(-2, -1)) /
            target.square().sum(dim=(-2, -1)).clamp_min(1e-12)).mean()


def tensors(samples, device: torch.device):
    images = torch.as_tensor(np.stack([s.compliant_mask for s in samples])[:, None], dtype=torch.float32, device=device)
    # This feature is derived from the released image, not metadata or labels.
    frac = 1.0 - images.mean(dim=(-2, -1, -3))
    target = torch.as_tensor(np.stack([s.reference_mandel for s in samples]), dtype=torch.float32, device=device)
    bounds = [voigt_reuss_bounds(s.stiffness_field()) for s in samples]
    reuss = torch.as_tensor(np.stack([b[0] for b in bounds]), dtype=torch.float32, device=device)
    voigt = torch.as_tensor(np.stack([b[1] for b in bounds]), dtype=torch.float32, device=device)
    return images, frac, target, reuss, voigt


def evaluate(model, data):
    images, frac, target, reuss, voigt = data
    with torch.no_grad():
        pred = model(images, frac, reuss, voigt)
        err = torch.linalg.norm(pred - target, dim=(-2, -1)) / torch.linalg.norm(target, dim=(-2, -1))
        eig_pred = torch.linalg.eigvalsh(pred).amin(dim=-1)
        low_margin = torch.linalg.eigvalsh(pred - reuss).amin(dim=-1)
        up_margin = torch.linalg.eigvalsh(voigt - pred).amin(dim=-1)
    return pred.cpu().numpy(), {
        "mean_relative_frobenius_error": float(err.mean()),
        "median_relative_frobenius_error": float(err.median()),
        "spd_rate": float((eig_pred > 0).float().mean()),
        "voigt_reuss_rate": float(((low_margin >= -1e-5) & (up_margin >= -1e-5)).float().mean()),
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but this PyTorch installation has no CUDA device. "
            "Use a CUDA-enabled Kaggle runtime or install a matching CUDA PyTorch build."
        )
    return torch.device(requested)


def prediction_rows(kind: str, samples, pred: np.ndarray, data) -> list[dict[str, object]]:
    """Write per-cell evidence instead of leaving only aggregate metrics."""
    _, _, target, reuss, voigt = data
    target_np = target.detach().cpu().numpy()
    reuss_np = reuss.detach().cpu().numpy()
    voigt_np = voigt.detach().cpu().numpy()
    rows = []
    for sample, estimated, reference, lower, upper in zip(samples, pred, target_np, reuss_np, voigt_np):
        rows.append({
            "model": kind,
            "split": sample.split,
            "mode_grid": sample.mode_grid[0],
            "index": sample.index,
            "relative_frobenius_error": float(np.linalg.norm(estimated - reference) /
                                                np.linalg.norm(reference)),
            "min_predicted_eigenvalue": float(np.linalg.eigvalsh(estimated).min()),
            "min_predicted_minus_reuss_eigenvalue": float(np.linalg.eigvalsh(estimated - lower).min()),
            "min_voigt_minus_predicted_eigenvalue": float(np.linalg.eigvalsh(upper - estimated).min()),
            **{f"pred_C{i}{j}": float(estimated[i, j]) for i in range(3) for j in range(3)},
            **{f"ref_C{i}{j}": float(reference[i, j]) for i in range(3) for j in range(3)},
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--train-per-grid", type=int, default=40)
    parser.add_argument("--val-per-grid", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="optional positive PyTorch CPU thread count; 0 leaves its default")
    parser.add_argument("--output-dir", default="outputs/vrnet_elasticity_baselines")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.train_per_grid < 1 or args.val_per_grid < 1:
        parser.error("--epochs, --batch, --train-per-grid, and --val-per-grid must be positive")
    if args.cpu_threads < 0:
        parser.error("--cpu-threads must be non-negative")
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    selection = selected_indices(args.h5, args.seed, args.train_per_grid, args.val_per_grid)
    train, val = [], []
    for mode in MODE_GRIDS:
        train.extend(load_vrnet_mechanical_samples(args.h5, split="train_set", mode_grid=(mode, mode),
                                                    indices=selection["train_set"][str(mode)], resolution=100))
        val.extend(load_vrnet_mechanical_samples(args.h5, split="val_set", mode_grid=(mode, mode),
                                                  indices=selection["val_set"][str(mode)], resolution=100))
    tr, va = tensors(train, device), tensors(val, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "h5": str(Path(args.h5).resolve()),
        "h5_sha256": _sha256(args.h5),
        "train_per_grid": args.train_per_grid,
        "val_per_grid": args.val_per_grid,
        "mode_grids": MODE_GRIDS,
        "epochs": args.epochs,
        "batch": args.batch,
        "selection": selection,
        "phase_fraction_feature": "1 - released compliant-mask mean",
        "note": ("Matched small-data diagnostic, not a reproduction of the "
                 "original full-data VoigtReussNet hyperparameter sweep."),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("# Matched DaRUS-5730 point-prediction baselines")
    print(f"# device={device}; train={len(train)}; validation={len(val)}; "
          f"epochs={args.epochs}; batch={args.batch}; seed={args.seed}")
    summary = []
    all_rows = []
    histories: dict[str, list[dict[str, float | int]]] = {}
    for kind in ("vanilla", "cholesky", "voigt_reuss"):
        model = Baseline(kind).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
        history = []
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(tr[0].shape[0], device=device)
            for idx in order.split(args.batch):
                prediction = model(tr[0][idx], tr[1][idx], tr[3][idx], tr[4][idx])
                loss = relative_mse(prediction, tr[2][idx])
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
                history.append({"epoch": epoch, "train_relative_mse": float(loss.item())})
                print(f"{kind:12s} epoch {epoch:4d} train relative MSE {loss.item():.5f}")
        elapsed = time.perf_counter() - started
        prediction, metrics = evaluate(model, va)
        _, train_metrics = evaluate(model, tr)
        metrics["model"] = kind
        metrics["train_mean_relative_frobenius_error"] = train_metrics["mean_relative_frobenius_error"]
        metrics["training_seconds"] = elapsed
        summary.append(metrics)
        all_rows.extend(prediction_rows(kind, val, prediction, va))
        histories[kind] = history
        torch.save({
            "kind": kind,
            "state_dict": model.state_dict(),
            "protocol": protocol,
            "history": history,
        }, output_dir / f"{kind}_model.pt")
    with (output_dir / "baseline_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    print("\n# Held-out baseline summary")
    for row in summary:
        print(f"{row['model']:12s} | mean error {row['mean_relative_frobenius_error']:.2%} | "
              f"SPD {row['spd_rate']:.0%} | VR {row['voigt_reuss_rate']:.0%} | "
              f"train {row['training_seconds']:.1f} s")


if __name__ == "__main__":
    main()
