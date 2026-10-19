"""Structural tests for the matched external-elasticity baselines."""

from __future__ import annotations

import torch

from examples.run_vrnet_elasticity_baselines import Baseline


def test_voigt_reuss_baseline_stays_in_analytic_envelope() -> None:
    """The parameterisation must preserve both Loewner inequalities."""
    torch.manual_seed(7)
    batch = 3
    image = torch.rand(batch, 1, 20, 20)
    fraction = 1.0 - image.mean(dim=(-3, -2, -1))
    reuss = torch.diag_embed(torch.tensor([[1.0, 1.5, 2.0]]).repeat(batch, 1))
    voigt = torch.diag_embed(torch.tensor([[3.0, 4.0, 5.0]]).repeat(batch, 1))

    predicted = Baseline("voigt_reuss")(image, fraction, reuss, voigt)

    assert torch.linalg.eigvalsh(predicted - reuss).amin().item() >= -1e-10
    assert torch.linalg.eigvalsh(voigt - predicted).amin().item() >= -1e-10


def test_cholesky_baseline_is_positive_definite() -> None:
    torch.manual_seed(11)
    image = torch.rand(2, 1, 20, 20)
    fraction = 1.0 - image.mean(dim=(-3, -2, -1))
    lower = torch.eye(3).repeat(2, 1, 1)
    upper = (2.0 * torch.eye(3)).repeat(2, 1, 1)

    predicted = Baseline("cholesky")(image, fraction, lower, upper)

    assert torch.linalg.eigvalsh(predicted).amin().item() > 0.0
