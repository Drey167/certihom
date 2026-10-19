"""Reader for the public Voigt--Reuss Net 2D mechanical benchmark.

The released mechanical cells are native 100 x 100 binary rasters.  This
reader reconstructs those rasters from the released Fourier coefficients and
checks bit-for-bit agreement with ``image_data`` at that native resolution;
it deliberately does *not* replace the labelled geometry by a new 400 x 400
rendering.

There is one non-obvious but material convention at the image/tensor boundary.
``image_data`` is a raster (row, column), while CertiHOM reads the returned
array in a right-handed Cartesian finite-element frame.  The public HDF5
elasticity tensor uses the opposite orientation of one spatial axis.  Thus,
after the ordinary engineering-Voigt-to-Mandel conversion, its shear
coordinate must be reflected using ``diag(1, 1, -1)``.  The transformation is
an exact coordinate change, not a fit or a modification of the label.  It was
validated by an independent periodic Q1 solve on the fixed 100-cell external
protocol: all normal and shear entries agree after this reflection; without
it the two shear-normal couplings have opposite signs.

Dataset: Keshav, Fritzen, Boddapati and Daraio, DaRUS-5730 (CC-BY-4.0).
The phase law is taken from the authors' public ``SpinodoidMechanical2D``
loader: solid (E=1000, nu=0.3), compliant (E=1, nu=0.49), plane strain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .elasticity import iso_stiffness_mandel, mandel_field_from_mask

SQRT2 = np.sqrt(2.0)
MANDEL_FROM_ENGINEERING_VOIGT = np.diag([1.0, 1.0, SQRT2])
# The coordinate change induced by reversing either one of the two Cartesian
# axes.  It leaves normal components unchanged and reverses Mandel shear.
VRNET_RASTER_TO_CARTESIAN_MANDEL = np.diag([1.0, 1.0, -1.0])


def voigt_to_mandel_stiffness(C_voigt: np.ndarray) -> np.ndarray:
    """Convert 2D engineering-Voigt stiffness to the Mandel convention.

    The public HDF5 target uses ``(eps11, eps22, gamma12)`` and
    ``(sig11, sig22, sig12)``.  CertiHOM uses orthonormal Mandel vectors
    ``(A11, A22, sqrt(2) A12)``.  Thus C_M = T C_V T with
    T = diag(1, 1, sqrt(2)).
    """
    C_voigt = np.asarray(C_voigt, dtype=float)
    if C_voigt.shape != (3, 3):
        raise ValueError(f"expected one 3x3 stiffness, got {C_voigt.shape}")
    return MANDEL_FROM_ENGINEERING_VOIGT @ C_voigt @ MANDEL_FROM_ENGINEERING_VOIGT


def vrnet_target_to_certihom_mandel(C_voigt: np.ndarray) -> np.ndarray:
    """Map a public VRNet HDF5 target into the returned raster's FE frame.

    The HDF5 target first undergoes the standard engineering-Voigt to Mandel
    conversion, then the one-axis raster-to-Cartesian reflection documented at
    module level.  Keeping this as a named operation prevents a comparison of
    an image in one handedness with a stiffness tensor in the other.
    """
    C_mandel_raster = voigt_to_mandel_stiffness(C_voigt)
    T = VRNET_RASTER_TO_CARTESIAN_MANDEL
    return T @ C_mandel_raster @ T


def render_thresholded_fourier(weights: np.ndarray, threshold: float,
                               resolution: int = 400) -> np.ndarray:
    """Reconstruct the authors' periodic thresholded-trigonometric image.

    Returns True for the compliant/TangoBlack phase.  Pixel centres and the
    Fourier ordering exactly follow ``RenderWrap2D._render`` in the public
    VoigtReussNet source.  The hard threshold is appropriate here because the
    stored FANS geometry is binary; the neural renderer's tiny-temperature
    sigmoid is only used to obtain design gradients.
    """
    A = np.asarray(weights, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] % 2 != 1:
        raise ValueError("weights must be an odd square Fourier coefficient matrix")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    k = A.shape[0]
    modes = np.arange(k, dtype=float) - (k - 1) / 2
    x = np.linspace(-0.5, 0.5, resolution)
    basis = np.exp(2j * np.pi * np.outer(modes, x))
    field = np.einsum("mn,mx,ny->xy", A, basis, basis).real.T
    fmin, fmax = float(field.min()), float(field.max())
    if not fmax > fmin:
        raise ValueError("degenerate Fourier field cannot be thresholded")
    return (field - fmin) / (fmax - fmin) >= float(threshold)


def vrnet_phase_stiffnesses() -> tuple[np.ndarray, np.ndarray]:
    """Return (C_solid, C_compliant) in plane-strain Mandel notation."""
    return iso_stiffness_mandel(1000.0, 0.3), iso_stiffness_mandel(1.0, 0.49)


def voigt_reuss_bounds(Cfield: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Reuss lower, Voigt upper) for a pixelwise Mandel field."""
    flat = np.asarray(Cfield, dtype=float).reshape(-1, 3, 3)
    C_up = flat.mean(axis=0)
    C_low = np.linalg.inv(np.linalg.inv(flat).mean(axis=0))
    return C_low, C_up


@dataclass(frozen=True)
class VRNetMechanicalSample:
    split: str
    mode_grid: tuple[int, int]
    index: int
    threshold: float
    declared_solid_fraction: float
    compliant_mask: np.ndarray
    reference_mandel: np.ndarray

    def stiffness_field(self) -> np.ndarray:
        C_solid, C_compliant = vrnet_phase_stiffnesses()
        return mandel_field_from_mask(self.compliant_mask, C_compliant, C_solid)


def load_vrnet_mechanical_samples(h5_path: str | Path, *, split: str = "val_set",
                                  mode_grid: tuple[int, int] = (3, 3),
                                  indices: Sequence[int] = tuple(range(10)),
                                  resolution: int = 400) -> list[VRNetMechanicalSample]:
    """Load selected public benchmark records and reconstruct native cells.

    The declared HDF5 ``volume_fraction`` is the solid fraction.  The stored
    native 100 x 100 preview is checked exactly when ``resolution==100``.
    ``reference_mandel`` is returned in the same Cartesian coordinate frame
    as ``compliant_mask`` through :func:`vrnet_target_to_certihom_mandel`.
    Other resolutions are geometry-only Fourier renders and must not be
    compared against the released 100 x 100 labels.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise ImportError("VRNet benchmark support needs h5py; install requirements.txt") from exc

    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    m, n = mode_grid
    root = f"{split}/mode_grid_{m}x{n}"
    target = f"{root}/effective_elasticity_tensor/DM8530_TangoBlack"
    out: list[VRNetMechanicalSample] = []
    with h5py.File(h5_path, "r") as f:
        for index in indices:
            weights = f[f"{root}/function_weights"][index]
            threshold = float(f[f"{root}/threshold"][0, index])
            solid_fraction = float(f[f"{root}/volume_fraction"][0, index])
            mask = render_thresholded_fourier(weights, threshold, resolution)
            if resolution == 100:
                preview = np.asarray(f[f"{root}/image_data"][index], dtype=bool)
                if not np.array_equal(mask, preview):
                    raise RuntimeError("Fourier reconstruction disagrees with released image_data")
            # Ones are the compliant phase; volume_fraction is solid phase.
            observed_solid_fraction = 1.0 - float(mask.mean())
            pixel_tolerance = 2.0 / resolution
            if abs(observed_solid_fraction - solid_fraction) > pixel_tolerance:
                raise RuntimeError(
                    f"phase-fraction mismatch for {root}[{index}]: "
                    f"declared solid {solid_fraction:.6f}, reconstructed {observed_solid_fraction:.6f}")
            C_ref = vrnet_target_to_certihom_mandel(f[target][index])
            out.append(VRNetMechanicalSample(split, mode_grid, int(index), threshold,
                                               solid_fraction, mask, C_ref))
    return out
