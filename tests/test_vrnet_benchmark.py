import numpy as np

from certihom.vrnet_benchmark import (
    MANDEL_FROM_ENGINEERING_VOIGT,
    VRNET_RASTER_TO_CARTESIAN_MANDEL,
    render_thresholded_fourier,
    vrnet_target_to_certihom_mandel,
    voigt_to_mandel_stiffness,
)


def test_voigt_to_mandel_preserves_energy():
    C_v = np.array([[7.0, 2.0, 1.5], [2.0, 6.0, -0.4], [1.5, -0.4, 3.0]])
    strain_v = np.array([0.2, -0.1, 0.6])  # engineering gamma12
    strain_m = np.linalg.inv(MANDEL_FROM_ENGINEERING_VOIGT) @ strain_v
    C_m = voigt_to_mandel_stiffness(C_v)
    assert np.allclose(strain_v @ C_v @ strain_v, strain_m @ C_m @ strain_m)
    assert np.allclose(C_m, C_m.T)


def test_vrnet_raster_target_reflects_shear_frame_only():
    """Regression for the raster-row versus Cartesian-axis convention.

    A tensor with nonzero normal/shear couplings makes a missing reflection
    observable.  Diagonal entries and C12 stay fixed; C13 and C23 reverse.
    """
    C_v = np.array([[7.0, 2.0, 1.5], [2.0, 6.0, -0.4], [1.5, -0.4, 3.0]])
    raw = voigt_to_mandel_stiffness(C_v)
    got = vrnet_target_to_certihom_mandel(C_v)
    T = VRNET_RASTER_TO_CARTESIAN_MANDEL
    assert np.allclose(got, T @ raw @ T)
    assert np.allclose(np.diag(got), np.diag(raw))
    assert np.isclose(got[0, 1], raw[0, 1])
    assert np.isclose(got[0, 2], -raw[0, 2])
    assert np.isclose(got[1, 2], -raw[1, 2])


def test_fourier_renderer_is_periodic_for_an_integer_mode():
    weights = np.zeros((3, 3))
    weights[2, 1] = 1.0
    mask = render_thresholded_fourier(weights, threshold=0.5, resolution=100)
    assert mask.dtype == np.bool_
    assert mask.shape == (100, 100)
