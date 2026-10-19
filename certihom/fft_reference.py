"""Moulinec-Suquet FFT homogenisation (VALIDATION / BASELINE only).

The basic scheme (Moulinec & Suquet 1994, 1998) is the standard workhorse
of computational homogenisation and the baseline any reviewer will ask
about. Implemented here for scalar conductivity so that the certified
bounds can be compared against it on cost AND on what each method actually
delivers.

Fixed-point iteration on the gradient field e(y) with a reference medium
k0:

    e_{n+1}(xi) = e_n(xi) - Gamma0(xi) q_n(xi)   for xi != 0,
    e_{n+1}(0)  = ebar,
    Gamma0(xi)  = xi (x) xi / (k0 |xi|^2),        q_n = k e_n.

Convergence is monotone for k0 chosen between the phase extremes; the
scheme is known to slow down as phase contrast grows, which is itself
worth reporting in the comparison.

IMPORTANT FRAMING for the paper: FFT returns an ESTIMATE with unknown
error. Certified bounds return an INTERVAL that provably contains the
truth. A cost-to-accuracy plot comparing them is not apples to apples, and
must be presented as "cost to a GUARANTEED answer" rather than "cost to an
accurate one" -- otherwise the comparison flatters FFT unfairly and invites
exactly the objection it is meant to answer.
"""

from __future__ import annotations

import numpy as np


def fft_effective(K: np.ndarray, tol: float = 1e-10, max_iter: int = 10000,
                  return_iters: bool = False):
    """Effective conductivity tensor by the FFT basic scheme.

    Parameters
    ----------
    K : (n, n) pixel conductivities.
    tol : relative equilibrium residual for convergence.
    max_iter : iteration cap (the scheme slows at high contrast).

    Returns
    -------
    (2, 2) effective tensor, and optionally the iteration counts per load.
    """
    n = K.shape[0]
    k0 = 0.5 * (K.min() + K.max())          # reference medium
    xi = np.fft.fftfreq(n, d=1.0 / n)
    X1, X2 = np.meshgrid(xi, xi, indexing="ij")
    X2sum = X1**2 + X2**2
    X2sum[0, 0] = 1.0                        # avoid divide-by-zero at xi=0

    K_hom = np.zeros((2, 2))
    iters = []
    for a, ebar in enumerate((np.array([1.0, 0.0]), np.array([0.0, 1.0]))):
        e1 = np.full((n, n), ebar[0])
        e2 = np.full((n, n), ebar[1])
        it = 0
        for it in range(1, max_iter + 1):
            q1, q2 = K * e1, K * e2
            Q1, Q2 = np.fft.fft2(q1), np.fft.fft2(q2)
            # equilibrium residual in Fourier: xi . q_hat
            div = X1 * Q1 + X2 * Q2
            res = np.linalg.norm(div) / max(abs(Q1[0, 0]) + abs(Q2[0, 0]), 1e-30)
            if res < tol:
                break
            fac = div / (k0 * X2sum)
            E1 = np.fft.fft2(e1) - X1 * fac
            E2 = np.fft.fft2(e2) - X2 * fac
            E1[0, 0] = ebar[0] * n * n       # pin the mean exactly
            E2[0, 0] = ebar[1] * n * n
            e1 = np.real(np.fft.ifft2(E1))
            e2 = np.real(np.fft.ifft2(E2))
        iters.append(it)
        K_hom[0, a] = (K * e1).mean()
        K_hom[1, a] = (K * e2).mean()
    K_hom = 0.5 * (K_hom + K_hom.T)
    return (K_hom, iters) if return_iters else K_hom
