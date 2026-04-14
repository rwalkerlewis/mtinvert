"""
Moment tensor decomposition following Jost & Herrmann (1989).

Decomposes a symmetric 3x3 moment tensor into isotropic (ISO),
compensated linear vector dipole (CLVD), and double-couple (DC)
components. Computes scalar moment, moment magnitude, and
fault plane solutions (strike, dip, rake).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MTDecomposition:
    """Result of moment tensor decomposition."""
    # Full tensor (symmetric, 3x3)
    mt_tensor: np.ndarray

    # Six independent elements: Mxx, Myy, Mzz, Mxy, Mxz, Myz
    mt_vector: np.ndarray

    # Eigenvalues sorted by absolute value descending
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    # Scalar moment (N*m)
    scalar_moment: float
    moment_magnitude: float

    # Percentage decomposition
    iso_percent: float
    clvd_percent: float
    dc_percent: float

    # Deviatoric eigenvalues
    dev_eigenvalues: np.ndarray

    # Fault planes (strike, dip, rake) in degrees
    strike1: float
    dip1: float
    rake1: float
    strike2: float
    dip2: float
    rake2: float

    # P, T, B axes (trend, plunge) in degrees
    p_axis: tuple[float, float]
    t_axis: tuple[float, float]
    b_axis: tuple[float, float]


def vector_to_tensor(m: np.ndarray) -> np.ndarray:
    """Convert 6-element vector [Mxx, Myy, Mzz, Mxy, Mxz, Myz] to 3x3 tensor."""
    return np.array([
        [m[0], m[3], m[4]],
        [m[3], m[1], m[5]],
        [m[4], m[5], m[2]],
    ])


def tensor_to_vector(M: np.ndarray) -> np.ndarray:
    """Convert 3x3 symmetric tensor to 6-element vector."""
    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]])


def _eigenvector_to_trend_plunge(v: np.ndarray) -> tuple[float, float]:
    """Convert eigenvector to trend and plunge (degrees)."""
    # Ensure downward pointing
    if v[2] > 0:
        v = -v
    plunge = np.degrees(np.arcsin(-v[2]))
    trend = np.degrees(np.arctan2(v[1], v[0]))
    if trend < 0:
        trend += 360.0
    return float(trend), float(plunge)


def _mt_to_nodal_planes(eigvals: np.ndarray, eigvecs: np.ndarray):
    """
    Compute nodal planes from eigenvectors.

    Returns (strike1, dip1, rake1), (strike2, dip2, rake2) in degrees.
    """
    # Sort by eigenvalue: T (most positive), B (intermediate), P (most negative)
    idx = np.argsort(eigvals)
    t_vec = eigvecs[:, idx[2]]  # most positive
    p_vec = eigvecs[:, idx[0]]  # most negative

    # Fault normal and slip vectors from P and T axes
    n1 = (t_vec + p_vec) / np.sqrt(2)
    n2 = (t_vec - p_vec) / np.sqrt(2)

    def _normal_to_sdr(n, s):
        """Convert normal and slip vectors to strike, dip, rake."""
        # Ensure normal points upward
        if n[2] > 0:
            n = -n
            s = -s

        # Dip
        dip = np.degrees(np.arccos(-n[2]))

        if abs(dip) < 1e-10:
            strike = 0.0
            rake = np.degrees(np.arctan2(s[1], s[0]))
        else:
            strike = np.degrees(np.arctan2(-n[0], n[1]))
            if strike < 0:
                strike += 360.0

            # Rake
            sin_strike = np.sin(np.radians(strike))
            cos_strike = np.cos(np.radians(strike))
            rake = np.degrees(
                np.arctan2(
                    -s[2] / np.sin(np.radians(dip)),
                    s[0] * cos_strike + s[1] * sin_strike,
                )
            )

        return float(strike % 360), float(dip), float(rake)

    sdr1 = _normal_to_sdr(n1, n2)
    sdr2 = _normal_to_sdr(n2, n1)
    return sdr1, sdr2


def decompose(mt_vector: np.ndarray) -> MTDecomposition:
    """
    Decompose a moment tensor into ISO + CLVD + DC.

    Parameters
    ----------
    mt_vector : array of shape (6,)
        Independent MT elements [Mxx, Myy, Mzz, Mxy, Mxz, Myz].

    Returns
    -------
    MTDecomposition
    """
    M = vector_to_tensor(mt_vector)
    eigvals, eigvecs = np.linalg.eigh(M)

    # Sort by absolute value, descending
    idx_abs = np.argsort(np.abs(eigvals))[::-1]
    eigvals_sorted = eigvals[idx_abs]
    eigvecs_sorted = eigvecs[:, idx_abs]

    # Isotropic part
    iso = np.trace(M) / 3.0

    # Deviatoric eigenvalues
    dev_eigvals = eigvals_sorted - iso

    # Sort deviatoric by absolute value
    idx_dev = np.argsort(np.abs(dev_eigvals))[::-1]
    dev_sorted = dev_eigvals[idx_dev]

    # Scalar moment (Jost & Herrmann eq. 15)
    m0 = np.sqrt(np.sum(eigvals**2) / 2.0)

    # Moment magnitude
    mw = (2.0 / 3.0) * (np.log10(m0) - 9.1) if m0 > 0 else 0.0

    # Decomposition percentages
    # Following Jost & Herrmann (1989) convention
    abs_total = np.abs(iso) + np.abs(dev_sorted[0]) + 1e-30

    # epsilon parameter for CLVD vs DC partition
    if abs(dev_sorted[0]) > 1e-30:
        epsilon = -dev_sorted[-1] / dev_sorted[0]
    else:
        epsilon = 0.0

    iso_percent = 100.0 * np.abs(iso) / abs_total
    dev_percent = 100.0 - iso_percent
    dc_percent = dev_percent * (1.0 - 2.0 * abs(epsilon))
    clvd_percent = dev_percent * 2.0 * abs(epsilon)

    # Nodal planes
    (s1, d1, r1), (s2, d2, r2) = _mt_to_nodal_planes(eigvals, eigvecs)

    # P, T, B axes
    idx_ptb = np.argsort(eigvals)
    p_axis = _eigenvector_to_trend_plunge(eigvecs[:, idx_ptb[0]])
    b_axis = _eigenvector_to_trend_plunge(eigvecs[:, idx_ptb[1]])
    t_axis = _eigenvector_to_trend_plunge(eigvecs[:, idx_ptb[2]])

    return MTDecomposition(
        mt_tensor=M,
        mt_vector=mt_vector.copy(),
        eigenvalues=eigvals_sorted,
        eigenvectors=eigvecs_sorted,
        scalar_moment=float(m0),
        moment_magnitude=float(mw),
        iso_percent=float(iso_percent),
        clvd_percent=float(clvd_percent),
        dc_percent=float(dc_percent),
        dev_eigenvalues=dev_sorted,
        strike1=s1, dip1=d1, rake1=r1,
        strike2=s2, dip2=d2, rake2=r2,
        p_axis=p_axis,
        t_axis=t_axis,
        b_axis=b_axis,
    )
