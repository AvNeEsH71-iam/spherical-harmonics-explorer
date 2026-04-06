"""
math_core.py
============
All mathematics for spherical harmonics Y_l^m(theta, phi).

Convention (physics)
--------------------
    theta : polar angle (colatitude), in [0, pi]       — from north pole
    phi   : azimuthal angle (longitude), in [0, 2*pi]  — around equator

scipy >= 1.15  provides sph_harm_y(n, m, theta, phi) which uses the SAME
physics convention, so no argument-swapping is needed.
"""

import numpy as np
from scipy.special import sph_harm_y, lpmv
from math import factorial


# ─── Hardcoded LaTeX strings for P_ℓ(x), ℓ = 0 … 10 ────────────────────────
# These appear in the equations panel so the user sees the explicit polynomial.

LEGENDRE_EXPLICIT = {
    0:  r"$1$",
    1:  r"$x$",
    2:  r"$\frac{1}{2}(3x^2-1)$",
    3:  r"$\frac{1}{2}(5x^3-3x)$",
    4:  r"$\frac{1}{8}(35x^4-30x^2+3)$",
    5:  r"$\frac{1}{8}(63x^5-70x^3+15x)$",
    6:  r"$\frac{1}{16}(231x^6-315x^4+105x^2-5)$",
    7:  r"$\frac{1}{16}(429x^7-693x^5+315x^3-35x)$",
    8:  r"$\frac{1}{128}(6435x^8-12012x^6+6930x^4-1260x^2+35)$",
    9:  r"$\frac{1}{128}(12155x^9-25740x^7+18018x^5-4620x^3+315x)$",
    10: r"$\frac{1}{256}(46189x^{10}-109395x^8+90090x^6-30030x^4+3465x^2-63)$",
}

L_MAX = 10   # maximum degree supported


# ─── Core Functions ──────────────────────────────────────────────────────────

def Y(l: int, m: int,
      theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Complex spherical harmonic Y_l^m(theta, phi).

    Parameters
    ----------
    l, m   : degree and order  (|m| <= l, l >= 0)
    theta  : polar angle  [0, pi]     (physics convention)
    phi    : azimuthal angle [0, 2pi] (physics convention)

    Returns
    -------
    Complex ndarray of the same shape as theta and phi.
    """
    if abs(m) > l:
        raise ValueError(f"|m|={abs(m)} must be ≤ l={l}")
    # scipy >= 1.15: sph_harm_y(n, m, theta_polar, phi_azimuthal)
    # Uses physics convention directly — no swapping needed.
    return sph_harm_y(l, m, theta, phi)


def normalization(l: int, m: int) -> float:
    """
    Normalization constant N_l^m.

    N_l^m = sqrt[ (2l+1)/(4π) * (l-|m|)! / (l+|m|)! ]

    Guarantees ∫ |Y_l^m|² dΩ = 1  over the full sphere.
    """
    m_abs = abs(m)
    return float(np.sqrt(
        (2*l + 1) / (4 * np.pi)
        * factorial(l - m_abs) / factorial(l + m_abs)
    ))


def assoc_legendre_normalized(l: int, m: int,
                              theta: np.ndarray) -> np.ndarray:
    """
    Return  N_l^m · P_l^|m|(cos θ)  as a function of theta.

    This is the real polar factor appearing in Y_l^m.  Used for the
    1-D polar-profile subplot.

    Parameters
    ----------
    theta : polar angle in [0, pi]
    """
    x = np.cos(theta)
    Plm = lpmv(abs(m), l, x)          # scipy lpmv(m, l, x)
    N   = normalization(l, m)
    return N * Plm


def spherical_grid(l: int, m: int, n: int = 120):
    """
    Evaluate Y_l^m on a regular (theta, phi) meshgrid.

    Returns
    -------
    THETA : (n, n) array  — polar angle,    varies along axis-1 (columns)
    PHI   : (n, n) array  — azimuthal angle, varies along axis-0 (rows)
    Ylm   : (n, n) complex array of Y_l^m values

    Meshgrid convention
    -------------------
    THETA[i, j] = theta_1d[j]   (theta is the 'x' argument of meshgrid)
    PHI  [i, j] = phi_1d  [i]   (phi   is the 'y' argument of meshgrid)
    → shape is (n_phi, n_theta) = (n, n)
    """
    theta_1d = np.linspace(0,        np.pi, n)
    phi_1d   = np.linspace(0, 2 * np.pi,   n)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d)   # shape (n, n)
    Ylm = Y(l, m, THETA, PHI)
    return THETA, PHI, Ylm


def mode_properties(l: int, m: int) -> dict:
    """
    Return a dict of physical properties for the mode (l, m).

    Keys (strings) → values (scalars or formatted strings):
        degree, order, eigenvalue, polar_nodes, azimuthal_nodes,
        total_nodes, degeneracy, normalization, parity, bending_factor
    """
    N       = normalization(l, m)
    eig     = -l * (l + 1)
    bending = l*(l+1) * (l*(l+1) - 2) if l >= 2 else 0

    return {
        "Degree  ℓ":               l,
        "Order   m":               m,
        "Eigenvalue  λ":           f"{eig}   [= -ℓ(ℓ+1)]",
        "Polar nodes":             l - abs(m),
        "Azimuthal nodes":         abs(m),
        "Total angular nodes":     l,
        "Degeneracy  (2ℓ+1)":     2*l + 1,
        "Normaliz.  N_ℓ^m":       f"{N:.6f}",
        "Parity  (−1)^ℓ":         int((-1)**l),
        "Bending factor":          bending,
    }


def m_sign_str(m: int) -> str:
    """Return '+1', '0', or '-2' — used in LaTeX titles."""
    return f"+{m}" if m > 0 else str(m)
