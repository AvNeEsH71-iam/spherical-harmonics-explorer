"""
plotter.py
==========
Rendering engine for the Spherical Harmonics Explorer.

Entry point
-----------
    update_figure(fig, l, m, component, cmap, n)

This clears the figure and draws five subplots:
    1. 3-D shape plot   (top-left, large)
    2. Polar cross-section  (top-right)
    3. Heatmap  θ vs φ  (bottom-left)
    4. 1-D polar profile  P_l^m(cos θ)  (bottom-centre)
    5. Equations & properties panel  (bottom-right)

All axes use the same dark colour palette defined at the top of this file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.special import sph_harm_y

try:
    from .math_core import (
        Y, normalization, assoc_legendre_normalized,
        spherical_grid, mode_properties,
        LEGENDRE_EXPLICIT, m_sign_str,
    )
except ImportError:
    from math_core import (
        Y, normalization, assoc_legendre_normalized,
        spherical_grid, mode_properties,
        LEGENDRE_EXPLICIT, m_sign_str,
    )

# ─── Colour palette (matches app.py) ─────────────────────────────────────────
BG     = '#0d1117'   # main background
PANEL  = '#161b22'   # sidebar
WIDGET = '#21262d'   # widget fill
TEXT   = '#c9d1d9'   # primary text
DIM    = '#8b949e'   # secondary / axis labels
ACCENT = '#58a6ff'   # blue highlight
GREEN  = '#7ee787'   # green highlight
ORANGE = '#ffa657'   # orange highlight
CYAN   = '#79c0ff'   # cyan lines
POS    = '#388bfd'   # positive fill
NEG    = '#f85149'   # negative fill
BORDER = '#2d333b'   # grid lines / pane edges


# ─── Main entry point ────────────────────────────────────────────────────────

def update_figure(fig,
                  l: int, m: int,
                  component: str = 'real',
                  cmap: str     = 'RdBu_r',
                  n: int        = 110) -> None:
    """
    Clear *fig* and redraw all subplots for Y_l^m.

    Parameters
    ----------
    fig       : matplotlib Figure already embedded in the tkinter canvas
    l, m      : spherical harmonic degree and order
    component : 'real' | 'imag' | 'abs' | 'phase'
    cmap      : any valid matplotlib colormap name
    n         : grid resolution (higher → smoother but slower)
    """
    fig.clear()
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 3,
        figure  = fig,
        hspace  = 0.52,
        wspace  = 0.40,
        left    = 0.04, right = 0.97,
        top     = 0.91, bottom= 0.07,
    )

    ax_3d    = fig.add_subplot(gs[0, :2], projection='3d')
    ax_polar = fig.add_subplot(gs[0,  2], projection='polar')
    ax_heat  = fig.add_subplot(gs[1,  0])
    ax_theta = fig.add_subplot(gs[1,  1])
    ax_eq    = fig.add_subplot(gs[1,  2])

    THETA, PHI, Ylm = spherical_grid(l, m, n)

    _draw_3d(ax_3d,    l, m, THETA, PHI, Ylm, component, cmap)
    _draw_polar(ax_polar, l, m)
    _draw_heatmap(ax_heat,  l, m, THETA, PHI, Ylm, cmap)
    _draw_theta_profile(ax_theta, l, m)
    _draw_equations(ax_eq, l, m)

    ms = m_sign_str(m)
    comp_labels = {
        'real': 'Real Part',
        'imag': 'Imaginary Part',
        'abs':  'Absolute Value',
        'phase':'Phase',
    }
    fig.suptitle(
        fr"$Y_{{{l}}}^{{{ms}}}(\theta,\phi)$ — Spherical Harmonic"
        f"  |  Component: {comp_labels.get(component, component)}",
        color='white', fontsize=12, fontweight='bold', y=0.97,
    )


# ─── Component extraction ────────────────────────────────────────────────────

def _extract(Ylm: np.ndarray, component: str) -> np.ndarray:
    """Return a real-valued array from complex Ylm."""
    return {
        'real':  Ylm.real,
        'imag':  Ylm.imag,
        'abs':   np.abs(Ylm),
        'phase': np.angle(Ylm),
    }.get(component, Ylm.real)


# ─── 1. 3-D shape plot ───────────────────────────────────────────────────────

def _draw_3d(ax, l, m, THETA, PHI, Ylm, component, cmap):
    """
    3-D shape plot: radius r = |component(Y_l^m)|, colour = component value.

    The sign of Re(Y) or Im(Y) is encoded in the colour (red = negative,
    blue = positive when using RdBu_r), while the radius encodes magnitude,
    giving the classic two-lobe / multi-lobe orbital shapes.
    """
    vals = _extract(Ylm, component)
    R    = np.abs(vals)

    # Cartesian coordinates
    X  = R * np.sin(THETA) * np.cos(PHI)
    Yc = R * np.sin(THETA) * np.sin(PHI)
    Z  = R * np.cos(THETA)

    # Colour normalisation — symmetric about zero for signed components
    if component in ('real', 'imag'):
        vmax = float(np.max(np.abs(vals))) + 1e-12
        norm = Normalize(-vmax, vmax)
    else:
        norm = Normalize(float(vals.min()), float(vals.max()) + 1e-12)

    cmap_obj   = plt.get_cmap(cmap)
    facecolors = cmap_obj(norm(vals))

    ax.set_facecolor(BG)
    ax.plot_surface(
        X, Yc, Z,
        facecolors  = facecolors,
        alpha       = 0.90,
        linewidth   = 0,
        antialiased = True,
        shade       = False,   # preserve our custom facecolors
    )

    # Equal aspect ratio (works for all matplotlib versions)
    _set_axes_equal_3d(ax, X, Yc, Z)

    # Scalar mappable for colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.022, pad=0.12, shrink=0.55)
    _style_colorbar(cb)

    _style_3d(ax, l, m, component)


def _set_axes_equal_3d(ax, X, Y, Z):
    """Force equal aspect ratio on a 3-D axis."""
    ranges = np.array([
        [X.min(), X.max()],
        [Y.min(), Y.max()],
        [Z.min(), Z.max()],
    ])
    centre   = ranges.mean(axis=1)
    half_rng = (ranges[:, 1] - ranges[:, 0]).max() / 2 + 1e-12
    ax.set_xlim(centre[0] - half_rng, centre[0] + half_rng)
    ax.set_ylim(centre[1] - half_rng, centre[1] + half_rng)
    ax.set_zlim(centre[2] - half_rng, centre[2] + half_rng)


def _style_3d(ax, l, m, component):
    labels = {
        'real': 'Re', 'imag': 'Im', 'abs': '|Y|', 'phase': '∠Y',
    }
    ms = m_sign_str(m)
    ax.set_title(
        fr"3-D: {labels.get(component,'Re')}$(Y_{{{l}}}^{{{ms}}})$",
        color=TEXT, fontsize=9, pad=8,
    )
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(BORDER)
        axis.set_tick_params(labelsize=6, colors=DIM)
        axis.label.set_color(DIM)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.grid(True, color=BORDER, alpha=0.5, linewidth=0.4)


# ─── 2. Polar cross-section at φ = 0 ─────────────────────────────────────────

def _draw_polar(ax, l, m):
    """
    Polar plot of |Y_l^m(θ, φ=0)| vs θ, mirrored to fill [0, 2π].

    Positive Re(Y) regions are filled blue, negative regions red.
    This shows the classic 'figure-8' (l=1,m=0) and multi-lobe patterns.
    """
    theta = np.linspace(0, np.pi, 500)
    Yvals = sph_harm_y(l, m, theta, np.zeros_like(theta))   # phi = 0
    r_raw = Yvals.real

    # Mirror to get a full [0, 2π] polar plot
    theta_f = np.concatenate([theta,  2*np.pi - theta[-2::-1]])
    r_raw_f = np.concatenate([r_raw,          r_raw[-2::-1]])
    r_f     = np.abs(r_raw_f)

    ax.set_facecolor(BG)
    ax.plot(theta_f, r_f, color=CYAN, lw=1.4, zorder=3)

    # Signed fill
    pos_mask = r_raw_f >= 0
    neg_mask = ~pos_mask

    for mask, colour in [(pos_mask, POS), (neg_mask, NEG)]:
        if mask.any():
            ax.fill(
                theta_f[mask], r_f[mask],
                alpha=0.28, color=colour, zorder=2,
            )

    ms = m_sign_str(m)
    ax.set_title(
        fr"Polar: $|Y_{{{l}}}^{{{ms}}}(\theta,0)|$",
        color=TEXT, fontsize=8, pad=6,
    )
    ax.tick_params(colors=DIM, labelsize=5)
    ax.set_yticklabels([])
    ax.grid(color=BORDER, linewidth=0.4)
    ax.spines['polar'].set_edgecolor(DIM)


# ─── 3. Heatmap θ vs φ ───────────────────────────────────────────────────────

def _draw_heatmap(ax, l, m, THETA, PHI, Ylm, cmap):
    """
    2-D colour map of Re(Y_l^m) on the (φ, θ) plane.

    x-axis : φ in [0°, 360°]   (azimuthal)
    y-axis : θ in [0°, 180°]   (polar, 0 = north pole)

    Meshgrid reminder
    -----------------
    THETA, PHI each have shape (n_phi, n_theta):
        PHI  [:, 0]  gives the φ values   → x-axis
        THETA[0, :]  gives the θ values   → y-axis
        vals .T      has shape (n_theta, n_phi) — matches pcolormesh expectation
    """
    vals = Ylm.real

    phi_1d   = PHI  [:, 0] * 180 / np.pi   # shape (n_phi,)
    theta_1d = THETA[0, :] * 180 / np.pi   # shape (n_theta,)

    vmax = float(np.max(np.abs(vals))) + 1e-12
    im   = ax.pcolormesh(
        phi_1d, theta_1d, vals.T,
        cmap    = cmap,
        shading = 'auto',
        vmin    = -vmax,
        vmax    =  vmax,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style_colorbar(cb)

    ms = m_sign_str(m)
    ax.set_xlabel('φ  (deg)', color=DIM, fontsize=7)
    ax.set_ylabel('θ  (deg)', color=DIM, fontsize=7)
    ax.set_title(
        fr"Heatmap: Re$(Y_{{{l}}}^{{{ms}}})$",
        color=TEXT, fontsize=9,
    )
    ax.tick_params(colors=DIM, labelsize=7)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


# ─── 4. 1-D polar profile P_l^m(cos θ) ──────────────────────────────────────

def _draw_theta_profile(ax, l, m):
    """
    1-D plot of the normalised polar factor N_l^m · P_l^|m|(cos θ) vs θ.

    Vertical dashed lines mark the angular nodes (zeros of P_l^m).
    The number of nodes equals l − |m|, confirming the analytical result.
    """
    theta   = np.linspace(0, np.pi, 600)
    profile = assoc_legendre_normalized(l, m, theta)

    ax.set_facecolor(BG)
    ax.axhline(0, color=DIM, lw=0.6, zorder=1)
    ax.plot(np.degrees(theta), profile, color=CYAN, lw=1.5, zorder=3)

    ax.fill_between(
        np.degrees(theta), profile, 0,
        where=(profile >= 0), alpha=0.22, color=POS, zorder=2,
    )
    ax.fill_between(
        np.degrees(theta), profile, 0,
        where=(profile <  0), alpha=0.22, color=NEG, zorder=2,
    )

    # Mark nodes with vertical dashed lines
    sign_changes = np.where(np.diff(np.sign(profile)))[0]
    for idx in sign_changes:
        ax.axvline(
            np.degrees(theta[idx]),
            color=DIM, lw=0.7, linestyle='--', alpha=0.65,
        )

    n_nodes = len(sign_changes)
    ms  = m_sign_str(m)
    ax.set_xlabel('θ  (deg)', color=DIM, fontsize=7)
    ax.set_ylabel(
        fr"$\mathcal{{N}}\,P_{{{l}}}^{{{abs(m)}}}(\cos\theta)$",
        color=DIM, fontsize=7,
    )
    ax.set_title(
        fr"Polar Profile  ({n_nodes} node{'s' if n_nodes != 1 else ''})",
        color=TEXT, fontsize=9,
    )
    ax.set_xlim(0, 180)
    ax.tick_params(colors=DIM, labelsize=7)
    ax.grid(color=BORDER, lw=0.4, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


# ─── 5. Equations & properties panel ─────────────────────────────────────────

def _draw_equations(ax, l, m):
    """
    Text panel rendered on a plain Axes using matplotlib mathtext.

    Shows:
    • General Y_l^m formula
    • This specific mode with numerical N
    • P_l(x) explicit polynomial (from LEGENDRE_EXPLICIT)
    • Associated Legendre and Rodrigues definitions
    • Eigenvalue equation
    • Orthonormality relation
    • Node count and bending energy factor
    """
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    N       = normalization(l, m)
    ms      = m_sign_str(m)
    eig     = -l*(l+1)
    bending = l*(l+1)*(l*(l+1)-2) if l >= 2 else 0
    polar_n = l - abs(m)
    azim_n  = abs(m)
    Pl_expr = LEGENDRE_EXPLICIT.get(l, fr"$P_{{{l}}}(x)$")

    # (label, LaTeX expression, colour)
    rows = [
        ("General",
         r"$Y_\ell^m = \mathcal{N}_\ell^m\,P_\ell^m(\cos\theta)\,e^{im\phi}$",
         ACCENT),

        ("This mode",
         fr"$Y_{{{l}}}^{{{ms}}} ="
         fr"\,{N:.4f}"
         fr"\cdot P_{{{l}}}^{{{abs(m)}}}(\cos\theta)"
         fr"\,e^{{{ms}i\phi}}$",
         GREEN),

        (fr"$P_{{{l}}}(x)$",
         Pl_expr,
         CYAN),

        ("Assoc. Leg.",
         fr"$P_\ell^m(x)=(-1)^m(1-x^2)^{{m/2}}"
         fr"\frac{{d^m}}{{dx^m}}P_\ell(x)$",
         TEXT),

        ("Rodrigues",
         fr"$P_\ell(x)=\frac{{1}}{{2^\ell\,\ell!}}"
         fr"\frac{{d^\ell}}{{dx^\ell}}(x^2-1)^\ell$",
         TEXT),

        ("Eigenvalue",
         fr"$\nabla^2_\Omega Y_{{{l}}}^{{{ms}}}"
         fr"={eig}\,Y_{{{l}}}^{{{ms}}}$",
         ORANGE),

        ("Orthonorm.",
         r"$\langle Y_{\ell'}^{{m'}}|Y_\ell^m\rangle"
         r"=\delta_{{\ell\ell'}}\delta_{{mm'}}$",
         TEXT),

        ("Nodes",
         fr"polar: {polar_n}   azimuthal: {azim_n}   total: {l}",
         GREEN),

        ("Bend factor",
         fr"$\ell(\ell+1)[\ell(\ell+1)-2]={bending}$",
         DIM),
    ]

    # Header
    ax.text(0.50, 0.985, "Equations  &  Properties",
            ha='center', va='top', fontsize=8.5,
            color=ACCENT, fontweight='bold',
            transform=ax.transAxes)

    # Thin separator line
    ax.plot([0, 1], [0.950, 0.950], color=BORDER, lw=0.7,
            transform=ax.transAxes)

    y  = 0.90
    dy = 0.096
    for label, expr, colour in rows:
        ax.text(0.01, y, label + ":",
                ha='left', va='top', fontsize=5.8,
                color=DIM, fontstyle='italic',
                transform=ax.transAxes)
        ax.text(0.50, y, expr,
                ha='center', va='top', fontsize=6.6,
                color=colour,
                transform=ax.transAxes)
        y -= dy


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _style_colorbar(cb):
    """Apply dark-theme styling to a colourbar."""
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=6)
    cb.outline.set_edgecolor(DIM)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
