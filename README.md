# ⟨Y|ℓ,m⟩ Spherical Harmonics Explorer

An interactive Python GUI for visualising **spherical harmonics Y_ℓ^m(θ,φ)**,
associated Legendre polynomials, and their physical meaning — designed for
physics students and researchers.

---

##  Features

| Feature | Detail |
|---|---|
| **3-D shape plot** | r = \|Re(Y)\|, colour = sign — the classic orbital lobes |
| **Polar cross-section** | \|Y(θ, φ=0)\| mirrored to full 2π |
| **Heatmap (θ–φ plane)** | Re(Y) as a colour map |
| **1-D polar profile** | N·P_ℓ^\|m\|(cos θ) with node markers |
| **Equations panel** | Live LaTeX render: Y_ℓ^m, P_ℓ, P_ℓ^m, eigenvalue, orthonormality |
| **Component selector** | Real · Imaginary · Absolute · Phase |
| **Colormap selector** | 10 colormaps (RdBu_r, coolwarm, viridis, plasma, …) |
| **Navigation buttons** | Step ℓ and m one at a time |
| **Properties sidebar** | λ, nodes, degeneracy, normalisation constant, bending factor |
| **Export** | Save as PNG / PDF / SVG at 200 DPI |
| **Physics reference** | Built-in popup with all key formulae |

---

##  Quick Start

```bash
# 1. Clone
git clone https://github.com/AvNeEsH71-iam/spherical-harmonics-explorer.git
cd spherical-harmonics-explorer

# 2. Install dependencies  (tkinter ships with Python — no pip needed)
pip install -r requirements.txt

# 3. Run
python main.py
```

> **Linux note** — if you see `_tkinter` errors, install tkinter first:
> ```bash
> sudo apt-get install python3-tk
> ```

---

##  Modern Interactive Web Dashboard

An interactive React + Tailwind + Plotly dashboard is included in `dashboard.html`.





### Included UI features

- Responsive dashboard grid with panel cards
- Click any panel to open expanded modal view with zoom animation
- Per-panel toolbar: Expand, Download, Reset (for 3D)
- Resizable cards (`resize: both`) for dynamic layout tuning
- Lazy loading for the heavy 3D panel
- Real / Imaginary / Absolute component toggle
- Sliders for dynamic `l` and `m`
- Hover tooltips on all plots

---

##  Project Structure

```
spherical_harmonics_explorer/
│
├── main.py                  ← entry point
├── requirements.txt
├── README.md
│
└── explorer/
    ├── __init__.py
    ├── math_core.py         ← all mathematics (Y_ℓ^m, P_ℓ^m, N, properties)
    ├── plotter.py           ← five subplot renderers
    └── app.py               ← tkinter GUI (SphericalHarmonicsApp)
```

---

##  Physics Background

### Spherical Harmonics

Spherical harmonics Y_ℓ^m(θ, φ) are the **natural basis functions on a sphere** —
the sphere's analogue of Fourier modes on a circle.  Any smooth function on a
spherical surface can be expanded as:

```
f(θ,φ) = Σ_{ℓ=0}^∞  Σ_{m=−ℓ}^{ℓ}  f_{ℓm} Y_ℓ^m(θ,φ)
```

They arise as the angular solutions to **Laplace's equation** in spherical
coordinates and appear in electrostatics, quantum mechanics, and soft-matter
physics.

### Definition

```
Y_ℓ^m(θ,φ) = N_ℓ^m · P_ℓ^m(cos θ) · e^{imφ}

N_ℓ^m = √[ (2ℓ+1)/4π · (ℓ−|m|)!/(ℓ+|m|)! ]
```

### Indices

| Symbol | Name | Range | Physical meaning |
|---|---|---|---|
| ℓ | degree | 0, 1, 2, … | total angular nodes |
| m | order | −ℓ … +ℓ | azimuthal oscillation count |

### Eigenvalue

```
∇²_Ω Y_ℓ^m = −ℓ(ℓ+1) · Y_ℓ^m
```

The factor ℓ(ℓ+1) appears in the bending energy of vesicle membranes,
the energy levels of hydrogen, and many other physical systems.

### Associated Legendre Polynomials

```
P_ℓ^m(x) = (−1)^m (1−x²)^{m/2} · d^m/dx^m  P_ℓ(x)

P_ℓ(x)   = 1/(2^ℓ ℓ!)  · d^ℓ/dx^ℓ  (x²−1)^ℓ    [Rodrigues]
```

### Why ℓ ≥ 2 for membrane physics?

| Mode | Energy | Reason excluded |
|---|---|---|
| ℓ = 0 | 0 (volume change) | Suppressed by incompressibility |
| ℓ = 1 | 0 (translation) | Rigid-body motion, no bending |
| ℓ ≥ 2 | κ · ℓ(ℓ+1)[ℓ(ℓ+1)−2] | **Physical deformation modes** |

---

##  Controls Reference

| Control | Function |
|---|---|
| ℓ spinbox / ◄ ► | Change degree (0–10) |
| m spinbox / ◄ ► | Change order (auto-clamped to \[−ℓ, +ℓ\]) |
| Component radio | Switch between Re, Im, \|Y\|, Phase |
| Colormap | Choose colourmap |
| Export | Save figure as PNG / PDF / SVG |
| Reset | Return to Y_2^0 |
| Physics Reference | Show formulae popup |
| Toolbar | Zoom, pan, rotate 3-D view |

---

##  Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Numerical arrays and meshgrid |
| `scipy` | ≥ 1.7 | `sph_harm`, `lpmv` — exact computation |
| `matplotlib` | ≥ 3.5 | All plotting + tkinter embedding |
| `tkinter` | stdlib | GUI framework |

---


##  Acknowledgements

- Scipy's `scipy.special.sph_harm` for exact Y_ℓ^m values.
- Jackson, *Classical Electrodynamics* — notation and conventions.
- Arfken & Weber, *Mathematical Methods for Physicists* — Legendre polynomials.
