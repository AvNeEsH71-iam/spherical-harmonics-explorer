"""
app.py
======
Main GUI application for the Spherical Harmonics Explorer.

Layout
------
┌──────────────────────────────────────────────────────────────────────┐
│  Left sidebar (fixed 270 px)  │  Matplotlib figure (expanding)      │
│                               │                                      │
│  • Title / branding           │  [3D shape] [Polar]                  │
│  • ℓ and m spinboxes          │  [Heatmap] [θ profile] [Equations]   │
│  • Component radio buttons    │                                      │
│  • Colormap dropdown          │                                      │
│  • ℓ / m navigation buttons   │                                      │
│  • Properties summary         │                                      │
│  • Export / Reset buttons     │                                      │
├──────────────────────────────┴──────────────────────────────────────┤
│  Status bar: Y_l^m | λ | nodes | N                                  │
└─────────────────────────────────────────────────────────────────────┘
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

try:
    from .math_core import mode_properties, L_MAX, m_sign_str, normalization
    from .plotter import update_figure
except ImportError:
    from math_core import mode_properties, L_MAX, m_sign_str, normalization
    from plotter import update_figure


# ─── Dark colour palette ──────────────────────────────────────────────────────
C = {
    'bg_dark':   '#0d1117',
    'bg_panel':  '#161b22',
    'bg_widget': '#21262d',
    'accent':    '#58a6ff',
    'accent2':   '#7ee787',
    'orange':    '#ffa657',
    'text':      '#c9d1d9',
    'dim':       '#8b949e',
    'border':    '#30363d',
    'red':       '#f85149',
}

CMAPS = [
    'RdBu_r', 'coolwarm', 'seismic', 'bwr',
    'viridis', 'plasma', 'inferno', 'magma',
    'twilight', 'hsv',
]

COMPONENTS = [
    ('Real  Re(Y)',       'real'),
    ('Imaginary  Im(Y)',  'imag'),
    ('Absolute  |Y|',     'abs'),
    ('Phase  ∠Y',         'phase'),
]


class SphericalHarmonicsApp:
    """
    Main application window.

    All UI state is stored in tk.*Var objects; changes are debounced
    (200 ms) before triggering a replot so rapid spinbox clicks remain smooth.
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self, root: tk.Tk):
        self.root = root
        self._configure_root()
        self._init_variables()
        self._apply_ttk_style()
        self._build_ui()
        self._update_m_bounds()
        self._schedule_update()   # initial plot

    def _configure_root(self):
        self.root.title("Spherical Harmonics Explorer  |  Y_ℓ^m(θ,φ)")
        self.root.geometry("1320x820")
        self.root.minsize(960, 640)
        self.root.configure(bg=C['bg_dark'])
        try:
            self.root.state('zoomed')   # maximise on Windows
        except Exception:
            pass

    def _init_variables(self):
        self.l_var         = tk.IntVar(value=2)
        self.m_var         = tk.IntVar(value=1)
        self.component_var = tk.StringVar(value='real')
        self.cmap_var      = tk.StringVar(value='RdBu_r')
        self._pending: str | None = None    # after() job id

    def _apply_ttk_style(self):
        s = ttk.Style()
        s.theme_use('clam')
        bg, fg, wb = C['bg_panel'], C['text'], C['bg_widget']

        s.configure('Dark.TFrame',
                     background=bg)
        s.configure('Dark.TLabel',
                     background=bg, foreground=fg,
                     font=('Helvetica', 9))
        s.configure('Header.TLabel',
                     background=bg, foreground=C['accent'],
                     font=('Helvetica', 13, 'bold'))
        s.configure('Sub.TLabel',
                     background=bg, foreground=C['dim'],
                     font=('Helvetica', 7))
        s.configure('Section.TLabel',
                     background=bg, foreground=C['accent2'],
                     font=('Helvetica', 8, 'bold'))
        s.configure('Dark.TRadiobutton',
                     background=bg, foreground=fg,
                     font=('Helvetica', 9),
                     indicatorcolor=C['accent'])
        s.map('Dark.TRadiobutton',
              background=[('active', bg)],
              foreground=[('active', C['accent'])])
        s.configure('Dark.TSpinbox',
                     fieldbackground=wb, foreground=fg,
                     background=wb, arrowcolor=fg,
                     font=('Helvetica', 11, 'bold'))
        s.configure('Dark.TCombobox',
                     fieldbackground=wb, foreground=fg,
                     background=wb, arrowcolor=fg,
                     font=('Helvetica', 9))
        s.map('Dark.TCombobox',
              fieldbackground=[('readonly', wb)])
        s.configure('Dark.TButton',
                     background='#2d333b', foreground=fg,
                     font=('Helvetica', 9),
                     padding=4, relief='flat')
        s.map('Dark.TButton',
              background=[('active', '#3d4450')],
              foreground=[('active', 'white')])
        s.configure('Nav.TButton',
                     background='#2d333b', foreground=C['accent'],
                     font=('Helvetica', 11, 'bold'),
                     padding=3, relief='flat')
        s.map('Nav.TButton',
              background=[('active', '#3d4450')])

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Left sidebar
        self.sidebar = tk.Frame(
            self.root, bg=C['bg_panel'],
            width=270, relief='flat',
        )
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        # Right canvas area
        self.canvas_frame = tk.Frame(self.root, bg=C['bg_dark'])
        self.canvas_frame.pack(side='right', fill='both', expand=True)

        # Status bar (bottom of root)
        self._build_status_bar()

        self._build_sidebar()
        self._build_canvas()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        p = self.sidebar

        # ── Branding header ──
        hdr = tk.Frame(p, bg='#1f2937', pady=12)
        hdr.pack(fill='x')
        ttk.Label(hdr, text="⟨Y|ℓ,m⟩  Explorer",
                  style='Header.TLabel',
                  background='#1f2937').pack()
        ttk.Label(hdr, text="Spherical Harmonics Visualiser",
                  style='Sub.TLabel',
                  background='#1f2937').pack()

        # Scrollable inner frame
        inner = tk.Frame(p, bg=C['bg_panel'])
        inner.pack(fill='both', expand=True, padx=10, pady=6)

        self._section(inner, "Quantum Numbers")
        self._build_lm_controls(inner)
        self._build_constraint_label(inner)

        self._section(inner, "Component")
        self._build_component_selector(inner)

        self._section(inner, "Colormap")
        self._build_cmap_selector(inner)

        self._section(inner, "Navigate  (←  →)")
        self._build_nav_buttons(inner)

        self._section(inner, "Mode Properties")
        self._build_props_display(inner)

        self._section(inner, "Actions")
        self._build_action_buttons(inner)

    def _section(self, parent, text):
        """Render a small section header with a horizontal rule."""
        f = tk.Frame(parent, bg=C['bg_panel'])
        f.pack(fill='x', pady=(8, 2))
        ttk.Label(f, text=f"── {text}", style='Section.TLabel').pack(anchor='w')

    def _build_lm_controls(self, parent):
        f = tk.Frame(parent, bg=C['bg_panel'])
        f.pack(fill='x', pady=4)

        # ℓ row
        row = tk.Frame(f, bg=C['bg_panel'])
        row.pack(fill='x', pady=2)
        ttk.Label(row, text="ℓ  (degree)  [0…10]",
                  style='Dark.TLabel').pack(anchor='w')

        self.l_spinbox = ttk.Spinbox(
            row, from_=0, to=L_MAX, width=6,
            textvariable=self.l_var,
            style='Dark.TSpinbox',
            command=self._on_l_spinbox,
        )
        self.l_spinbox.pack(anchor='w', pady=2)

        # m row
        row2 = tk.Frame(f, bg=C['bg_panel'])
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="m  (order)   [−ℓ … +ℓ]",
                  style='Dark.TLabel').pack(anchor='w')

        self.m_spinbox = ttk.Spinbox(
            row2, from_=-L_MAX, to=L_MAX, width=6,
            textvariable=self.m_var,
            style='Dark.TSpinbox',
            command=self._on_m_spinbox,
        )
        self.m_spinbox.pack(anchor='w', pady=2)

    def _build_constraint_label(self, parent):
        self.constraint_var = tk.StringVar()
        lbl = ttk.Label(parent,
                        textvariable=self.constraint_var,
                        style='Dark.TLabel')
        lbl.pack(anchor='w', pady=(0, 4))
        self._refresh_constraint_label()

    def _build_component_selector(self, parent):
        f = tk.Frame(parent, bg=C['bg_panel'])
        f.pack(fill='x', pady=4)
        for display, value in COMPONENTS:
            rb = ttk.Radiobutton(
                f, text=display, value=value,
                variable=self.component_var,
                style='Dark.TRadiobutton',
                command=self._schedule_update,
            )
            rb.pack(anchor='w', pady=1)

    def _build_cmap_selector(self, parent):
        f = tk.Frame(parent, bg=C['bg_panel'])
        f.pack(fill='x', pady=4)
        cb = ttk.Combobox(
            f, textvariable=self.cmap_var,
            values=CMAPS, state='readonly',
            style='Dark.TCombobox', width=18,
        )
        cb.pack(anchor='w')
        cb.bind('<<ComboboxSelected>>', lambda _: self._schedule_update())

    def _build_nav_buttons(self, parent):
        """ℓ and m navigation with ◄ ► buttons."""
        f = tk.Frame(parent, bg=C['bg_panel'])
        f.pack(fill='x', pady=4)

        for label, callback in [
            ("◄ ℓ", lambda: self._step_l(-1)),
            ("ℓ ►", lambda: self._step_l(+1)),
        ]:
            ttk.Button(f, text=label, style='Nav.TButton',
                       command=callback).pack(side='left', padx=2)

        tk.Frame(f, bg=C['bg_panel'], width=6).pack(side='left')

        for label, callback in [
            ("◄ m", lambda: self._step_m(-1)),
            ("m ►", lambda: self._step_m(+1)),
        ]:
            ttk.Button(f, text=label, style='Nav.TButton',
                       command=callback).pack(side='left', padx=2)

    def _build_props_display(self, parent):
        """Read-only text widget showing mode properties."""
        self.props_text = tk.Text(
            parent,
            height=10, width=28,
            bg='#0d1117', fg=C['text'],
            font=('Courier', 8),
            relief='flat', state='disabled',
            insertbackground=C['text'],
        )
        self.props_text.pack(fill='x', pady=4)
        self._refresh_props_display()

    def _build_action_buttons(self, parent):
        f = tk.Frame(parent, bg=C['bg_panel'])
        f.pack(fill='x', pady=6)

        ttk.Button(f, text="💾  Export PNG / PDF",
                   style='Dark.TButton',
                   command=self._export).pack(fill='x', pady=2)

        ttk.Button(f, text="🔄  Reset to ℓ=2, m=0",
                   style='Dark.TButton',
                   command=self._reset).pack(fill='x', pady=2)

        ttk.Button(f, text="ℹ️  Physics Reference",
                   style='Dark.TButton',
                   command=self._show_reference).pack(fill='x', pady=2)

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _build_canvas(self):
        self.fig = plt.figure(
            figsize=(10.5, 7.0),
            dpi=100,
            facecolor=C['bg_dark'],
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True,
                                         padx=2, pady=2)

        # Navigation toolbar (dark-ish)
        tb_frame = tk.Frame(self.canvas_frame, bg=C['bg_dark'])
        tb_frame.pack(fill='x')
        toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)
        toolbar.config(background=C['bg_panel'])
        for child in toolbar.winfo_children():
            try:
                child.config(background=C['bg_panel'],
                             foreground=C['text'])
            except Exception:
                pass
        toolbar.update()

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Initialising …")
        bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg='#161b22', fg=C['dim'],
            font=('Courier', 8),
            anchor='w', padx=10, pady=3,
            relief='flat',
        )
        bar.pack(side='bottom', fill='x')

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_l_spinbox(self):
        self._clamp_l()
        self._update_m_bounds()
        self._clamp_m()
        self._refresh_constraint_label()
        self._refresh_props_display()
        self._schedule_update()

    def _on_m_spinbox(self):
        self._clamp_m()
        self._refresh_constraint_label()
        self._refresh_props_display()
        self._schedule_update()

    def _step_l(self, delta: int):
        self.l_var.set(max(0, min(L_MAX, self.l_var.get() + delta)))
        self._on_l_spinbox()

    def _step_m(self, delta: int):
        l = self.l_var.get()
        self.m_var.set(max(-l, min(l, self.m_var.get() + delta)))
        self._on_m_spinbox()

    def _clamp_l(self):
        try:
            val = int(self.l_spinbox.get())
        except ValueError:
            val = 2
        self.l_var.set(max(0, min(L_MAX, val)))

    def _clamp_m(self):
        l = self.l_var.get()
        try:
            val = int(self.m_spinbox.get())
        except ValueError:
            val = 0
        self.m_var.set(max(-l, min(l, val)))

    def _update_m_bounds(self):
        l = self.l_var.get()
        self.m_spinbox.configure(from_=-l, to=l)

    def _refresh_constraint_label(self):
        l = self.l_var.get()
        m = self.m_var.get()
        ok = abs(m) <= l
        sym  = "✓" if ok else "✗"
        clr  = C['accent2'] if ok else C['red']
        txt  = f"  {sym}  |m|={abs(m)} ≤ ℓ={l}"
        self.constraint_var.set(txt)
        # Recolour: find the label widget
        for w in self.sidebar.winfo_children():
            pass  # we use textvariable; colour is static (handled via ttk)

    def _refresh_props_display(self):
        l   = self.l_var.get()
        m   = self.m_var.get()
        props = mode_properties(l, m)
        lines = []
        for key, val in props.items():
            lines.append(f"  {key:<22} {val}")
        text = "\n".join(lines)

        self.props_text.configure(state='normal')
        self.props_text.delete('1.0', 'end')
        self.props_text.insert('end', text)
        self.props_text.configure(state='disabled')

    # ── Debounced update ──────────────────────────────────────────────────────

    def _schedule_update(self, delay_ms: int = 180):
        """Cancel any pending replot and schedule a new one after delay_ms."""
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(delay_ms, self._do_update)

    def _do_update(self):
        self._pending = None
        l = self.l_var.get()
        m = self.m_var.get()

        self.status_var.set(
            f"  Plotting  Y_{l}^{m_sign_str(m)}  |  "
            f"component: {self.component_var.get()}  |  "
            f"colormap: {self.cmap_var.get()}  …"
        )
        self.root.update_idletasks()

        update_figure(
            self.fig, l, m,
            component = self.component_var.get(),
            cmap      = self.cmap_var.get(),
        )
        self.canvas.draw()
        self._update_status(l, m)

    def _update_status(self, l: int, m: int):
        ms   = m_sign_str(m)
        eig  = -l*(l+1)
        pn   = l - abs(m)
        an   = abs(m)
        N    = normalization(l, m)
        self.status_var.set(
            f"  Y_{l}^{ms}(θ,φ)  │  λ = {eig}  │  "
            f"polar nodes: {pn}  │  azimuthal nodes: {an}  │  "
            f"N = {N:.5f}  │  degeneracy: {2*l+1}"
        )

    # ── Actions ───────────────────────────────────────────────────────────────

    def _export(self):
        l  = self.l_var.get()
        m  = self.m_var.get()
        ms = m_sign_str(m)
        path = filedialog.asksaveasfilename(
            title          = "Save figure",
            defaultextension='.png',
            initialfile    = f"Y_{l}_{ms}.png",
            filetypes      = [
                ('PNG image',  '*.png'),
                ('PDF document', '*.pdf'),
                ('SVG vector',  '*.svg'),
            ],
        )
        if path:
            try:
                self.fig.savefig(
                    path, dpi=200,
                    bbox_inches='tight',
                    facecolor=C['bg_dark'],
                )
                self.status_var.set(f"  ✓  Saved to {path}")
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def _reset(self):
        self.l_var.set(2)
        self.m_var.set(0)
        self.component_var.set('real')
        self.cmap_var.set('RdBu_r')
        self._update_m_bounds()
        self._refresh_constraint_label()
        self._refresh_props_display()
        self._schedule_update()

    def _show_reference(self):
        """Pop-up window with quick physics reference."""
        win = tk.Toplevel(self.root)
        win.title("Physics Reference")
        win.geometry("560x680")
        win.configure(bg=C['bg_dark'])

        text = tk.Text(
            win,
            bg=C['bg_dark'], fg=C['text'],
            font=('Courier', 9),
            wrap='word', relief='flat',
            padx=16, pady=16,
        )
        text.pack(fill='both', expand=True)

        ref = """
SPHERICAL HARMONICS  Y_ℓ^m(θ, φ)
══════════════════════════════════════════════════════════

DEFINITION
  Y_ℓ^m(θ,φ) = N_ℓ^m · P_ℓ^m(cos θ) · e^{imφ}

  N_ℓ^m = √[ (2ℓ+1)/4π · (ℓ−|m|)!/(ℓ+|m|)! ]

INDICES
  ℓ ∈ {0, 1, 2, …}       degree  (number of angular nodes)
  m ∈ {−ℓ, …, +ℓ}        order   (azimuthal quantum number)

LEGENDRE POLYNOMIAL  P_ℓ(x)  — Rodrigues' formula
  P_ℓ(x) = 1/(2^ℓ ℓ!) · d^ℓ/dx^ℓ (x²−1)^ℓ

ASSOCIATED LEGENDRE  P_ℓ^m(x)
  P_ℓ^m(x) = (−1)^m (1−x²)^{m/2} · d^m/dx^m P_ℓ(x)

EIGENVALUE (angular Laplacian)
  ∇²_Ω Y_ℓ^m = −ℓ(ℓ+1) · Y_ℓ^m        λ = −ℓ(ℓ+1)

ORTHONORMALITY
  ∫ [Y_ℓ'^{m'}]* Y_ℓ^m dΩ = δ_{ℓℓ'} δ_{mm'}

COMPLETENESS  (sphere's Fourier theorem)
  f(θ,φ) = Σ_{ℓ=0}^∞  Σ_{m=−ℓ}^{ℓ}  f_{ℓm} Y_ℓ^m(θ,φ)

PARITY
  Y_ℓ^m(π−θ, φ+π) = (−1)^ℓ Y_ℓ^m(θ,φ)

CONJUGATE SYMMETRY
  [Y_ℓ^m]* = (−1)^m Y_ℓ^{−m}

NODE COUNT
  polar nodes    = ℓ − |m|   (zeros along θ)
  azimuthal nodes = |m|       (zeros along φ)
  total nodes    = ℓ

PHYSICAL EXCLUSIONS (membrane fluctuations)
  ℓ = 0 : uniform volume change → suppressed by incompressibility
  ℓ = 1 : rigid translation     → zero bending energy
  ℓ ≥ 2 : genuine shape modes   → bending factor ℓ(ℓ+1)[ℓ(ℓ+1)−2]

SUBPLOT GUIDE
  3-D shape   : radius r = |Re(Y)|; colour = sign (red −, blue +)
  Polar       : |Y(θ, φ=0)| vs θ, mirrored to full 2π
  Heatmap     : Re(Y) as a function of φ (x) and θ (y)
  Polar profile: N·P_ℓ^|m|(cos θ) vs θ  (dashed = nodes)
  Equations   : key formulae and numerical values for this mode
"""
        text.insert('end', ref)
        text.configure(state='disabled')
