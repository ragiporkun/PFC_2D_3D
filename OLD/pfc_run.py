"""
pfc_run.py  --  shared non-isothermal PFC engine (2D, FiPy) for the revision runs.

This is the verified rev9 solver wrapped in run_simulation(cfg). The thermal
physics is the corrected one: constant heat capacity c_p = alpha, latent heat as
the RHS source (beta + psi)*dpsi/dt, physical far-field boundary. run1.py and
run2.py are thin drivers that only set the Config.

Outputs (per run, in cfg.outdir):
    psi_<step>.npy, T_<step>.npy   snapshots for post-processing
    GR_log.csv                     solidification conditions G, cooling rate, R
    frame_<step>.png               quick-look psi/T figures

The G/R log is the key physics output: columnar grain width and columnar-vs-
equiaxed selection are governed by G (gradient at the melt isotherm) and R
(solidification-front speed), NOT by raw scan speed. Reporting G, R, G/R lets
the regime-mismatch finding (M5/R2-02) be answered on the controlling variables.
"""
import os
from dataclasses import dataclass, field
import numpy as np

os.environ.setdefault('FIPY_SOLVERS', 'scipy')
import fipy as fp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A0 = 4.0 * np.pi / np.sqrt(3.0)        # triangular lattice spacing ~7.255


@dataclass
class Config:
    # --- domain ---
    nx: int = 384
    ny: int = 128
    dx: float = 1600 / 2**11           # ~0.78125  -> ~9.3 nodes per lattice spacing
    # --- time ---
    dt: float = 1.0
    # --- structural model (unchanged from rev9) ---
    psi0: float = 0.0
    T0: float = 0.6
    M: float = 0.6
    tau_psi: float = 1.0
    # --- THERMAL knobs (physical values -- the corrected coupling) ---
    alpha_cp: float = 3.0              # heat capacity c_p (= alpha). NOT 0.1.
    beta_latent: float = 0.5           # latent-heat coupling. NOT 0.025.
    # --- process / source ---
    v_x: float = 1.0                   # dimensionless scan speed (map to m/s in paper)
    power: float = 1200.0              # tune via thermal_diag so peak T ~ T_m
    absorption: float = 0.3
    sigma_x_a0: float = 3.0            # beam half-widths in LATTICE UNITS (>> a0)
    sigma_y_a0: float = 6.0
    # --- seeding (programmatic polycrystal, fits any domain) ---
    seed: int = 7
    seed_spacing: float = 28.0
    grain_radius: float = 16.0
    # --- schedule (steps) ---
    warmup_steps: int = 200            # let seeds crystallize into a polycrystal
    cooldown_steps: int = 250          # solidify after the source leaves
    sink_warm: float = 0.01
    sink_scan: float = 0.01
    sink_cool: float = 0.05
    # --- io ---
    outdir: str = "run_out"
    save_every: int = 10
    gr_every: int = 10                 # log G/R cadence during scan

    @property
    def Lx(self): return self.nx * self.dx
    @property
    def Ly(self): return self.ny * self.dx


def seed_polycrystal(cfg, xv, yv):
    """Tile small oriented seeds across the domain; they grow to fill it."""
    rng = np.random.default_rng(cfg.seed)
    psi = np.full_like(xv, cfg.psi0)
    p = A0
    A = cfg.psi0
    B = ((4 - 12*A) + 4*np.sqrt(-15*(1 - 1/cfg.T0) + 24*A + 1 - 36*A**2)) / 15
    pre = -0.5
    sx = np.arange(cfg.seed_spacing/2, cfg.Lx, cfg.seed_spacing)
    sy = np.arange(cfg.seed_spacing/2, cfg.Ly, cfg.seed_spacing)
    for cx in sx:
        for cy in sy:
            ang = rng.uniform(-np.pi/6, np.pi/6)     # within triangular symmetry
            xs, ys = xv - cx, yv - cy
            wave = A + B*(np.cos(2*np.pi*(xs*np.cos(ang)+ys*np.sin(ang))/p) *
                          np.cos(2*np.pi*(xs*np.sin(ang)-ys*np.cos(ang))/(np.sqrt(3)*p)) -
                          pre*np.cos(4*np.pi*(xs*np.sin(ang)-ys*np.cos(ang))/(np.sqrt(3)*p)))
            m = (xs**2 + ys**2) <= cfg.grain_radius**2
            psi[m] = wave[m]
    return psi


def _front_GR(Tnew, Told, cfg, Tm=1.0, band=0.06):
    """G (|grad T| at the melt isotherm), cooling rate, and R = cooling/G."""
    T2 = Tnew.reshape(cfg.ny, cfg.nx)
    gy, gx = np.gradient(T2, cfg.dx)
    G = np.sqrt(gx**2 + gy**2)
    front = np.abs(T2 - Tm) < band
    if front.sum() == 0:
        return None
    Gf = float(G[front].mean())
    cooling = float(np.abs((Tnew - Told).reshape(cfg.ny, cfg.nx)[front]).mean() / cfg.dt)
    R = cooling / max(Gf, 1e-12)
    return Gf, cooling, R


def run_simulation(cfg: Config):
    os.makedirs(cfg.outdir, exist_ok=True)
    mesh = fp.Grid2D(nx=cfg.nx, ny=cfg.ny, dx=cfg.dx, dy=cfg.dx)
    x, y = mesh.x, mesh.y
    xv, yv = x.value, y.value

    psi = fp.CellVariable(mesh=mesh, name="psi", value=cfg.psi0, hasOld=True)
    mu  = fp.CellVariable(mesh=mesh, name="mu")
    xi  = fp.CellVariable(mesh=mesh, name="xi")
    T   = fp.CellVariable(mesh=mesh, name="T", value=cfg.T0, hasOld=True)
    src = fp.CellVariable(mesh=mesh, value=0.0)
    source_coeff = fp.Variable(0.)
    sink = fp.Variable(0.)

    sigx, sigy = cfg.sigma_x_a0 * A0, cfg.sigma_y_a0 * A0
    y_laser = cfg.Ly

    def gaussian(xc):
        return (cfg.absorption * cfg.power / (2*np.pi*sigx*sigy)) * \
               np.exp(-((xv-xc)**2)/(2*sigx**2)) * np.exp(-((yv-y_laser)**2)/(2*sigy**2))

    # --- corrected temperature equation (constant c_p, RHS latent heat) ---
    T_eq = (fp.TransientTerm(coeff=cfg.alpha_cp, var=T)
            == fp.DiffusionTerm(coeff=cfg.M, var=T)
            + fp.TransientTerm(coeff=(cfg.beta_latent), var=psi)
            + source_coeff * src
            + sink * cfg.T0 - fp.ImplicitSourceTerm(coeff=sink, var=T))
    # --- structural operator (identical to rev9) ---
    xi_eq = (fp.ImplicitSourceTerm(coeff=1, var=xi)
             == fp.ImplicitSourceTerm(coeff=2, var=psi) + fp.DiffusionTerm(coeff=1, var=psi))
    mu_eq = (fp.ImplicitSourceTerm(coeff=1, var=mu)
             == fp.ImplicitSourceTerm(coeff=1 - psi + psi**2, var=psi)
             + fp.DiffusionTerm(coeff=1/T, var=xi) - cfg.beta_latent / T)
    psi_eq = (fp.TransientTerm(coeff=cfg.tau_psi, var=psi) == fp.DiffusionTerm(coeff=1, var=mu))
    eq = T_eq & xi_eq & mu_eq & psi_eq

    psi.setValue(seed_polycrystal(cfg, xv, yv))
    T.setValue(cfg.T0)
    T.constrain(cfg.T0, where=mesh.facesLeft | mesh.facesRight | mesh.facesBottom)

    n_scan = int(np.ceil(cfg.Lx / (cfg.v_x * cfg.dt))) + 1
    n_steps = cfg.warmup_steps + n_scan + cfg.cooldown_steps
    print(f"[{cfg.outdir}] grid {cfg.nx}x{cfg.ny}  v_x={cfg.v_x}  power={cfg.power}  "
          f"scan_steps={n_scan}  total_steps={n_steps}")

    gr_rows = ["step,elapsed,x_laser,G,cooling_rate,R,Tmax"]
    x_laser, elapsed = 0.0, 0.0
    for step in range(n_steps):
        scanning = cfg.warmup_steps <= step < cfg.warmup_steps + n_scan
        if step < cfg.warmup_steps:
            source_coeff.value = 0.; sink.value = cfg.sink_warm; src.setValue(0.)
        elif scanning:
            source_coeff.value = 1.; sink.value = cfg.sink_scan
            x_laser += cfg.v_x * cfg.dt
            src.setValue(gaussian(x_laser))
        else:
            source_coeff.value = 0.; sink.value = cfg.sink_cool; src.setValue(0.)

        T.updateOld(); psi.updateOld()
        T_prev = T.value.copy()
        eq.solve(dt=cfg.dt)
        elapsed += cfg.dt

        if scanning and (step % cfg.gr_every == 0):
            gr = _front_GR(T.value, T_prev, cfg)
            if gr:
                Gf, cool, R = gr
                gr_rows.append(f"{step},{elapsed:.2f},{x_laser:.2f},{Gf:.4e},{cool:.4e},{R:.4e},{T.value.max():.4f}")

        if step % cfg.save_every == 0 or step == n_steps - 1:
            np.save(os.path.join(cfg.outdir, f"psi_{step}.npy"), psi.value)
            np.save(os.path.join(cfg.outdir, f"T_{step}.npy"), T.value)
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8))
            a1.imshow(psi.value.reshape(cfg.ny, cfg.nx), cmap='rainbow', origin='lower',
                      extent=[0, cfg.Lx, 0, cfg.Ly]); a1.set_ylabel("psi")
            im = a2.imshow(T.value.reshape(cfg.ny, cfg.nx), cmap='hot', origin='lower',
                           vmin=0, vmax=2, extent=[0, cfg.Lx, 0, cfg.Ly])
            a2.contour(T.value.reshape(cfg.ny, cfg.nx), levels=[1.0], colors='k',
                       origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly]); a2.set_ylabel("T")
            #fig.colorbar(im, ax=a2); plt.tight_layout()
            plt.savefig(os.path.join(cfg.outdir, f"frame_{step}.png"), dpi=110); plt.close()

        print(f"step={step} elapsed={elapsed:.1f} T=({T.value.min():.3f},{T.value.max():.3f}) x_laser={x_laser:.1f}")

    with open(os.path.join(cfg.outdir, "GR_log.csv"), "w") as f:
        f.write("\n".join(gr_rows))
    print(f"[{cfg.outdir}] done. G/R log: {cfg.outdir}/GR_log.csv")
