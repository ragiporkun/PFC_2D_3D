"""
convergence_verification.py
============================
Mesh / time-step convergence (referee M7, R3-20) and implicit-vs-explicit
cross-verification (referee M8, R3-05) for the corrected non-isothermal PFC
thermal solver.

The thermal physics is taken verbatim from New_PFC_FV_rev10.py:
    constant heat capacity  c_p = ALPHA_CP
    conductivity            M
    normalized 2D Gaussian source, widths in lattice units (sigma = k * R0)
    Newton volumetric sink toward the far-field T0
    insulated top scan surface, far-field Dirichlet T0 on the other faces

To isolate the discretization error of the thermal solver, the source is held
STATIONARY at the top-centre and the structural field is frozen uniform (the
paper's "thermal_diag" mode). This is standard practice for a code-verification
convergence study.

Three studies are run:
  1. SPATIAL convergence  -- steady state solved directly (the steady thermal
     equation is linear), metric = peak temperature T_max and melt area; the
     mesh is refined by factors of two. Observed order from a Richardson triplet.
  2. TEMPORAL convergence -- transient backward-Euler integration to a fixed
     time t*, fixed (production) mesh, time step refined by factors of two;
     metric = probe temperature at the source centre.
  3. IMPLICIT vs EXPLICIT -- the implicit finite-volume FiPy solver vs an
     independent explicit finite-difference integrator of the SAME PDE on a
     common reference problem (verifies scheme independence; the 3D PyTorch
     production code solves the identical temperature equation).

Reproduces (FiPy 4.0.2, scipy solvers):
    spatial : p(T_max) ~= 2.0 ; production grid dx=0.781 within ~0.01% of
              the Richardson-extrapolated T_max
    temporal: p ~= 1.0 (backward-Euler)
    cross   : ||T_imp - T_exp||_inf ~= 1e-7  (relative L2 ~= 3e-8)

Usage:
    python convergence_verification.py            # all three studies + figure
Outputs: console tables, conv_results.npz, Figure_convergence.{pdf,png}
"""
import os
os.environ['FIPY_SOLVERS'] = 'scipy'
import time
import numpy as np
import fipy as fp

# ---------------------------------------------------------------------------
# Physics constants  (verbatim from New_PFC_FV_rev10.py)
# ---------------------------------------------------------------------------
R0         = 4 * np.pi / np.sqrt(3)     # dimensionless lattice spacing (~7.255)
T0         = 0.6                        # ambient / preheat (and far-field) temperature
M          = 0.6                        # thermal conductivity in the T equation
ALPHA_CP   = 5.0                        # constant heat capacity c_p
power      = 1800.0
absorption = 0.3
sigma_x    = 3.0 * R0                   # beam half-widths in LATTICE UNITS
sigma_y    = 6.0 * R0
sink_val   = 0.1                        # Newton volumetric sink coefficient
Lx, Ly     = 300.0, 100.0              # = 384*0.78125 , 128*0.78125  (paper domain)


def gauss(xc, yc, X, Y):
    """Normalized 2D Gaussian heat source (same form as rev10.gaussian_source)."""
    return (absorption * power / (2 * np.pi * sigma_x * sigma_y)) * \
           np.exp(-((X - xc) ** 2) / (2 * sigma_x ** 2)) * \
           np.exp(-((Y - yc) ** 2) / (2 * sigma_y ** 2))


def order_from_triplet(f_coarse, f_med, f_fine, r=2.0):
    """Observed order of accuracy from three solutions on grids refined by r."""
    den = (f_med - f_fine)
    if den == 0:
        return float('nan')
    return np.log(abs((f_coarse - f_med) / den)) / np.log(r)


# ===========================================================================
# 1. SPATIAL convergence -- direct steady-state solve (linear thermal problem)
#    0 = M lap(T) - sink (T - T0) + S
# ===========================================================================
def steady_fipy(nx, ny, n_sweeps=5, save_field=False):
    dx, dy = Lx / nx, Ly / ny
    mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    X, Y = mesh.x.value, mesh.y.value
    T = fp.CellVariable(mesh=mesh, value=T0)
    src = fp.CellVariable(mesh=mesh, value=gauss(Lx / 2, Ly, X, Y))   # stationary
    T.constrain(T0, where=mesh.facesLeft | mesh.facesRight | mesh.facesBottom)
    eq = (fp.DiffusionTerm(coeff=M, var=T)
          - fp.ImplicitSourceTerm(coeff=sink_val, var=T)
          + sink_val * T0 + src == 0)
    for _ in range(n_sweeps):       # linear -> converges immediately; sweeps polish residual
        eq.solve(var=T)
    Tg = T.value.reshape(ny, nx)
    Tmax = float(Tg.max())
    melt_area = float((T.value > 1.0).sum()) * dx * dy
    return (Tmax, melt_area, Tg, (dx, dy)) if save_field else (Tmax, melt_area)


def run_spatial():
    print("=" * 72)
    print("1. SPATIAL CONVERGENCE  (direct steady-state solve)")
    print("=" * 72)
    grids = [(192, 64), (384, 128), (768, 256), (1536, 512)]
    res, Tg_med = [], None
    for nx, ny in grids:
        t0 = time.time()
        if nx == 384:
            Tmax, A, Tg_med, _ = steady_fipy(nx, ny, save_field=True)
        else:
            Tmax, A = steady_fipy(nx, ny)
        res.append((nx, ny, Lx / nx, Tmax, A))
        print(f"  nx={nx:5d} ny={ny:4d}  dx={Lx/nx:.4f}  "
              f"Tmax={Tmax:.6f}  melt_area={A:.4f}   ({time.time()-t0:.1f}s)")
    pT = order_from_triplet(res[1][3], res[2][3], res[3][3])
    pA = order_from_triplet(res[1][4], res[2][4], res[3][4])
    f2, f3, f4 = res[1][3], res[2][3], res[3][3]
    T_rich = f4 + (f4 - f3) / (2 ** pT - 1) if np.isfinite(pT) else float('nan')
    print(f"  observed order  p(Tmax)={pT:.2f}   p(melt_area)={pA:.2f}")
    print(f"  Richardson Tmax(dx->0) ~= {T_rich:.6f}")
    print(f"  production grid (dx=0.781) Tmax error vs extrapolated "
          f"= {abs(res[1][3]-T_rich)/T_rich*100:.3f}%")
    return res, pT, pA, T_rich, Tg_med


# ===========================================================================
# 2. TEMPORAL convergence -- transient backward-Euler to t*, fixed mesh
# ===========================================================================
def probe_transient(nx, ny, dt, tstar=120.0):
    dx, dy = Lx / nx, Ly / ny
    mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    X, Y = mesh.x.value, mesh.y.value
    T = fp.CellVariable(mesh=mesh, value=T0, hasOld=True)
    src = fp.CellVariable(mesh=mesh, value=gauss(Lx / 2, Ly, X, Y))
    T_eq = (fp.TransientTerm(coeff=ALPHA_CP, var=T)
            == fp.DiffusionTerm(coeff=M, var=T)
            + sink_val * T0 - fp.ImplicitSourceTerm(coeff=sink_val, var=T) + src)
    T.constrain(T0, where=mesh.facesLeft | mesh.facesRight | mesh.facesBottom)
    for _ in range(int(round(tstar / dt))):
        T.updateOld()
        T_eq.solve(dt=dt)
    Tg = T.value.reshape(ny, nx)
    return float(Tg[-1, nx // 2])           # probe = top-centre cell


def run_temporal():
    print("\n" + "=" * 72)
    print("2. TEMPORAL CONVERGENCE  (transient probe at t*=120, mesh 384x128)")
    print("=" * 72)
    temporal = []
    for dt in [4.0, 2.0, 1.0, 0.5]:
        v = probe_transient(384, 128, dt)
        temporal.append((dt, v))
        print(f"  dt={dt:4.1f}  T_probe(t*=120)={v:.6f}")
    pt = order_from_triplet(temporal[0][1], temporal[1][1], temporal[2][1])
    print(f"  observed temporal order p ~= {pt:.2f}   (backward-Euler -> ~1)")
    return temporal, pt


# ===========================================================================
# 3. IMPLICIT (FiPy) vs EXPLICIT (finite-difference) -- M8 / R3-05
#    explicit forward-Euler of  alpha dT/dt = M lap T - sink (T - T0) + S
#    Dirichlet T0 on L/R/bottom (ghost = 2 T0 - T_in); insulated top (ghost = T_in)
# ===========================================================================
def steady_explicit(nx, ny, dt=0.5, max_steps=20000, tol=1e-9):
    dx, dy = Lx / nx, Ly / ny
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    S = gauss(Lx / 2, Ly, X, Y)
    T = np.full((ny, nx), T0)
    for _ in range(max_steps):
        Tp = np.empty((ny + 2, nx + 2))
        Tp[1:-1, 1:-1] = T
        Tp[1:-1, 0]  = 2 * T0 - T[:, 0]     # left  Dirichlet
        Tp[1:-1, -1] = 2 * T0 - T[:, -1]    # right Dirichlet
        Tp[0, 1:-1]  = 2 * T0 - T[0, :]     # bottom Dirichlet
        Tp[-1, 1:-1] = T[-1, :]             # top   Neumann (insulated)
        lap = ((Tp[1:-1, 2:] - 2 * T + Tp[1:-1, :-2]) / dx ** 2
               + (Tp[2:, 1:-1] - 2 * T + Tp[:-2, 1:-1]) / dy ** 2)
        Tn = T + (dt / ALPHA_CP) * (M * lap - sink_val * (T - T0) + S)
        if np.abs(Tn - T).max() < tol:
            T = Tn
            break
        T = Tn
    return T


def run_cross_check(Tg_med):
    print("\n" + "=" * 72)
    print("3. IMPLICIT (FiPy) vs EXPLICIT (finite-difference)  -- M8 / R3-05")
    print("=" * 72)
    nx, ny = 384, 128
    if Tg_med is None:
        _, _, Tg_med, _ = steady_fipy(nx, ny, save_field=True)
    Tg_e = steady_explicit(nx, ny)
    diff = np.abs(Tg_med - Tg_e)
    linf = float(diff.max())
    l2rel = float(np.linalg.norm(diff) / np.linalg.norm(Tg_med))
    print(f"  FiPy implicit : Tmax={Tg_med.max():.6f}")
    print(f"  explicit FD   : Tmax={Tg_e.max():.6f}")
    print(f"  Tmax relative difference = {abs(Tg_med.max()-Tg_e.max())/Tg_med.max()*100:.4f}%")
    print(f"  ||T_imp - T_exp||_inf            = {linf:.3e}")
    print(f"  ||T_imp - T_exp||_2 / ||T_imp||_2 = {l2rel:.3e}")
    return linf, l2rel


# ===========================================================================
# Figure
# ===========================================================================
def make_figure(res, pT, T_rich, temporal, pt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dx = np.array([r[2] for r in res]); Tmax = np.array([r[3] for r in res])
    dt = np.array([t[0] for t in temporal]); Tpr = np.array([t[1] for t in temporal])
    Tpr_ex = Tpr[-1] + (Tpr[-1] - Tpr[-2]) / (2 ** 1 - 1)     # order-1 extrapolation
    err_s = np.abs(Tmax - T_rich)
    err_t = np.abs(Tpr - Tpr_ex)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(dx, err_s, 'o-', color='#1f4e79', lw=2, ms=7,
                 label=r'$|T_{\max}(\Delta x)-T_{\max}^{\ast}|$')
    ax[0].loglog(dx, err_s[1] * (dx / dx[1]) ** 2, '--', color='gray', label='slope 2 (reference)')
    ax[0].set_xlabel(r'mesh size $\Delta x$ (dimensionless)')
    ax[0].set_ylabel('peak-temperature error')
    ax[0].set_title(f'(a) spatial convergence   $p\\approx{pT:.2f}$')
    ax[0].grid(True, which='both', alpha=0.3); ax[0].legend(frameon=False, fontsize=9); ax[0].invert_xaxis()
    ax[1].loglog(dt, err_t, 's-', color='#7a2d2d', lw=2, ms=7,
                 label=r'$|T_{\rm probe}(\Delta t)-T_{\rm probe}^{\ast}|$')
    ax[1].loglog(dt, err_t[1] * (dt / dt[1]) ** 1, '--', color='gray', label='slope 1 (reference)')
    ax[1].set_xlabel(r'time step $\Delta t$ (dimensionless)')
    ax[1].set_ylabel('probe-temperature error')
    ax[1].set_title(f'(b) temporal convergence   $p\\approx{pt:.2f}$')
    ax[1].grid(True, which='both', alpha=0.3); ax[1].legend(frameon=False, fontsize=9); ax[1].invert_xaxis()
    plt.tight_layout()
    plt.savefig("Figure_convergence.pdf", bbox_inches="tight")
    plt.savefig("Figure_convergence.png", dpi=300, bbox_inches="tight")
    print("\nsaved Figure_convergence.pdf and Figure_convergence.png")


if __name__ == "__main__":
    res, pT, pA, T_rich, Tg_med = run_spatial()
    temporal, pt = run_temporal()
    linf, l2rel = run_cross_check(Tg_med)
    np.savez("conv_results.npz",
             spatial=np.array([(r[2], r[3], r[4]) for r in res]),
             temporal=np.array(temporal),
             pT=pT, pA=pA, pt=pt, T_rich=T_rich, linf=linf, l2rel=l2rel)
    make_figure(res, pT, T_rich, temporal, pt)
