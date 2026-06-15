"""
New_PFC_FV_rev9.py  --  revised non-isothermal PFC (2D, FiPy)

Revision of New_PFC_FV_rev8.py addressing referee findings on the thermal field.
Only the THERMAL physics, boundary conditions, parameters and run structure are
changed; the structural (psi / mu / xi) operator is kept as in rev8 so the
density dynamics are unchanged (decide the polynomial-coefficient question --
referee R3-02 -- separately against Free_Energy.ipynb).

KEY FIX (closes R3-06, R3-07*, R3-11, R2-03, R3-10, M2, M10)
------------------------------------------------------------
rev8 multiplied dT/dt by a STRUCTURE-DEPENDENT "heat capacity"
        q = alpha + 2*psi**2/T + div((psi/T) grad xi)
Deriving c_p from the model free energy in Free_Energy.ipynb gives, exactly,
        e = alpha*T - beta*psi - psi**2/2   ->   c_p = de/dT = alpha   (CONSTANT)
        latent-heat coupling = -de/dpsi = (beta + psi)
So the structural terms must NOT sit in the heat capacity. In voids psi~0 made
q~alpha=0.1, ~30x too small, so dT/dt = S/q overshot (peak T>5) and tracked the
lattice (voids hotter than atoms). rev9 uses a CONSTANT physical c_p and moves
the latent heat to the RHS as a source ~ (beta+psi)*dpsi/dt.

*R3-07 (thermal expansion) is only PARTLY a code issue: this free energy fixes
 the wavelength via the (1+lap)^2 operator (q0=1, T-independent), so the lattice
 constant does not depend on T by construction. Either soften the expansion
 claim or introduce a T-dependent q0; see the report.

USAGE
-----
    MODE = "thermal_diag"   # Phase-A: pure thermal, uniform medium (no PFC) -> verify peak T, gradients
    MODE = "full"           # coupled AM run
Set the two physical knobs ALPHA_CP and BETA_LATENT to exercise the coupling (M2).


Figure 7
"""

import os
os.environ['FIPY_SOLVERS'] = 'scipy'
import numpy as np
import fipy as fp
import matplotlib.pyplot as plt

os.makedirs("Plots_P1800_v0.1_alpha3_beta0.5_M0.6_1.6", exist_ok=True)

# ======================================================================
# PARAMETERS  (single consolidated block -- closes m3; documents R3-03)
# ======================================================================
# --- mesoscale mapping (document once, here -- closes m2) ---------------
#   length unit  : R0 = 4*pi/sqrt(3) (dimensionless) <-> one lattice spacing.
#   To map to physical units choose ONE physical lattice spacing a_phys [m]
#   and one physical time unit; report v, T in physical units in the paper.
R0 = 4 * np.pi / np.sqrt(3)            # ~7.255 dimensionless lattice spacing

# --- domain -------------------------------------------------------------
dx = dy = 1600 / 2**11                 # ~0.781  -> ~9.3 nodes per lattice spacing
nx, ny = 384, 128
Lx, Ly = nx * dx, ny * dy

# --- time ---------------------------------------------------------------
dt = 1.0                               # solver step (rev8 used timeStep=100*dt=1.0)

# --- structural model (unchanged from rev8) -----------------------------
psi0 = 0.0
T0   = 0.6                             # ambient / preheat temperature
M    = 0.6                           # thermal conductivity (diffusion coeff in T_eq)
tau_psi = 1/0.6

# --- seeding (programmatic polycrystal, fits any domain -- from pfc_run) -
seed         = 7
seed_spacing = 28.0
grain_radius = 16.0

# --- THERMAL model knobs (the physically meaningful ones -- M2, R3-17) --
# Set these to physical values to actually exercise the non-isothermal
# coupling; the dimensionless rev8 values are kept here only as a baseline.
ALPHA_CP   = 3.    # heat capacity c_p. rev8 used 0.1 (~30x too small). USE ~3.
BETA_LATENT = 0.5   # latent-heat coupling magnitude. rev8 used -0.025 (~0 latent).
#   For a side-by-side "coupling on/off" figure, run once with
#   (ALPHA_CP, BETA_LATENT) = (0.1, 0.025) and once with (3.0, 0.5).

# --- process / heat source (documented, parametrized -- R3-14, R3-19) ---
v_x    = 0.1                           # scan speed (map to m/s in paper -- M5/R2-02)
power  = 1800.0
absorption = 0.3
sigma_x = 3.0 * R0                     # beam half-widths in LATTICE UNITS (was 20, 50
sigma_y = 6.0 * R0                     # in raw units); >> a0 so we never melt 1 atom
x_laser_center = 0.0
y_laser_center = Ly

# --- run control --------------------------------------------------------
MODE = "full"                          # "full" or "thermal_diag"
N_STEPS = 3751
SAVE_TIMES = (1, 5, 10, 25, 50, 100, 250, 300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 3750)

# ======================================================================
# MESH AND FIELDS
# ======================================================================
mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
x, y = mesh.x, mesh.y          # CellVariables: support .value and numpy ufuncs

psi = fp.CellVariable(mesh=mesh, name=r"$\psi$", value=psi0, hasOld=True)
mu  = fp.CellVariable(mesh=mesh, name=r"$\mu$")
xi  = fp.CellVariable(mesh=mesh, name=r"$\xi$")
T   = fp.CellVariable(mesh=mesh, name=r"$T$", value=T0, hasOld=True)
source_term = fp.CellVariable(mesh=mesh, value=0.0)

source_coeff = fp.Variable(0.)
sink         = fp.Variable(0.)

# ======================================================================
# HEAT SOURCE  (normalized 2D Gaussian; width tied to lattice spacing)
# ======================================================================
def gaussian_source(xc, yc):
    return (absorption * power / (2 * np.pi * sigma_x * sigma_y)) * \
           np.exp(-((x - xc)**2) / (2 * sigma_x**2)) * \
           np.exp(-((y - yc)**2) / (2 * sigma_y**2))

# ======================================================================
# SEEDING  (programmatic polycrystal -- copied from pfc_run.py)
# ======================================================================
def seed_polycrystal(xv, yv):
    """Tile small oriented seeds across the domain; they grow to fill it.
    Replaces the hardcoded xc/yc/ang lists with a domain-agnostic grid of
    randomly oriented triangular seeds (same construction as pfc_run.py)."""
    rng = np.random.default_rng(seed)
    field = np.full_like(xv, psi0)
    p = R0
    A = psi0
    B = ((4 - 12*A) + 4*np.sqrt(-15*(1 - 1/T0) + 24*A + 1 - 36*A**2)) / 15
    pre = -0.5
    sx = np.arange(seed_spacing/2, Lx, seed_spacing)
    sy = np.arange(seed_spacing/2, Ly, seed_spacing)
    for cx in sx:
        for cy in sy:
            ang = rng.uniform(-np.pi/6, np.pi/6)     # within triangular symmetry
            xs, ys = xv - cx, yv - cy
            wave = A + B*(np.cos(2*np.pi*(xs*np.cos(ang)+ys*np.sin(ang))/p) *
                          np.cos(2*np.pi*(xs*np.sin(ang)-ys*np.cos(ang))/(np.sqrt(3)*p)) -
                          pre*np.cos(4*np.pi*(xs*np.sin(ang)-ys*np.cos(ang))/(np.sqrt(3)*p)))
            m = (xs**2 + ys**2) <= grain_radius**2
            field[m] = wave[m]
    return field

# ======================================================================
# EQUATIONS
# ======================================================================
# --- CORRECTED temperature equation -----------------------------------
#   c_p dT/dt = M lap(T) + (beta+psi) dpsi/dt + S - sink*(T - T0)
#   * constant c_p = ALPHA_CP                         (removes the q bug)
#   * latent heat as a RHS source (beta+psi) dpsi/dt  (de/dpsi from F)
#   * Newton volumetric relaxation toward ambient T0  (physical, not 0.01)
T_eq = (fp.TransientTerm(coeff=ALPHA_CP, var=T)
        == fp.DiffusionTerm(coeff=M, var=T)
        + fp.TransientTerm(coeff=BETA_LATENT, var=psi)  # (beta+psi) dpsi/dt
        + source_coeff * source_term
        + sink * T0 - fp.ImplicitSourceTerm(coeff=sink, var=T))

# --- structural operator (IDENTICAL to rev8; q_eq removed) -------------
xi_eq = (fp.ImplicitSourceTerm(coeff=1, var=xi)
         == fp.ImplicitSourceTerm(coeff=2, var=psi)
         + fp.DiffusionTerm(coeff=1, var=psi))

mu_eq = (fp.ImplicitSourceTerm(coeff=1, var=mu)
         == fp.ImplicitSourceTerm(coeff=1 - psi + psi**2, var=psi)
         + fp.DiffusionTerm(coeff=1/T, var=xi) - BETA_LATENT / T)

psi_eq = (fp.TransientTerm(coeff=tau_psi, var=psi)
          == fp.DiffusionTerm(coeff=1, var=mu))

if MODE == "full":
    eq = T_eq & xi_eq & mu_eq & psi_eq
else:                                   # thermal-only diagnostic: psi frozen uniform
    eq = T_eq

# ======================================================================
# INITIAL CONDITION
# ======================================================================
if MODE == "full":
    # programmatic polycrystalline triangular seeds (from pfc_run.py)
    psi.setValue(seed_polycrystal(x.value, y.value))
else:
    psi.setValue(psi0)                  # uniform medium for Phase-A thermal check
T.setValue(T0)

# ======================================================================
# BOUNDARY CONDITIONS  (physical far-field, not artificial pinning -- M10/R3-19)
# ======================================================================
# Top face (y=Ly) is the free/scan surface -> insulated (default Neumann).
# Far boundaries relax to the ambient/substrate temperature T0 (NOT 0.01).
T.constrain(T0, where=mesh.facesLeft | mesh.facesRight | mesh.facesBottom)

# ======================================================================
# MAIN LOOP
# ======================================================================
elapsed = 0.0
for step in range(N_STEPS):
    # ---- scan schedule ----
    if elapsed > 250.:
        sink.value = 0.1
        source_coeff.value = 1.
        x_laser_center += v_x * dt
        source_term.setValue(gaussian_source(x_laser_center, y_laser_center))
        if x_laser_center >= 300.:
            source_coeff.value = 0.
    # elif 300 < elapsed <= 1000:
    #     source_coeff.value = 0.
    #     sink.value = 0.01
    #     source_term.setValue(gaussian_source(x_laser_center, y_laser_center))
    else:
        source_coeff.value = 0.
        source_term.setValue(0.)

    T.updateOld()
    if MODE == "full":
        psi.updateOld()
    eq.solve(dt=dt)
    elapsed += dt

    # ---- light-weight figure ----
    if step % 10 == 0:
        fig, axes = plt.subplots(2 if MODE == "full" else 1, 1, figsize=(12, 12))
        axes = np.atleast_1d(axes)
        if MODE == "full":
            im0 = axes[0].imshow(psi.value.reshape(ny, nx), cmap='rainbow',
                                 origin='lower', extent=[0, Lx, 0, Ly])
            axes[0].set_ylabel(r'$\psi$'); fig.colorbar(im0, ax=axes[0])
        axT = axes[-1]
        imT = axT.imshow(T.value.reshape(ny, nx), cmap='hot', origin='lower',
                         vmin=0, vmax=1.5, extent=[0, Lx, 0, Ly])  # physical scale ~T_m
        axT.contour(T.value.reshape(ny, nx), levels=[1.0], colors='black',
                    origin='lower', extent=[0, Lx, 0, Ly])
        axT.set_ylabel('T'); fig.colorbar(imT, ax=axT)
        plt.tight_layout(); plt.savefig(f"Plots_P1800_v0.1_alpha3_beta0.5_M0.6_1.6/psiT_{step}.png"); plt.close()

    if int(round(elapsed)) in SAVE_TIMES:
        np.save(f"Plots_P1800_v0.1_alpha3_beta0.5_M0.6_1.6/psi_{int(round(elapsed))}.npy", psi.value)
        np.save(f"Plots_P1800_v0.1_alpha3_beta0.5_M0.6_1.6/T_{int(round(elapsed))}.npy", T.value)

    print(f"step={step} elapsed={elapsed:.1f} "
          f"T[min,max]=({T.value.min():.3f},{T.value.max():.3f}) "
          f"x_laser={x_laser_center:.1f}")