"""
PFC_dendrite_benchmark.py  --  non-isothermal PFC (2D, FiPy)

Derived from New_PFC_FV_rev9.py. The laser/scan additive-manufacturing setup has
been stripped out and replaced by the classic solidification benchmark: a SINGLE
crystalline seed at the centre of the domain growing into an undercooled melt.
The structural (psi / mu / xi) operator is UNCHANGED from rev8/rev9.

----------------------------------------------------------------------------------
WHAT CHANGED RELATIVE TO rev9
----------------------------------------------------------------------------------
1. Heat source removed.  No Gaussian beam, no scan schedule, no Newton sink.
   The melt loses latent heat only by conduction to the far-field boundaries,
   which are held at the ambient/undercooling temperature T0 (constant far field).
2. Single centred seed.  seed_polycrystal() -> seed_single(): one oriented
   triangular nucleus at (Lx/2, Ly/2).
3. Square domain so the crystal can grow isotropically in every direction.

----------------------------------------------------------------------------------
PHYSICS YOU MUST GET RIGHT (this is what actually decides whether you see a
dendrite, a hexagon, or a domain that just freezes everywhere)
----------------------------------------------------------------------------------
The structural dynamics here are CONSERVED (Cahn-Hilliard type): psi_eq is
TransientTerm = DiffusionTerm(mu).  For a *seeded* crystal you therefore need a
uniform background density PSI_MEAN that is METASTABLE: linearly stable (it will
not crystallise on its own) yet thermodynamically undercooled (the seed grows).

Linearising the conserved dynamics about a uniform density gives a uniform state
that is UNSTABLE (freezes everywhere) when
        g'(psi) < 1/T ,     with g'(psi) = 1 - 2*psi + 3*psi**2
i.e. the spinodal densities are the roots of  3*psi**2 - 2*psi + (1 - 1/T) = 0,
        psi_spinodal = ( 1 +/- sqrt(3/T - 2) ) / 3 .
At T0 = 0.6 these are  psi = -0.244  and  psi = +0.911 .  The uniform melt is only
metastable OUTSIDE that interval.  (rev9 used psi0 = 0, which is INSIDE it -- with
no laser the whole domain crystallises spontaneously and there is no seeded
growth.  That is the single most important fix here.)

The one-mode seed-amplitude formula also needs a non-negative discriminant,
        disc = -15*(1 - 1/T) + 24*A + 1 - 36*A**2  >= 0 ,
which at T0 = 0.6 holds only for  A > -0.312 .

==> Usable window for a seeded, metastable, undercooled melt at T0 = 0.6:
            -0.312 < PSI_MEAN < -0.244
    Near the spinodal (~ -0.25) the driving force is high -> fast, COMPACT,
    HEXAGONAL-FACETED growth.  Near the binodal (~ -0.31) growth nearly stalls.
    DENDRITES (faceted 6-armed stars: corners outrun the diffusion-shadowed
    facet centres) appear in between AND require a LARGE domain + long run so the
    crystal can grow well past the facet stage.  Start at PSI_MEAN = -0.28 and
    scan -0.26 ... -0.31 on your own hardware.
    (run_regime_check() below prints the window for any T0 you choose.)

----------------------------------------------------------------------------------
ISOTHERMAL-STYLE vs THERMALLY-COUPLED dendrite  (LATENT_HEAT_MODE)
----------------------------------------------------------------------------------
rev9's latent-heat term is the LINEARISED  beta*dpsi/dt.  Because psi oscillates
at the lattice scale, dpsi/dt averages to ~0 over a unit cell, so this term
carries almost no NET latent heat: T stays pinned at ~T0 and the run is
effectively isothermal.  The dendrite is then driven purely by the conserved
DENSITY field (the standard isothermal PFC dendrite of Tegze/Granasy/Provatas).

    LATENT_HEAT_MODE = "linear"  -> beta*dpsi/dt           (rev9 default; robust,
                                                            ~isothermal, density-driven)
    LATENT_HEAT_MODE = "full"    -> (beta + psi)*dpsi/dt    (the de/dpsi the rev9
                                                            header derives; the
                                                            psi*dpsi/dt = d(psi^2/2)/dt
                                                            piece carries the real
                                                            net latent heat)
With "full" you MUST raise the heat capacity: c_p = 0.1 lets the latent heat spike
T by O(3); use ALPHA_CP ~ 3 so the temperature rise is ~0.1 and stays below the
model melting point (~1.07 at these parameters).  "full" is the path that actually
exercises the non-isothermal coupling; the nonlinear coefficient is frozen within
each solve, so add a short sweep loop if you need tight accuracy.
"""

import os
os.environ['FIPY_SOLVERS'] = 'scipy'
import numpy as np
import fipy as fp
import matplotlib.pyplot as plt

# ======================================================================
# PARAMETERS
# ======================================================================
R0 = 4 * np.pi / np.sqrt(3)            # ~7.255 dimensionless lattice spacing

# --- domain (SQUARE; large enough for arms to develop) -----------------
dx = dy = 1600 / 2**11                 # ~0.781  -> ~9.3 nodes per lattice spacing
nx = ny = 384                          # ~41 lattice spacings per side.
#   384^2 ~ 1.5e5 cells. For a quick look use 256 (~27 spacings, shows the facet
#   -> star onset); for crisp arms use 512. Cost scales with the cell count.
Lx, Ly = nx * dx, ny * dy

# --- time --------------------------------------------------------------
dt = 1.0

# --- structural model (UNCHANGED from rev8/rev9) -----------------------
T0   = 0.6                             # ambient / far-field melt temperature (undercooling)
M    = 0.6                             # thermal conductivity (diffusion coeff in T_eq)
tau_psi = 1.0

# --- BACKGROUND DENSITY = the regime knob (see header) -----------------
PSI_MEAN = -0.28                       # uniform melt + seed average density.
#   MUST sit in the metastable window (-0.312, -0.244) at T0 = 0.6.
#   This replaces rev9's psi0 = 0.0, which freezes the whole domain.

# --- single-seed geometry ----------------------------------------------
seed_angle   = 0.0                     # crystallographic orientation of the nucleus
grain_radius = 16.0                    # seed radius (~2.2 spacings). Enlarge if the
#                                        seed dissolves at the more negative PSI_MEAN.

# --- THERMAL coupling --------------------------------------------------
LATENT_HEAT_MODE = "linear"            # "linear" (rev9, ~isothermal) or "full"
ALPHA_CP    = 0.1                      # heat capacity c_p. Use ~3.0 if MODE="full".
BETA_LATENT = 0.025                    # latent-heat coupling constant.

# --- run control -------------------------------------------------------
N_STEPS    = 2001
PLOT_EVERY = 10
SAVE_TIMES = (1, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000)

OUTDIR = f"Plots_dendrite_psimean{PSI_MEAN}_T{T0}_{LATENT_HEAT_MODE}"
os.makedirs(OUTDIR, exist_ok=True)


# ======================================================================
# REGIME CHECK  (prints spinodal window + validates PSI_MEAN)
# ======================================================================
def run_regime_check(T, psi_mean):
    inv = 1.0 / T
    # spinodal roots of 3*psi^2 - 2*psi + (1 - 1/T) = 0
    rad = 3.0 * inv - 2.0
    if rad <= 0:
        print(f"[regime] T={T}: no spinodal (T>=1.5); uniform melt stable everywhere.")
        lo = hi = None
    else:
        lo = (1.0 - np.sqrt(rad)) / 3.0
        hi = (1.0 + np.sqrt(rad)) / 3.0
        print(f"[regime] T={T}: spinodal at psi = {lo:.3f} and {hi:.3f} "
              f"(uniform melt metastable for psi<{lo:.3f} or psi>{hi:.3f}).")
    # seed-amplitude discriminant
    disc = -15 * (1 - inv) + 24 * psi_mean + 1 - 36 * psi_mean**2
    print(f"[regime] PSI_MEAN={psi_mean}: seed-amplitude discriminant = {disc:.3f} "
          f"({'OK' if disc >= 0 else 'NEGATIVE -> seed formula invalid'}).")
    if lo is not None and not (psi_mean < lo):
        print(f"[regime] !! WARNING: PSI_MEAN={psi_mean} is NOT below the spinodal "
              f"{lo:.3f} -> the whole domain will crystallise spontaneously "
              f"(no seeded growth).")
    if disc < 0:
        print(f"[regime] !! WARNING: discriminant<0 -> reduce |PSI_MEAN| "
              f"(stay above -0.312 at T0=0.6).")


# ======================================================================
# MESH AND FIELDS
# ======================================================================
mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
x, y = mesh.x, mesh.y

psi = fp.CellVariable(mesh=mesh, name=r"$\psi$", value=PSI_MEAN, hasOld=True)
mu  = fp.CellVariable(mesh=mesh, name=r"$\mu$")
xi  = fp.CellVariable(mesh=mesh, name=r"$\xi$")
T   = fp.CellVariable(mesh=mesh, name=r"$T$", value=T0, hasOld=True)


# ======================================================================
# SINGLE SEED  (one oriented triangular nucleus at the domain centre)
# ======================================================================
def seed_single(xv, yv, cx, cy, ang=0.0):
    """One oriented triangular seed of radius grain_radius centred at (cx, cy),
    embedded in a uniform melt of density PSI_MEAN. Same one-mode construction as
    the rev9 polycrystal seeder, reduced to a single grain."""
    A = PSI_MEAN
    p = R0
    pre = -0.5
    disc = -15 * (1 - 1/T0) + 24*A + 1 - 36*A**2
    B = ((4 - 12*A) + 4*np.sqrt(disc)) / 15      # equilibrium one-mode amplitude
    field = np.full_like(xv, A)
    xs, ys = xv - cx, yv - cy
    wave = A + B * (np.cos(2*np.pi*(xs*np.cos(ang) + ys*np.sin(ang)) / p) *
                    np.cos(2*np.pi*(xs*np.sin(ang) - ys*np.cos(ang)) / (np.sqrt(3)*p)) -
                    pre*np.cos(4*np.pi*(xs*np.sin(ang) - ys*np.cos(ang)) / (np.sqrt(3)*p)))
    m = (xs**2 + ys**2) <= grain_radius**2
    field[m] = wave[m]
    return field


# ======================================================================
# EQUATIONS
# ======================================================================
# --- temperature ------------------------------------------------------
#   c_p dT/dt = M lap(T) + (latent heat) ;  far-field heat sink = boundaries
if LATENT_HEAT_MODE == "full":
    latent_coeff = BETA_LATENT          # (beta + psi) dpsi/dt
else:
    latent_coeff = BETA_LATENT                # beta dpsi/dt   (rev9 linearised)

T_eq = (fp.TransientTerm(coeff=ALPHA_CP, var=T)
        == fp.DiffusionTerm(coeff=M, var=T)
        + fp.TransientTerm(coeff=latent_coeff, var=psi))

# --- structural operator (IDENTICAL to rev8/rev9) ----------------------
xi_eq = (fp.ImplicitSourceTerm(coeff=1, var=xi)
         == fp.ImplicitSourceTerm(coeff=2, var=psi)
         + fp.DiffusionTerm(coeff=1, var=psi))

mu_eq = (fp.ImplicitSourceTerm(coeff=1, var=mu)
         == fp.ImplicitSourceTerm(coeff=1 - psi + psi**2, var=psi)
         + fp.DiffusionTerm(coeff=1/T, var=xi) - BETA_LATENT / T)

psi_eq = (fp.TransientTerm(coeff=tau_psi, var=psi)
          == fp.DiffusionTerm(coeff=1., var=mu))

eq = T_eq & xi_eq & mu_eq & psi_eq


# ======================================================================
# INITIAL AND BOUNDARY CONDITIONS
# ======================================================================
run_regime_check(T0, PSI_MEAN)
psi.setValue(seed_single(x.value, y.value, Lx/2, Ly/2, seed_angle))
T.setValue(T0)

# Far-field melt held at the undercooling temperature on ALL four faces
# (constant-far-field thermal BC; psi keeps its default no-flux -> mass conserved).
T.constrain(T0, where=(mesh.facesLeft | mesh.facesRight
                       | mesh.facesTop | mesh.facesBottom))


# ======================================================================
# MAIN LOOP
# ======================================================================
elapsed = 0.0
for step in range(N_STEPS):
    T.updateOld()
    psi.updateOld()
    eq.solve(dt=dt)
    elapsed += dt

    if step % PLOT_EVERY == 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        im0 = axes[0].imshow(psi.value.reshape(ny, nx), cmap='rainbow',
                             origin='lower', extent=[0, Lx, 0, Ly],
                             vmin=PSI_MEAN - 0.1, vmax=1.7)
        axes[0].set_title(r'$\psi$'); fig.colorbar(im0, ax=axes[0])
        imT = axes[1].imshow(T.value.reshape(ny, nx), cmap='hot', origin='lower',
                             extent=[0, Lx, 0, Ly],
                             vmin=T0 - 0.02, vmax=T0 + 0.15)   # adjust for MODE="full"
        axes[1].set_title('T'); fig.colorbar(imT, ax=axes[1])
        fig.suptitle(f"step={step}  t={elapsed:.0f}")
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/psiT_{step:05d}.png", dpi=90)
        plt.close()

    if int(round(elapsed)) in SAVE_TIMES:
        np.save(f"{OUTDIR}/psi_{int(round(elapsed))}.npy", psi.value)
        np.save(f"{OUTDIR}/T_{int(round(elapsed))}.npy",   T.value)

    print(f"step={step} t={elapsed:.0f} "
          f"psi[min,max]=({psi.value.min():.3f},{psi.value.max():.3f}) "
          f"T[min,max]=({T.value.min():.3f},{T.value.max():.3f})")