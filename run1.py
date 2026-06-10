"""
run1.py  --  Run 1: coupled non-isothermal PFC at PHYSICAL alpha/beta.

Purpose (closes / supports M2, R3-03, R3-17, and feeds M3, M11, R3-16):
  * Demonstrates the non-isothermal coupling is now ON (physical c_p and latent
    heat), not switched off as in rev8.
  * Produces a polycrystalline single-track microstructure to post-process for
    the quantitative validation metrics (pfc_postprocess.py).
  * Logs the solidification conditions G, R at the melt isotherm -- the physics
    that actually controls columnar morphology (used to answer M5/R2-02).

This is the BASELINE regime (v_x = 1 -> ~7.255 m/s in the mesoscale mapping).
Run 2 repeats it mapped toward the experimental scan speed.

BEFORE running: confirm peak T sits near T_m ~ 1 with these settings. If it is
too high, lower `power` (do NOT touch alpha_cp -- that is the heat capacity).
Use the thermal_diag mode in New_PFC_FV_rev9.py / the Rosenthal check in
pfc_diagnostics.py to pick `power`.
"""
from pfc_run import Config, run_simulation

cfg = Config(
    # domain: small but wide enough for several columnar grains
    nx=256, ny=128, dx=1600/2**11,        # Lx ~ 300, Ly ~ 100 (units)
    dt=1.0,
    # physical thermal coupling (the corrected, ON regime)
    alpha_cp=3.0,                          # heat capacity c_p (= alpha)
    beta_latent=0.5,                       # latent-heat coupling
    # process: BASELINE regime
    v_x=1.0,                               # ~7.255 m/s in the mesoscale mapping
    power=1200.0,                          # tune so peak T ~ T_m (see note above)
    # seeding / schedule
    seed=7, seed_spacing=28.0, grain_radius=16.0,
    warmup_steps=150, cooldown_steps=200,
    # io
    outdir="run1_baseline_P1200",
    save_every=10, gr_every=10,
)

if __name__ == "__main__":
    run_simulation(cfg)
    print("\nRun 1 finished. Next:")
    print("  from pfc_postprocess import analyze")
    print("  analyze('run1_baseline/psi_<last>.npy', ny=128, nx=256, dx=1600/2**11,")
    print("          build_dir_deg=90.0, out_prefix='run1_baseline/val')")
    print("  -> width / tilt / misorientation distributions + orientation map")
    print("  Also read run1_baseline/GR_log.csv for G, R at the melt isotherm.")
