"""
run2.py  --  Run 2: same model mapped toward the EXPERIMENTAL scan speed (M5/R2-02).

Mesoscale mapping (state this in the paper):
    v_x = 1 (dimensionless)  <->  ~7.255 m/s
    experiment 1000 mm/min   =    0.01667 m/s   <->   v_x ~ 0.0023

COST WARNING (read before running)
----------------------------------
A slow scan needs MANY more steps to cross the same track:
    scan_steps = Lx / (v_x * dt)
At the literal experimental speed v_x = 0.0023 this is ~400x more steps than
Run 1 -- often infeasible under time pressure. Three honest ways to handle it,
in order of recommended preference:

  (A) PHYSICS-FIRST (recommended, cheapest): keep Run 1, and answer M5 with the
      solidification conditions. Columnar width and columnar-vs-equiaxed
      selection are governed by G and R (G/R log), NOT raw scan speed. If Run 1's
      G, R already overlap the experiment's solidification regime, the speed
      mismatch is moot. Use this run only as corroboration on a SHORT track.

  (B) SHORT TRACK at the true speed: set MAP='experimental' and a small nx so
      the step count stays tractable (a few grains, correct speed).

  (C) INTERMEDIATE speed: set MAP='intermediate' (e.g. 50x slower than Run 1)
      and argue trend transfer via the dimensionless G/R analysis.

Tune `power` DOWN relative to Run 1: a slower scan dwells longer and deposits
more energy per unit length, so the same power overheats. Pick power so peak T
again sits near T_m (thermal_diag / Rosenthal check).
"""
from pfc_run import Config, run_simulation

MAP = "experimental"     # 'experimental' (true speed, short track) or 'intermediate'

if MAP == "experimental":
    v_x = 0.0023         # ~0.01667 m/s  (literal experimental speed)
    nx = 96              # SHORT track to keep scan_steps tractable
    dt = 1.0             # larger step allowed by the slower transient
    power = 150.0        # much lower: slow dwell -> retune to peak T ~ T_m
    outdir = "run2_experimental_P150_v0.0023"
else:                    # intermediate: 50x slower than Run 1
    v_x = 0.02           # ~0.145 m/s
    nx = 192
    dt = 1.0
    power = 150.0
    outdir = "run2_intermediate_P150_v0.02"

cfg = Config(
    nx=nx, ny=128, dx=1600/2**11,
    dt=dt,
    alpha_cp=3.0, beta_latent=0.5,        # same physical coupling as Run 1
    v_x=v_x, power=power,
    seed=7, seed_spacing=28.0, grain_radius=16.0,
    warmup_steps=150, cooldown_steps=300,
    outdir=outdir,
    save_every=100, gr_every=20,
)

if __name__ == "__main__":
    n_scan = int(cfg.Lx / (cfg.v_x * cfg.dt))
    print(f"MAP={MAP}: ~{n_scan} scan steps "
          f"({'tractable' if n_scan < 5000 else 'EXPENSIVE -- consider option (A) or (C)'}).")
    run_simulation(cfg)
    print("\nRun 2 finished. Compare G, R, G/R against Run 1 and the experiment:")
    print(f"  {outdir}/GR_log.csv  vs  run1_baseline/GR_log.csv")
    print("  Then post-process the microstructure the same way as Run 1.")
