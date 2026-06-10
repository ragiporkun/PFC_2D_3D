"""
postprocess_usage.py  --  how to run pfc_postprocess on a finished run.

Put this next to pfc_postprocess.py and your run folders, then run it.
All outputs go next to the snapshot (the out_prefix folder is auto-created).

  analyze(...)            -> orientation map (with orientation key) + width/tilt/
                             misorientation metrics + histograms   [Figs 4-5 col 3, validation]
  analyze_deviation(...)  -> lattice-spacing deviation map, same imshow style
                             as the orientation map (no white gaps)  [Fig 6]

IMPORTANT: always use a FULLY COOLED snapshot (e.g. psi_3750), where the track
has resolidified -- grain structure and spacing deviation are only defined in
solid material.
"""
from pfc_postprocessor import analyze, analyze_deviation

# ---- common settings (your grid) ----
NY, NX, DX = 128, 384, 1600 / 2**11      # ny, nx, dx
BUILD_DIR  = 90.0                         # build/normal direction in degrees (y-axis)

# =====================================================================
# 1) ORIENTATION MAP + GRAIN METRICS   (Fig. 4 / Fig. 5 third column)
#    Run once per case you want as a row of Fig. 4 / Fig. 5.
# =====================================================================
res = analyze(
    "Plots_P1800_v0.1_alpha3_beta0.5/psi_3750.npy",
    ny=NY, nx=NX, dx=DX,
    build_dir_deg=BUILD_DIR,
    out_prefix="P1800_v0.1_alpha3_beta0.5/grains",   # folder auto-created
)
# printed: grains detected, width (units & in a0), tilt, misorientation
# written: grains_orientation_rgb.png/pdf  (with orientation key)
#          grains_metrics_hist.png
# returned dict: res["width"], res["tilt"], res["misorientation"], ...

# =====================================================================
# 2) LATTICE-SPACING DEVIATION MAP   (Fig. 6)
#    signed=False -> copper magnitude map  (matches the original Fig. 6 look)
#    signed=True  -> red/blue compression vs tension
# =====================================================================
analyze_deviation("Plots_P1800_v0.1_alpha3_beta0.5/psi_3750.npy",
                  ny=128, nx=384, dx=1600/2**11, signed=True,
                  out_prefix="P1800_v0.1_alpha3_beta0.5/grains")
# written: fig6_deviation.png/pdf  (filled imshow, named colorbar)

# =====================================================================
# 3) BATCH: orientation maps for every sweep case (Fig. 4 rows)
#    Edit the list to the runs you want; each needs a cooled psi snapshot.
# =====================================================================
sweep_cases = [
    "Plots_P1800_v0.1_alpha3_beta0.5",
    "Plots_P1800_v0.5_alpha3_beta0.5",
    "Plots_P1800_v1_alpha3_beta0.5",
    # add the rest as needed
]
for folder in sweep_cases:
    analyze(f"{folder}/psi_3750.npy", ny=NY, nx=NX, dx=DX,
            build_dir_deg=BUILD_DIR, out_prefix=f"{folder}/grains")