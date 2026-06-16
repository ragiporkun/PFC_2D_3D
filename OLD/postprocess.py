import os; os.makedirs("P1800_v0.1_alpha5_beta0.5", exist_ok=True)

from pfc_postprocess import analyze
res = analyze("Plots_P1800_v0.1_alpha5_beta0.5/psi_3750.npy", ny=128, nx=384, dx=1600/2**11,
              build_dir_deg=90.0, out_prefix="P1800_v0.1_alpha5_beta0.5/grains")