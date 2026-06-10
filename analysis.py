#from pfc_postprocess import analyze
#analyze('run1_baseline_P1200/psi_550.npy', ny=128, nx=256, dx=1600/2**11, build_dir_deg=90.0, out_prefix='run1_baseline_P1200/val')


import os
os.makedirs(r"C:\Users\orkun.secer\PycharmProjects\PythonProject\val", exist_ok=True)
from pfc_postprocess import analyze

res = analyze(r"C:\Users\orkun.secer\PycharmProjects\PythonProject\Plots_P1800_v0.1_alpha3_beta0.5_M0.6\psi_3750.npy",
              ny=128, nx=384, dx=1600/2**11,
              build_dir_deg=90.0,
              out_prefix=r"C:\Users\orkun.secer\PycharmProjects\PythonProject\val\cooled_3750")

print("columnar width (a0):", res["width_columnar"].mean(), "±", res["width_columnar"].std())
print("tilt (deg):", res["tilt_columnar"].mean(), "±", res["tilt_columnar"].std())
print("misorientation (deg):", res["misorientation"].mean(), "±", res["misorientation"].std())