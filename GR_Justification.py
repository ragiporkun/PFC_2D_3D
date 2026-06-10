"""Reproduce the G-R justification figure (T + psi with the T=1 isotherm) for the
three scan-speed runs behind Figure 5 / Table 1. Arrays are (ny=128, nx) flattened."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
dx = 0.78125
runs = [("(a) $v=1$", "T_300.npy", "psi_300.npy"),
        ("(b) $v=0.02$", "T_3000.npy", "psi_3000.npy"),
        ("(c) $v=0.0023$", "T_14000.npy", "psi_14000.npy")]
fig, ax = plt.subplots(3, 2, figsize=(12, 9))
for r, (lbl, tf, pf) in enumerate(runs):
    T = np.load(tf); P = np.load(pf); nx = T.size // 128
    T = T.reshape(128, nx); P = P.reshape(128, nx)
    ext = [0, nx*dx, 0, 128*dx]; X = np.linspace(0, nx*dx, nx); Y = np.linspace(0, 128*dx, 128)
    hT = ax[r, 0].imshow(T, origin="lower", cmap="inferno", extent=ext, aspect="auto", vmin=0.6, vmax=2.6)
    ax[r, 0].contour(X, Y, T, levels=[1.0], colors="cyan", linewidths=1.5)
    ax[r, 0].set_title(f"{lbl}: temperature $T$  ($T=1$ in cyan)", fontsize=10); ax[r, 0].set_ylabel("depth $y$")
    plt.colorbar(hT, ax=ax[r, 0], fraction=0.025, label="$T$")
    hP = ax[r, 1].imshow(P, origin="lower", cmap="jet", extent=ext, aspect="auto", vmin=-0.5, vmax=1.5)
    ax[r, 1].contour(X, Y, T, levels=[1.0], colors="k", linewidths=1.0)
    ax[r, 1].set_title(f"{lbl}: density $\\psi$  (liquid inside $T=1$)", fontsize=10)
    plt.colorbar(hP, ax=ax[r, 1], fraction=0.025, label="$\\psi$")
    if r == 2:
        ax[r, 0].set_xlabel("scan $x$"); ax[r, 1].set_xlabel("scan $x$")
plt.tight_layout(); plt.savefig("gr_justification_fields.png", dpi=140); plt.close()