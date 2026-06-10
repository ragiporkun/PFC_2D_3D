import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# --- Grid and constants -----------------------------------------------------
# psi_3750.npy is a flat array of 49152 = 384 * 128 values.
dx, dy = 1600 / 2**11, 1600 / 2**11
nx, ny = 384, 128
R = 4 * math.pi / math.sqrt(3)

PSI_FILE = "Plots_P1800_v0.5_alpha3_beta0.5_M0.6\psi_1000.npy"
DISP_CSV = "local_maxima_with_avg_angles_floored_1000_with_disp.csv"


def load_psi(path, ny, nx):
    """Load the psi field from a .npy file and reshape to (ny, nx) if flat."""
    arr = np.load(path)
    if arr.ndim == 1:
        if arr.size != ny * nx:
            raise ValueError(
                f"{path} has {arr.size} values, cannot reshape to ({ny},{nx})"
            )
        arr = arr.reshape((ny, nx))
    elif arr.shape != (ny, nx):
        raise ValueError(f"{path} has shape {arr.shape}, expected ({ny},{nx})")
    return arr


# --- Load data --------------------------------------------------------------
psi = load_psi(PSI_FILE, ny, nx)
df = pd.read_csv(DISP_CSV)

# --- Coordinates for imshow extent ---
x = np.arange(nx) * dx
y = np.arange(ny) * dy

# --- Find neighbor coordinate columns and compute displacement diffs ---
nbr_pairs = []
disp_cols = []
for i in range(1, 7):
    xcol, ycol = f"nbr{i}_x", f"nbr{i}_y"
    if xcol in df.columns and ycol in df.columns:
        nbr_pairs.append((i, xcol, ycol))
        dx_i = df[xcol] - df["x"]
        dy_i = df[ycol] - df["y"]
        dist_i = np.sqrt(dx_i**2 + dy_i**2)
        diff_col = f"nbr{i}_dist_diff"
        df[diff_col] = dist_i - R
        disp_cols.append(diff_col)

# Average displacement per center
df["avg_disp"] = df[disp_cols].mean(axis=1, skipna=True)

# =========================
# Plot 1: Average displacement scatter
# =========================
fig, ax = plt.subplots(figsize=(12, 6))

ax.imshow(
    psi,
    origin='lower',
    aspect='equal',
    cmap='rainbow',
    extent=[x.min(), x.max(), y.min(), y.max()]
)

sc_disp = ax.scatter(
    df['x'], df['y'],
    c=df['avg_disp'],
    cmap='coolwarm',
    s=165,
    edgecolors='black',
    linewidths=0.3,
    zorder=3, vmax=1, vmin=-1
)

ax.set_xlabel("X (units)")
ax.set_ylabel("Y (units)")
ax.set_title("Average Displacement Field")

cbar = fig.colorbar(sc_disp, ax=ax)
cbar.set_label("Average displacement (Delta dist from R)")

plt.tight_layout()
plt.show()

# =========================
# Plot 2: Displacement vectors
# =========================
fig, ax = plt.subplots(figsize=(12, 6))

ax.imshow(
    psi,
    origin='lower',
    aspect='equal',
    cmap='rainbow',
    extent=[x.min(), x.max(), y.min(), y.max()]
)

ax.scatter(
    df['x'], df['y'],
    c='white',
    s=8,
    edgecolors='black',
    linewidths=0.2,
    zorder=3
)

plt.tight_layout()
plt.show()
