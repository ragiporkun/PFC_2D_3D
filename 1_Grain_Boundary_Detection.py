import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Grid / domain params ---------------------------------------------------
# NOTE: psi_3750.npy is a flat array of 49152 = 384 * 128 values.
# The grid spacing (dx, dy) and lattice constant R are unchanged from before;
# only the number of grid points differs from the old psi_1000 data.
dx, dy   = 1600 / 2**11, 1600 / 2**11
nx, ny   = 384, 128
R        = 4 * np.pi / np.sqrt(3)
frac_R   = 0.20

PSI_FILE = "/home/debian/PycharmProjects/PFC_2D_3D/Plots_P1800_v0.02_alpha0.1_beta0.025_M0.6_1/psi_15750.npy"
OUT_PATH = "local_maxima_with_avg_angles_floored_15000_1.csv"


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


psi = load_psi(PSI_FILE, ny, nx)

# --- Local maxima detection (3x3 neighbourhood) -----------------------------
local_max = np.zeros_like(psi, dtype=bool)
for j in range(ny):
    for i in range(nx):
        j0, j1 = max(j-1, 0), min(j+2, ny)
        i0, i1 = max(i-1, 0), min(i+2, nx)
        window = psi[j0:j1, i0:i1]
        if psi[j, i] >= window.max():
            local_max[j, i] = True

mask     = local_max & (psi > 1.0)
ys, xs   = np.where(mask)
x_coords = xs * dx
y_coords = ys * dy
psi_vals = psi[ys, xs]
ids      = np.arange(1, len(x_coords) + 1)

ring_lo, ring_hi = R * (1 - frac_R), R * (1 + frac_R)
rows = []

for idx, (x0, y0) in enumerate(zip(x_coords, y_coords), start=1):
    dists = np.hypot(x_coords - x0, y_coords - y0)
    dists[idx-1] = np.inf

    cand = np.where((dists >= ring_lo) & (dists <= ring_hi))[0]
    order = np.argsort(np.abs(dists[cand] - R))
    sel   = cand[order][:6]

    angles = []
    for j in sel:
        dx_ij = x_coords[j] - x0
        dy_ij = y_coords[j] - y0
        ang = np.arctan2(dy_ij, dx_ij)
        #if ang < 0:
        #    ang += 2 * np.pi
        #ang_floored = np.floor(ang * 1000) / 1000
        #ang_floored = min(ang_floored, 2*np.pi - ang_floored)
        angles.append(ang)

    num_neighbor = len(angles)
    if num_neighbor == 0:
        # no neighbours within the ring -> skip this point
        continue

    sin_sum = np.sum(np.sin(angles)) / num_neighbor
    cos_sum = np.sum(np.cos(angles)) / num_neighbor
    avg_angle = np.arctan2(sin_sum, cos_sum)
    avg_angle = np.floor(avg_angle * 1000) / 1000

    psi6 = np.sum(np.exp(1j * 6 * np.array(angles))) / num_neighbor
    phi6 = np.angle(psi6, deg=False) / num_neighbor

    if phi6 < -np.pi / 6:
        phi6 += np.pi / 3
    elif phi6 >= np.pi / 6:
        phi6 -= np.pi / 3

    phi6 = np.floor(phi6 * 1000) / 1000

    # average distance of the selected neighbors (floored to 0.001)
    avg_dist = np.floor(np.mean(dists[sel]) * 1000) / 1000

    row = {
        "id": idx,
        "x": x0,
        "y": y0,
        "psi": psi_vals[idx - 1],
        "avg_angle": avg_angle,
        "phi6": phi6,
        "avg_dist": avg_dist,
    }
    for ni, j in enumerate(sel, start=1):
        row.update({
            f"nbr{ni}_id": ids[j],
            f"nbr{ni}_x": x_coords[j],
            f"nbr{ni}_y": y_coords[j],
            f"nbr{ni}_dist": np.floor(dists[j] * 1000) / 1000,
            f"nbr{ni}_angle": angles[ni - 1]
        })
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df)} records with wrapped angles to {OUT_PATH}")
