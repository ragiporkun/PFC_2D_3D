import numpy as np, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree

# --- Grid and constants (v0.1 case: 384 x 128) ---
dx = dy = 1600 / 2**11
nx, ny = 384, 128
R = 4 * math.pi / math.sqrt(3)            # reference spacing ~7.255 (PHYSICAL units)

# --- Load cooled snapshot ---
psi = np.load("Plots_P1800_v0.1_alpha5_beta0.5/psi_3750.npy").reshape(ny, nx)
x = np.arange(nx) * dx
y = np.arange(ny) * dy

# --- Detect atoms (density peaks) -> PHYSICAL coordinates ---
win = max(3, int(round(0.6 * R / dx)))
mx = ndimage.maximum_filter(psi, size=win, mode="nearest")
thr = psi.min() + 0.4 * (psi.max() - psi.min())
iy, ix = np.where((psi == mx) & (psi > thr))
atoms = np.column_stack([ix * dx, iy * dx])        # same convention as your CSV (index*dx)

# --- Six-neighbour average spacing deviation (your method, computed from psi) ---
tree = cKDTree(atoms)
d, nbr = tree.query(atoms, k=7)                    # self + 6 neighbours
avg_disp = np.full(len(atoms), np.nan)
for j in range(len(atoms)):
    dist = d[j, 1:]                                # distances to 6 neighbours
    dist = dist[dist <= 1.4 * R]                   # drop spurious far neighbours (edges)
    if len(dist):
        avg_disp[j] = np.mean(dist - R)            # deviation from reference spacing
frac = avg_disp / R                                # dimensionless deviation

# --- crystallinity mask (psi6) so liquid/defect cells render transparent ---
psi6 = np.zeros(len(atoms), complex)
for j, pj in enumerate(atoms):
    idx = tree.query_ball_point(pj, 1.4 * R)
    dd = atoms[[k for k in idx if k != j]] - pj
    if len(dd):
        psi6[j] = np.mean(np.exp(6j * np.arctan2(dd[:, 1], dd[:, 0])))
mag = np.abs(psi6)
frac[mag < 0.6] = np.nan

# --- Rasterise onto grid (nearest atom) -> filled map, no white gaps ---
Xc, Yc = np.meshgrid(x, y)
_, gidx = tree.query(np.column_stack([Xc.ravel(), Yc.ravel()]))
field = frac[gidx].reshape(ny, nx)                 # NaN -> transparent

# --- Plot: coolwarm filled deviation map (same style as orientation map) ---
ext = [x.min(), x.max(), y.min(), y.max()]
vmax = np.nanpercentile(np.abs(field), 98)
fig = plt.figure(figsize=(12, 3.2))
gs = fig.add_gridspec(1, 2, width_ratios=[40, 1], wspace=0.04)
axm = fig.add_subplot(gs[0, 0])
im = axm.imshow(field, origin="lower", extent=ext, cmap="coolwarm",
                vmin=-vmax, vmax=vmax, aspect="auto")
axm.set_xlabel("x (units)"); axm.set_ylabel("y (units)")
axm.set_title("lattice-spacing deviation", fontsize=10)
axc = fig.add_subplot(gs[0, 1])
cb = fig.colorbar(im, cax=axc); cb.set_label(r"deviation  $(\bar d - R)/R$", fontsize=9)
plt.tight_layout()
plt.savefig("P1800_v0.1_alpha5_beta0.5/fig6_v01_deviation.png", dpi=130)
plt.savefig("P1800_v0.1_alpha5_beta0.5/fig6_v01_deviation.pdf")
plt.close()

print(f"atoms={len(atoms)}  crystalline={int((mag>=0.6).sum())}")
print(f"deviation (fraction of R): mean|.|={np.nanmean(np.abs(frac)):.4f}  "
      f"range=[{np.nanmin(frac):.3f}, {np.nanmax(frac):.3f}]")