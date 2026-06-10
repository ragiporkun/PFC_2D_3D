"""
pfc_postprocess.py  --  quantitative validation metrics from a finished 2D PFC run.

Turns a saved density field psi (ny x nx) into the three like-for-like numbers
the reviewers ask to compare against the EBSD map in Fig. 2:

    * columnar grain WIDTH      distribution        (M3, R3-16)
    * columnar TILT angle       distribution        (M3, R3-16)
    * grain-boundary MISORIENTATION distribution     (M3, R3-16)

It uses the STANDARD PFC post-processing route (R3-08): detect density peaks
(atoms) -> per-atom bond-orientational order psi6 -> orientation field ->
union-find grain segmentation -> shape statistics. It also emits a genuine
orientation->RGB map (M4) instead of the wrapped phi6 scalar, and a
compare_to_experiment() that produces a side-by-side table + overlay plots so
the comparison to Fig. 2 is quantitative, not visual.

Dependencies: numpy, scipy, matplotlib.

Typical use after a run that saved Plots/psi_2500.npy:

    from pfc_postprocess import analyze
    res = analyze("Plots/psi_2500.npy", ny=256, nx=1024, dx=1600/2**11,
                  build_dir_deg=90.0, out_prefix="Plots/val_2500")
    # res["width"], res["tilt"], res["misorientation"] are arrays
    # then: compare_to_experiment(res, exp_csv="ebsd_fig2_metrics.csv")
"""
import os
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

A0_2D = 4.0 * np.pi / np.sqrt(3.0)          # triangular lattice constant ~7.255
SYM_DEG = 60.0                               # triangular orientation symmetry


# ----------------------------------------------------------------------
# 1. atom detection (density peaks)
# ----------------------------------------------------------------------
def detect_atoms(psi2d, dx, a0=A0_2D, rel_thresh=0.4):
    """Return (N,2) atom coordinates [x,y] in physical units from local maxima."""
    fp_size = max(3, int(round(0.6 * a0 / dx)))          # ~0.6 lattice spacing window
    mx = ndimage.maximum_filter(psi2d, size=fp_size, mode="nearest")
    thr = psi2d.min() + rel_thresh * (psi2d.max() - psi2d.min())
    peaks = (psi2d == mx) & (psi2d > thr)
    iy, ix = np.where(peaks)                              # row=y, col=x
    return np.column_stack([(ix + 0.5) * dx, (iy + 0.5) * dx])


# ----------------------------------------------------------------------
# 2. per-atom bond-orientational order psi6  ->  local orientation
# ----------------------------------------------------------------------
def bond_orientation(atoms, a0=A0_2D):
    """psi6 magnitude (crystallinity 0..1) and orientation in [0,60) deg per atom."""
    tree = cKDTree(atoms)
    rmax = 1.4 * a0
    psi6 = np.zeros(len(atoms), dtype=complex)
    for j, pj in enumerate(atoms):
        idx = tree.query_ball_point(pj, rmax)
        idx = [k for k in idx if k != j]
        if not idx:
            continue
        d = atoms[idx] - pj
        ang = np.arctan2(d[:, 1], d[:, 0])
        psi6[j] = np.mean(np.exp(6j * ang))
    mag = np.abs(psi6)
    theta = (np.angle(psi6) / 6.0) % np.deg2rad(SYM_DEG)
    return mag, np.rad2deg(theta)


def _ang_diff(a_deg, b_deg, sym=SYM_DEG):
    """Smallest orientation difference under sym-fold symmetry, in degrees."""
    d = np.abs(a_deg - b_deg) % sym
    return np.minimum(d, sym - d)


# ----------------------------------------------------------------------
# 3. grain segmentation (union-find on the atom neighbour graph)
# ----------------------------------------------------------------------
def segment_grains(atoms, theta_deg, mag, a0=A0_2D,
                   mag_min=0.6, mis_tol_deg=10.0, min_atoms=15):
    """Cluster crystalline atoms into grains by orientation.

    Region-grow with a running GRAIN-MEAN orientation (circular mean under the
    triangular 6-fold symmetry) so growth stops at the true boundary instead of
    leaking across the smooth misorientation transition between grains. This is
    the standard EBSD/PFC grain-reconstruction rule (R3-08).
    """
    from collections import deque
    n = len(atoms)
    crystalline = mag >= mag_min
    tree = cKDTree(atoms)
    rmax = 1.3 * a0
    neigh = [tree.query_ball_point(p, rmax) for p in atoms]

    labels = np.full(n, -1, dtype=int)
    order = np.argsort(-mag)                      # seed from the most crystalline atoms
    grains, gid = [], 0
    sym_rad = np.deg2rad(SYM_DEG)
    for s in order:
        if labels[s] != -1 or not crystalline[s]:
            continue
        members = [s]
        labels[s] = gid
        zsum = np.exp(1j * 6 * np.deg2rad(theta_deg[s]))   # running orientation sum
        q = deque([s])
        while q:
            i = q.popleft()
            mean_deg = np.rad2deg((np.angle(zsum) / 6.0) % sym_rad)
            for k in neigh[i]:
                if labels[k] != -1 or k == i or not crystalline[k]:
                    continue
                if _ang_diff(theta_deg[k], mean_deg) <= mis_tol_deg:
                    labels[k] = gid
                    members.append(k)
                    zsum += np.exp(1j * 6 * np.deg2rad(theta_deg[k]))
                    q.append(k)
        if len(members) >= min_atoms:
            grains.append(np.array(members))
            gid += 1
        else:
            labels[members] = -1                  # too small -> unassigned
    return labels, grains


# ----------------------------------------------------------------------
# 4. per-grain shape metrics
# ----------------------------------------------------------------------
def grain_metrics(atoms, grains, theta_deg, build_dir_deg=90.0):
    """Width, length, aspect, tilt (deg from build dir), mean orientation per grain."""
    out = []
    for members in grains:
        P = atoms[members]
        c = P.mean(axis=0)
        Q = P - c
        C = np.cov(Q.T)
        w, V = np.linalg.eigh(C)               # ascending eigenvalues
        e_major = V[:, 1]
        e_minor = V[:, 0]
        length = np.ptp(Q @ e_major)           # extent along major axis
        width = np.ptp(Q @ e_minor)            # extent perpendicular -> columnar width
        major_deg = np.rad2deg(np.arctan2(e_major[1], e_major[0]))
        tilt = _ang_diff(major_deg, build_dir_deg, sym=180.0)   # 0..90 from build dir
        ori = theta_deg[members]
        out.append(dict(centroid=c, n=len(members), width=width, length=length,
                        aspect=length / max(width, 1e-9), tilt=tilt,
                        orient=float(np.mean(ori)), orient_spread=float(np.std(ori))))
    return out


# ----------------------------------------------------------------------
# 5. grain-boundary misorientation distribution
# ----------------------------------------------------------------------
def misorientation_distribution(atoms, labels, gmet, a0=A0_2D):
    tree = cKDTree(atoms)
    pairs = tree.query_pairs(1.4 * a0, output_type="ndarray")
    seen, mis = set(), []
    ori = [g["orient"] for g in gmet]
    for i, j in pairs:
        gi, gj = labels[i], labels[j]
        if gi >= 0 and gj >= 0 and gi != gj:
            key = (min(gi, gj), max(gi, gj))
            if key not in seen:
                seen.add(key)
                mis.append(_ang_diff(ori[gi], ori[gj]))   # 0..30 deg (triangular)
    return np.array(mis)


# ----------------------------------------------------------------------
# 6. orientation -> RGB map  (a REAL 2D orientation key, not wrapped phi6)
# ----------------------------------------------------------------------
def orientation_rgb(atoms, theta_deg, mag, ny, nx, dx, mag_min=0.6):
    """Nearest-atom orientation map; hue=orientation/60, value=crystallinity."""
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx
    GX, GY = np.meshgrid(xs, ys)
    tree = cKDTree(atoms)
    _, idx = tree.query(np.column_stack([GX.ravel(), GY.ravel()]))
    H = (theta_deg[idx] / SYM_DEG).reshape(ny, nx)
    Vv = np.clip(mag[idx].reshape(ny, nx) / 1.0, 0, 1)
    Vv[Vv < mag_min] = 0.0
    hsv = np.dstack([H, np.ones_like(H), Vv])
    return mcolors.hsv_to_rgb(hsv)


def deviation_field(atoms, dev, mag, ny, nx, dx, mag_min=0.6):
    """Nearest-atom raster of the spacing deviation, same scheme as orientation_rgb.
    Returns a 2D array (NaN where not crystalline, so it renders transparent)."""
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx
    GX, GY = np.meshgrid(xs, ys)
    tree = cKDTree(atoms)
    _, idx = tree.query(np.column_stack([GX.ravel(), GY.ravel()]))
    field = dev[idx].reshape(ny, nx)
    crys = (mag[idx] >= mag_min).reshape(ny, nx)
    field = np.where(crys, field, np.nan)          # liquid/defect -> transparent
    return field


# ----------------------------------------------------------------------
# top-level driver
# ----------------------------------------------------------------------
def analyze(psi_path, ny, nx, dx, a0=A0_2D, build_dir_deg=90.0, out_prefix=None):
    psi = np.load(psi_path)
    psi2d = psi.reshape(ny, nx) if psi.ndim == 1 else psi
    atoms = detect_atoms(psi2d, dx, a0)
    mag, theta = bond_orientation(atoms, a0)
    labels, grains = segment_grains(atoms, theta, mag, a0)
    gmet = grain_metrics(atoms, grains, theta, build_dir_deg)
    mis = misorientation_distribution(atoms, labels, gmet, a0)

    width = np.array([g["width"] for g in gmet])
    tilt = np.array([g["tilt"] for g in gmet])
    columnar = np.array([g["aspect"] for g in gmet]) >= 1.5

    res = dict(atoms=atoms, theta=theta, mag=mag, labels=labels, grains=gmet,
               width=width, tilt=tilt, misorientation=mis,
               width_columnar=width[columnar], tilt_columnar=tilt[columnar],
               a0=a0, ny=ny, nx=nx, dx=dx)

    print(f"grains detected: {len(gmet)}  (columnar, aspect>=1.5: {int(columnar.sum())})")
    if len(width):
        print(f"width [units]      mean={width.mean():.2f}  std={width.std():.2f}  (in a0: {width.mean()/a0:.2f})")
        print(f"tilt  [deg]        mean={tilt.mean():.1f}  std={tilt.std():.1f}")
    if len(mis):
        print(f"misorientation[deg] mean={mis.mean():.1f}  std={mis.std():.1f}  (N_boundaries={len(mis)})")

    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        rgb = orientation_rgb(atoms, theta, mag, ny, nx, dx)
        # map panel + orientation key (hue = orientation 0..60 deg)
        fig = plt.figure(figsize=(12, 3.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[40, 1], wspace=0.04)
        axm = fig.add_subplot(gs[0, 0])
        axm.imshow(rgb, origin="lower", extent=[0, nx*dx, 0, ny*dx], aspect="auto")
        axm.set_xlabel("x (lattice units)"); axm.set_ylabel("y (lattice units)")
        axm.set_title("orientation map (hue = orientation mod 60$\\degree$, brightness = crystallinity)",
                      fontsize=10)
        # vertical orientation key
        axk = fig.add_subplot(gs[0, 1])
        grad = np.linspace(0, 1, 256).reshape(-1, 1)          # 0..1 -> 0..60 deg
        key = mcolors.hsv_to_rgb(np.dstack([np.repeat(grad, 8, axis=1),
                                            np.ones((256, 8)), np.ones((256, 8))]))
        axk.imshow(key, origin="lower", extent=[0, 1, 0, SYM_DEG], aspect="auto")
        axk.set_xticks([]); axk.yaxis.tick_right(); axk.yaxis.set_label_position("right")
        axk.set_yticks([0, 15, 30, 45, 60])
        axk.set_ylabel("orientation (deg)", rotation=90, labelpad=8)
        plt.tight_layout(); plt.savefig(f"{out_prefix}_orientation_rgb.png", dpi=130); plt.close()

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for a, data, lab in zip(ax, [width/a0, tilt, mis],
                                ["width / a0", "tilt [deg]", "misorientation [deg]"]):
            if len(data): a.hist(data, bins=15, color="#1F4E79", alpha=0.85)
            a.set_xlabel(lab); a.set_ylabel("count")
        plt.tight_layout(); plt.savefig(f"{out_prefix}_metrics_hist.png", dpi=130); plt.close()
    return res


# ----------------------------------------------------------------------
# experiment comparison  (Fig. 2)
# ----------------------------------------------------------------------
def metrics_from_orientation_map(theta_map_deg, mask, dx, build_dir_deg=90.0,
                                 sym=SYM_DEG, min_px=50, mis_tol_deg=8.0):
    """Same metrics from an EBSD per-pixel orientation map, for like-for-like
    comparison. theta_map_deg: 2D orientations; mask: valid (indexed) pixels."""
    ny, nx = theta_map_deg.shape
    lbl = np.zeros((ny, nx), int); cur = 0
    visited = np.zeros((ny, nx), bool)
    from collections import deque
    for i in range(ny):
        for j in range(nx):
            if visited[i, j] or not mask[i, j]:
                continue
            cur += 1; q = deque([(i, j)]); visited[i, j] = True; lbl[i, j] = cur
            while q:
                a, b = q.popleft()
                for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                    na, nb = a+da, b+db
                    if 0<=na<ny and 0<=nb<nx and not visited[na,nb] and mask[na,nb] \
                       and _ang_diff(theta_map_deg[a,b], theta_map_deg[na,nb], sym) <= mis_tol_deg:
                        visited[na,nb]=True; lbl[na,nb]=cur; q.append((na,nb))
    widths, tilts, oris = [], [], []
    for g in range(1, cur+1):
        ys, xs = np.where(lbl == g)
        if len(xs) < min_px:
            continue
        P = np.column_stack([xs*dx, ys*dx]); Q = P - P.mean(0)
        w, V = np.linalg.eigh(np.cov(Q.T)); emaj = V[:,1]; emin = V[:,0]
        widths.append(np.ptp(Q@emin))
        major_deg = np.rad2deg(np.arctan2(emaj[1], emaj[0]))
        tilts.append(_ang_diff(major_deg, build_dir_deg, 180.0))
        oris.append(np.mean(theta_map_deg[ys, xs]))
    return dict(width=np.array(widths), tilt=np.array(tilts), orient=np.array(oris))


def compare_to_experiment(sim_res, exp_metrics, out_prefix="Plots/sim_vs_exp"):
    """exp_metrics: dict with arrays 'width','tilt','misorientation' (or summary
    dict with *_mean/*_std). Writes a comparison CSV + overlay histograms."""
    rows = [("metric", "sim_mean", "sim_std", "exp_mean", "exp_std", "rel_err_%")]
    keys = [("width", sim_res["width"]), ("tilt", sim_res["tilt"]),
            ("misorientation", sim_res["misorientation"])]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, (name, sim) in enumerate(keys):
        exp = exp_metrics.get(name, np.array([]))
        em = exp.mean() if len(np.atleast_1d(exp)) and np.ndim(exp) else \
             exp_metrics.get(f"{name}_mean", np.nan)
        es = exp.std() if len(np.atleast_1d(exp)) and np.ndim(exp) else \
             exp_metrics.get(f"{name}_std", np.nan)
        sm, ss = (sim.mean(), sim.std()) if len(sim) else (np.nan, np.nan)
        rel = 100*abs(sm-em)/em if (em and np.isfinite(em)) else np.nan
        rows.append((name, f"{sm:.3f}", f"{ss:.3f}", f"{em:.3f}", f"{es:.3f}", f"{rel:.1f}"))
        if len(sim): ax[k].hist(sim, bins=15, alpha=0.6, label="sim", color="#1F4E79")
        if np.ndim(exp) and len(np.atleast_1d(exp)) > 1:
            ax[k].hist(exp, bins=15, alpha=0.6, label="exp (Fig.2)", color="#C55A11")
        ax[k].set_xlabel(name); ax[k].legend()
    plt.tight_layout(); plt.savefig(f"{out_prefix}.png", dpi=130); plt.close()
    with open(f"{out_prefix}.csv", "w") as f:
        f.write("\n".join(",".join(r) for r in rows))
    print("comparison written:", f"{out_prefix}.csv")
    return rows


# ----------------------------------------------------------------------
#  Figure 6:  lattice-spacing deviation map  (M11, R3-09)
# ----------------------------------------------------------------------
def analyze_deviation(psi_path, ny, nx, dx, a0=A0_2D, signed=True,
                      mag_min=0.6, out_prefix=None):
    """Per-atom nearest-neighbour spacing deviation (d-a0)/a0 -- the well-defined
    scalar that replaces the undefined 'displacement' of the original Fig. 6.

    signed=True  -> diverging map, compression (-) vs tension (+)  [(d-a0)/a0]
    signed=False -> magnitude only, paper-style brown 'enhanced displacement' [|d-a0|/a0]
    Returns dict with atoms, deviation, and the reference spacing a0_ref.
    """
    psi = np.load(psi_path)
    psi2d = psi.reshape(ny, nx) if psi.ndim == 1 else psi
    atoms = detect_atoms(psi2d, dx, a0)
    mag, _ = bond_orientation(atoms, a0)

    tree = cKDTree(atoms)
    d, _ = tree.query(atoms, k=2)
    nn = d[:, 1]                                   # nearest-neighbour distance
    a0_ref = float(np.median(nn))                  # equilibrium spacing (reference)
    dev = (nn - a0_ref) / a0_ref                   # signed spacing deviation
    keep = mag >= mag_min                          # crystalline atoms only

    res = dict(atoms=atoms, deviation=dev, crystalline=keep, a0_ref=a0_ref,
               ny=ny, nx=nx, dx=dx)
    print(f"atoms={len(atoms)} crystalline={int(keep.sum())}  a0_ref={a0_ref:.3f}")
    print(f"spacing deviation: mean|dev|={np.mean(np.abs(dev[keep])):.4f}  "
          f"max|dev|={np.max(np.abs(dev[keep])):.4f}")

    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        field = deviation_field(atoms, dev, mag, ny, nx, dx, mag_min)
        if not signed:
            field = np.abs(field)
        ext = [0, nx*dx, 0, ny*dx]
        vmax = np.nanpercentile(np.abs(field), 98)
        fig, ax = plt.subplots(figsize=(12, 3.2))
        if signed:
            im = ax.imshow(field, origin="lower", extent=ext, cmap="coolwarm",
                           vmin=-vmax, vmax=vmax, aspect="auto")
            lab = r"lattice-spacing deviation  $(d-a_0)/a_0$"
        else:
            im = ax.imshow(field, origin="lower", extent=ext, cmap="copper",
                           vmin=0, vmax=vmax, aspect="auto")
            lab = r"lattice-spacing deviation  $|d-a_0|/a_0$"
        ax.set_xlabel("x (lattice units)"); ax.set_ylabel("y (lattice units)")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label(lab)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_deviation.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"{out_prefix}_deviation.pdf", bbox_inches="tight")
        plt.close()
    return res