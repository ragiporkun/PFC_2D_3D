"""
pfc3d_postprocess.py  --  interpretable visualization of a 3D PFC run.

A raw volume render of the density psi is unreadable: every grain has the same
red peaks / blue valleys, so grains (which differ by ORIENTATION, not by psi
value) are invisible. This module turns a 3D psi field into:

  1. clean 2D cross-section slices of psi  (longitudinal / transverse / plan)
  2. a per-voxel ORIENTATION field via the 3D structure tensor, coloured RGB by
     the local lattice direction -- the analog of an EBSD inverse-pole-figure
     map. Grains appear as uniformly coloured regions; boundaries as colour
     changes. This is robust at the modest resolution (~16 cells per lattice
     constant) where per-atom orientation fitting is too noisy.
  3. a VTK export of the orientation field (orient_rgb as 3 scalars) so the
     polycrystal can be volume-rendered by orientation in ParaView.

Reads either a .npy (nz,ny,nx) array or an ASCII STRUCTURED_POINTS .vtk with a
'psi' scalar.

Usage:
    from pfc3d_postprocess import load_field, slices, orientation_field, save_orientation_vtk
    psi, sp = load_field("fields_000400.vtk")          # or .npy + spacing
    slices(psi, sp, out_prefix="post/cooled")
    rgb = orientation_field(psi, sp)                   # (nz,ny,nx,3)
    save_orientation_vtk(rgb, sp, "post/orientation.vtk")
"""
import numpy as np
from scipy.ndimage import gaussian_filter

A0 = 2 * np.pi * np.sqrt(2.0)          # bcc lattice constant used in the model


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def load_field(path, field="psi"):
    """Return (array (nz,ny,nx), spacing). Accepts .npy or ASCII VTK structured points."""
    if path.endswith(".npy"):
        a = np.load(path)
        return a, 1.0
    dims = None; sp = 1.0; fields = {}; cur = None; buf = []; reading = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("DIMENSIONS"):
                nx, ny, nz = map(int, s.split()[1:4]); dims = (nz, ny, nx)
            elif s.startswith("SPACING"):
                sp = float(s.split()[1])
            elif s.startswith("SCALARS"):
                if cur and buf: fields[cur] = np.array(buf, dtype=np.float32)
                cur = s.split()[1]; buf = []; reading = False
            elif s.startswith("LOOKUP_TABLE"):
                reading = True
            elif reading and s and not s[0].isalpha():
                buf.extend(float(v) for v in s.split())
    if cur and buf: fields[cur] = np.array(buf, dtype=np.float32)
    return fields[field].reshape(dims), sp


# ----------------------------------------------------------------------
# 1. cross-section slices of psi
# ----------------------------------------------------------------------
def slices(psi, sp=1.0, out_prefix="post/field", vmin=-0.5, vmax=1.5, cmap="jet"):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    nz, ny, nx = psi.shape
    lon = psi[nz // 2, :, :]          # scan(x) x depth(y)
    tra = psi[:, :, nx // 2]          # width(z) x depth(y)
    top = psi[:, max(ny - 4, 0), :]   # scan(x) x width(z), near top surface
    fig, ax = plt.subplots(3, 1, figsize=(9, 10))
    for a, (im, ttl, ext) in zip(ax, [
        (lon, "longitudinal (scan x vs depth y)", [0, nx*sp, 0, ny*sp]),
        (tra, "transverse (depth y vs width z)",  [0, ny*sp, 0, nz*sp]),
        (top, "plan near top (scan x vs width z)",[0, nx*sp, 0, nz*sp])]):
        h = a.imshow(im, origin="lower", cmap=cmap, extent=ext, vmin=vmin, vmax=vmax, aspect="auto")
        a.set_title(ttl, fontsize=10); plt.colorbar(h, ax=a, fraction=0.025, label=r"$\psi$")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_slices.png", dpi=130); plt.close()
    return f"{out_prefix}_slices.png"


# ----------------------------------------------------------------------
# 1b. single slice in the x-y plane (normal to z) at z = z_frac * Lz
# ----------------------------------------------------------------------
def slice_normal_z(psi, sp=1.0, z_frac=0.5, out_prefix="post/slice_z",
                   vmin=-0.5, vmax=1.5, cmap="jet", orientation=True):
    """Plane normal to the z-axis at z = z_frac*Lz (default mid-plane, z = nz*dz/2).
    x (scan) is horizontal, y (depth) is vertical. Saves the psi slice and,
    if orientation=True, the structure-tensor orientation map on the same plane.
    Returns (k_index, list_of_png_paths)."""
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    nz, ny, nx = psi.shape
    k = int(round(z_frac * nz)); k = min(max(k, 0), nz - 1)   # z = nz*dz/2 -> k = nz//2
    sl = psi[k, :, :]                                          # (ny, nx): rows=y, cols=x
    ext = [0, nx * sp, 0, ny * sp]
    out = []
    plt.figure(figsize=(9, 4))
    h = plt.imshow(sl, origin="lower", cmap=cmap, extent=ext, vmin=vmin, vmax=vmax, aspect="auto")
    plt.xlabel("x (scan)"); plt.ylabel("y (depth)")
    plt.title(rf"$\psi$  |  plane $\perp z$ at $z={k*sp:.2f}$  (k={k} of {nz})")
    plt.colorbar(h, fraction=0.03, label=r"$\psi$")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_psi.png", dpi=130); plt.close()
    out.append(f"{out_prefix}_psi.png")
    if orientation:
        rgb = orientation_field(psi, sp)
        plt.figure(figsize=(9, 4))
        plt.imshow(rgb[k, :, :], origin="lower", extent=ext, aspect="auto")
        plt.xlabel("x (scan)"); plt.ylabel("y (depth)")
        plt.title(rf"orientation  |  plane $\perp z$ at $z={k*sp:.2f}$")
        plt.tight_layout(); plt.savefig(f"{out_prefix}_orient.png", dpi=130); plt.close()
        out.append(f"{out_prefix}_orient.png")
    return k, out


# ----------------------------------------------------------------------
# 2. structure-tensor orientation field -> RGB
# ----------------------------------------------------------------------
def orientation_field(psi, sp=1.0, smooth_cells=None):
    """Per-voxel principal direction of the smoothed structure tensor, |components|->RGB.
    Within a grain the dominant lattice direction is uniform -> uniform colour."""
    psi = psi.astype(float)
    gz, gy, gx = np.gradient(psi)
    sig = (A0 / sp) * 0.9 if smooth_cells is None else smooth_cells
    Jxx = gaussian_filter(gx*gx, sig); Jyy = gaussian_filter(gy*gy, sig); Jzz = gaussian_filter(gz*gz, sig)
    Jxy = gaussian_filter(gx*gy, sig); Jxz = gaussian_filter(gx*gz, sig); Jyz = gaussian_filter(gy*gz, sig)
    nz, ny, nx = psi.shape
    # assemble per-voxel 3x3 tensors and take the principal eigenvector (vectorized eigh)
    J = np.zeros((nz, ny, nx, 3, 3))
    J[..., 0, 0] = Jxx; J[..., 1, 1] = Jyy; J[..., 2, 2] = Jzz
    J[..., 0, 1] = J[..., 1, 0] = Jxy
    J[..., 0, 2] = J[..., 2, 0] = Jxz
    J[..., 1, 2] = J[..., 2, 1] = Jyz
    w, v = np.linalg.eigh(J)                     # ascending; principal = last
    e = np.abs(v[..., -1])                       # |principal eigenvector| components
    e = e / (e.max(axis=-1, keepdims=True) + 1e-9)
    return e                                     # (nz,ny,nx,3) RGB in [0,1]


def orientation_slices(psi, sp=1.0, out_prefix="post/orient"):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    rgb = orientation_field(psi, sp)
    nz, ny, nx = psi.shape
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].imshow(rgb[:, max(ny-6, 0), :], origin="lower", extent=[0, nx*sp, 0, nz*sp], aspect="auto")
    ax[0].set_title("orientation map — plan near top")
    ax[1].imshow(rgb[nz//2, :, :], origin="lower", extent=[0, nx*sp, 0, ny*sp], aspect="auto")
    ax[1].set_title("orientation map — longitudinal section")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_slices.png", dpi=130); plt.close()
    return rgb, f"{out_prefix}_slices.png"


# ----------------------------------------------------------------------
# 3. VTK export of the orientation field (for Paraway volume render)
# ----------------------------------------------------------------------
def save_orientation_vtk(rgb, sp, path):
    nz, ny, nx, _ = rgb.shape
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nPFC orientation field\nASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN 0 0 0\nSPACING {sp} {sp} {sp}\n")
        f.write(f"POINT_DATA {nx*ny*nz}\n")
        for ci, cname in enumerate(["orient_r", "orient_g", "orient_b"]):
            f.write(f"SCALARS {cname} float 1\nLOOKUP_TABLE default\n")
            vals = rgb[..., ci].ravel(order="C")   # x fastest within (nz,ny,nx) C-order matches VTK
            for i in range(0, len(vals), 12):
                f.write(" ".join(f"{x:.4f}" for x in vals[i:i+12]) + "\n")
    return path


# ----------------------------------------------------------------------
# 4. EXACT 3D misorientation from rotation matrices (e.g. grain_data.csv)
#    NOTE: this is the reliable, quotable route. FFT-based orientation
#    recovery from the VTK psi field was tested and is NOT reliable at this
#    resolution (a one-mode bcc density is centrosymmetric, so its {110}
#    diffraction star cannot pin a unique orientation; recovery error ~30 deg
#    even on clean synthetic data). Use rotation matrices you already know.
# ----------------------------------------------------------------------
import math as _math, itertools as _it


def _euler_to_matrix(a, b, g):
    """R = Rz(g) Ry(b) Rx(a). Must match the seeding convention in the solver."""
    ca, sa = _math.cos(a), _math.sin(a)
    cb, sb = _math.cos(b), _math.sin(b)
    cg, sg = _math.cos(g), _math.sin(g)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _cubic_ops():
    ops = []
    for perm in _it.permutations(range(3)):
        for sgn in _it.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = sgn[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-6:
                ops.append(M)
    return ops


_CUBIC = _cubic_ops()   # 24 proper rotations of the cube


def misorientation_3d(Ri, Rj):
    """Cubic-symmetry-folded misorientation angle (deg) between two orientations.
    theta = min_g  arccos((tr(g Ri^T Rj) - 1)/2), with the [-1,1] clip."""
    dR = Ri.T @ Rj
    best = 180.0
    for g in _CUBIC:
        c = np.clip((np.trace(g @ dR) - 1.0) / 2.0, -1.0, 1.0)
        best = min(best, _math.degrees(_math.acos(c)))
    return best


def misorientation_from_grain_data(csv_path, k_neighbors=1):
    """Read grain_data.csv (columns: x,y,z,euler_a,euler_b,euler_c) and return the
    distribution of misorientation angles between each grain and its k nearest
    neighbours (k_neighbors=1 approximates the boundary-misorientation set).
    Returns (angles_array, summary_dict). Exact -- uses the seeded orientations."""
    from scipy.spatial import cKDTree
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    centers = data[:, 0:3]
    R = [_euler_to_matrix(*data[i, 3:6]) for i in range(len(data))]
    tree = cKDTree(centers)
    kk = min(k_neighbors + 1, len(data))
    _, nbr = tree.query(centers, k=kk)
    seen = set(); ang = []
    for i in range(len(data)):
        for j in np.atleast_1d(nbr[i])[1:]:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            ang.append(misorientation_3d(R[i], R[j]))
    ang = np.array(ang)
    summary = dict(n_grains=len(data), n_pairs=len(ang),
                   mean=float(ang.mean()) if len(ang) else float("nan"),
                   std=float(ang.std()) if len(ang) else float("nan"),
                   min=float(ang.min()) if len(ang) else float("nan"),
                   max=float(ang.max()) if len(ang) else float("nan"))
    return ang, summary


def grain_misorientation_map(grain_data_csv, like_vtk=None, dims=None, sp=None,
                             ref_grain=0, ref_orientation=None, z_frac=0.5,
                             out_path="post/misori_map.png", angle_unit="rad", cmap="jet"):
    """Mid-z slice coloured by the EXACT cubic-folded misorientation of each grain to
    a reference, using the seeded orientations in grain_data.csv. Colourbar 0..pi/3
    (angle_unit='rad') or 0..60 deg (angle_unit='deg').

    Grains are approximated as the Voronoi regions of the seed centres -- exact for
    the as-seeded polycrystal; for a post-melt frame the region shapes are only
    approximate (orientations are still exact). Reference is grain `ref_grain`
    (default 0) or an explicit 3x3 `ref_orientation`.

    Provide the grid via `like_vtk` (a VTK file to copy nz,ny,nx,sp from) OR via
    explicit `dims=(nz,ny,nx)` and `sp`. Returns the output PNG path."""
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import os
    from scipy.spatial import cKDTree
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    data = np.loadtxt(grain_data_csv, delimiter=",", skiprows=1)
    centers = data[:, 0:3]
    R = [_euler_to_matrix(*data[i, 3:6]) for i in range(len(data))]

    if like_vtk is not None:
        _psi, sp = load_field(like_vtk); nz, ny, nx = _psi.shape
    elif dims is not None and sp is not None:
        nz, ny, nx = dims
    else:
        raise ValueError("Provide either like_vtk, or both dims=(nz,ny,nx) and sp.")

    Rref = ref_orientation if ref_orientation is not None else R[ref_grain]
    ang_g = np.array([misorientation_3d(Rref, R[i]) for i in range(len(R))])   # degrees

    zc = (np.arange(nz) + 0.5) * sp; yc = (np.arange(ny) + 0.5) * sp; xc = (np.arange(nx) + 0.5) * sp
    Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])    # match centres (x,y,z)
    _, lab = cKDTree(centers).query(pts); lab = lab.reshape(nz, ny, nx)
    angle_vol = ang_g[lab]                                       # degrees per voxel

    if angle_unit == "rad":
        angle_vol = np.radians(angle_vol); vmax = np.pi / 3.0; lbl = "misorientation angle (rad)"
        ticks = [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3]
        ticklab = ["0", r"$\pi/12$", r"$\pi/6$", r"$\pi/4$", r"$\pi/3$"]
    else:
        vmax = 60.0; lbl = "misorientation angle (deg)"; ticks = [0, 15, 30, 45, 60]; ticklab = None

    k = int(round(z_frac * nz)); k = min(max(k, 0), nz - 1)
    sl = angle_vol[k, :, :]
    plt.figure(figsize=(9, 4))
    h = plt.imshow(sl, origin="lower", cmap=cmap, extent=[0, nx*sp, 0, ny*sp],
                   vmin=0.0, vmax=vmax, aspect="auto")
    plt.xlabel("x (scan)"); plt.ylabel("y (depth)")
    plt.title(rf"grain misorientation to ref grain {ref_grain}  (plane $\perp z$, z={k*sp:.2f})")
    cb = plt.colorbar(h, fraction=0.03, label=lbl, ticks=ticks)
    if ticklab is not None: cb.ax.set_yticklabels(ticklab)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    return out_path


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "fields_000020.vtk"
    psi, sp = load_field(src)
    print("loaded", src, "shape", psi.shape, "spacing", sp, "psi range", psi.min(), psi.max())
    print("slices ->", slices(psi, sp, out_prefix="post/field"))
    rgb, fig = orientation_slices(psi, sp, out_prefix="post/orient")
    print("orientation slices ->", fig)
    print("orientation vtk ->", save_orientation_vtk(rgb, sp, "post/orientation.vtk"))