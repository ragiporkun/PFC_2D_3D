import os
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable


# ============================================================
# Runtime configuration (device + dtype)
# ============================================================
# Set REQUIRE_CUDA=1 (default) to fail fast if CUDA is unavailable instead of
# silently running on CPU. Override with REQUIRE_CUDA=0 to allow CPU fallback.
REQUIRE_CUDA = os.environ.get("REQUIRE_CUDA", "1") != "0"

# Default is float64 for numerical fidelity. On consumer Blackwell (RTX 50xx)
# FP64 throughput is throttled (~1:64 of FP32) and FP64 doubles memory traffic,
# so float32 is typically 2-4x faster end-to-end. Switch with TORCH_DTYPE=float32.
_DTYPE_MAP = {"float32": torch.float32, "float64": torch.float64}
TORCH_DTYPE = _DTYPE_MAP[os.environ.get("TORCH_DTYPE", "float64").lower()]


# ============================================================
# Params
# ============================================================
@dataclass
class Params:
    dx: float
    dy: float
    dz: float
    nx: int
    ny: int
    nz: int

    dt: float

    # initial
    psi0: float
    T0: float

    # model
    alpha: float
    beta: float
    M: float
    tau_psi: float

    # process
    v_x: float
    power: float
    a: float = 5
    b: float = 10
    c: float = 5
    absorption: float = 0.3

    # laser center initial
    x_laser_center: float = 0.0
    y_laser_center: float = 0.0
    z_laser_center: float = 0.0

    # IMEX ref temperature
    T_ref: float = 0.6

    # seeds
    seed_np: int = 56805
    num_grains: int = 32
    grain_radius: float = 2 * math.pi * math.sqrt(2)
    p_wave: float = 2 * math.pi * math.sqrt(2)
    B_amp: float = 0.102722199982204
    # substrate-layer seeding (polycrystalline base + liquid above -> columnar growth)
    substrate_frac: float = 0.4
    sub_nx: int = 3
    sub_nz: int = 3

    # --- BCC stability parameters ---
    # Maximum allowed temperature (prevents BCC -> stripe transition)
    T_max: float = 1.2
    # Three-body stabilization coefficient (widens BCC stability region)
    gamma_3b: float = 0.05


# ============================================================
# VTK writer (STRUCTURED_POINTS) with Fortran-order flattening
# ============================================================
def flatten_fortran(u: torch.Tensor) -> np.ndarray:
    return u.permute(2, 1, 0).contiguous().view(-1).detach().cpu().numpy()


def write_vtk_structured_points(filename: str, p: Params, fields: Dict[str, torch.Tensor]):
    nx, ny, nz = p.nx, p.ny, p.nz
    npts = nx * ny * nz
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("PyTorch phase-field output\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {p.dx/2.0:.8f} {p.dy/2.0:.8f} {p.dz/2.0:.8f}\n")
        f.write(f"SPACING {p.dx:.8f} {p.dy:.8f} {p.dz:.8f}\n")
        f.write(f"POINT_DATA {npts}\n")

        for name, arr3 in fields.items():
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            vals = flatten_fortran(arr3.to(torch.float32))
            per_line = 8
            for i in range(0, npts, per_line):
                f.write(" ".join(f"{v:.6e}" for v in vals[i:i + per_line]) + "\n")


# ============================================================
# Boundary padding helper for cell-centered FV
# ============================================================
def pad_scalar_mixed(u: torch.Tensor, bc: Dict[str, Tuple[str, Optional[float]]]) -> torch.Tensor:
    up = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="replicate")
    up = up.squeeze(0).squeeze(0)

    def enforce(face: str, ghost_slice, interior_slice):
        typ, val = bc.get(face, ("N", None))
        if typ == "D":
            if val is None:
                raise ValueError(f"Dirichlet BC for {face} requires a value.")
            up[ghost_slice] = 2.0 * float(val) - up[interior_slice]

    enforce("x0", (0, slice(None), slice(None)), (1, slice(None), slice(None)))
    enforce("x1", (-1, slice(None), slice(None)), (-2, slice(None), slice(None)))
    enforce("y0", (slice(None), 0, slice(None)), (slice(None), 1, slice(None)))
    enforce("y1", (slice(None), -1, slice(None)), (slice(None), -2, slice(None)))
    enforce("z0", (slice(None), slice(None), 0), (slice(None), slice(None), 1))
    enforce("z1", (slice(None), slice(None), -1), (slice(None), slice(None), -2))

    return up


def pad_neumann(u: torch.Tensor) -> torch.Tensor:
    return pad_scalar_mixed(u, {})


def pad_coeff_neumann(k: torch.Tensor) -> torch.Tensor:
    return pad_neumann(k)


# ============================================================
# Cell-centered FV diffusion: div( k grad u )
# ============================================================
def diffusion_var(
    u: torch.Tensor,
    k: torch.Tensor,
    p: Params,
    pad_u: Callable[[torch.Tensor], torch.Tensor],
    pad_k: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    up = pad_u(u)
    kp = pad_k(k)

    dx2 = p.dx * p.dx
    dy2 = p.dy * p.dy
    dz2 = p.dz * p.dz

    uc = up[1:-1, 1:-1, 1:-1]
    kc = kp[1:-1, 1:-1, 1:-1]

    kxp = 0.5 * (kp[2:, 1:-1, 1:-1] + kc)
    kxm = 0.5 * (kc + kp[:-2, 1:-1, 1:-1])
    divx = (kxp * (up[2:, 1:-1, 1:-1] - uc) - kxm * (uc - up[:-2, 1:-1, 1:-1])) / dx2

    kyp = 0.5 * (kp[1:-1, 2:, 1:-1] + kc)
    kym = 0.5 * (kc + kp[1:-1, :-2, 1:-1])
    divy = (kyp * (up[1:-1, 2:, 1:-1] - uc) - kym * (uc - up[1:-1, :-2, 1:-1])) / dy2

    kzp = 0.5 * (kp[1:-1, 1:-1, 2:] + kc)
    kzm = 0.5 * (kc + kp[1:-1, 1:-1, :-2])
    divz = (kzp * (up[1:-1, 1:-1, 2:] - uc) - kzm * (uc - up[1:-1, 1:-1, :-2])) / dz2

    return divx + divy + divz


def laplacian(u: torch.Tensor, p: Params, pad_u: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    ones = torch.ones_like(u)
    return diffusion_var(u, ones, p, pad_u, pad_coeff_neumann)


# ============================================================
# |grad u|^2 (central differences + Neumann padding)
# ============================================================
@torch.no_grad()
def grad_sq_neumann(u: torch.Tensor, p: Params) -> torch.Tensor:
    up = pad_neumann(u)
    dudx = (up[2:, 1:-1, 1:-1] - up[:-2, 1:-1, 1:-1]) / (2.0 * p.dx)
    dudy = (up[1:-1, 2:, 1:-1] - up[1:-1, :-2, 1:-1]) / (2.0 * p.dy)
    dudz = (up[1:-1, 1:-1, 2:] - up[1:-1, 1:-1, :-2]) / (2.0 * p.dz)
    return dudx * dudx + dudy * dudy + dudz * dudz


# ============================================================
# Matrix-free BiCGStab
# ============================================================
@torch.no_grad()
def bicgstab(matvec, b, x0=None, tol=1e-8, maxiter=200, verbose=False):
    device = b.device
    dtype = b.dtype

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    rhat = r.clone()

    rho_old = torch.tensor(1.0, device=device, dtype=dtype)
    alpha = torch.tensor(1.0, device=device, dtype=dtype)
    omega = torch.tensor(1.0, device=device, dtype=dtype)

    v = torch.zeros_like(b)
    pvec = torch.zeros_like(b)

    bnorm = torch.linalg.norm(b)
    bnorm = torch.clamp(bnorm, min=torch.tensor(1e-30, device=device, dtype=dtype))

    for k in range(1, maxiter + 1):
        rho_new = torch.sum(rhat * r)
        if torch.abs(rho_new) < 1e-30:
            break

        if k == 1:
            pvec = r
        else:
            beta = (rho_new / rho_old) * (alpha / omega)
            pvec = r + beta * (pvec - omega * v)

        v = matvec(pvec)
        denom = torch.sum(rhat * v)
        if torch.abs(denom) < 1e-30:
            break

        alpha = rho_new / denom
        s = r - alpha * v

        snorm = torch.linalg.norm(s)
        if snorm / bnorm < tol:
            return x + alpha * pvec

        t = matvec(s)
        tt = torch.sum(t * t)
        if tt < 1e-30:
            break

        omega = torch.sum(t * s) / tt
        x = x + alpha * pvec + omega * s
        r = s - omega * t

        rnorm = torch.linalg.norm(r)
        if verbose and (k == 1 or k % 20 == 0):
            print(f"  BiCGStab iter {k:4d}: relres={float(rnorm / bnorm):.3e}")
        if rnorm / bnorm < tol:
            break

        rho_old = rho_new

        if not torch.isfinite(x).all() or not torch.isfinite(r).all():
            break

    return x


# ============================================================
# Heat source
# ============================================================
@torch.no_grad()
def gaussian_source_3d(p: Params, x_center: float, y_center: float, z_center: float, device, dtype) -> torch.Tensor:
    x = (torch.arange(p.nx, device=device, dtype=dtype) + 0.5) * p.dx
    y = (torch.arange(p.ny, device=device, dtype=dtype) + 0.5) * p.dy
    z = (torch.arange(p.nz, device=device, dtype=dtype) + 0.5) * p.dz
    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    pref = (p.absorption * p.power) / (((2 * math.pi) ** 1.5) * p.a * p.b * p.c)
    return (
        pref
        * torch.exp(-((X - x_center) ** 2) / (2 * p.a ** 2))
        * torch.exp(-((Y - y_center) ** 2) / (2 * p.b ** 2))
        * torch.exp(-((Z - z_center) ** 2) / (2 * p.c ** 2))
    )


# ============================================================
# Seeding (fixed centers version)
# ============================================================

# ------------------------------------------------------------
# per-grain 3D orientation (random Euler rotation matrix)
# ------------------------------------------------------------
def euler_to_matrix(a, b, g):
    """Rotation matrix R = Rz(g) Ry(b) Rx(a) for seeding distinct grain orientations."""
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    Rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    Ry = np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
    Rz = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
    return Rz @ Ry @ Rx


@torch.no_grad()
def seed_grains_like_fipy(p: Params, device, dtype) -> Tuple[torch.Tensor, np.ndarray]:
    np.random.seed(p.seed_np)

    Lx, Ly, Lz = p.nx * p.dx, p.ny * p.dy, p.nz * p.dz
    R = float(p.grain_radius)

    q = 2.0 * math.pi / p.p_wave
    B = p.B_amp

    centers = np.array(
        [
            (1 * R, 1 * R, 1 * R),
            (3 * R, 1 * R, 1 * R),
            (5 * R, 1 * R, 1 * R),
            (7 * R, 1 * R, 1 * R),
            (1 * R, 1 * R, 3 * R),
            (3 * R, 1 * R, 3 * R),
            (5 * R, 1 * R, 3 * R),
            (7 * R, 1 * R, 3 * R),
            (1 * R, 1 * R, 5 * R),
            (3 * R, 1 * R, 5 * R),
            (5 * R, 1 * R, 5 * R),
            (7 * R, 1 * R, 5 * R),
            (1 * R, 1 * R, 7 * R),
            (3 * R, 1 * R, 7 * R),
            (5 * R, 1 * R, 7 * R),
            (7 * R, 1 * R, 7 * R),
            (1 * R, 3 * R, 1 * R),
            (3 * R, 3 * R, 1 * R),
            (5 * R, 3 * R, 1 * R),
            (7 * R, 3 * R, 1 * R),
            (1 * R, 3 * R, 3 * R),
            (3 * R, 3 * R, 3 * R),
            (5 * R, 3 * R, 3 * R),
            (7 * R, 3 * R, 3 * R),
            (1 * R, 3 * R, 5 * R),
            (3 * R, 3 * R, 5 * R),
            (5 * R, 3 * R, 5 * R),
            (7 * R, 3 * R, 5 * R),
            (1 * R, 3 * R, 7 * R),
            (3 * R, 3 * R, 7 * R),
            (5 * R, 3 * R, 7 * R),
            (7 * R, 3 * R, 7 * R),
        ],
        dtype=np.float64,
    )

    if p.num_grains < centers.shape[0]:
        centers = centers[: p.num_grains]
    elif p.num_grains > centers.shape[0]:
        raise ValueError(f"p.num_grains={p.num_grains}, but only {centers.shape[0]} fixed centers are defined.")

    eps = 1e-12
    if (
        np.any(centers[:, 0] < R - eps) or np.any(centers[:, 0] > Lx - R + eps)
        or np.any(centers[:, 1] < R - eps) or np.any(centers[:, 1] > Ly - R + eps)
        or np.any(centers[:, 2] < R - eps) or np.any(centers[:, 2] > Lz - R + eps)
    ):
        raise ValueError(
            "One or more fixed centers fall outside the valid domain [R, L-R]. "
            "Increase nx/ny/nz or dx/dy/dz, or adjust the center list."
        )

    # three random Euler angles per grain -> distinct crystallographic orientations
    euler = np.random.uniform(0, 2 * np.pi, size=(centers.shape[0], 3))

    x = (torch.arange(p.nx, device=device, dtype=dtype) + 0.5) * p.dx
    y = (torch.arange(p.ny, device=device, dtype=dtype) + 0.5) * p.dy
    z = (torch.arange(p.nz, device=device, dtype=dtype) + 0.5) * p.dz
    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    psi = torch.full((p.nx, p.ny, p.nz), p.psi0, device=device, dtype=dtype)

    for (xc, yc, zc), ang in zip(centers, euler):
        xShift = X - xc
        yShift = Y - yc
        zShift = Z - zc

        # Rotate THIS grain's lattice by its own random orientation, so that
        # neighbouring seeds grow with different crystallographic orientations.
        # Without this rotation every seed shares one orientation and the domain
        # is effectively a single crystal (referee M9 / R3-21). ang = (a, b, g).
        Rg = euler_to_matrix(float(ang[0]), float(ang[1]), float(ang[2]))
        xr = float(Rg[0, 0]) * xShift + float(Rg[0, 1]) * yShift + float(Rg[0, 2]) * zShift
        yr = float(Rg[1, 0]) * xShift + float(Rg[1, 1]) * yShift + float(Rg[1, 2]) * zShift
        zr = float(Rg[2, 0]) * xShift + float(Rg[2, 1]) * yShift + float(Rg[2, 2]) * zShift

        wave = 4.0 * B * (
            torch.cos(q * xr / math.sqrt(2)) * torch.cos(q * yr / math.sqrt(2))
            + torch.cos(q * yr / math.sqrt(2)) * torch.cos(q * zr / math.sqrt(2))
            + torch.cos(q * xr / math.sqrt(2)) * torch.cos(q * zr / math.sqrt(2))
        )

        mask = (xShift ** 2 + yShift ** 2 + zShift ** 2) <= (R ** 2)
        psi = torch.where(mask, wave, psi)

    grain_data = np.column_stack((centers, euler))
    return psi, grain_data


# ============================================================
# mu and q  (WITH three-body BCC stabilization)
# ============================================================
@torch.no_grad()
def compute_mu(psi: torch.Tensor, T: torch.Tensor, p: Params) -> torch.Tensor:
    T_safe = torch.clamp(T, min=1e-8)
    invT = 1.0 / T_safe

    lap_psi = laplacian(psi, p, pad_neumann)
    bilap_psi = laplacian(lap_psi, p, pad_neumann)

    # Standard PFC chemical potential
    mu = (psi - psi * psi + psi * psi * psi) - (p.beta * invT) + invT * (2.0 * lap_psi + bilap_psi)

    # ---- THREE-BODY BCC STABILIZATION ----
    # Adds dF_3b/dpsi = -gamma_3b * 2*psi*(psi^2 - <psi^2>)
    # This penalizes stripe formation and widens the BCC stability window.
    if p.gamma_3b > 0.0:
        psi_sq_mean = torch.mean(psi * psi)
        mu_3b = -p.gamma_3b * 2.0 * psi * (psi * psi - psi_sq_mean)
        mu = mu + mu_3b

    return mu


@torch.no_grad()
def compute_q(psi: torch.Tensor, T: torch.Tensor, p: Params) -> torch.Tensor:
    # Constant heat capacity c_p = alpha (consistent with the 2D solver).
    return torch.full_like(T, p.alpha)


# ============================================================
# psi IMEX step  (includes three-body correction in mu_corr)
# ============================================================
@torch.no_grad()
def step_psi_imex(psi_n: torch.Tensor, T_n: torch.Tensor, p: Params, tol=1e-8, maxiter=120, verbose=False) -> torch.Tensor:
    dt = p.dt
    kref = 1.0 / p.T_ref

    lap_psi_n = laplacian(psi_n, p, pad_neumann)
    bilap_psi_n = laplacian(lap_psi_n, p, pad_neumann)

    mu_n = compute_mu(psi_n, T_n, p)  # now includes three-body term
    mu_lin_ref_n = psi_n + kref * (2.0 * lap_psi_n + bilap_psi_n)
    mu_corr_n = mu_n - mu_lin_ref_n

    rhs = psi_n + (dt / p.tau_psi) * laplacian(mu_corr_n, p, pad_neumann)

    def A_mv(x: torch.Tensor) -> torch.Tensor:
        lapx = laplacian(x, p, pad_neumann)
        bilapx = laplacian(lapx, p, pad_neumann)
        inside = x + kref * (2.0 * lapx + bilapx)
        return x - (dt / p.tau_psi) * laplacian(inside, p, pad_neumann)

    return bicgstab(A_mv, rhs, x0=psi_n, tol=tol, maxiter=maxiter, verbose=verbose)


# ============================================================
# T semi-implicit step  (WITH T_max clamping)
# ============================================================
@torch.no_grad()
def step_T_semiimplicit(
    T_n: torch.Tensor,
    q: torch.Tensor,
    dpsi_dt: torch.Tensor,
    source_term: torch.Tensor,
    source_coeff: float,
    sink: float,
    p: Params,
    T_bc: Dict[str, Tuple[str, Optional[float]]],
    tol=1e-8,
    maxiter=200,
    verbose=False,
) -> torch.Tensor:
    dt = p.dt
    padT = lambda u: pad_scalar_mixed(u, T_bc)

    q_eff = torch.clamp(q, min=1e-8)

    rhs = (q_eff / dt) * T_n + p.beta * dpsi_dt + (source_coeff * source_term) + (sink * p.T0)

    def A_mv(x: torch.Tensor) -> torch.Tensor:
        return (q_eff / dt + sink) * x - p.M * laplacian(x, p, padT)

    T_new = bicgstab(A_mv, rhs, x0=T_n, tol=tol, maxiter=maxiter, verbose=verbose)

    # ---- CLAMP TEMPERATURE TO PREVENT BCC -> STRIPE TRANSITION ----
    T_new = torch.clamp(T_new, min=1e-8, max=p.T_max)

    return T_new


# ============================================================
# One full timestep (Picard)
# ============================================================
@torch.no_grad()
def advance_one_step(
    psi: torch.Tensor,
    T: torch.Tensor,
    source_term: torch.Tensor,
    source_coeff: float,
    sink: float,
    T_bc: Dict[str, Tuple[str, Optional[float]]],
    p: Params,
    n_picard: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    psi_old = psi
    T_old = T

    psi_k = psi
    T_k = T

    for _ in range(n_picard):
        psi_new = step_psi_imex(psi_old, T_k, p, tol=1e-8, maxiter=200, verbose=False)
        dpsi_dt = (psi_new - psi_old) / p.dt

        q_new = compute_q(psi_new, T_k, p)

        T_new = step_T_semiimplicit(
            T_old,
            q_new,
            dpsi_dt,
            source_term,
            source_coeff,
            sink,
            p,
            T_bc,
            tol=1e-8,
            maxiter=200,
            verbose=False,
        )

        psi_k, T_k = psi_new, T_new

    return psi_k, T_k


# ============================================================
# Main
# ============================================================
@torch.no_grad()
def seed_substrate_layer(p: Params, device, dtype):
    """Polycrystalline substrate filling the bottom (y < substrate_frac*Ly) with liquid
    above. The bottom x-z plane is tiled into sub_nx x sub_nz differently-oriented bcc
    grains. A moving source applied at the top surface then drives columnar epitaxial
    regrowth upward from these substrate grains -- the canonical AM columnar structure,
    far more legible than a fully-seeded equiaxed cube (referee M9 / R3-21)."""
    np.random.seed(p.seed_np)
    Lx, Ly, Lz = p.nx * p.dx, p.ny * p.dy, p.nz * p.dz
    q = 2.0 * math.pi / p.p_wave
    B = p.B_amp
    y_sub = p.substrate_frac * Ly

    x = (torch.arange(p.nx, device=device, dtype=dtype) + 0.5) * p.dx
    y = (torch.arange(p.ny, device=device, dtype=dtype) + 0.5) * p.dy
    z = (torch.arange(p.nz, device=device, dtype=dtype) + 0.5) * p.dz
    X = x[:, None, None]; Y = y[None, :, None]; Z = z[None, None, :]

    psi = torch.full((p.nx, p.ny, p.nz), p.psi0, device=device, dtype=dtype)
    in_sub = (Y < y_sub)
    grain_data = []
    for ix in range(p.sub_nx):
        for iz in range(p.sub_nz):
            x0, x1 = ix * Lx / p.sub_nx, (ix + 1) * Lx / p.sub_nx
            z0, z1 = iz * Lz / p.sub_nz, (iz + 1) * Lz / p.sub_nz
            ang = np.random.uniform(0, 2 * np.pi, size=3)
            Rg = euler_to_matrix(float(ang[0]), float(ang[1]), float(ang[2]))
            xc, zc = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
            xs = X - xc; ys = Y; zs = Z - zc
            xr = float(Rg[0, 0]) * xs + float(Rg[0, 1]) * ys + float(Rg[0, 2]) * zs
            yr = float(Rg[1, 0]) * xs + float(Rg[1, 1]) * ys + float(Rg[1, 2]) * zs
            zr = float(Rg[2, 0]) * xs + float(Rg[2, 1]) * ys + float(Rg[2, 2]) * zs
            wave = 4.0 * B * (
                torch.cos(q * xr / math.sqrt(2)) * torch.cos(q * yr / math.sqrt(2))
                + torch.cos(q * yr / math.sqrt(2)) * torch.cos(q * zr / math.sqrt(2))
                + torch.cos(q * xr / math.sqrt(2)) * torch.cos(q * zr / math.sqrt(2))
            )
            region = in_sub & (X >= x0) & (X < x1) & (Z >= z0) & (Z < z1)
            psi = torch.where(region, wave, psi)
            grain_data.append([xc, 0.0, zc, float(ang[0]), float(ang[1]), float(ang[2])])
    return psi, np.array(grain_data, dtype=np.float64)


def main():
    os.makedirs("VTK", exist_ok=True)

    dx = dy = dz = 2 * math.pi * math.sqrt(2) / 16
    nx = 2**7
    ny = 2**6
    nz = 2**7
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz

    dt = 0.1

    p = Params(
        dx=dx, dy=dy, dz=dz,
        nx=nx, ny=ny, nz=nz,
        dt=dt,
        a=2*math.pi*math.sqrt(2),       # beam half-widths = a0 (FWHM ~2.4 a0 -> spans several unit cells,
        b=2*math.pi*math.sqrt(2),       # answering referee R3-14) while keeping enough peak intensity to melt.
        c=2*math.pi*math.sqrt(2),       # NOTE: source prefactor ~ power/(a*b*c), so wider beam needs more power.
        psi0=0.0,
        T0=0.6,
        alpha=3.,
        beta=0.5,
        M=0.6,
        tau_psi=1.5,                 # CHANGED: slower psi kinetics (was 0.6) to suppress stripe nucleation
        v_x=1.,
        power=7000.0,               # scaled up for the a0-wide beam: pref~power/(a*b*c). Tune so
                                    # T.max reaches ~1.3-1.5 during the laser pass (watch printout).
        x_laser_center=Lx / 5.0,
        y_laser_center=Ly,
        z_laser_center=Lz / 2.0,
        T_ref=0.6,
        seed_np=56805,
        num_grains=32,
        grain_radius=2 * math.pi * math.sqrt(2),
        substrate_frac=0.45,   # bottom 45% is solid polycrystalline substrate
        sub_nx=3, sub_nz=3,    # 3x3 = 9 substrate grains of distinct orientation
        p_wave=2 * math.pi * math.sqrt(2),
        B_amp=0.102722199982204,
        # --- BCC stability controls ---
        T_max=3.0,                   # loose safety cap only (was 1.2, which clamped the superheat).
                                    # Set very high / remove for pure-physics superheat once stable.
        gamma_3b=0.05,              # NEW: three-body term strengthening BCC over stripes
    )

    # --- Device selection (fail fast unless REQUIRE_CUDA=0) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Device=cuda ({gpu_name}, sm_{cap[0]}{cap[1]}, {total_mem_gb:.1f} GB)")
        print(f"PyTorch={torch.__version__}, CUDA runtime={torch.version.cuda}")
    else:
        if REQUIRE_CUDA:
            raise RuntimeError(
                "CUDA is not available, but REQUIRE_CUDA=1.\n"
                f"  torch={torch.__version__} (note: a '+cpu' build will NEVER see the GPU).\n"
                "  Install a CUDA-enabled wheel that supports your GPU's compute capability, e.g.:\n"
                "    pip install --index-url https://download.pytorch.org/whl/cu128 torch\n"
                "  Or set REQUIRE_CUDA=0 to allow CPU fallback."
            )
        device = torch.device("cpu")
        print(f"Device=cpu  (REQUIRE_CUDA=0; running on CPU — this will be slow)")

    dtype = TORCH_DTYPE
    print(f"dtype={dtype}")
    print(f"Grid nx,ny,nz = {p.nx},{p.ny},{p.nz}  dx={p.dx}")
    print(f"BCC stability: T_max={p.T_max}, gamma_3b={p.gamma_3b}, tau_psi={p.tau_psi}, power={p.power}")

    psi, grain_data = seed_substrate_layer(p, device, dtype)
    T = torch.full((p.nx, p.ny, p.nz), p.T0, device=device, dtype=dtype)

    np.savetxt("VTK/grain_data.csv", grain_data, delimiter=",",
               header="x_center,y_center,z_center,euler_a,euler_b,euler_c", comments="")
    np.save("VTK/grain_data.npy", grain_data)

    source_coeff = 0.0
    sink = 0.0
    source_term = torch.zeros_like(T)

    elapsed = 0.0
    x_laser_center = p.x_laser_center

    nsteps = 2001
    vtk_every = 10
    n_picard = 3
    # Print cadence — each print calls .item() 4x, forcing a CPU sync that
    # stalls the GPU pipeline. Keep this >= 1; 10 matches vtk_every cheaply.
    print_every = 10

    t_start = time.perf_counter()

    x_min = 0.5 * p.dx
    x_max = (p.nx - 0.5) * p.dx

    for step in range(nsteps):
        if (step % vtk_every) == 0:
            vtk_filename = os.path.join("VTK", f"fields_{step:06d}.vtk")
            write_vtk_structured_points(vtk_filename, p, {"psi": psi, "T": T})

        T_bc: Dict[str, Tuple[str, Optional[float]]] = {}

        if elapsed > 15.0:
            sink = 0.1               # weak sink during laser pass (let T build to superheat)
            source_coeff = 1.0

            x_laser_center = x_laser_center + p.v_x * p.dt

            if x_laser_center < x_min or x_laser_center > x_max:
                source_coeff = 0.0
                source_term = torch.zeros_like(T)
                sink = 3.0               # CHANGED: stronger post-laser sink (was 2.0)
            else:
                source_term = gaussian_source_3d(
                    p, x_laser_center, p.y_laser_center, p.z_laser_center, device, dtype
                )
        else:
            source_term = torch.zeros_like(T)
            source_coeff = 0.0
            sink = 2.0

        psi, T = advance_one_step(
            psi, T,
            source_term=source_term,
            source_coeff=source_coeff,
            sink=sink,
            T_bc=T_bc,
            p=p,
            n_picard=n_picard
        )

        elapsed += p.dt

        if (step % print_every) == 0:
            print(
                f"step={step:7d} elapsed={elapsed:.4f} "
                f"psi[min,max]=({psi.min().item():.4f},{psi.max().item():.4f}) "
                f"T[min,max]=({T.min().item():.4f},{T.max().item():.4f}) "
                f"source_coeff={source_coeff:.2f} sink={sink:.3f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    print(f"Done. wall={wall:.2f}s ({wall / max(nsteps, 1) * 1e3:.2f} ms/step)")


if __name__ == "__main__":
    main()