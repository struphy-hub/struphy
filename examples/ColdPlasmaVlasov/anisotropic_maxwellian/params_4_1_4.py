
"""
Thesis Figure 4.1--4.4 reproduction: historical Standard FEM + PIC Run 1.

This is intentionally NOT a Struphy Simulation params file. The thesis
figures 4.1--4.4 were generated with the historical Standard FEM code,
not the current geometric ColdPlasmaVlasov implementation.

Exact thesis Run-1 physical/numerical parameters:
    c = 1, q_e = -1, m_e = 1, B0z = 1
    Omega_ce = -1, Omega_pe = 2
    nu_h = 0.06, w_parallel = 0.2, w_perp = 0.53
    k = 2, L = pi
    N_el = 32, p = 1, N_p = 1e5
    dt = 0.0125, T_end = 200
    Bx(z,0) = 1e-4 sin(2 z)
    all other wave/current perturbations = 0
    control variate = ON

The historical code did not fix a NumPy RNG seed. For reproducibility of
this wrapper, RNG_SEED is fixed. This makes future reruns deterministic,
but the particle realization will not be literally the same as Florian's
historical run unless its original RNG state/particle file is recovered.
"""

from pathlib import Path
import time
from copy import deepcopy

import numpy as np
from scipy.linalg import block_diag

# Exact historical utility module is restored from git history by the
# accompanying setup command/script.
from Utilitis_HybridCode import (
    borisPush,
    createBasis,
    fieldInterpolation,
    hotCurrent,
    IC,
    L2proj,
    matrixAssembly,
    solveDispersionHybrid,
)

# ---------------------------
# Reproduction parameters
# ---------------------------
EPS0 = 1.0
MU0 = 1.0
C = 1.0
QE = -1.0
ME = 1.0
B0Z = 1.0

WCE = QE * B0Z / ME
WPE = 2.0 * abs(WCE)

NUH = 0.06
NH = NUH * WPE**2
WPAR = 0.2 * C
WPERP = 0.53 * C

K = 2.0
AMP = 1.0e-4
EPS_DIST = 0.0

LZ = 2.0 * np.pi / K
NEL = 32
TEND = 200.0
DT = 0.0125
P = 1
NP = 100_000

# Historical Standard FEM uses the auxiliary velocity-domain metadata below.
LV = 8.0
NV = 76

CONTROL_VARIATE = 1
SAVING_STEP = 1

# Reproducibility choice for a new run. The historical code itself did not
# explicitly set this seed.
RNG_SEED = 1234

OUT = Path("thesis_run1_standard_cv")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Initial conditions
# ---------------------------
def Bx0(z):
    return AMP * np.sin(K * z)

def By0(z):
    return 0.0 * z

def Ex0(z):
    return 0.0 * z

def Ey0(z):
    return 0.0 * z

def jx0(z):
    return 0.0 * z

def jy0(z):
    return 0.0 * z

def fh0(z, vx, vy, vz):
    return (
        (1.0 + EPS_DIST * np.cos(K * z))
        * NH
        / ((2.0 * np.pi) ** 1.5 * WPAR * WPERP**2)
        * np.exp(
            -vz**2 / (2.0 * WPAR**2)
            - (vx**2 + vy**2) / (2.0 * WPERP**2)
        )
    )

def Maxwell(vx, vy, vz):
    return (
        NH
        / ((2.0 * np.pi) ** 1.5 * WPAR * WPERP**2)
        * np.exp(
            -vz**2 / (2.0 * WPAR**2)
            - (vx**2 + vy**2) / (2.0 * WPERP**2)
        )
    )

def g_sampling(vx, vy, vz):
    return (
        1.0
        / ((2.0 * np.pi) ** 1.5 * WPAR * WPERP**2)
        * np.exp(
            -vz**2 / (2.0 * WPAR**2)
            - (vx**2 + vy**2) / (2.0 * WPERP**2)
        )
        / LZ
    )

def B_background_z(z):
    return B0Z * (1.0 + 0.0 * (z - LZ / 2.0) ** 2)

# ---------------------------
# Historical 6x6 field system
# ---------------------------
A1 = np.array(
    [
        [0, 0, 0, +C**2, 0, 0],
        [0, 0, -C**2, 0, 0, 0],
        [0, -1, 0, 0, 0, 0],
        [+1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=float,
)

A2 = np.array(
    [
        [0, 0, 0, 0, MU0 * C**2, 0],
        [0, 0, 0, 0, 0, MU0 * C**2],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [-EPS0 * WPE**2, 0, 0, 0, 0, -WCE],
        [0, -EPS0 * WPE**2, 0, 0, +WCE, 0],
    ],
    dtype=float,
)
S = 6

# ---------------------------
# Main run
# ---------------------------
def run():
    np.random.seed(RNG_SEED)

    zj = np.linspace(0.0, LZ, NEL + 1)

    # Historical periodic B-spline basis.
    bsp, N, quad_points, quad_weights = createBasis(LZ, NEL, P)
    Nb = N - P

    # Global state.
    uj = np.zeros(S * Nb)
    Fh = np.zeros(S * Nb)
    u0 = np.zeros((Nb, S))

    A1block = block_diag(*([A1] * Nb))
    A2block = block_diag(*([A2] * Nb))

    t0 = time.perf_counter()
    M, Cmat = matrixAssembly(
        bsp, quad_weights, quad_points, B_background_z, 1
    )[0:2]
    print(f"matrix assembly: {time.perf_counter() - t0:.3f} s")

    for qu in range(S):
        def initial(z, qu=qu):
            return IC(z, 3, AMP, K, omega=0.0)[qu]
        u0[:, qu] = L2proj(
            bsp, LZ, quad_points, quad_weights, M, initial
        )

    uj = np.reshape(u0, S * Nb)

    # Historical block matrices and Crank-Nicolson update.
    Mblock = np.zeros((S * Nb, S * Nb))
    Cblock = np.zeros((S * Nb, S * Nb))
    for i in range(S):
        Mblock[i::S, i::S] = M
        Cblock[i::S, i::S] = Cmat

    LHS = (
        Mblock
        + 0.5 * DT * np.dot(Cblock, A1block)
        + 0.5 * DT * np.dot(Mblock, A2block)
    )
    RHS = (
        Mblock
        - 0.5 * DT * np.dot(Cblock, A1block)
        - 0.5 * DT * np.dot(Mblock, A2block)
    )
    LHSinv = np.linalg.inv(LHS)

    # Particles.
    particles = np.zeros((NP, 5))
    particles[:, 0] = np.random.rand(NP) * LZ
    particles[:, 1] = np.random.randn(NP) * WPERP
    particles[:, 2] = np.random.randn(NP) * WPERP
    particles[:, 3] = np.random.randn(NP) * WPAR

    g0 = g_sampling(
        particles[:, 1], particles[:, 2], particles[:, 3]
    )
    w0 = fh0(
        particles[:, 0],
        particles[:, 1],
        particles[:, 2],
        particles[:, 3],
    ) / g0

    # Initial fields at particle positions.
    Ep = np.zeros((NP, 3))
    Bp = np.zeros((NP, 3))
    Bp[:, 2] = B0Z
    Ep[:, 0:2], Bp[:, 0:2] = fieldInterpolation(
        particles[:, 0], zj, bsp, uj
    )

    # Historical half-step initialization.
    particles[:, 1:4] = borisPush(
        particles, -DT / 2.0, Bp, Ep, QE, ME, LZ
    )[1]
    particles[:, 4] = (
        w0
        - CONTROL_VARIATE
        * Maxwell(
            particles[:, 1],
            particles[:, 2],
            particles[:, 3],
        )
        / g0
    )

    # Diagnostics.
    Eh_eq = LZ * NH * ME / 2.0 * (WPAR**2 + 2.0 * WPERP**2)

    en_E = []
    en_B = []
    en_Bx = []
    en_jc = []
    en_jh = []

    # Save initial and final particle states, and 100 tracer particles.
    particles_initial = particles.copy()
    tracer_vpar = [particles[:100, 3].copy()]
    tracer_z = [particles[:100, 0].copy()]

    def diagnostics():
        e = EPS0 / 2.0 * (
            np.dot(uj[0::S], M @ uj[0::S])
            + np.dot(uj[1::S], M @ uj[1::S])
        )
        b = EPS0 / (2.0 * MU0) * (
            np.dot(uj[2::S], M @ uj[2::S])
            + np.dot(uj[3::S], M @ uj[3::S])
        )
        bx = EPS0 / (2.0 * MU0) * np.dot(
            uj[2::S], M @ uj[2::S]
        )
        jc = 1.0 / (2.0 * EPS0 * WPE**2) * (
            np.dot(uj[4::S], M @ uj[4::S])
            + np.dot(uj[5::S], M @ uj[5::S])
        )
        jh = ME / (2.0 * NP) * np.dot(
            particles[:, 4],
            particles[:, 1] ** 2
            + particles[:, 2] ** 2
            + particles[:, 3] ** 2,
        ) + CONTROL_VARIATE * Eh_eq
        return e, b, bx, jc, jh

    e, b, bx, jc, jh = diagnostics()
    en_E.append(e); en_B.append(b); en_Bx.append(bx)
    en_jc.append(jc); en_jh.append(jh)

    n_steps = int(round(TEND / DT))
    print(f"Starting historical Standard FEM Run 1: {n_steps} steps")

    start = time.perf_counter()

    for step in range(n_steps):
        zold = particles[:, 0].copy()

        znew, vnew = borisPush(
            particles, DT, Bp, Ep, QE, ME, LZ
        )

        wnew = (
            w0
            - CONTROL_VARIATE
            * Maxwell(vnew[:, 0], vnew[:, 1], vnew[:, 2])
            / g0
        )

        jhnew = hotCurrent(
            vnew[:, 0:2],
            0.5 * (znew + zold),
            wnew,
            zj,
            bsp,
            QE,
            C,
        )

        Fh[0::S] = -C**2 * MU0 * jhnew[0::2]
        Fh[1::S] = -C**2 * MU0 * jhnew[1::2]

        uj = LHSinv @ (RHS @ uj + DT * Fh)

        Ep[:, 0:2], Bp[:, 0:2] = fieldInterpolation(
            znew, zj, bsp, uj
        )

        particles[:, 0] = znew
        particles[:, 1:4] = vnew
        particles[:, 4] = wnew

        e, b, bx, jc, jh = diagnostics()
        en_E.append(e); en_B.append(b); en_Bx.append(bx)
        en_jc.append(jc); en_jh.append(jh)

        tracer_vpar.append(particles[:100, 3].copy())
        tracer_z.append(particles[:100, 0].copy())

        if step % 500 == 0 or step == n_steps - 1:
            print(
                f"step {step+1:5d}/{n_steps} "
                f"t={(step+1)*DT:9.4f} "
                f"elapsed={(time.perf_counter()-start)/60:.2f} min"
            )

    elapsed = time.perf_counter() - start
    print(f"Time integration: {elapsed/60:.2f} min")

    times = np.arange(n_steps + 1, dtype=float) * DT

    np.savez_compressed(
        OUT / "run1_standard_cv.npz",
        time=times,
        en_E=np.asarray(en_E),
        en_B=np.asarray(en_B),
        en_Bx=np.asarray(en_Bx),
        en_jc=np.asarray(en_jc),
        en_jh=np.asarray(en_jh),
        particles_initial=particles_initial,
        particles_final=particles,
        tracer_vpar=np.asarray(tracer_vpar),
        tracer_z=np.asarray(tracer_z),
        parameters=np.array(
            [
                EPS0, MU0, C, QE, ME, B0Z, WCE, WPE,
                NUH, NH, WPAR, WPERP, K, LZ, NEL,
                TEND, DT, P, NP, CONTROL_VARIATE, RNG_SEED,
            ],
            dtype=float,
        ),
    )

    with open(OUT / "run_metadata.txt", "w") as f:
        f.write("Historical Standard FEM + PIC thesis Run 1\n")
        f.write("control_variate=1\n")
        f.write(f"rng_seed={RNG_SEED}\n")
        f.write(f"n_steps={n_steps}\n")
        f.write(f"runtime_seconds={elapsed:.6f}\n")
        f.write("Physical parameters: nu_h=0.06, wpar=0.2, wperp=0.53, "
                "Omega_pe=2, k=2, L=pi\n")
        f.write("Numerical parameters: Nel=32, p=1, Np=100000, "
                "dt=0.0125, Tend=200\n")

if __name__ == "__main__":
    run()
