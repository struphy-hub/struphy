"""
Historical Standard FEM + PIC reproduction of thesis Figure 4.7.

Run 3:
    c=1, qe=-1, me=1, B0z=1
    Omega_ce=-1, Omega_pe=2
    nu_h=0.002, w_parallel=0.1, w_perp=0.1
    L=80, N_el=256, p=3, Np=50000
    dt=0.05, T_end=300
    initial EM perturbation = zero
    distribution perturbation = zero
    control variate = ON

The particle noise is therefore the seed of the electromagnetic spectrum.
"""

from pathlib import Path
from copy import deepcopy
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import Utilitis_HybridCode as utils


# ============================================================
# Physical parameters
# ============================================================
eps0 = 1.0
mu0 = 1.0
c = 1.0
qe = -1.0
me = 1.0
B0z = 1.0

wce = qe * B0z / me
wpe = 2.0 * abs(wce)

nuh = 0.002
nh = nuh * wpe**2

wpar = 0.1 * c
wperp = 0.1 * c


# ============================================================
# Historical Run-3 parameters
# ============================================================
k = 2.0
ini = 6          # all wave-field perturbations zero
amp = 1.0e-4     # irrelevant for ini=6
eps = 0.0

Lz = 80.0
Nz = 256
T = 300.0
dt = 0.05
p = 3
Np = 50000

control = 1
saving_step = 1

# Historical notebook did not explicitly fix an RNG seed.
# We fix one here solely for reproducibility of the modern rerun.
RNG_SEED = 1234

OUT = ROOT / "thesis_fig4_7_run3"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Model definitions
# ============================================================
def B_background_z(z):
    return B0z * (1.0 + 0.0 * (z - Lz / 2.0) ** 2)


def fh0(z, vx, vy, vz):
    return (
        (1.0 + eps * np.cos(k * z))
        * nh / ((2.0 * np.pi) ** 1.5 * wpar * wperp**2)
        * np.exp(
            -vz**2 / (2.0 * wpar**2)
            - (vx**2 + vy**2) / (2.0 * wperp**2)
        )
    )


def Maxwell(vx, vy, vz):
    return (
        nh / ((2.0 * np.pi) ** 1.5 * wpar * wperp**2)
        * np.exp(
            -vz**2 / (2.0 * wpar**2)
            - (vx**2 + vy**2) / (2.0 * wperp**2)
        )
    )


def g_sampling(vx, vy, vz):
    return (
        1.0 / ((2.0 * np.pi) ** 1.5 * wpar * wperp**2)
        * np.exp(
            -vz**2 / (2.0 * wpar**2)
            - (vx**2 + vy**2) / (2.0 * wperp**2)
        )
        / Lz
    )


# ============================================================
# Historical 6x6 FEM system
# ============================================================
A1 = np.array([
    [0, 0, 0, +c**2, 0, 0],
    [0, 0, -c**2, 0, 0, 0],
    [0, -1, 0, 0, 0, 0],
    [+1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=float)

A2 = np.array([
    [0, 0, 0, 0, mu0*c**2, 0],
    [0, 0, 0, 0, 0, mu0*c**2],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [-eps0*wpe**2, 0, 0, 0, 0, -wce],
    [0, -eps0*wpe**2, 0, 0, +wce, 0],
], dtype=float)

s = 6


# ============================================================
# Main simulation
# ============================================================
def run():

    np.random.seed(RNG_SEED)

    # Periodic spatial grid.
    zj = np.linspace(0.0, Lz, Nz + 1)

    # Historical periodic B-spline basis.
    bsp, N, quad_points, weights = utils.createBasis(
        Lz, Nz, p
    )

    Nb = N - p

    # FEM vectors.
    uj = np.zeros(s * Nb)
    Fh = np.zeros(s * Nb)
    u0 = np.zeros((Nb, s))

    # --------------------------------------------------------
    # FEM matrices
    # --------------------------------------------------------
    t0 = time.perf_counter()

    M, C = utils.matrixAssembly(
        bsp,
        weights,
        quad_points,
        B_background_z,
        1,
    )[0:2]

    print(
        f"Matrix assembly: "
        f"{time.perf_counter() - t0:.2f} s"
    )

    # --------------------------------------------------------
    # Initial field: ini=6 => exactly zero
    # --------------------------------------------------------
    for qu in range(s):

        def initial(z, qu=qu):
            return utils.IC(
                z,
                ini,
                amp,
                k,
                omega=0,
            )[qu]

        u0[:, qu] = utils.L2proj(
            bsp,
            Lz,
            quad_points,
            weights,
            M,
            initial,
        )

    uj = np.reshape(u0, s * Nb)

    # --------------------------------------------------------
    # Block FEM system
    # --------------------------------------------------------
    M_block = np.kron(M, np.identity(6))
    C_tilde = np.kron(C, A1)
    M_tilde = np.kron(M, A2)

    LHS = (
        M_block
        + 0.5 * dt * C_tilde
        + 0.5 * dt * M_tilde
    )

    RHS = (
        M_block
        - 0.5 * dt * C_tilde
        - 0.5 * dt * M_tilde
    )

    print("Computing field update inverse...")
    LHSinv = np.linalg.inv(LHS)

    # --------------------------------------------------------
    # Particles
    # --------------------------------------------------------
    particles = np.zeros((Np, 5))

    particles[:, 0] = np.random.rand(Np) * Lz
    particles[:, 1] = np.random.randn(Np) * wperp
    particles[:, 2] = np.random.randn(Np) * wperp
    particles[:, 3] = np.random.randn(Np) * wpar

    g0 = g_sampling(
        particles[:, 1],
        particles[:, 2],
        particles[:, 3],
    )

    w0 = (
        fh0(
            particles[:, 0],
            particles[:, 1],
            particles[:, 2],
            particles[:, 3],
        )
        / g0
    )

    # --------------------------------------------------------
    # Initial fields at particles
    # --------------------------------------------------------
    Ep = np.zeros((Np, 3))
    Bp = np.zeros((Np, 3))

    Bp[:, 2] = B0z

    Ep[:, 0:2], Bp[:, 0:2] = utils.fieldInterpolation(
        particles[:, 0],
        zj,
        bsp,
        uj,
    )

    # Historical half-step particle initialization.
    particles[:, 1:4] = utils.borisPush(
        particles,
        -dt / 2.0,
        Bp,
        Ep,
        qe,
        me,
        Lz,
    )[1]

    particles[:, 4] = (
        w0
        - control
        * Maxwell(
            particles[:, 1],
            particles[:, 2],
            particles[:, 3],
        )
        / g0
    )

    # ========================================================
    # IMPORTANT: save Bx exactly as historical notebook did
    #
    #   bx_save = append(uj[2::s], uj[2])
    #
    # The final duplicate point enforces the periodic grid
    # representation used by the historical FFT diagnostic.
    # ========================================================
    bx_save = np.empty((int(T / dt) + 1, Nz + 1))

    bx_save[0] = np.append(
        deepcopy(uj[2::s]),
        uj[2],
    )

    # Energy diagnostics retained for integrity checking.
    Eh_eq = (
        Lz * nh * me / 2.0
        * (wpar**2 + 2.0 * wperp**2)
    )

    en_E = np.empty(int(T / dt) + 1)
    en_B = np.empty(int(T / dt) + 1)
    en_jc = np.empty(int(T / dt) + 1)
    en_jh = np.empty(int(T / dt) + 1)

    en_E[0] = eps0 / 2.0 * (
        np.dot(uj[0::s], M @ uj[0::s])
        + np.dot(uj[1::s], M @ uj[1::s])
    )

    en_B[0] = eps0 / (2.0 * mu0) * (
        np.dot(uj[2::s], M @ uj[2::s])
        + np.dot(uj[3::s], M @ uj[3::s])
    )

    en_jc[0] = 1.0 / (2.0 * eps0 * wpe**2) * (
        np.dot(uj[4::s], M @ uj[4::s])
        + np.dot(uj[5::s], M @ uj[5::s])
    )

    en_jh[0] = (
        me / (2.0 * Np)
        * np.dot(
            particles[:, 4],
            particles[:, 1]**2
            + particles[:, 2]**2
            + particles[:, 3]**2,
        )
        + control * Eh_eq
    )

    # ========================================================
    # Time integration
    # ========================================================
    Nt = int(round(T / dt))

    print()
    print("==============================================")
    print("STARTING THESIS FIGURE 4.7 RUN 3")
    print("==============================================")
    print(f"Particles       : {Np}")
    print(f"Elements        : {Nz}")
    print(f"B-spline degree : {p}")
    print(f"dt              : {dt}")
    print(f"Tend            : {T}")
    print(f"Timesteps       : {Nt}")
    print(f"RNG seed        : {RNG_SEED}")
    print("Initial EM field: ZERO")
    print("Control variate : ON")
    print("==============================================")
    print()

    t_start = time.perf_counter()

    for time_step in range(Nt):

        zold = deepcopy(particles[:, 0])

        # Boris particle advance.
        znew, vnew = utils.borisPush(
            particles,
            dt,
            Bp,
            Ep,
            qe,
            me,
            Lz,
        )

        # Control-variate weights.
        wnew = (
            w0
            - control
            * Maxwell(
                vnew[:, 0],
                vnew[:, 1],
                vnew[:, 2],
            )
            / g0
        )

        # Hot-electron current.
        jhnew = utils.hotCurrent(
            vnew[:, 0:2],
            0.5 * (znew + zold),
            wnew,
            zj,
            bsp,
            qe,
            c,
        )

        # RHS source.
        Fh[0::s] = (
            -c**2 * mu0 * jhnew[0::2]
        )

        Fh[1::s] = (
            -c**2 * mu0 * jhnew[1::2]
        )

        # Crank-Nicolson field update.
        uj = LHSinv @ (
            RHS @ uj + dt * Fh
        )

        # Fields at particle locations.
        Ep[:, 0:2], Bp[:, 0:2] = (
            utils.fieldInterpolation(
                znew,
                zj,
                bsp,
                uj,
            )
        )

        particles[:, 0] = znew
        particles[:, 1:4] = vnew
        particles[:, 4] = wnew

        idx = time_step + 1

        # Historical Figure-4.7 diagnostic:
        # Bx evaluated on the periodic spatial grid.
        bx_save[idx] = np.append(
            deepcopy(uj[2::s]),
            uj[2],
        )

        en_E[idx] = eps0 / 2.0 * (
            np.dot(uj[0::s], M @ uj[0::s])
            + np.dot(uj[1::s], M @ uj[1::s])
        )

        en_B[idx] = eps0 / (2.0 * mu0) * (
            np.dot(uj[2::s], M @ uj[2::s])
            + np.dot(uj[3::s], M @ uj[3::s])
        )

        en_jc[idx] = 1.0 / (2.0 * eps0 * wpe**2) * (
            np.dot(uj[4::s], M @ uj[4::s])
            + np.dot(uj[5::s], M @ uj[5::s])
        )

        en_jh[idx] = (
            me / (2.0 * Np)
            * np.dot(
                particles[:, 4],
                particles[:, 1]**2
                + particles[:, 2]**2
                + particles[:, 3]**2,
            )
            + control * Eh_eq
        )

        if idx % 500 == 0 or idx == Nt:
            elapsed = (
                time.perf_counter() - t_start
            ) / 60.0

            print(
                f"step {idx:5d}/{Nt}  "
                f"t={idx*dt:8.2f}  "
                f"elapsed={elapsed:8.2f} min"
            )

    elapsed = time.perf_counter() - t_start

    # ========================================================
    # Save everything needed for Figure 4.7
    # ========================================================
    time_array = np.arange(Nt + 1) * dt

    np.savez_compressed(
        OUT / "run3_fig4_7.npz",
        time=time_array,
        z=zj,
        Bx=bx_save,
        en_E=en_E,
        en_B=en_B,
        en_jc=en_jc,
        en_jh=en_jh,
        parameters=np.array([
            eps0, mu0, c, qe, me, B0z,
            wce, wpe, nuh, nh,
            wpar, wperp, k, Lz,
            Nz, T, dt, p, Np,
            control, RNG_SEED,
        ]),
    )

    with open(OUT / "run_metadata.txt", "w") as f:
        f.write("THESIS FIGURE 4.7 / HISTORICAL RUN 3\n")
        f.write("Standard FEM + PIC\n")
        f.write(f"rng_seed={RNG_SEED}\n")
        f.write(f"L={Lz}\n")
        f.write(f"Nel={Nz}\n")
        f.write(f"p={p}\n")
        f.write(f"Np={Np}\n")
        f.write(f"dt={dt}\n")
        f.write(f"Tend={T}\n")
        f.write("nu_h=0.002\n")
        f.write("wpar=0.1\n")
        f.write("wperp=0.1\n")
        f.write("Omega_pe=2\n")
        f.write("Omega_ce=-1\n")
        f.write("initial_EM_perturbation=ZERO\n")
        f.write("distribution_perturbation=ZERO\n")
        f.write("control_variate=1\n")
        f.write(f"runtime_seconds={elapsed:.6f}\n")

    print()
    print("==============================================")
    print("RUN 3 COMPLETE")
    print("==============================================")
    print(f"Runtime: {elapsed/60.0:.2f} min")
    print(f"Saved:   {OUT / 'run3_fig4_7.npz'}")
    print(f"Bx shape: {bx_save.shape}")
    print("==============================================")


if __name__ == "__main__":
    run()
