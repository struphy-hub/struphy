"""Verification of VlasovPoissonAdiabaticPauli: classical Pauli particles on the slow manifold follow the
guiding-center orbits of the GuidingCenter model in an ad hoc tokamak (banana and passing orbits), with
time steps larger than the cyclotron period (Xiao and Qin, Comput. Phys. Commun. 265, 107981 (2021)).

Run the MPI test with::

    mpirun -n 4 pytest -p no:xdist --with-mpi src/struphy/models/tests/verification/test_verif_VlasovPoissonAdiabaticPauli.py
"""

import numpy as np
import pytest
from maybempi import MPI

# normalized cyclotron period 2 pi epsilon = 0.063
EPSILON = 0.01
R0 = 3.0
NUM_ELEMENTS = (8, 16, 4)
DEGREE = (3, 3, 1)

# test markers: guiding-center position (eta1, eta2, eta3), parallel velocity, magnetic moment;
# deeply and barely trapped (banana orbits) and passing
MARKERS = np.array(
    [
        [0.5, 0.0, 0.0, 0.2, 0.5],
        [0.5, 0.0, 0.0, 0.4, 0.5],
        [0.4, 0.25, 0.0, 1.0, 0.1],
        [0.6, 0.5, 0.0, -0.8, 0.2],
    ]
)


def _simulation(tmp_folder, sim_folder, model, background, dt, comm):
    from struphy import (
        BoundaryParameters,
        DerhamOptions,
        EnvironmentOptions,
        LoadingParameters,
        Simulation,
        Time,
        domains,
        equils,
        grids,
    )

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=tmp_folder, sim_folder=sim_folder),
        time_opts=Time(dt=dt, Tend=dt),
        domain=domains.HollowTorus(a1=0.1, a2=1.0, R0=R0, tor_period=1),
        equil=equils.AdhocTorus(a=1.0, R0=R0, B0=1.0),
        grid=grids.TensorProductGrid(num_elements=NUM_ELEMENTS),
        derham_opts=DerhamOptions(degree=DEGREE, bcs=(("free", "free"), None, None)),
        comm=comm,
    )
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=1, seed=1234),
        boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
        bufsize=1.0,
    )
    model.kinetic_ions.var.add_background(background)
    sim.allocate()
    return sim


def _place_markers(particles, fill, comm):
    """Replace the loaded markers by the test markers (on rank 0, then sorted to their ranks), with marker IDs
    0, 1, ...; ``fill(markers, eta)`` sets the velocity columns of the test markers (the first columns of
    ``markers``) at the logical positions ``eta``."""
    markers = particles.markers
    markers[:, :] = -1.0
    if comm.Get_rank() == 0:
        n = MARKERS.shape[0]
        eta = MARKERS[:, :3].T.copy()
        markers[:, :n] = 0.0
        markers[:3, :n] = eta
        fill(markers, eta)
        markers[particles.index["weights"], :n] = 1.0
        markers[particles.index["ids"], :n] = np.arange(n)
    particles.update_holes()
    particles._update_ghost_particles()
    if comm.Get_size() > 1:
        particles.mpi_sort_markers()


def _gather_RZ(particles, domain, comm):
    """(R, Z) of the test markers, ordered by marker ID, on all ranks."""
    valid = particles.valid_mks
    ids = particles.markers[-1, valid].astype(int)
    x, y, z = domain(particles.markers[:3, valid].copy(), remove_outside=False)
    local = np.stack((ids, np.sqrt(x**2 + y**2), z), axis=1)
    rows = np.concatenate(comm.allgather(local)) if comm.Get_size() > 1 else local
    out = np.full((MARKERS.shape[0], 2), np.nan)
    out[rows[:, 0].astype(int)] = rows[:, 1:]
    return out


# orbits over about one bounce period of the trapped markers
T_END = 150.0
DT_GC = 0.05


def gc_orbits(tmp_folder, comm, t_end=T_END, dt_gc=DT_GC):
    """Orbits (R, Z) of the test markers at all time steps of the GuidingCenter model (explicit RK4 pushers)."""
    from struphy import maxwellians
    from struphy.models import GuidingCenter

    # guiding centers: v_par and mu as velocity columns
    gc = GuidingCenter(epsilon=EPSILON)
    gc.propagators.push_bxe.options = gc.propagators.push_bxe.Options(algo="explicit")
    gc.propagators.push_parallel.options = gc.propagators.push_parallel.Options(algo="explicit")
    sim = _simulation(tmp_folder, "gc", gc, maxwellians.GyroMaxwellian2D(), dt_gc, comm)
    particles = gc.kinetic_ions.var.particles

    def fill_gc(markers, eta):
        markers[3:5, : MARKERS.shape[0]] = MARKERS[:, 3:5].T

    _place_markers(particles, fill_gc, comm)

    orbits = [_gather_RZ(particles, sim.domain, comm)]
    for _ in range(int(round(t_end / dt_gc))):
        gc.propagators.push_bxe(dt_gc)
        gc.propagators.push_parallel(dt_gc)
        orbits += [_gather_RZ(particles, sim.domain, comm)]
    return np.array(orbits)


def pauli_orbits(tmp_folder, comm, dt_pauli, t_end=T_END):
    """Orbits (R, Z) of the test markers at all time steps of VlasovPoissonAdiabaticPauli (Boris scheme, phi = 0),
    the largest relative error of the energy v^2/2 + mu |B0| and the largest mu' / mu."""
    from struphy import maxwellians
    from struphy.models import VlasovPoissonAdiabaticPauli

    pauli = VlasovPoissonAdiabaticPauli(epsilon=EPSILON)
    pauli.propagators.push_vxb.options = pauli.propagators.push_vxb.Options(algo="implicit")
    sim = _simulation(tmp_folder, f"pauli_{dt_pauli}", pauli, maxwellians.Maxwellian3D(), dt_pauli, comm)
    particles = pauli.kinetic_ions.var.particles

    # classical Pauli particles at the guiding-center positions, v = v_par b0, magnetic moment in its column
    def fill_pauli(markers, eta):
        b, _ = particles._unit_b_and_absB(eta)
        markers[3:6, : MARKERS.shape[0]] = MARKERS[:, 3] * b
        markers[particles.mu_idx, : MARKERS.shape[0]] = MARKERS[:, 4]

    _place_markers(particles, fill_pauli, comm)

    def energies():
        _, absB = particles._unit_b_and_absB(particles.positions)
        en = 0.5 * np.sum(particles.velocities**2, axis=0) + particles.magnetic_moments * absB
        return particles.markers[-1, particles.valid_mks].astype(int), en

    # initial energies of all test markers (they move between processes)
    ids0, en0 = energies()
    pairs = list(zip(ids0, en0))
    if comm.Get_size() > 1:
        pairs = [pair for rank_pairs in comm.allgather(pairs) for pair in rank_pairs]
    en0 = dict(pairs)

    # no electric field: the Strang step of the model without the Poisson solve
    pauli.em_fields.phi.spline.vector *= 0.0
    props = pauli.propagators
    orbits = [_gather_RZ(particles, sim.domain, comm)]
    max_en_err = 0.0
    max_mu_prime = 0.0
    for _ in range(int(round(t_end / dt_pauli))):
        props.push_eta(dt_pauli / 2)
        props.push_v_E(dt_pauli / 2)
        props.push_vxb(dt_pauli)
        props.push_v_E(dt_pauli / 2)
        props.push_eta(dt_pauli / 2)
        orbits += [_gather_RZ(particles, sim.domain, comm)]

        ids, en = energies()
        if ids.size > 0:
            max_en_err = max(max_en_err, np.max([abs(e - en0[i]) / en0[i] for i, e in zip(ids, en)]))
            max_mu_prime = max(max_mu_prime, np.max(particles.mu_prime() / particles.magnetic_moments))

    if comm.Get_size() > 1:
        max_en_err = comm.allreduce(max_en_err, op=MPI.MAX)
        max_mu_prime = comm.allreduce(max_mu_prime, op=MPI.MAX)
    return np.array(orbits), max_en_err, max_mu_prime


def orbit_errors(orbits_gc, orbits_pauli):
    """Per test marker: the largest distance of the Pauli positions from the guiding-center positions at the same
    time and from the guiding-center orbit (curve), and the radial width of that orbit."""
    stride = (orbits_gc.shape[0] - 1) // (orbits_pauli.shape[0] - 1)
    dist_time = np.max(np.linalg.norm(orbits_pauli - orbits_gc[::stride], axis=2), axis=0)
    dist_curve = np.array(
        [
            np.max(np.min(np.linalg.norm(orbits_pauli[:, k, None] - orbits_gc[None, :, k], axis=2), axis=1))
            for k in range(orbits_gc.shape[1])
        ]
    )
    r_gc = np.linalg.norm(orbits_gc - np.array([R0, 0.0]), axis=2)
    width = np.max(r_gc, axis=0) - np.min(r_gc, axis=0)
    return dist_time, dist_curve, width


@pytest.mark.mpi(min_size=1)
def test_pauli_vs_guiding_center(tmp_path_factory):
    """Classical Pauli particles pushed with the Boris scheme with dt = 4 and 8 cyclotron periods follow the
    guiding-center orbits (explicit RK4 pushers of GuidingCenter, dt = 0.8 cyclotron periods) of trapped and passing
    particles, with second-order convergence in dt, conserve the energy v^2/2 + mu |B0| up to O(dt^2) and stay on the
    slow manifold (mu' << mu)."""
    comm = MPI.COMM_WORLD
    tmp_folder = comm.bcast(str(tmp_path_factory.mktemp("pauli_gc")) if comm.Get_rank() == 0 else None)

    orbits_gc = gc_orbits(tmp_folder, comm)
    assert np.all(np.isfinite(orbits_gc))

    errors = {}
    for dt in (0.5, 0.25):
        orbits, en_err, mu_prime = pauli_orbits(tmp_folder, comm, dt)
        assert np.all(np.isfinite(orbits))
        dist_time, dist_curve, width = orbit_errors(orbits_gc, orbits)
        errors[dt] = dist_curve
        if comm.Get_rank() == 0:
            print(f"{dt = }: {dist_time = }, {dist_curve = }, {width = }, {en_err = }, {mu_prime = }")

        # bounds of the second-order errors, about 1.5 times the measured ones
        # (dt = 0.5: relative distance from the orbit 0.009 - 0.43, at the same time 0.011 - 1.1)
        assert mu_prime < 1e-3
        assert en_err < 0.01 * dt**2
        assert np.all(dist_curve < 2.5 * dt**2 * width)
        assert np.all(dist_time < 6.0 * dt**2 * width)

    # second order in dt, for the passing markers (the errors of the trapped ones are at the level of the reference)
    assert np.all(errors[0.5][2:] > 3.0 * errors[0.25][2:])


if __name__ == "__main__":
    import sys
    import tempfile

    from matplotlib import pyplot as plt

    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    comm = MPI.COMM_WORLD
    with tempfile.TemporaryDirectory() as d:
        orbits_gc = gc_orbits(d, comm)
        orbits_pauli, en_err, mu_prime = pauli_orbits(d, comm, dt)
    if comm.Get_rank() == 0:
        dist_time, dist_curve, width = orbit_errors(orbits_gc, orbits_pauli)
        print(f"{dist_time = }\n{dist_curve = }\n{width = }\n{en_err = }\n{mu_prime = }")
        fig, ax = plt.subplots()
        for k in range(MARKERS.shape[0]):
            ax.plot(orbits_gc[:, k, 0], orbits_gc[:, k, 1], "-", label=f"guiding center {k}")
            ax.plot(orbits_pauli[:, k, 0], orbits_pauli[:, k, 1], ".", ms=2, label=f"Pauli {k}, dt = {dt}")
        ax.set(xlabel="R", ylabel="Z", aspect="equal")
        ax.legend(fontsize=6)
        fig.savefig("pauli_vs_gc.png", dpi=150)
