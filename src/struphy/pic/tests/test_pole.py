"""Markers on domains with a pole at eta1=0 (no hole around the axis).

Run the MPI tests with::

    mpirun -n 4 pytest -p no:xdist --with-mpi src/struphy/pic/tests/test_pole.py
"""

import numpy as np
import pytest
from maybempi import MPI

from struphy.pic.pushing.pusher_utilities_kernels import apply_kinetic_bc_marker

# column layout of the hand-made marker arrays below (as MarkerArguments indices)
N_COLS = 25
FIRST_PUSHER_IDX = 8
FIRST_SHIFT_IDX = 14
RESIDUAL_IDX = 17
FIRST_FREE_IDX = 18


def _marker_args(markers, vdim, bc_type, pole):
    from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments

    n_mks = markers.shape[0]
    return MarkerArguments(
        markers,
        np.ones(n_mks, dtype=bool),
        n_mks,
        vdim,
        3 + vdim,
        7,
        FIRST_PUSHER_IDX,
        FIRST_SHIFT_IDX,
        RESIDUAL_IDX,
        FIRST_FREE_IDX,
        4,
        np.array(bc_type, dtype=int),
        pole,
    )


@pytest.mark.mpi_skip
def test_bc_kernel_axis_map():
    """A marker with eta1 < 0 is mapped through the axis; eta1 is kept off the pole."""
    from struphy import domains

    domain = domains.HollowCylinder(a1=0.0, a2=1.0, Lz=4.0)

    markers = np.zeros((3, N_COLS), order="F")
    markers[:, -1] = np.arange(3)
    markers[0, :6] = (-0.2, 0.3, 0.5, 0.1, 0.2, 0.3)
    markers[1, :6] = (-0.1, 0.7, 0.5, 0.1, 0.2, 0.3)
    markers[2, :6] = (0.0, 0.7, 0.5, 0.1, 0.2, 0.3)
    args_markers = _marker_args(markers, 3, (2, 0, 0), True)

    for ip in range(3):
        apply_kinetic_bc_marker(ip, args_markers, domain.args_domain, False)

    assert np.allclose(markers[0, :3], (0.2, 0.8, 0.5))
    assert np.allclose(markers[1, :3], (0.1, 0.2, 0.5))
    assert markers[2, 0] == 1e-10
    # Cartesian velocities are unchanged, the marker is not removed
    assert np.all(markers[:, 3:6] == (0.1, 0.2, 0.3))

    # without a pole, the same marker is removed
    markers[0, :6] = (-0.2, 0.3, 0.5, 0.1, 0.2, 0.3)
    apply_kinetic_bc_marker(0, _marker_args(markers, 3, (2, 0, 0), False), domain.args_domain, False)
    assert np.all(markers[0, :-1] == -1.0)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("vdim", [2, 3])
def test_bc_kernel_reflect_vdim(vdim):
    """Reflection mirrors the position; only Cartesian velocities (vdim=3) are reflected."""
    from struphy import domains

    domain = domains.HollowCylinder(a1=0.2, a2=1.0, Lz=4.0)

    markers = np.zeros((1, N_COLS), order="F")
    markers[0, :6] = (1.1, 0.3, 0.5, 0.4, 0.2, 0.7)
    apply_kinetic_bc_marker(0, _marker_args(markers, vdim, (1, 0, 0), False), domain.args_domain, False)

    assert np.isclose(markers[0, 0], 0.9)
    assert markers[0, FIRST_PUSHER_IDX] == -1.0
    if vdim == 3:
        # the radial velocity is reversed, the speed is kept (orthogonal mapping)
        assert not np.allclose(markers[0, 3:6], (0.4, 0.2, 0.7))
        assert np.isclose(np.linalg.norm(markers[0, 3:6]), np.linalg.norm((0.4, 0.2, 0.7)))
    else:
        # v_parallel, mu and the weight are unchanged
        assert np.all(markers[0, 3:6] == (0.4, 0.2, 0.7))


def _particles_on_pole_domain(comm, domain, ppc=10, num_elements=(8, 9, 4), bufsize=0.25):
    from struphy import BoundaryParameters, LoadingParameters
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=(2, 2, 1)), comm=comm)
    domain_decomp = (derham.domain_array, derham.domain_decomposition.nprocs)

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=LoadingParameters(ppc=ppc, seed=1234, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
        boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
        bufsize=bufsize,
        domain=domain,
    )
    particles.draw_markers()
    particles.mpi_sort_markers()
    return particles


def _by_unique_id(comm, ids, values):
    """Dictionary marker ID -> values over all ranks, without IDs that occur more than once
    (pseudo-random loading can give duplicate IDs at rank boundaries)."""
    all_ids = np.concatenate(comm.allgather(ids))
    all_values = np.concatenate(comm.allgather(values))
    unique, counts = np.unique(all_ids, return_counts=True)
    keep = set(unique[counts == 1])
    return {i: v for i, v in zip(all_ids, all_values) if i in keep}


@pytest.mark.mpi(min_size=1)
def test_python_bc_axis_map():
    """Particles.apply_kinetic_bc maps markers with eta1 < 0 through the axis."""
    from struphy import domains

    comm = MPI.COMM_WORLD
    domain = domains.HollowCylinder(a1=0.0, a2=1.0, Lz=4.0)
    particles = _particles_on_pole_domain(comm, domain)

    assert particles.args_markers.pole
    assert np.all(particles.positions[:, 0] >= 1e-10)

    valid = np.nonzero(particles.valid_mks)[0][:4]
    markers = particles.markers
    markers[valid[:3], 0] = -0.25
    markers[valid[:3], 1] = 0.6
    # outside at eta1=1: removed (boundary condition of axis 0), not mapped through the axis
    markers[valid[3], 0] = 1.05
    n_valid = np.count_nonzero(particles.valid_mks)

    particles.apply_kinetic_bc()

    assert np.allclose(markers[valid[:3], 0], 0.25)
    assert np.allclose(markers[valid[:3], 1], 0.1)
    assert np.all(markers[valid[3], :-1] == -1.0)
    assert particles.n_lost_markers == 1
    assert np.count_nonzero(particles.valid_mks) == n_valid - 1


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("algo", ["forward_euler", "heun2", "rk4"])
def test_push_eta_through_pole(algo):
    """Straight lines through the axis of a cylinder without hole are exact, since the mapping is linear in xi."""
    from struphy import domains
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage
    from struphy.pic.pushing.pusher import Pusher

    comm = MPI.COMM_WORLD
    a2 = 1.0
    domain = domains.HollowCylinder(a1=0.0, a2=a2, Lz=4.0)
    # all markers cross the axis, i.e. most of them change the process
    particles = _particles_on_pole_domain(comm, domain, bufsize=2.0)
    markers = particles.markers

    # aim all markers at (a point next to) the axis, such that they cross it and end inside the disc
    dt, n_steps = 0.1, 10
    T = dt * n_steps
    valid = particles.valid_mks
    x0, y0, z0 = domain(markers[valid, :3].copy())
    markers[valid, 3] = (-1.8 * x0 + 0.01) / T
    markers[valid, 4] = (-1.8 * y0 - 0.02) / T
    markers[valid, 5] = 0.3

    # expected positions per marker ID (markers change rank while being pushed)
    expected = np.stack((x0 + markers[valid, 3] * T, y0 + markers[valid, 4] * T, z0 + markers[valid, 5] * T), axis=1)
    expected_by_id = _by_unique_id(comm, markers[valid, -1], expected)
    n_valid_0 = comm.allreduce(np.count_nonzero(valid))

    butcher = ButcherTableau(algo=algo)
    pusher = Pusher(
        particles,
        push_eta_stage,
        (butcher.a_stage, butcher.b, butcher.c, butcher.n_stages),
        domain.args_domain,
        alpha_in_kernel=1.0,
        n_stages=butcher.n_stages,
        pushes_eta=True,
        local_eval_only=True,
    )
    for _ in range(n_steps):
        pusher(dt)

    valid = particles.valid_mks
    assert comm.allreduce(np.count_nonzero(valid)) == n_valid_0
    assert particles.n_lost_markers == 0

    ids = markers[valid, -1]
    compare = np.array([i in expected_by_id for i in ids], dtype=bool)
    assert comm.allreduce(np.count_nonzero(compare)) > 0.9 * n_valid_0
    x, y, z = domain(markers[valid, :3][compare].copy())
    x_expected, y_expected, z_expected = np.array([expected_by_id[i] for i in ids[compare]]).T
    assert np.allclose(x, x_expected, atol=1e-10)
    assert np.allclose(y, y_expected, atol=1e-10)
    # z is periodic with period Lz
    assert np.allclose(np.mod(z - z_expected + 2.0, 4.0) - 2.0, 0.0, atol=1e-10)


# grid of the simulation tests: the number of eta2 elements must be a multiple of 3 for polar splines,
# and the IGAPolar* mappings must have the same number of elements and degree in (eta1, eta2)
NUM_ELEMENTS = (8, 9, 4)
DEGREE = (3, 3, 1)


def _run_pole_simulation(tmp_folder, model, domain, equil, background, bc, dt, n_steps, comm):
    from struphy import (
        BoundaryParameters,
        DerhamOptions,
        EnvironmentOptions,
        LoadingParameters,
        Simulation,
        Time,
        grids,
    )

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=tmp_folder, sim_folder="sim_pole"),
        time_opts=Time(dt=dt, Tend=dt * n_steps),
        domain=domain,
        equil=equil,
        grid=grids.TensorProductGrid(num_elements=NUM_ELEMENTS),
        derham_opts=DerhamOptions(degree=DEGREE, bcs=(("free", "free"), None, None), polar_splines=True),
        comm=comm,
    )
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=4, seed=4321),
        boundary_params=BoundaryParameters(bc=bc),
        # few markers per process: the discrete-gradient pushers (sorting at mid-points in each iteration)
        # need a larger buffer, also on domains without a pole
        bufsize=3.0,
    )
    model.kinetic_ions.var.add_background(background)
    return sim


def _read_scalars(sim, names):
    import h5py

    with h5py.File(sim.data.file_path, "r") as file:
        return {name: file["scalar/" + name][()] for name in names}


def _inner_density(*etas):
    # called with (eta1, eta2, eta3) or with one (n, 3) array of positions
    eta1 = etas[0] if len(etas) == 3 else etas[0][:, 0]
    return np.where(eta1 < 0.6, 1.0, 0.0)


@pytest.mark.mpi(min_size=1)
def test_vlasov_through_pole(tmp_path_factory):
    """Vlasov (PushVxB with polar splines and PushEta through the axis) conserves the kinetic energy."""
    from struphy import domains, equils, maxwellians
    from struphy.models import Vlasov

    comm = MPI.COMM_WORLD
    tmp_folder = comm.bcast(str(tmp_path_factory.mktemp("vlasov_pole")) if comm.Get_rank() == 0 else None)

    R0 = 5.0
    model = Vlasov()
    sim = _run_pole_simulation(
        tmp_folder,
        model,
        domains.IGAPolarCylinder(num_elements=NUM_ELEMENTS[:2], degree=DEGREE[:2], a=1.0, Lz=2 * np.pi * R0),
        equils.ScrewPinch(a=1.0, R0=R0, B0=1.0),
        # only markers that cannot reach the outer wall carry weight (reflection on a spline mapping changes |v|)
        maxwellians.Maxwellian3D(n=(_inner_density, None), vth1=(0.1, None), vth2=(0.1, None), vth3=(0.1, None)),
        ("reflect", "periodic", "periodic"),
        dt=0.05,
        n_steps=10,
        comm=comm,
    )
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    sim.run()

    assert model.kinetic_ions.var.particles.n_lost_markers == 0
    if comm.Get_rank() == 0:
        en = _read_scalars(sim, ["kinetic_energy"])["kinetic_energy"]
        assert np.all(np.isfinite(en))
        assert np.allclose(en, en[0], rtol=1e-12)


def _gc_energies(sim, particles):
    """Marker IDs, positions and guiding-center energies v_par^2/2 + mu |B0| of the valid markers,
    with the projected (spline) |B0| that the pushers use."""
    particles.save_magnetic_background_energy()
    markers = particles.markers
    valid = particles.valid_mks
    eta = markers[valid, :3].copy()
    energy = 0.5 * markers[valid, 3] ** 2 + markers[valid, particles.first_diagnostics_idx]
    return markers[valid, -1].copy(), eta, energy


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("algo", ["explicit", "discrete_gradient_1st_order"])
def test_guiding_center_through_pole(tmp_path_factory, algo):
    """GuidingCenter on a torus without hole, polar splines: markers drift through the axis.

    The marker energies are conserved (explicit pushers in the regular coordinates xi);
    the discrete-gradient pushers only map markers through the axis and are less accurate near it."""
    from struphy import domains, equils, maxwellians
    from struphy.models import GuidingCenter

    comm = MPI.COMM_WORLD
    tmp_folder = comm.bcast(str(tmp_path_factory.mktemp(f"gc_pole_{algo}")) if comm.Get_rank() == 0 else None)

    # large epsilon: large drifts, such that markers cross the axis
    model = GuidingCenter(epsilon=1.0)
    sim = _run_pole_simulation(
        tmp_folder,
        model,
        domains.IGAPolarTorus(num_elements=NUM_ELEMENTS[:2], degree=DEGREE[:2], a=1.0, R0=3.0, tor_period=1),
        equils.AdhocTorus(a=1.0, R0=3.0, B0=1.0),
        maxwellians.GyroMaxwellian2D(n=(1.0, None), vth_para=(0.5, None), vth_perp=(0.5, None)),
        ("remove", "periodic", "periodic"),
        dt=0.05,
        n_steps=10,
        comm=comm,
    )
    model.propagators.push_bxe.options = model.propagators.push_bxe.Options(algo=algo, evaluate_e_field=True)
    model.propagators.push_parallel.options = model.propagators.push_parallel.Options(algo=algo, evaluate_e_field=True)
    sim.allocate()

    particles = model.kinetic_ions.var.particles
    ids, eta0, energy0 = _gc_energies(sim, particles)
    initial = _by_unique_id(comm, ids, np.concatenate((eta0, energy0[:, None]), axis=1))

    dt = sim.time_opts.dt
    for _ in range(10):
        model.propagators.push_bxe(dt)
        model.propagators.push_parallel(dt)

    ids, eta, energy = _gc_energies(sim, particles)
    assert np.all(np.isfinite(eta)) and np.all(np.isfinite(energy))
    assert np.all(eta[:, 0] > 0.0) and np.all(eta[:, 0] <= 1.0)

    compare = np.array([i in initial for i in ids], dtype=bool)
    start = np.array([initial[i] for i in ids[compare]]).reshape(-1, 4)
    rel_err = np.abs(energy[compare] - start[:, 3]) / start[:, 3]

    # markers that went through the axis region: close to the axis and eta2 changed by about 1/2
    near_axis = (start[:, 0] < 0.1) & (eta[compare, 0] < 0.1)
    jump = np.abs(np.mod(eta[compare, 1] - start[:, 1] + 0.5, 1.0) - 0.5) > 0.25
    crossed = near_axis & jump
    assert comm.allreduce(np.count_nonzero(crossed)) > 0

    max_err = comm.allreduce(np.max(rel_err, initial=0.0), op=MPI.MAX)
    max_err_axis = comm.allreduce(np.max(rel_err[near_axis], initial=0.0), op=MPI.MAX)
    if algo == "explicit":
        assert max_err < 1e-3
        assert max_err_axis < 1e-4
    else:
        # the discrete-gradient schemes evaluate at mid-points in eta, which are not regular at the axis
        assert max_err < 1e-1
