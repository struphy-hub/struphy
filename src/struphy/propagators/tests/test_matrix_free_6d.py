"""The matrix-free particle matrices (MarkerOperator) of the 6D coupling propagators give the same steps as the
assembled ones (Accumulator)."""

import logging

import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


def _solver_params():
    from struphy.linear_algebra.solver import SolverParameters

    return SolverParameters(tol=1e-14, maxiter=3000)


def _steps(prop, get_state, n: int = 2, dt: float = 0.1):
    """Changes of the arrays ``get_state()`` in ``n`` steps of the propagator ``prop``."""
    state0 = [a.copy() for a in get_state()]
    for _ in range(n):
        prop(dt)
    return [a - a0 for a, a0 in zip(get_state(), state0)]


def _compare(run):
    """Runs ``run(matrix_free)`` for both variants and compares the returned changes, which must not vanish."""
    import numpy as np

    res_mf = run(True)
    res_as = run(False)
    for a, b in zip(res_mf, res_as):
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a - b) < 1e-10 * np.linalg.norm(b)


def test_vlasov_ampere_coupling_matrix_free():
    """VlasovAmpereCoupling: e-field and velocities after two steps (in a curved domain)."""
    from struphy import DerhamOptions, LoadingParameters, Simulation, domains, grids, maxwellians, perturbations
    from struphy.models import VlasovAmpereOneSpecies

    def run(matrix_free: bool):
        model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
        model.kinetic_ions.set_markers(loading_params=LoadingParameters(Np=1000, seed=1234))
        model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D())
        model.em_fields.e_field.add_perturbation(
            perturbations.ModesCos(ls=(1,), ms=(1,), amps=(1e-2,), given_in_basis="1", comp=0)
        )
        prop = model.propagators.coupling_va
        prop.options = prop.Options(matrix_free=matrix_free, solver_params=_solver_params())
        Simulation(
            model=model,
            domain=domains.Colella(alpha=0.1),
            grid=grids.TensorProductGrid(num_elements=(6, 4, 2)),
            derham_opts=DerhamOptions(degree=(2, 2, 1)),
        ).allocate()
        particles = model.kinetic_ions.var.particles
        return _steps(
            prop,
            lambda: (model.em_fields.e_field.spline.vector.toarray(), particles.markers[3:6, particles.valid_mks]),
        )

    _compare(run)


def test_efield_weights_coupling_matrix_free():
    """EfieldWeightsCoupling: e-field and weights after two steps (in a curved domain)."""
    from struphy import DerhamOptions, LoadingParameters, Simulation, domains, grids, maxwellians, perturbations
    from struphy.models import LinearVlasovAmpereOneSpecies

    def run(matrix_free: bool):
        model = LinearVlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False, with_E0=False)
        model.kinetic_ions.set_markers(loading_params=LoadingParameters(Np=1000, seed=1234))
        # delta-f: the initial e-field is the Poisson solution of the density perturbation
        model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
        perturbation = perturbations.ModesCos(ls=(1,), ms=(1,), amps=(1e-2,))
        model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))
        prop = model.propagators.coupling_Eweights
        prop.options = prop.Options(matrix_free=matrix_free, solver_params=_solver_params())
        Simulation(
            model=model,
            domain=domains.Colella(alpha=0.1),
            grid=grids.TensorProductGrid(num_elements=(6, 4, 2)),
            derham_opts=DerhamOptions(degree=(2, 2, 1)),
        ).allocate()
        particles = model.kinetic_ions.var.particles
        return _steps(
            prop, lambda: (model.em_fields.e_field.spline.vector.toarray(), particles.markers[6, particles.valid_mks])
        )

    _compare(run)


def _linear_mhd_vlasov_sim(model, u_space: str):
    """Allocates a Simulation of LinearMHDVlasovCC/PC in a curved domain, with perturbed u and b."""
    from struphy import DerhamOptions, LoadingParameters, Simulation, domains, equils, grids, maxwellians, perturbations

    model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=1000, seed=1234))
    model.energetic_ions.var.add_background(maxwellians.Maxwellian3D())
    basis = {"Hcurl": "1", "Hdiv": "2", "H1vec": "v"}[u_space]
    model.mhd.velocity.add_perturbation(
        perturbations.ModesCos(ls=(1,), ms=(1,), amps=(1e-2,), given_in_basis=basis, comp=1)
    )
    model.em_fields.b_field.add_perturbation(
        perturbations.ModesSin(ls=(1,), ns=(1,), amps=(1e-2,), given_in_basis="2", comp=0)
    )
    Simulation(
        model=model,
        domain=domains.Colella(alpha=0.1),
        equil=equils.HomogenSlab(B0x=0.3, B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=(6, 4, 2)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    ).allocate()


def _set_u_space(model, u_space: str):
    """The model fixes u in Hdiv; the other spaces test the remaining kernel branches."""
    model.mhd.velocity._space = u_space
    props = model.propagators
    for prop in (props.couple_dens, props.shear_alf, props.couple_curr, props.mag_sonic):
        prop.options = prop.Options(u_space=u_space)


@pytest.mark.parametrize("u_space", ["Hdiv", "Hcurl", "H1vec"])
def test_current_coupling_6d_density_matrix_free(u_space: str):
    """CurrentCoupling6DDensity: u after two steps."""
    from struphy.models import LinearMHDVlasovCC

    def run(matrix_free: bool):
        model = LinearMHDVlasovCC()
        _set_u_space(model, u_space)
        prop = model.propagators.couple_dens
        prop.options = prop.Options(u_space=u_space, matrix_free=matrix_free, solver_params=_solver_params())
        _linear_mhd_vlasov_sim(model, u_space)
        return _steps(prop, lambda: (model.mhd.velocity.spline.vector.toarray(),))

    _compare(run)


@pytest.mark.parametrize("u_space", ["Hdiv", "Hcurl", "H1vec"])
def test_current_coupling_6d_current_matrix_free(u_space: str):
    """CurrentCoupling6DCurrent: u and velocities after two steps."""
    from struphy.models import LinearMHDVlasovCC

    def run(matrix_free: bool):
        model = LinearMHDVlasovCC()
        _set_u_space(model, u_space)
        prop = model.propagators.couple_curr
        prop.options = prop.Options(u_space=u_space, matrix_free=matrix_free, solver_params=_solver_params())
        _linear_mhd_vlasov_sim(model, u_space)
        particles = model.energetic_ions.var.particles
        return _steps(
            prop, lambda: (model.mhd.velocity.spline.vector.toarray(), particles.markers[3:6, particles.valid_mks])
        )

    _compare(run)


@pytest.mark.parametrize("use_perp_model", [True, False])
def test_pressure_coupling_6d_matrix_free(use_perp_model: bool):
    """PressureCoupling6D: u and velocities after two steps."""
    from struphy.models import LinearMHDVlasovPC

    def run(matrix_free: bool):
        model = LinearMHDVlasovPC(turn_off=("PushEtaPC", "PushVxB", "ShearAlfven", "Magnetosonic"))
        prop = model.propagators.pc6d
        prop.options = prop.Options(
            use_perp_model=use_perp_model, matrix_free=matrix_free, solver_params=_solver_params()
        )
        _linear_mhd_vlasov_sim(model, "Hdiv")
        particles = model.energetic_ions.var.particles
        return _steps(
            prop, lambda: (model.mhd.velocity.spline.vector.toarray(), particles.markers[3:6, particles.valid_mks])
        )

    _compare(run)


@pytest.mark.parametrize("space_id", ["Hcurl", "Hdiv", "H1vec"])
@pytest.mark.parametrize("kind", ["symm", "asym", "rank1"])
def test_marker_operator_dot(space_id: str, kind: str):
    """The generic dot kernels of MarkerOperator applied to random coefficients equal the assembled particle matrix
    with the same (random) fillings, and the transposes are consistent."""
    import numpy as np

    from struphy import DerhamOptions, LoadingParameters, Simulation, domains, grids, maxwellians
    from struphy.models import VlasovAmpereOneSpecies
    from struphy.pic.accumulation.kernels.marker_operator_asym_dot import marker_operator_asym_dot
    from struphy.pic.accumulation.kernels.marker_operator_rank1_dot import marker_operator_rank1_dot
    from struphy.pic.accumulation.kernels.marker_operator_symm_dot import marker_operator_symm_dot
    from struphy.pic.accumulation.particles_to_grid import MarkerOperator

    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
    model.kinetic_ions.set_markers(loading_params=LoadingParameters(Np=200, seed=1234))
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D())
    Simulation(
        model=model,
        domain=domains.Colella(alpha=0.1),
        grid=grids.TensorProductGrid(num_elements=(4, 4, 2)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    ).allocate()
    prop = model.propagators.coupling_va
    particles = model.kinetic_ions.var.particles
    derham, mass_ops, args_domain = prop.derham, prop.mass_ops, prop.domain.args_domain
    basis_u = {"Hcurl": 1, "Hdiv": 2, "H1vec": 0}[space_id]
    column = particles.args_markers.first_free_idx

    # random fillings F_p in the marker columns
    rng = np.random.default_rng(0)
    n_cols = {"symm": 6, "asym": 3, "rank1": 4}[kind]
    valid = particles.valid_mks
    particles.markers[column : column + n_cols, valid] = rng.standard_normal((n_cols, int(np.sum(valid))))
    kernel = {"symm": marker_operator_symm_dot, "asym": marker_operator_asym_dot, "rank1": marker_operator_rank1_dot}
    symmetry = "asym" if kind == "asym" else "symm"
    op = MarkerOperator(particles, space_id, kernel[kind], mass_ops, args_domain, (basis_u, column), symmetry=symmetry)

    # reference: the matrix of the basis functions at the markers, (A s)_i = sum_p Lambda_i(eta_p) F_p S_h(eta_p)
    field = derham.create_spline_function("s", space_id)
    eta = particles.markers[:3, valid]

    def apply_ref(x):
        field.vector = x
        s_h = np.array(field(eta, local=True))
        f = particles.markers[column : column + n_cols, valid]
        if kind == "symm":
            mat = np.array([[f[0], f[1], f[2]], [f[1], f[3], f[4]], [f[2], f[4], f[5]]])
        elif kind == "asym":
            z = np.zeros_like(f[0])
            mat = np.array([[z, f[0], f[1]], [-f[0], z, f[2]], [-f[1], -f[2], z]])
        else:
            mat = f[3] * f[:3, None, :] * f[None, :3, :]
        return np.einsum("ijp,jp->ip", mat, s_h)

    def random_vector():
        v = op.domain.zeros()
        for blk in v.blocks:
            blk._data[:] = rng.standard_normal(blk._data.shape)
        v.update_ghost_regions()
        return v

    x, y = random_vector(), random_vector()
    ax = op.dot(x)
    # (A x, y) = sum_p (F_p S_x(eta_p)) . S_y(eta_p)
    field.vector = y
    s_y = np.array(field(eta, local=True))
    ref = float(np.sum(apply_ref(x) * s_y))
    if derham.comm is not None and derham.comm.Get_size() > 1:
        from maybempi import MPI

        ref = derham.comm.allreduce(ref, op=MPI.SUM)
    assert abs(ax.inner(y) - ref) < 1e-11 * abs(ref)
    # transpose
    sign = -1.0 if kind == "asym" else 1.0
    assert abs(ax.inner(y) - sign * x.inner(op.dot(y))) < 1e-11 * abs(ref)
