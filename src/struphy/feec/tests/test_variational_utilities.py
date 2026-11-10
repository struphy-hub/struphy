import pytest


@pytest.mark.parametrize(
    "name",
    [
        "evaluate_discrete_de_drho_grid",
        "evaluate_discrete_de_ds_grid",
        "evaluate_discrete_d2e_drho2_grid",
        "evaluate_discrete_d2e_ds2_grid",
    ],
)
def test_internal_energy_evaluator_out(name):
    """The discrete (second) derivatives of InternalEnergyEvaluator return their result, with and without out."""

    import cunumpy as xp
    from maybempi import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.variational_utilities import InternalEnergyEvaluator
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(
        grid=TensorProductGrid(num_elements=[4, 3, 2]),
        options=DerhamOptions(degree=[2, 2, 1]),
        comm=MPI.COMM_WORLD,
    )
    evaluator = InternalEnergyEvaluator(derham, 5 / 3)

    # the three arguments are 3-forms: (rho, rho1, s) or (rho, s, s1)
    args = []
    for n, val in enumerate([1.0, 1.1, 0.3]):
        f = derham.create_spline_function(f"f{n}", "L2")
        f.vector[:] = val
        f.vector.update_ghost_regions()
        args += [f.vector]

    func = getattr(evaluator, name)

    out = xp.zeros_like(evaluator._tmp_int_grid)
    res = func(*args, out=out)
    assert res is out
    assert xp.all(xp.isfinite(out))

    res_none = func(*args)
    assert res_none is not None
    assert xp.allclose(res_none, out)


if __name__ == "__main__":
    test_internal_energy_evaluator_out("evaluate_discrete_d2e_ds2_grid")


@pytest.mark.parametrize("u_space", ["Hdiv", "H1vec"])
@pytest.mark.parametrize("mapping", ["Cuboid", "HollowCylinder", "HollowTorus"])
def test_transport_adjoint_matrix(mapping: str, u_space: str):
    """The assembled matrix of b -> T[b]^T l (TransportAdjointMatrix) equals the matrix-free transposed transport
    operator Hdiv0_transport_operator for random b and l, with Dirichlet conditions in eta1 (also under MPI, where
    the projection cells near the domain decomposition cuts contribute to the rows of the neighbouring rank)."""

    import numpy as np
    from feectools.linalg.utilities import array_to_psydac
    from maybempi import MPI

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.variational_utilities import Hdiv0_transport_operator, TransportAdjointMatrix
    from struphy.geometry import domains
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    if mapping == "Cuboid":
        domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=2.0, l3=0.0, r3=3.0)
    elif mapping == "HollowCylinder":
        domain = domains.HollowCylinder(a1=0.1, a2=1.0, Lz=5.0)
    else:
        domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=3.0, sfl=False, pol_period=1, tor_period=1)
    derham = Derham(
        TensorProductGrid(num_elements=[8, 6, 4]),
        DerhamOptions(degree=[3, 3, 3], bcs=(("dirichlet", "dirichlet"), None, None)),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    mass_ops = WeightedMassOperators(derham, domain)
    u_form = derham.space_to_form[u_space]

    def random_vector(space, seed, bop=None):
        v = space.zeros()
        n = len(v.toarray())
        array_to_psydac(np.random.default_rng(seed).standard_normal(n), v.space).copy(out=v)
        if bop is not None:
            bop.dot(v).copy(out=v)
        v.update_ghost_regions()
        return v

    lf = random_vector(derham.V2pol, 1)
    b = random_vector(derham.V2pol, 2, derham.boundary_ops["2"])

    F = TransportAdjointMatrix(derham, domain, mass_ops, u_space, lf)
    TwT = Hdiv0_transport_operator(derham, u_space=u_space, phys_domain=domain).T
    TwT.update_coeffs(b)
    ref = derham.boundary_ops[u_form].dot(TwT.dot(lf))
    diff = F.dot(b) - ref
    rel = np.sqrt(diff.inner(diff) / ref.inner(ref))
    assert rel < 1e-12, rel


@pytest.mark.parametrize("u_space", ["Hdiv", "H1vec"])
@pytest.mark.parametrize("mapping", ["Cuboid", "HollowCylinder", "HollowTorus"])
def test_pressure_transport_adjoint_matrix(mapping: str, u_space: str):
    """The assembled matrix of p -> S[p]^T l (PressureTransportAdjointMatrix) equals the matrix-free transposed
    Pressure_transport_operator for random p and l, with Dirichlet conditions in eta1 (also under MPI)."""

    import numpy as np
    from feectools.linalg.utilities import array_to_psydac
    from maybempi import MPI

    from struphy.feec.basis_projection_ops import BasisProjectionOperators
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.variational_utilities import Pressure_transport_operator, PressureTransportAdjointMatrix
    from struphy.geometry import domains
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    if mapping == "Cuboid":
        domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=2.0, l3=0.0, r3=3.0)
    elif mapping == "HollowCylinder":
        domain = domains.HollowCylinder(a1=0.1, a2=1.0, Lz=5.0)
    else:
        domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=3.0, sfl=False, pol_period=1, tor_period=1)
    derham = Derham(
        TensorProductGrid(num_elements=[8, 6, 4]),
        DerhamOptions(degree=[3, 3, 3], bcs=(("dirichlet", "dirichlet"), None, None)),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    mass_ops = WeightedMassOperators(derham, domain)
    basis_ops = BasisProjectionOperators(derham, domain)
    u_form = derham.space_to_form[u_space]
    gamma = 5 / 3

    def random_vector(space, seed):
        v = space.zeros()
        array_to_psydac(np.random.default_rng(seed).standard_normal(len(v.toarray())), v.space).copy(out=v)
        v.update_ghost_regions()
        return v

    lf = random_vector(derham.V3pol, 1)
    p = random_vector(derham.V3pol, 2)

    A = PressureTransportAdjointMatrix(derham, domain, mass_ops, basis_ops, u_space, gamma, lf)
    ST = Pressure_transport_operator(derham, domain, basis_ops.Uv, gamma, u_space=u_space).T
    ST.update_coeffs(p)
    ref = derham.boundary_ops[u_form].dot(ST.dot(lf))
    diff = derham.boundary_ops[u_form].dot(A.dot(p)) - ref
    rel = np.sqrt(diff.inner(diff) / ref.inner(ref))
    assert rel < 1e-12, rel
