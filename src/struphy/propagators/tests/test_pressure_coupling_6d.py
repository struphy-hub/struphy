import cunumpy as xp
import pytest
from feectools.linalg.basic import IdentityOperator

from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.propagators.pressure_coupling_6d import PressureCoupling6D
from struphy.topology.grids import TensorProductGrid


def test_gt_mat_g_transpose():
    """GT_MAT_G.transpose() must return a GT_MAT_G with flipped transposed flag (symmetric MAT)."""
    derham = Derham(TensorProductGrid(num_elements=[4, 3, 2]), DerhamOptions(degree=[2, 1, 1]))
    Id = IdentityOperator(derham.V1)
    MAT = [[Id, Id, Id], [Id, Id, Id], [Id, Id, Id]]

    op = PressureCoupling6D.GT_MAT_G(derham, MAT)
    opT = op.transpose()

    assert not op.transposed
    assert isinstance(opT, PressureCoupling6D.GT_MAT_G)
    assert opT.transposed
    assert not opT.transpose().transposed

    v = derham.Vv.zeros()
    for vi in v:
        vi._data[:] = xp.random.rand(*vi._data.shape)
    assert xp.allclose(op.dot(v).copy().toarray(), opT.dot(v).toarray())


def _basis_ops(domain, equil, num_elements=(6, 4, 3), degree=(2, 2, 2)):
    from struphy.feec.basis_projection_ops import BasisProjectionOperators

    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree))
    equil.domain = domain
    return derham, BasisProjectionOperators(derham, domain, eq_mhd=equil)


@pytest.mark.parametrize("space_id", ["Hcurl", "Hdiv", "H1vec"])
def test_pperp_slab(space_id: str):
    """For B0 along z in a Cuboid, Pperp drops the third component exactly (the former Cartesian-xy model)."""
    from struphy import domains, equils

    derham, basis_ops = _basis_ops(domains.Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=3.0, l3=0.0, r3=4.0), equils.HomogenSlab())
    space = {"Hcurl": derham.V1, "Hdiv": derham.V2, "H1vec": derham.Vv}[space_id]
    u = space.zeros()
    for ui in u:
        ui._data[:] = xp.random.rand(*ui._data.shape)
    u.update_ghost_regions()

    pu = basis_ops.Pperp(space_id).dot(u)
    for i in range(3):
        expected = u[i].toarray() if i < 2 else 0.0 * u[i].toarray()
        assert xp.allclose(pu[i].toarray(), expected, atol=1e-10)

    # Xperp: the Cartesian components of X1/X2/Xv without the z-component
    X = getattr(basis_ops, {"Hcurl": "X1", "Hdiv": "X2", "H1vec": "Xv"}[space_id])
    xu = X.dot(u)
    xpu = basis_ops.Xperp(space_id).dot(u)
    for i in range(3):
        expected = xu[i].toarray() if i < 2 else 0.0 * xu[i].toarray()
        assert xp.allclose(xpu[i].toarray(), expected, atol=1e-10)


@pytest.mark.parametrize("space_id", ["Hcurl", "Hdiv", "H1vec"])
def test_pperp_curved_tilted(space_id: str):
    """In a curved domain with B0 tilted against the coordinate lines, Pperp u (transport in PushEtaPC) and Xperp u
    (force in PressureCoupling6D) are the projection of u perpendicular to b0, up to the projection error, which
    converges with the resolution. This is the consistency the energy conservation of the model needs; the energy test
    below checks PressureCoupling6D alone."""
    from struphy import domains, equils, perturbations

    form = {"Hcurl": "1", "Hdiv": "2", "H1vec": "v"}[space_id]
    e = xp.linspace(0.05, 0.95, 7)
    ee1, ee2, ee3 = xp.meshgrid(e, e, e, indexing="ij")

    def errors(num_elements):
        """Max. Cartesian error of Pperp u and max. parallel component b0 . (Pperp u), relative to max |u|."""
        domain = domains.Colella(alpha=0.1)
        equil = equils.HomogenSlab(B0x=0.3, B0z=1.0)
        derham, basis_ops = _basis_ops(domain, equil, num_elements=num_elements, degree=(3, 3, 2))
        pert = [perturbations.ModesCos(ls=(1,), ms=(1,), amps=(1.0,), given_in_basis=form, comp=c) for c in range(3)]
        u = derham.create_spline_function("u", space_id, perturbations=pert, domain=domain)
        u_perp = derham.create_spline_function("u_perp", space_id, coeffs=basis_ops.Pperp(space_id).dot(u.vector))
        # Cartesian components of U_perp as used by PressureCoupling6D (0-forms)
        x_perp = derham.create_spline_function("x_perp", "H1vec", coeffs=basis_ops.Xperp(space_id).dot(u.vector))

        cart = lambda f: xp.array(domain.push(f(e, e, e), e, e, e, kind=form))
        U, U_perp = cart(u), cart(u_perp)
        b = equil.unit_b_cart(ee1, ee2, ee3)[0]
        exact = U - xp.sum(b * U, axis=0) * b
        scale = xp.max(xp.abs(U))
        # the transport (PushEtaPC, Pperp) and the force (PressureCoupling6D, Xperp) must use the same U_perp,
        # up to the projection error
        X_perp = xp.array(x_perp(e, e, e))
        err = max(xp.max(xp.abs(U_perp - exact)), xp.max(xp.abs(X_perp - exact)))
        return err / scale, xp.max(xp.abs(xp.sum(b * U_perp, axis=0))) / scale

    err_coarse, par_coarse = errors((12, 12, 4))
    err_fine, par_fine = errors((24, 24, 4))
    # observed (degree 3): err 7e-3 -> 2e-4 (Hcurl), 3e-2 -> 4e-3 (Hdiv), 4e-3 -> 1e-4 (H1vec)
    assert err_fine < 5e-3 and par_fine < 1e-3
    assert err_coarse / err_fine > 4.0 and par_coarse / par_fine > 4.0


@pytest.mark.parametrize("use_perp_model", [True, False])
def test_pressure_coupling_6d_energy(use_perp_model: bool):
    """In a curved domain with tilted B0, a step of PressureCoupling6D conserves the fluid kinetic energy plus the
    particle kinetic energy up to the solver tolerance, while exchanging energy between them."""
    import numpy as np
    from maybempi import MPI

    from struphy.linear_algebra.solver import SolverParameters
    from struphy.models import LinearMHDVlasovPC
    from struphy.propagators.tests.test_matrix_free_6d import _linear_mhd_vlasov_sim

    model = LinearMHDVlasovPC()
    prop = model.propagators.pc6d
    prop.options = prop.Options(use_perp_model=use_perp_model, solver_params=SolverParameters(tol=1e-14, maxiter=3000))
    _linear_mhd_vlasov_sim(model, "Hdiv")
    particles = model.energetic_ions.var.particles
    u = model.mhd.velocity.spline.vector
    comm = MPI.COMM_WORLD

    def energies():
        e_u = 0.5 * u.inner(prop.mass_ops.M2n.dot(u))
        e_f = np.array([0.5 * np.sum(particles.weights * np.sum(particles.velocities**2, axis=0))])
        if comm.Get_size() > 1:
            comm.Allreduce(MPI.IN_PLACE, e_f, op=MPI.SUM)
        return e_u, e_f[0]

    e_u0, e_f0 = energies()
    for _ in range(3):
        prop(0.1)
    e_u1, e_f1 = energies()

    exchanged = abs(e_u1 - e_u0)
    assert exchanged > 1e-6 * e_u0
    assert abs((e_u1 + e_f1) - (e_u0 + e_f0)) < 1e-8 * exchanged


if __name__ == "__main__":
    test_gt_mat_g_transpose()
