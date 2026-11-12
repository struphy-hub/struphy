"""Tests of the bracket of the variational momentum advection for a velocity in H(div)
(:class:`~struphy.feec.variational_utilities.HdivBracketOperator`)."""

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from struphy import domains
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.feec.variational_utilities import (
    HdivBracketMomentumDerivative,
    HdivBracketOperator,
)
from struphy.io.options import DerhamOptions
from struphy.topology.grids import TensorProductGrid

BCS = (("dirichlet", "dirichlet"), None, None)

MAPS = {
    "Cuboid": lambda: domains.Cuboid(),
    "Colella": lambda: domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0),
    # the geometry of the TAE test cases, where the velocity mass matrix in H1vec is ill-conditioned
    "HollowTorus": lambda: domains.HollowTorus(a1=0.1, a2=1.0, R0=5.0, sfl=False, pol_period=1, tor_period=1),
}


def _derham(domain, num_elements, degree, bcs=BCS):
    grid = TensorProductGrid(num_elements=num_elements)
    return Derham(grid, DerhamOptions(degree=degree, bcs=bcs), comm=MPI.COMM_WORLD, domain=domain)


def _random_vector(space, seed):
    rank = MPI.COMM_WORLD.Get_rank()
    rng = np.random.default_rng(seed + 100 * rank)
    v = space.zeros()
    for block in v.blocks if hasattr(v, "blocks") else [v]:
        block._data[:] = xp.asarray(rng.standard_normal(block._data.shape))
    v.update_ghost_regions()
    return v


@pytest.mark.parametrize("map_name", ["Cuboid", "Colella", "HollowTorus"])
@pytest.mark.parametrize("degree", [(2, 2, 2), (3, 2, 3)])
def test_hdiv_bracket_skew(map_name, degree):
    """The bracket is skew-symmetric for any momentum (conservation of kinetic energy), and its result satisfies the
    boundary conditions of the velocity."""
    domain = MAPS[map_name]()
    derham = _derham(domain, (6, 6, 4), degree)
    mass_ops = WeightedMassOperators(derham, domain)
    B = HdivBracketOperator(derham, domain, mass_ops)

    B.update_momentum(mass_ops.M2.dot(_random_vector(derham.V2pol, 7)))
    bc = derham.boundary_ops["2"]
    u = bc.dot(_random_vector(derham.V2pol, 3))
    v = bc.dot(_random_vector(derham.V2pol, 5))
    Bu = B.dot(u)
    Bv = B.dot(v)
    scale = (Bu.inner(Bu) * v.inner(v)) ** 0.5
    assert abs(v.inner(Bu) + u.inner(Bv)) / scale < 1e-12
    assert abs(u.inner(Bu)) / scale < 1e-12

    # B.T = -B, and the output has vanishing boundary coefficients
    assert B.T.dot(u).inner(v) == pytest.approx(-Bu.inner(v))
    Bu_bc = bc.dot(Bu)
    diff = Bu_bc - Bu
    assert diff.inner(diff) == 0.0


@pytest.mark.parametrize("map_name", ["Cuboid", "Colella"])
def test_hdiv_bracket_convergence(map_name):
    r"""v^T B u converges to int m . [u, v] dx for smooth fields (in logical coordinates:
    int (G u_m) . [u, v]_eta sqrt g d eta, with contravariant components u, v, u_m that vanish in direction 1 at the
    walls, 2-form coefficients sqrt g u and momentum m = M2 omega_m)."""
    from numpy import cos, pi, sin

    domain = MAPS[map_name]()

    def u_f(x, y, z):
        return [sin(pi * x) * cos(2 * pi * y), cos(pi * x) * sin(2 * pi * y + 0.3) + 0.5, cos(2 * pi * z) * x]

    def v_f(x, y, z):
        return [sin(2 * pi * x) * sin(2 * pi * z), x**2 + cos(2 * pi * y), sin(pi * x) * cos(2 * pi * y)]

    def um_f(x, y, z):
        return [sin(pi * x) * (1.0 + 0.5 * cos(2 * pi * y)), sin(2 * pi * y) * x + 1.0, 1.0 + 0 * x]

    def jac(fun, x, y, z, h=1e-6):
        steps = [(h, 0, 0), (0, h, 0), (0, 0, h)]
        return [
            [(fun(x + a, y + b, z + c)[i] - fun(x - a, y - b, z - c)[i]) / (2 * h) for a, b, c in steps]
            for i in range(3)
        ]

    xg, wg = np.polynomial.legendre.leggauss(24)
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    X, Y, Z = np.meshgrid(xg, xg, xg, indexing="ij")
    wts = np.einsum("i,j,k->ijk", wg, wg, wg)
    u, v, um = u_f(X, Y, Z), v_f(X, Y, Z), um_f(X, Y, Z)
    Ju, Jv = jac(u_f, X, Y, Z), jac(v_f, X, Y, Z)
    br = [sum(u[j] * Jv[i][j] - v[j] * Ju[i][j] for j in range(3)) for i in range(3)]
    G = domain.metric(X, Y, Z)
    sqrt_g = domain.jacobian_det(X, Y, Z)
    exact = np.sum(wts * sqrt_g * sum(G[i, j] * um[j] * br[i] for i in range(3) for j in range(3)))

    errors = []
    for n in (4, 8):
        derham = _derham(domain, (n, n, n), (2, 2, 2))
        mass_ops = WeightedMassOperators(derham, domain)
        B = HdivBracketOperator(derham, domain, mass_ops)

        def two_form(f):
            comps = [lambda e1, e2, e3, i=i: domain.jacobian_det(e1, e2, e3) * f(e1, e2, e3)[i] for i in range(3)]
            return derham.boundary_ops["2"].dot(derham.P2(comps))

        B.update_momentum(mass_ops.M2.dot(two_form(um_f)))
        val = two_form(v_f).inner(B.dot(two_form(u_f)))
        errors += [abs(val - exact) / abs(exact)]

    # the sign of the bracket: v^T B u = <m, [u, v]>
    assert errors[-1] < 1e-2, f"{errors = }"
    # convergence (about fourth order for p = 2)
    assert errors[0] / errors[1] > 8.0, f"{errors = }"


@pytest.mark.parametrize("map_name", ["Colella", "HollowTorus"])
def test_hdiv_bracket_matrix_free(map_name):
    """The assembled bracket (N, R) equals its matrix-free application, and the derivative with respect to the
    momentum is consistent with the bracket (bilinearity)."""
    domain = MAPS[map_name]()
    derham = _derham(domain, (6, 6, 4), (3, 2, 2))
    mass_ops = WeightedMassOperators(derham, domain)
    B = HdivBracketOperator(derham, domain, mass_ops)
    bc = derham.boundary_ops["2"]

    m = mass_ops.M2.dot(_random_vector(derham.V2pol, 11))
    B.update_momentum(m)
    u = bc.dot(_random_vector(derham.V2pol, 13))
    Bu = B.dot(u)

    shape = B.shape_q
    u_q = [xp.zeros(shape) for _ in range(3)]
    divu_q = xp.zeros(shape)
    B.velocity_values(u, u_q, divu_q)
    Bu_free = B.apply_values(u_q, divu_q, B._y_q, B._c_q)
    diff = Bu - Bu_free
    assert (diff.inner(diff) / Bu.inner(Bu)) ** 0.5 < 1e-12

    # D[u] m = B[y(m)] u
    D = HdivBracketMomentumDerivative(B)
    D.set_velocity(u)
    Dm = D.dot(m)
    diff = Dm - Bu
    assert (diff.inner(diff) / Bu.inner(Bu)) ** 0.5 < 1e-10


def test_hdiv_bracket_rejects_polar():
    """Polar splines are not supported."""
    domain = domains.IGAPolarCylinder(num_elements=(6, 9), degree=(2, 2), a=1.0, Lz=3.0)
    grid = TensorProductGrid(num_elements=(6, 9, 2))
    derham = Derham(
        grid,
        DerhamOptions(degree=(2, 2, 2), bcs=(("free", "dirichlet"), None, None), polar_splines=True),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    mass_ops = WeightedMassOperators(derham, domain)
    with pytest.raises(AssertionError):
        HdivBracketOperator(derham, domain, mass_ops)


if __name__ == "__main__":
    test_hdiv_bracket_skew("HollowTorus", (3, 2, 3))
    test_hdiv_bracket_convergence("Colella")
    test_hdiv_bracket_matrix_free("HollowTorus")
