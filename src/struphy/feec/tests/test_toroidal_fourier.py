import numpy as np
import pytest
from maybempi import MPI


def _setup(polar: bool, num_elements: tuple[int, int, int]):
    """Derham and mass operators of an (axisymmetric) screw pinch in a cylinder, with polar or tensor-product splines."""
    from struphy import domains, equils
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    degree = (2, 2, 2)
    if polar:
        domain = domains.IGAPolarCylinder(num_elements=num_elements[:2], degree=degree[:2], a=1.0, Lz=3.0)
    else:
        domain = domains.HollowCylinder(a1=0.1, a2=1.0, Lz=3.0)
    equil = equils.ScrewPinch(a=1.0, R0=3.0 / (2 * np.pi), B0=1.0)
    equil.domain = domain
    derham_opts = DerhamOptions(degree=degree, bcs=(("free", "dirichlet"), None, None), polar_splines=polar)
    derham = Derham(TensorProductGrid(num_elements=num_elements), derham_opts, comm=MPI.COMM_WORLD, domain=domain)
    return derham, WeightedMassOperators(derham, domain, eq_mhd=equil)


@pytest.mark.parametrize("polar", [False, True])
@pytest.mark.parametrize("num_elements", [(4, 6, 6), (8, 6, 2)])
def test_toroidal_fourier_solver(polar, num_elements):
    """For the axisymmetric screw pinch, the solver is the exact inverse (residual on a random vector) of symmetric
    (M2, M1 + G M0 G^T, Cholesky) and of non-symmetric operators (M2 + 0.3 M2B, LU), also when the grid is decomposed
    along eta_3 or eta_1 (the two grids on 2 processes)."""
    from struphy.feec.toroidal_fourier import ToroidalFourierSolver

    derham, mass_ops = _setup(polar, num_elements)
    cases = [
        ("M2", mass_ops.M2, True),
        ("M1 + G M0 G^T", mass_ops.M1 + derham.grad @ mass_ops.M0 @ derham.grad.T, True),
        ("M2 + 0.3 M2B", mass_ops.M2 + 0.3 * mass_ops.M2B, False),
    ]
    for name, A, symmetric in cases:
        solver = ToroidalFourierSolver(A)
        assert solver.residual < 1e-10, (name, solver.residual)
        assert solver.symmetric == symmetric, name
        if symmetric:
            assert solver.transpose() is solver
        else:
            with pytest.raises(NotImplementedError):
                solver.transpose()
