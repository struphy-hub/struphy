import cunumpy as xp
import pytest

pytest.importorskip("petsc4py")

from feectools.ddm.mpi import mpi as MPI

from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.fields_background.equils import HomogenSlab
from struphy.geometry.domains import Cuboid
from struphy.io.options import DerhamOptions
from struphy.topology.grids import TensorProductGrid


@pytest.mark.parametrize("space_id", ["H1", "L2"])
def test_l2_projector_petsc_matches_pcg(space_id):
    """L2Projector(solver_name="petsc") must match L2Projector(solver_name="pcg") for a real mass matrix."""
    comm = MPI.COMM_WORLD

    domain = Cuboid()
    equil = HomogenSlab(n0=2.0)
    equil.domain = domain

    grid = TensorProductGrid(num_elements=[8, 8, 8])
    derham_opts = DerhamOptions(degree=[2, 2, 2])
    derham = Derham(grid, derham_opts, comm=comm, domain=domain)

    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=equil)

    def rhs(e1, e2, e3):
        return xp.sin(2 * xp.pi * e1) * xp.cos(2 * xp.pi * e2) * xp.cos(2 * xp.pi * e3)

    proj_pcg = L2Projector(space_id, mass_ops, solver_name="pcg")
    proj_petsc = L2Projector(space_id, mass_ops, solver_name="petsc")

    b = proj_pcg.get_dofs(rhs, apply_bc=True)

    x_pcg = proj_pcg.solve(b)
    x_petsc = proj_petsc.solve(b)

    assert xp.linalg.norm((x_petsc - x_pcg).toarray()) < 1e-6


if __name__ == "__main__":
    test_l2_projector_petsc_matches_pcg("H1")
