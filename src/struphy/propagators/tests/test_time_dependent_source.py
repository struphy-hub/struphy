import cunumpy as xp
import pytest
from maybempi import MPI

from struphy import domains
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.time_dependent_source import TimeDependentSource
from struphy.topology.grids import TensorProductGrid


@pytest.mark.parametrize("space", ["H1", "Hcurl", "Hdiv", "L2", "H1vec"])
@pytest.mark.parametrize("hfun", ["cos", "sin"])
def test_time_dependent_source(space: str, hfun: str):
    """The source coefficients are the initial ones times h(omega t), in every FEEC space."""
    comm = MPI.COMM_WORLD
    domain = domains.Cuboid()
    grid = TensorProductGrid(num_elements=(6, 5, 4))
    derham = Derham(grid=grid, options=DerhamOptions(degree=(2, 2, 1)), comm=comm)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = WeightedMassOperators(derham=derham, domain=domain)

    source = FEECVariable(space=space)
    source.allocate(derham=derham, domain=domain)
    c0 = source.spline.vector
    _set_from_array(c0, xp.arange(c0.toarray().size, dtype=float) + 1.0)
    # owned part of the initial coefficients (zero elsewhere)
    c0_arr = c0.toarray().copy()

    omega = 3.0
    prop = TimeDependentSource()
    prop.variables.source = source
    prop.options = prop.Options(omega=omega, hfun=hfun)
    prop.add_time_state(xp.array([0.0]))
    prop.allocate()

    h = xp.cos if hfun == "cos" else xp.sin
    for t in (0.0, 0.3, 1.7):
        prop.time_state[0] = t
        prop(0.1)
        assert xp.allclose(source.spline.vector.toarray(), c0_arr * h(omega * t))


def _set_from_array(v, arr):
    """Write the global array ``arr`` into the (stencil or block) vector ``v``."""
    from struphy.linear_algebra.multigrid.preconditioner import _scatter

    _scatter(arr, v)
    v.update_ghost_regions()
