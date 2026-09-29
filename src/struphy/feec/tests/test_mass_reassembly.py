"""Regression for reassembling mass operators with an evolving spline weight."""

import numpy as np
from feectools.ddm.mpi import mpi as MPI

from struphy import DerhamOptions, domains, grids
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham


def test_density_weighted_mass_reassembly():
    # Non-unit Jacobian and non-unit density expose accumulation of the spline
    # factor into the cached geometric weights on subsequent assemblies.
    domain = domains.Cuboid(r1=2.0, r2=3.0)
    derham = Derham(
        grids.TensorProductGrid(num_elements=(4, 4, 1)),
        DerhamOptions(degree=(2, 2, 1)),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    masses = WeightedMassOperators(derham, domain)
    weighted = masses.WMMnew
    rho = weighted.spline_functions["l2_field"]
    rho.vector = derham.projectors["3"](lambda e1, e2, e3: 2.0 + 0 * e1)
    probe = derham.projectors["v"](
        [
            lambda e1, e2, e3: 1.0 + 0 * e1,
            lambda e1, e2, e3: 2.0 + 0 * e1,
            lambda e1, e2, e3: 3.0 + 0 * e1,
        ]
    )
    reference = masses.Mv.dot(probe).toarray() * (2.0 / 6.0)
    weighted.assemble()
    np.testing.assert_allclose(weighted.dot(probe).toarray(), reference, rtol=1e-12, atol=1e-12)
    weighted.assemble()
    np.testing.assert_allclose(weighted.dot(probe).toarray(), reference, rtol=1e-12, atol=1e-12)
    rho.vector *= 1.5
    weighted.assemble()
    np.testing.assert_allclose(weighted.dot(probe).toarray(), 1.5 * reference, rtol=1e-12, atol=1e-12)
    weighted.assemble()
    np.testing.assert_allclose(weighted.dot(probe).toarray(), 1.5 * reference, rtol=1e-12, atol=1e-12)
