import logging
import os
import shutil

import cunumpy as xp
import pytest
from maybempi import MPI

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
    perturbations,
)
from struphy.models import VlasovMaxwellOneSpecies

logger = logging.getLogger("struphy")


@pytest.mark.clones
@pytest.mark.parametrize(
    "control_variate, Tend, rel_tol",
    [(True, 0.0, 0.1), (False, 0.0, 0.1), (False, 1.0, 0.5)],
)
def test_gauss_error(control_variate: bool, Tend: float, rel_tol: float, nclones: int, exit_before_run: bool = False):
    """The ``gauss_error`` diagnostic: right after the initial Poisson solve the weak Gauss law holds, and for
    full-f markers the residual stays at the level of the time stepping.

    ``post_allocate`` solves :math:`\\mathbb G^\\top \\mathbb M^1 \\mathbb G \\boldsymbol \\phi = \\frac{\\alpha^2}{\\varepsilon} \\boldsymbol \\rho`
    and sets :math:`\\mathbf e = -\\mathbb G \\boldsymbol \\phi`, so the ``gauss_error`` scalar
    :math:`\\max |\\mathbb G^\\top \\mathbb M^1 \\mathbf e + \\frac{\\alpha^2}{\\varepsilon} \\boldsymbol \\rho|` must be small
    against :math:`\\max |\\mathbb G^\\top \\mathbb M^1 \\mathbf e|` (what is left is the Monte-Carlo net charge,
    which the periodic Poisson problem cannot represent), independently of the number of MPI ranks.

    The cases with ``Tend = 0`` run zero time steps, with and without the control variate. For full-f markers the
    initial solve uses the delta-f weights, and the diagnostic deposits the full-f weights minus the cached offset
    of :math:`t=0`; the case with ``Tend = 1`` (20 steps) checks that the residual then stays of the order of the
    Gauss-law violation of the time stepping (about 0.1 of the field term here), instead of the noise of the
    :math:`f_0` deposit (about 3 times the field term without the offset).
    """
    # alpha^2/epsilon != 1 to check the coefficient of the charge density
    model = VlasovMaxwellOneSpecies(alpha=2.0, epsilon=0.5, measure_gauss_law=True)

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "VlasovMaxwellOneSpecies")
    env = EnvironmentOptions(
        out_folders=out_folders, sim_folder=f"gauss_error_{control_variate}_{Tend}", num_clones=nclones
    )

    time_opts = Time(dt=0.05, Tend=Tend)

    domain = domains.Cuboid(r1=12.56)
    grid = grids.TensorProductGrid(num_elements=(16, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=1000, seed=1234),
        weights_params=WeightsParameters(control_variate=control_variate),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(8, 1, 1), do_sort=True),
        saving_params=SavingParameters(),
        bufsize=0.4,
    )

    model.propagators.maxwell.options = model.propagators.maxwell.Options()
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options()

    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    perturbation = perturbations.ModesCos(ls=(1,), amps=(1e-2,))
    model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    if exit_before_run:
        logger.info("Exiting before running simulation.")
        return sim

    sim.run()
    assert (sim.clone_config.num_clones if sim.clone_config else 1) == nclones

    gauss_error = model.scalars.dct["gauss_error"]
    gauss_error.update()
    error = float(gauss_error.value[0])

    lhs = model.op.dot(model.em_fields.e_field.spline.vector)
    lhs_max = xp.array([xp.max(xp.abs(lhs.toarray()))])
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        comm.Allreduce(MPI.IN_PLACE, lhs_max, op=MPI.MAX)

    assert error < rel_tol * lhs_max[0], (
        f"Assertion for the initial Gauss-law residual failed: {error =} vs. max|G^T M1 e| = {lhs_max[0]}."
    )
    logger.info(f"Assertion for the initial Gauss-law residual passed ({error =}, {lhs_max[0] =}).")

    # all ranks are done with this case before the shared folder is removed
    comm.Barrier()
    if comm.Get_rank() == 0:
        shutil.rmtree(test_folder)
    comm.Barrier()


if __name__ == "__main__":
    test_gauss_error(True, 0.0, 0.1, nclones=1)
    test_gauss_error(False, 1.0, 0.5, nclones=1)
