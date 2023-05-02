import numpy as np
from mpi4py import MPI

from struphy.propagators.base import Propagator
from struphy.psydac_api import preconditioner
from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import IdentityOperator as Identity
from struphy.polar.basic import PolarVector

from psydac.linalg.iterative_solvers import pcg
from psydac.linalg.stencil import StencilVector


class PoissonSolver(Propagator):
    r"""
    Solve the Poisson equation

    ..math:
        - \Delta \phi = \rho

    for the electric potential :math:`\phi\,\in H^1` given a charge-neutral right-hand side :math:`\rho\,\in H1`.

    Parameters
    ----------
    rho : psydac.linalg.block.StencilVector
        FE coefficients of a 0-form (optional, can be set via setter later).

    **solver_params : dict
        Parameters for this solver
    """

    def __init__(self, rho=None, x0=None, **solver_params):

        self._rho = StencilVector(self.derham.Vh['0'])
        if rho is not None:
            # check solvability condition
            solvability = np.zeros(1)
            self.derham.comm.Allreduce(np.sum(rho.toarray()), solvability, op=MPI.SUM)
            assert solvability[0] <= 1e-15, f'Solvability condition not met: {solvability[0]}'

            assert isinstance(rho, (StencilVector, PolarVector))
            # assert rho.space.space_id == "H1" TODO: doesn't work, but we should check that rho is in H1
            self._rho[:] = rho[:]

        self._phi = StencilVector(self.derham.Vh['0'])

        self._x0 = x0

        # Set lhs matrix
        self._A = Compose(self.derham.grad.T,
                          self.mass_ops.M1,
                          self.derham.grad) \
                    + 1e-14 * Identity(self.derham.Vh['0'])
                    # + 1e-12 * self.mass_ops.M0

        self._solver_params = solver_params
        # preconditioner
        if self._solver_params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, self._solver_params['pc'])
            self._pc = pc_class(self.mass_ops.M0)

    @property
    def variables(self):
        return [self._phi]

    @property
    def rho(self):
        """
        psydac.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector.
        """
        return self._rho

    @rho.setter
    def rho(self, value):
        """ In-place setter for StencilVector/PolarVector.
        """
        assert type(value) == type(self._rho)
        assert value.space.space_id == "H1"

        # check solvability condition
        solvability = np.zeros(1)
        self.derham.comm.Allreduce(np.sum(value.toarray()), solvability, op=MPI.SUM)
        assert solvability[0] <= 1e-15

        self._rho[:] = value[:]

    @property
    def x0(self):
        """
        psydac.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector. First guess of the iterative solver.
        """
        return self._rho

    @x0.setter
    def x0(self, value):
        """ In-place setter for StencilVector/PolarVector. First guess of the iterative solver.
        """
        assert type(value) == type(self._rho)
        assert value.space.space_id == "H1"

        if self._x0 is None:
            self._x0 = value
        else:
            self._x0[:] = value[:]

    def __call__(self, dt):

        res, info = pcg(self._A,
                        self._rho,
                        self._pc,
                        x0=self._x0,
                        tol=self._solver_params['tol'],
                        maxiter=self._solver_params['maxiter'],
                        verbose=self._solver_params['verbose']
                        )

        if self._solver_params['info']:
            print(info)

        self.in_place_update(res)
