from struphy.propagators.base import Propagator

from struphy.psydac_api import preconditioner

from psydac.linalg.iterative_solvers import pcg

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector


class PoissonSolver(Propagator):
    r"""
    Solve the Poisson equation

    ..math:
        \nabla \cdot \mathbf{E} = \rho

    by using the electric potential :math:`\phi\,\in H^1` as an auxiliary field:

    ..math:

        \nabla \cdot \mathbf{E} = \rho \,, \qquad \mathbf{E} = - \nabla \phi \,,

    for the FE coefficients of :math:`\mathbf{E} \, \in H(\text{curl})`.

    Parameters
    ----------
    **solver_params : dict
        Parameters for this solver
    """

    def __init__(self, solver_params):

        # Set lhs matrix
        self._A = self.derham.grad.T.dot(
            self.mass_ops.M1.dot(
                self.derham.grad))

        self._solver_params = solver_params
        # preconditioner
        if self._solver_params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, self._solver_params['pc'])
            self._pc = pc_class(self.mass_ops.M1)

    def __call__(self, rhs, out=None):
        """
        TODO

        Parameters
        ----------
        rhs : psydac.linalg.block.StencilVector
            FE coefficients of a 0-form.

        rhs : psydac.linalg.block.StencilVector
            FE coefficients of a 0-form.
        """
        assert isinstance(rhs, StencilVector)

        res, info = pcg(self._A,
                        self.mass_ops.M0.dot(rhs),
                        self._pc,
                        tol=self._solver_params['tol'],
                        maxiter=self._solver_params['maxiter'],
                        verbose=self._solver_params['verbose']
                        )

        if self._solver_params['info']:
            print(info)

        if out is not None:
            assert isinstance(out, BlockVector)
            out = - self.derham.grad.dot(res)
        else:
            return - self.derham.grad.dot(res)
