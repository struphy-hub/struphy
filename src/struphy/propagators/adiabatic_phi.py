import logging

from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilVector
from scope_profiler import ProfileManager

from struphy.feec import preconditioner
from struphy.feec.mass import L2Projector, WeightedMassOperator, WeightedMassOperators
from struphy.fields_background.equils import set_defaults
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.pic.base import Particles
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")


class AdiabaticPhi(Propagator):
    r"""
    Electrostatic potential for adiabatic electrons, computed from

    .. math::

        n_e = n_{e0}\,\exp \left( \frac{e \phi}{k_B T_e} \right) \approx n_{e0} \left( 1 + \frac{e \phi}{k_B T_{e0}} \right)\,,

    where :math:`n_{e0}` and :math:`T_{e0}` denote electron equilibrium density and temperature, respectively.

    This is solved in weak form: find :math:`\phi \in H^1` such that

    .. math::

        \int_\Omega \psi\, \frac{n_{e0}(\mathbf x)}{T_{e0}(\mathbf x)}\phi \,\textrm d \mathbf x  = \int_\Omega \psi\, (n_{e}(\mathbf x) - n_{e0}(\mathbf x))\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,.

    The equation is discretized as

    .. math::

        \sigma_1 \mathbb M^0_{n/T} \boldsymbol \phi = (\Lambda^0, n_{e} - n_{e0} )_{L^2}\,,

    where :math:`M^0_{n/T}` is a :class:`~struphy.feec.mass.WeightedMassOperator` and :math:`\sigma_1`
    is a normalization parameter.

    Parameters
    ----------
    phi : StencilVector
        FE coefficients of the solution as a discrete 0-form.

    A_mat : WeightedMassOperator
        The matrix to invert.

    rho : StencilVector or tuple
        Right-hand side FE coefficients of a 0-form (optional, can be set with a setter later).
        Can be either a) StencilVector or b) 2-tuple.
        In case b) the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.

    sigma_1 : float
        Normalization parameter.

    x0 : StencilVector
        Initial guess for the iterative solver (optional, can be set with a setter later).

    **params : dict
        Parameters for the iterative solver (see ``__init__`` for details).
    """

    def __init__(
        self,
        phi: StencilVector,
        *,
        A_mat: WeightedMassOperator = "M0",
        rho: StencilVector | tuple = None,
        sigma_1: float = 1.0,
        x0: StencilVector = None,
        **params,
    ):
        assert phi.space == self.derham.V0

        super().__init__(phi)

        # solver parameters
        params_default = {
            "type": ("pcg", "MassMatrixPreconditioner"),
            "tol": 1e-8,
            "maxiter": 3000,
            "info": False,
            "verbose": False,
            "recycle": False,
        }

        params = set_defaults(params, params_default)

        # collect rhs
        if rho is None:
            rho = phi.space.zeros()
        else:
            if isinstance(rho, tuple):
                assert isinstance(rho[0], AccumulatorVector)
                assert isinstance(rho[1], Particles)
                # assert rho[0].space_id == 'H1'
            else:
                assert rho.space == phi.space
        self._rho = rho

        # initial guess and solver params
        self._x0 = x0
        self._params = params
        A_mat = getattr(self.mass_ops, A_mat)

        # Set lhs matrices
        self._A = sigma_1 * A_mat

        # preconditioner and solver for Ax=b
        if params["type"][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params["type"][1])
            pc = pc_class(A_mat)

        # solver just with A_2, but will be set during call with dt
        self._solver = inverse(
            self._A,
            params["type"][0],
            pc=pc,
            x0=self.x0,
            tol=params["tol"],
            maxiter=params["maxiter"],
            verbose=params["verbose"],
            recycle=params["recycle"],
        )

        # allocate memory for solution
        self._tmp = phi.space.zeros()
        self._tmp2 = phi.space.zeros()
        self._rhs = phi.space.zeros()
        self._rhs2 = phi.space.zeros()

    @property
    def rho(self):
        """
        Right-hand side FE coefficients of a 0-form.
        Can be either a) StencilVector or b) 2-tuple.
        In the latter case, the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.
        """
        return self._rho

    @rho.setter
    def rho(self, value):
        """In-place setter for StencilVector/PolarVector."""
        if isinstance(value, tuple):
            assert isinstance(value[0], AccumulatorVector)
            assert isinstance(value[1], Particles)
            self._rho = value
        else:
            assert value.space == self.derham.V0
            self._rho[:] = value[:]

    @property
    def x0(self):
        """
        feectools.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector. First guess of the iterative solver.
        """
        return self._x0

    @x0.setter
    def x0(self, value):
        """In-place setter for StencilVector/PolarVector. First guess of the iterative solver."""
        assert value.space == self.derham.V0
        assert value.space.symbolic_space == "H1", (
            f"Right-hand side must be in H1, but is in {value.space.symbolic_space}."
        )

        if self._x0 is None:
            self._x0 = value
        else:
            self._x0[:] = value[:]

    def __call__(self, dt):
        self._rhs *= 0.0
        if isinstance(self._rho, tuple):
            self._rho[0]()  # accumulate
            self._rhs += self._rho[0].vectors[0]
        else:
            self._rhs += self._rho

        # solve
        with ProfileManager.profile_region(self._solve_region, functions=[self._solver.solve]):
            out = self._solver.solve(self._rhs, out=self._tmp)
        info = self._solver._info

        if self._lin_solver["info"]:
            logger.info(info)

        dphi = self.feec_vars_update(out)

    @classmethod
    def options(cls):
        dct = {}
        dct["solver"] = {
            "type": [
                ("pcg", "MassMatrixPreconditioner"),
                ("cg", None),
            ],
            "tol": 1.0e-8,
            "maxiter": 3000,
            "info": False,
            "verbose": False,
            "recycle": True,
        }
        return dct
