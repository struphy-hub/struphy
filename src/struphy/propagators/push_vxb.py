"Only particle variables are updated."

from dataclasses import dataclass
from typing import Literal

from feectools.linalg.basic import LinearOperator
from feectools.linalg.block import BlockVector
from line_profiler import profile

from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option
class PushVxB(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf{v}_p(t)}{\textnormal d t} =  \frac{1}{\varepsilon} \, \mathbf{v}_p(t) \times (\mathbf{B} + \mathbf{B}_{\text{add}}) \,,

    where :math:`\varepsilon = 1/(\hat{\Omega}_c \hat t)` is a constant scaling factor, and for rotation vector :math:`\mathbf{B}` and optional, additional fixed rotation vector :math:`\mathbf{B}_{\text{add}}`, both given as a 2-form:

    .. math::

        \mathbf{B} =  \frac{DF\, \hat{\mathbf{B}}^2}{\sqrt{g}}\,.

    Available algorithms:

    - ``analytic``
    - ``implicit``.
    """

    class Variables:
        """Container for variables advanced by :class:`PushVxB`.

        Attributes
        ----------
        ions : PICVariable or SPHVariable
            Particle variable whose velocities are advanced.
        """
        def __init__(self):
            self._ions: PICVariable | SPHVariable = None

        @property
        def ions(self) -> PICVariable | SPHVariable:
            return self._ions

        @ions.setter
        def ions(self, new):
            assert isinstance(new, PICVariable | SPHVariable)
            assert new.space in ("Particles6D", "DeltaFParticles6D", "ParticlesSPH")
            self._ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        """Configuration options for :class:`PushVxB`.

        Parameters
        ----------
        algo : {"analytic", "implicit"}, default="analytic"
            Time stepping algorithm used for the velocity-rotation update.

        b2_var : FEECVariable, default=None
            Optional additional magnetic-field 2-form added to the projected
            equilibrium field before pushing.
        """
        # specific literals
        OptsAlgo = Literal["analytic", "implicit"]
        # propagator options
        algo: OptsAlgo = "analytic"
        b2_var: FEECVariable = None

        def __post_init__(self):
            # checks
            check_option(self.algo, self.OptsAlgo)

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new

    @profile
    def allocate(self, verbose: bool = False):
        # scaling factor
        self._epsilon = self.variables.ions.species.equation_params.epsilon
        assert self.derham is not None, f"{self.__class__.__name__} needs a Derham object."

        # TODO: treat PolarVector as well, but polar splines are being reworked at the moment
        if self.projected_equil is not None:
            self._b2 = self.projected_equil.b2
            assert self._b2.space == self.derham.V2
        else:
            self._b2 = self.derham.V2.zeros()

        if self.options.b2_var is None:
            self._b2_var = None
        else:
            assert self.options.b2_var.spline.vector.space == self.derham.V2
            self._b2_var = self.options.b2_var.spline.vector

        # allocate dummy vectors to avoid temporary array allocations
        self._tmp = self.derham.V2.zeros()
        self._b_full = self.derham.V2.zeros()

        # define pusher kernel
        if self.options.algo == "analytic":
            kernel = Pyccelkernel(pusher_kernels.push_vxb_analytic)
        elif self.options.algo == "implicit":
            kernel = Pyccelkernel(pusher_kernels.push_vxb_implicit)
        else:
            raise ValueError(f"{self.options.algo =} not supported.")

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self._b_full[0]._data,
            self._b_full[1]._data,
            self._b_full[2]._data,
        )

        self._pusher = Pusher(
            self.variables.ions.particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T: LinearOperator = self.derham.extraction_ops["2"].transpose()

    @profile
    def __call__(self, dt):
        # sum up total magnetic field
        tmp = self._b2.copy(out=self._tmp)
        if self._b2_var is not None:
            tmp += self._b2_var

        # extract coefficients to tensor product space
        b_full: BlockVector = self._E2T.dot(tmp, out=self._b_full)
        b_full.update_ghost_regions()
        b_full /= self._epsilon

        # call pusher kernel
        self._pusher(dt)

        # update_weights
        if self.variables.ions.particles.control_variate:
            self.variables.ions.particles.update_weights()


