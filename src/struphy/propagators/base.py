"Propagator base class."

from abc import ABCMeta, abstractmethod
import numpy as np
from mpi4py import MPI

from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain
from struphy.models.variables import Variable, FEECVariable, PICVariable, SPHVariable


class Propagator(metaclass=ABCMeta):
    """Base class for propagators used in StruphyModels.

    Note
    ----
    All Struphy propagators are subclasses of ``Propagator`` and must be added to ``struphy/propagators``
    in one of the modules ``propagators_fields.py``, ``propagators_markers.py`` or ``propagators_coupling.py``.
    Only propagators that update both a FEEC and a PIC species go into ``propagators_coupling.py``.
    """

    def __init__(self, *vars):
        """Create an instance of a Propagator.

        Parameters
        ----------
        vars : Variable
            Variables to be updated.
        """

        comm = None

        # for iterative particle push
        self._init_kernels = []
        self._eval_kernels = []

        self._rank = comm.Get_rank() if comm is not None else 0

    def set_variables(self, *vars):
        for var in vars:
            assert isinstance(var, Variable)
            if isinstance(var, (PICVariable, SPHVariable)):
                # comm = var.obj.mpi_comm
                pass
            elif isinstance(var, FEECVariable):
                # comm = var.obj.comm
                pass
        self._vars = vars

    @property
    def vars(self):
        """List of Variables to be updated by the propagator.
        """
        return self._vars

    @property
    def init_kernels(self):
        r"""List of initialization kernels for evaluation at
        :math:`\boldsymbol \eta^n`
        in an iterative :class:`~struphy.pic.pushing.pusher.Pusher`.
        """
        return self._init_kernels

    @property
    def eval_kernels(self):
        r"""List of evaluation kernels for evaluation at
        :math:`\alpha_i \eta_{i}^{n+1,k} + (1 - \alpha_i) \eta_{i}^n`
        for :math:`i=1, 2, 3` and different :math:`\alpha_i \in [0,1]`,
        in an iterative :class:`~struphy.pic.pushing.pusher.Pusher`.
        """
        return self._eval_kernels

    @property
    def rank(self):
        """MPI rank, is 0 if no communicator."""
        return self._rank

    @abstractmethod
    def set_options(self, **opts):
        """Set the dynamical options of the propagator (kwargs).
        """
        
    @abstractmethod
    def allocate():
        """Allocate all data/objects for an instance.
        """
    
    @abstractmethod
    def __call__(self, dt):
        """Update variables from t -> t + dt.
        Use ``Propagators.feec_vars_update`` to write to FEEC variables to ``Propagator.feec_vars``.

        Parameters
        ----------
        dt : float
            Time step size.
        """

    @property
    def derham(self):
        """Derham spaces and projectors."""
        assert hasattr(
            self,
            "_derham",
        ), "Derham not set. Please do obj.derham = ..."
        assert isinstance(self._derham, Derham)
        return self._derham

    @derham.setter
    def derham(self, derham):
        self._derham = derham

    @property
    def domain(self):
        """Domain object that characterizes the mapping from the logical to the physical domain."""
        assert hasattr(self, "_domain"), "Domain for analytical MHD equilibrium not set. Please do obj.domain = ..."
        assert isinstance(self._domain, Domain)
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain

    @property
    def mass_ops(self):
        """Weighted mass operators."""
        assert hasattr(self, "_mass_ops"), "Weighted mass operators not set. Please do obj.mass_ops = ..."
        assert isinstance(self._mass_ops, WeightedMassOperators)
        return self._mass_ops

    @mass_ops.setter
    def mass_ops(self, mass_ops):
        self._mass_ops = mass_ops

    @property
    def basis_ops(self):
        """Basis projection operators."""
        assert hasattr(self, "_basis_ops"), "Basis projection operators not set. Please do obj.basis_ops = ..."
        assert isinstance(self._basis_ops, BasisProjectionOperators)
        return self._basis_ops

    @basis_ops.setter
    def basis_ops(self, basis_ops):
        self._basis_ops = basis_ops

    @property
    def projected_equil(self):
        """Fluid equilibrium projected on 3d Derham sequence with commuting projectors."""
        assert hasattr(
            self,
            "_projected_equil",
        ), "Projected MHD equilibrium not set."
        return self._projected_equil

    @projected_equil.setter
    def projected_equil(self, projected_equil):
        self._projected_equil = projected_equil

    @property
    def time_state(self):
        """A pointer to the time variable of the dynamics ('t')."""
        return self._time_state 

    def add_time_state(self, time_state):
        """Add a pointer to the time variable of the dynamics ('t').

        Parameters
        ----------
        time_state : ndarray
            Of size 1, holds the current physical time 't'.
        """
        assert time_state.size == 1
        self._time_state = time_state

    def feec_vars_update(self, *variables_new):
        r"""Return :math:`\textrm{max}_i |x_i(t + \Delta t) - x_i(t)|` for each unknown in list,
        update :method:`~struphy.propagators.base.Propagator.feec_vars`
        and update ghost regions.

        Parameters
        ----------
        variables_new : list[StencilVector | BlockVector]
            Same sequence as in :method:`~struphy.propagators.base.Propagator.feec_vars`
            but with the updated variables,
            i.e. for feec_vars = [e, b] we must have variables_new = [e_updated, b_updated].

        Returns
        -------
        diffs : list
            A list [max(abs(self.feec_vars - variables_new)), ...] for all variables in self.feec_vars and variables_new.
        """

        diffs = []

        for i, new in enumerate(variables_new):
            assert type(new) is type(self.feec_vars[i])

            # calculate maximum of difference abs(old - new)
            diffs += [np.max(np.abs(self.feec_vars[i].toarray() - new.toarray()))]

            # copy new variables into self.feec_vars
            new.copy(out=self.feec_vars[i])

            # important: sync processes!
            self.feec_vars[i].update_ghost_regions()

        return diffs

    def add_init_kernel(
        self,
        kernel,
        column_nr: int,
        comps: tuple | int,
        args_init: tuple,
    ):
        """Add an initialization kernel to self.init_kernels.

        Parameters
        ----------
        kernel : pyccel func
            The kernel function.

        column_nr : int
            The column index at which the result is stored in marker array.

        comps : tuple | int
            None or (0) for scalar-valued function evaluation.
            In vector valued case, allows to specify which components to save
            at column_nr:column_nr + len(comps).

        args_init : tuple
            The arguments for the kernel function.
        """
        if comps is None:
            comps = np.array([0])  # case for scalar evaluation
        else:
            comps = np.array(comps, dtype=int)

        self._init_kernels += [
            (
                kernel,
                column_nr,
                comps,
                args_init,
            )
        ]

    def add_eval_kernel(
        self,
        kernel,
        column_nr: int,
        comps: tuple | int,
        args_eval: tuple,
        alpha: float | int | tuple | list = 1.0,
    ):
        """Add an evaluation kernel to self.eval_kernels.

        Parameters
        ----------
        kernel : pyccel func
            The kernel function.

        column_nr : int
            The column index at which the result is stored in marker array.

        comps : tuple | int
            None for scalar-valued function evaluation. In vecotr valued case,
            allows to specify which components to save
            at column_nr:column_nr + len(comps).

        args_init : tuple
            The arguments for the kernel function.

        alpha : float | int | tuple | list
            Evaluations in kernel are at the weighted average
            alpha[i]*markers[:, i] + (1 - alpha[i])*markers[:, buffer_idx + i],
            for i=0,1,2. If float or int or then alpha = [alpha]*dim,
            where dim is the dimension of the phase space (<=6).
            alpha[i] must be between 0 and 1.
        """
        if isinstance(alpha, int) or isinstance(alpha, float):
            alpha = [alpha] * 6
        alpha = np.array(alpha)

        if comps is None:
            comps = np.array([0])  # case for scalar evaluation
        else:
            comps = np.array(comps, dtype=int)

        self._eval_kernels += [
            (
                kernel,
                alpha,
                column_nr,
                comps,
                args_eval,
            )
        ]
       
    @property
    def opts(self):
        if not hasattr(self, "_opts"):
            self._opts = self.Options()
        return self._opts
        
    class Options:
        def __init__(self, outer_prop, verbose=False):
            self._outer_prop = outer_prop.__class__.__name__ # outer class
            self._verbose = verbose
        
        @property
        def all(self):
            return self.__dict__
        
        def add(self, name: str, opt):
            setattr(self, name, opt)
            if self._verbose and MPI.COMM_WORLD.Get_rank() == 0:
                print(f"Propagator '{self._outer_prop}': added option '{name}' with value '{opt}'")