from abc import ABCMeta, abstractmethod
from typing import Callable, Union

import cunumpy as xp
import numpy as np
from maybempi import MPI

from struphy.feec.mass import WeightedMassOperator
from struphy.feec.psydac_derham import space_to_form
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.pic.particles import Particles5D
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.utils.docstring_converter import auto_convert_docstring

_DUMMY_VARIABLE = object()


def _scalar_value(value):
    """`value` (of size 1) as something that can be written into a scalar's buffer without leaving the device.

    A device result (e.g. a reduction on the CuPy backend) stays a 0-d device array; ``float()`` would wait for the
    device and copy it to the host at every update. Host values (Python or NumPy scalars, such as the result of an
    inner product that already went through MPI on the host) are plain floats.
    """
    if isinstance(value, xp.ndarray):
        return value.reshape(())
    return float(np.asarray(value).reshape(()))


class Scalar(metaclass=ABCMeta):
    """Abstract base class for scalar quantities in MPI parallel simulations.

    Parameters
    ----------
    variables : Variable or Scalar
        The variable(s) associated with the scalar, or scalars for summation."""

    def __init__(self, *variables: Union[Variable, "Scalar"]):
        self.variables = variables
        self.local_value = xp.empty(1, dtype=float)
        self.value = xp.empty(1, dtype=float)
        self.uptodate = False

    @abstractmethod
    def _local_update(self):
        """Update self.local_value[0] on the current process."""
        pass

    @abstractmethod
    def _mpi_sum(self):
        """Sum the local values over MPI processes."""
        pass

    def update(self):
        """Update the scalar quantity by performing local update and then summing over MPI processes."""
        if not self.uptodate:
            self._local_update()
            self._mpi_sum()
            self.uptodate = True

    def __add__(self, other):
        return SumOfScalars(self, other)


class SumOfScalars(Scalar):
    """Scalar representing the sum of other scalars. An update of this scalar will also update all its summands."""

    def __init__(self, *scalars):
        for scalar in scalars:
            assert isinstance(scalar, Scalar)
        super().__init__(*scalars)

    def _local_update(self):
        "Local updates for each summands are performed in _mpi_sum via .update()."
        pass

    def _mpi_sum(self):
        for scalar in self.variables:
            scalar.update()
        energy = sum(scalar.value[0] for scalar in self.variables)
        self.value[0] = energy


class PICScalar(Scalar):
    """Base class for scalar quantities computed from PIC variables.
    Handles MPI communication within and between clones, but requires subclasses to implement the local update of self.local_value[0]."""

    def __init__(
        self,
        pic_variable: PICVariable,
        normalization: float = 1.0,
    ):
        assert isinstance(pic_variable, PICVariable), "variable must be an instance of PICVariable"
        super().__init__(pic_variable)
        self.normalization = normalization

    def _local_update(self):
        raise NotImplementedError(
            "Subclasses of PICScalar must implement _local_update to compute self.local_value[0]."
        )

    def _mpi_sum(self):
        self.value[0] = self.local_value[0]

        # sum within clone
        if Propagator.derham.comm is not None:
            Propagator.derham.comm.Allreduce(
                MPI.IN_PLACE,
                self.value,
                op=MPI.SUM,
            )

        # sum between clones
        if not hasattr(self, "clone_config"):
            self.clone_config = self.variables[0].particles.clone_config

        if self.clone_config is not None:
            self.clone_config.inter_comm.Allreduce(
                MPI.IN_PLACE,
                self.value,
                op=MPI.SUM,
            )


class SPHScalar(Scalar):
    """Base class for scalar quantities computed from SPH variables.
    Handles MPI communication, but requires subclasses to implement the local update of self.local_value[0]."""

    def __init__(
        self,
        sph_variable: SPHVariable,
        normalization: float = 1.0,
    ):
        assert isinstance(sph_variable, SPHVariable), "variable must be an instance of SPHVariable"
        super().__init__(sph_variable)
        self.normalization = normalization

    def _local_update(self):
        raise NotImplementedError(
            "Subclasses of SPHScalar must implement _local_update to compute self.local_value[0]."
        )

    def _mpi_sum(self):
        self.value[0] = self.local_value[0]

        MPI.COMM_WORLD.Allreduce(
            MPI.IN_PLACE,
            self.value,
            op=MPI.SUM,
        )


class Scalars:
    """Container for multiple Scalar objects.
    Calling .update() on this container will update all contained scalars.

    The scalars are computed where the data lives (on the device on the CuPy backend). The ``value`` of each scalar
    in the container is a view into one array of the container, and :meth:`update` ends with :meth:`to_host`, which
    copies that array to the host in one transfer. The output (:meth:`host_value`, printing) reads the host copy.
    """

    def __init__(self, **scalars: dict[str, Scalar]):
        for name, scalar in scalars.items():
            assert isinstance(scalar, Scalar)
        if scalars:
            self._dct = scalars
        else:
            self._dct = {}

        # one buffer for all values, so that they reach the host in a single copy
        self._device_values = xp.zeros(len(self._dct), dtype=float)
        self._host_values = np.zeros(len(self._dct), dtype=float)
        self._index = {}
        for i, (name, scalar) in enumerate(self._dct.items()):
            scalar.value = self._device_values[i : i + 1]
            self._index[name] = i

    @property
    def dct(self) -> dict[str, Scalar]:
        return self._dct

    def update(self):
        for scalar in self.dct.values():
            scalar.update()
        # reset status to False for next update, including the summands of sums: `a + b + c` nests
        # SumOfScalars(SumOfScalars(a, b), c), and an inner sum left up to date would keep its first value
        for scalar in self.dct.values():
            _mark_outdated(scalar)
        self.to_host()

    def to_host(self):
        """Copy the values of all scalars to the host, in one transfer (the only one of the scalar diagnostics)."""
        self._host_values[:] = xp.to_numpy(self._device_values)

    def host_value(self, name: str) -> np.ndarray:
        """The value of scalar `name` after the last :meth:`update`, as a NumPy array of size 1.

        A view into the host buffer of the container: it follows later updates (the output registers it once).
        """
        i = self._index[name]
        return self._host_values[i : i + 1]


def _mark_outdated(scalar: Scalar):
    scalar.uptodate = False
    for variable in scalar.variables:
        if isinstance(variable, Scalar):
            _mark_outdated(variable)


@auto_convert_docstring
class BilinearEnergyFEEC(Scalar):
    """Scalar from a bilinear FEEC form evaluated on one or two FEEC variables."""

    def __init__(
        self,
        left_variable: FEECVariable,
        right_variable: FEECVariable | str | None = None,
        bilinear_form_name: str | None = None,
        normalization: float = 1.0,
    ):
        assert isinstance(left_variable, FEECVariable), "left_variable must be an instance of FEECVariable"
        if right_variable is None:
            right_variable = left_variable
        assert isinstance(right_variable, (FEECVariable, str)), (
            "right_variable must be an instance of FEECVariable or a string"
        )

        if bilinear_form_name is None:
            # assert left_variable.space == right_variable.space, (
            #     "If bilinear_form_name is not provided, left and right variables must be in the same space to infer the bilinear form."
            # )
            form = space_to_form[left_variable.space]
            bilinear_form_name = f"M{form}"

        super().__init__(left_variable, right_variable)
        self.bilinear_form_name = bilinear_form_name
        self.normalization = normalization

    def _local_update(self):
        if not hasattr(self, "left_vec"):
            self.left_vec = self.variables[0].spline.vector
            if isinstance(self.variables[1], str):
                self.right_vec = getattr(Propagator.projected_equil, self.variables[1])
            else:
                self.right_vec = self.variables[1].spline.vector
            self.vec_space = self.left_vec.space
        if not hasattr(self, "bilinear_form"):
            self.bilinear_form: WeightedMassOperator = getattr(Propagator.mass_ops, self.bilinear_form_name)
            assert self.bilinear_form.codomain == self.vec_space, "bilinear_form codomain must match variable space"

        value = self.normalization * 0.5 * self.bilinear_form.dot_inner(self.right_vec, self.left_vec)
        self.local_value[0] = value

    def _mpi_sum(self):
        """Communication has been handled by psydac's .dot_inner, so no additional MPI operations are needed."""
        self.value[0] = self.local_value[0]

    __doc_rst__ = r"""For example, for a vector-valued variable :math:`\mathbf{u}` the computed energy when right_variable is None reads

.. math::

    \mathcal E = \alpha \frac{1}{2} \int_{\Omega} \mathbf{u}^\top A \mathbf u  \, d \mathbf x\,,
    
where :math:`\alpha` is a normalization constant and :math:`A` is a symmetric positive definite matrix (the identity by default)."""


class VolumeFormEnergyFEEC(Scalar):
    """Scalar from a volume form integrated over the domain."""

    def __init__(
        self,
        feec_variable: FEECVariable,
        normalization: float = 1.0,
    ):
        assert isinstance(feec_variable, FEECVariable), "variable must be an instance of FEECVariable"
        super().__init__(feec_variable)
        self.normalization = normalization

    def _local_update(self):
        if not hasattr(self, "vec"):
            self.vec = self.variables[0].spline.vector
            if isinstance(self.vec, PolarVector):
                self.ones = Propagator.derham.V3pol.zeros()
                self.ones.tp[:] = 1.0
            else:
                self.ones = Propagator.derham.V3.zeros()
                self.ones[:] = 1.0

        self.local_value[0] = self.normalization * self.vec.inner(self.ones)

    def _mpi_sum(self):
        """Communication has been handled by psydac's .dot_inner, so no additional MPI operations are needed."""
        self.value[0] = self.local_value[0]

    __doc_rst__ = r"""For example, for a volume form :math:`p` the computed energy reads

.. math::

    \mathcal E = \alpha \int_{\Omega} p  \, d \mathbf x\,,
    
where :math:`\alpha` is a normalization constant."""


class FunctionScalarFEEC(Scalar):
    """Scalar defined by a callable working on FEEC variables."""

    def __init__(
        self,
        function: Callable[[], float],
    ):
        self.function = function
        Scalar.__init__(self, _DUMMY_VARIABLE)

    def _local_update(self):
        self.local_value[0] = _scalar_value(self.function())

    def _mpi_sum(self):
        """Communication has been handled by psydac, so no additional MPI operations are needed."""
        self.value[0] = self.local_value[0]


class KineticEnergyPIC(PICScalar):
    r"""Scalar representing the kinetic energy computed from a PIC variable, according to

    :math:

        \mathcal E = \frac{\alpha}{2} \sum_{i=0}^{N_p-1} w_i v_i^2\,,

    where :math:`\alpha` is a normalization constant and :math:`w_i` and :math:`v_i` are the weight and
    velocity of particle :math:`i`. The marker weights already carry the :math:`1/N_p` of the Monte-Carlo
    estimate (:meth:`~struphy.pic.base.Particles.initialize_weights` sets
    :math:`w_i = f_i / (s_i N_p)`), so the sum must not be divided by :math:`N_p` again.

    For :class:`~struphy.pic.particles.Particles5D` the velocity coordinates are :math:`(v_\parallel, \mu)`,
    so only the parallel kinetic energy :math:`v_i^2 = v_{\parallel,i}^2` is summed; the magnetic-moment
    part :math:`\mu_i |B_0|` is tracked by a separate scalar in the models (e.g. ``en_fB``).
    """

    def _local_update(self):
        # `particles.velocities` and `.weights` are fancy-indexed copies of the marker array, not views,
        # so they must be read at every update: a cached copy keeps the state of the first call forever.
        particles = self.variables[0].particles
        velocities = particles.velocities
        weights = particles.weights

        # Particles5D velocities are (v_par, mu): mu is not a velocity, only v_par contributes here.
        if isinstance(particles, Particles5D):
            velocities = velocities[:, :1]

        energy = self.normalization * 0.5 * xp.sum(weights * xp.sum(velocities**2, axis=1))
        self.local_value[0] = energy


class LostMarkersPIC(PICScalar):
    r"""Scalar representing the number of lost markers, computed from a PIC variable."""

    def _local_update(self):
        particles = self.variables[0].particles
        self.local_value[0] = particles.n_lost_markers


class FunctionScalarPIC(PICScalar):
    """Scalar defined by a callable working on a Particle variable."""

    def __init__(
        self,
        function: Callable[[], float],
        pic_variable: PICVariable,
    ):
        self.function = function
        super().__init__(pic_variable)

    def _local_update(self):
        self.local_value[0] = _scalar_value(self.function())


class KineticEnergySPH(SPHScalar):
    r"""Scalar representing the kinetic energy computed from a SPH variable, according to

    :math:

        \mathcal E = \frac{\alpha}{2} \sum_{i=0}^{N_p-1} w_i v_i^2\,,

    where :math:`\alpha` is a normalization constant and :math:`w_i` and :math:`v_i` are the weight and
    velocity of particle :math:`i`. The marker weights already carry the :math:`1/N_p` of the Monte-Carlo
    estimate (:meth:`~struphy.pic.base.Particles.initialize_weights` sets
    :math:`w_i = f_i / (s_i N_p)`), so the sum must not be divided by :math:`N_p` again.
    """

    def _local_update(self):
        # As in KineticEnergyPIC: the marker arrays are copies, so they are read at every update.
        particles = self.variables[0].particles
        velocities = particles.velocities
        weights = particles.weights

        energy = self.normalization * 0.5 * xp.sum(weights * xp.sum(velocities**2, axis=1))
        self.local_value[0] = energy


class FunctionScalarSPH(SPHScalar):
    """Scalar defined by a callable working on a SPH variable."""

    def __init__(
        self,
        function: Callable[[], float],
        sph_variable: SPHVariable,
    ):
        self.function = function
        super().__init__(sph_variable)

    def _local_update(self):
        self.local_value[0] = _scalar_value(self.function())
