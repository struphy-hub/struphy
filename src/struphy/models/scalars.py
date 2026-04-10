from abc import ABCMeta, abstractmethod
from typing import Callable, Union

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.mass import WeightedMassOperator
from struphy.feec.psydac_derham import space_to_form
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.utils.docstring_converter import auto_convert_docstring

_DUMMY_VARIABLE = object()


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
    Calling .update() on this container will update all contained scalars."""

    def __init__(self, **scalars: dict[str, Scalar]):
        for name, scalar in scalars.items():
            assert isinstance(scalar, Scalar)
        self._dct = scalars

    @property
    def dct(self) -> dict[str, Scalar]:
        return self._dct

    def update(self):
        for scalar in self.dct.values():
            scalar.update()
        # reset status to False for next update
        for scalar in self.dct.values():
            scalar.uptodate = False


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
            assert left_variable.space == right_variable.space, (
                "If bilinear_form_name is not provided, left and right variables must be in the same space to infer the bilinear form."
            )
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
        self.local_value[0] = float(self.function())

    def _mpi_sum(self):
        """Communication has been handled by psydac, so no additional MPI operations are needed."""
        self.value[0] = self.local_value[0]


class KineticEnergyPIC(PICScalar):
    r"""Scalar representing the kinetic energy computed from a PIC variable, according to

    :math:

        \mathcal E = \frac{\alpha}{2 N_p} \sum_{i=0}^{N_p-1} w_i v_i^2\,,

    where :math:`\alpha` is a normalization constant, :math:`N_p` is the total number of particles,
    and :math:`w_i` and :math:`v_i` are the weight and velocity of particle :math:`i`, respectively.
    """

    def _local_update(self):
        if not hasattr(self, "velocities"):
            self.velocities = self.variables[
                0
            ].particles.velocities  # TODO: velocities need to redefined for Particles5d? Put magnetic moment as COM.
            self.weights = self.variables[0].particles.weights
            self.Np = self.variables[0].particles.Np

        energy = self.normalization * 0.5 / self.Np * xp.sum(self.weights * xp.sum(self.velocities**2, axis=1))
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
        self.local_value[0] = float(self.function())


class KineticEnergySPH(SPHScalar):
    r"""Scalar representing the kinetic energy computed from a SPH variable, according to

    :math:

        \mathcal E = \frac{\alpha}{2 N_p} \sum_{i=0}^{N_p-1} w_i v_i^2\,,

    where :math:`\alpha` is a normalization constant, :math:`N_p` is the total number of particles,
    and :math:`w_i` and :math:`v_i` are the weight and velocity of particle :math:`i`, respectively.
    """

    def _local_update(self):
        if not hasattr(self, "velocities"):
            self.velocities = self.variables[0].particles.velocities
            self.weights = self.variables[0].particles.weights
            self.Np = self.variables[0].particles.Np

        energy = self.normalization * 0.5 / self.Np * xp.sum(self.weights * xp.sum(self.velocities**2, axis=1))
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
        self.local_value[0] = float(self.function())
