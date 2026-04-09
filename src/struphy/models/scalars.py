from abc import ABCMeta, abstractmethod
from struphy.models.variables import Variable, FEECVariable, PICVariable, SPHVariable
import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from struphy.propagators.base import Propagator
from struphy.feec.mass import WeightedMassOperator
from struphy.feec.psydac_derham import space_to_form
from typing import Union
from struphy.utils.docstring_converter import auto_convert_docstring


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
class QuadraticEnergyFEEC(Scalar):
    """Scalar representing a quadratic energy computed from a FEEC variable."""

    def __init__(self, feec_variable: FEECVariable, bilinear_form_name: str=None, normalization: float=1.0,):
        assert isinstance(feec_variable, FEECVariable), "variable must be an instance of FEECVariable"
        if bilinear_form_name is None:
            form = space_to_form[feec_variable.space]
            bilinear_form_name = f"M{form}"
        
        super().__init__(feec_variable)
        
        self.bilinear_form_name = bilinear_form_name
        self.normalization = normalization

    def _local_update(self):
        if not hasattr(self, "vec"):
            self.vec = self.variables[0].spline.vector
            self.vec_space = self.vec.space
        if not hasattr(self, "bilinear_form"):
            self.bilinear_form: WeightedMassOperator = getattr(Propagator.mass_ops, self.bilinear_form_name)
            assert self.bilinear_form.domain == self.vec_space, "bilinear_form domain must match variable space"
        energy = self.normalization * 0.5 * self.bilinear_form.dot_inner(self.vec, self.vec)
        self.local_value[0] = energy
        
    def _mpi_sum(self):
        """Communication has been handled by psydac's .dot_inner, so no additional MPI operations are needed."""
        self.value[0] = self.local_value[0]
        
    __doc_rst__ = r"""For example, for a vector-valued variable :math:`\mathbf{u}` the computed energy is

.. math::

    \mathcal E = \alpha \frac{1}{2} \int_{\Omega} \mathbf{u}^\top A \mathbf u  \, d \mathbf x\,,
    
where :math:`\alpha` is a normalization constant and :math:`A` is a symmetric positive definite matrix (the identity by default)."""
    
class PICScalar(Scalar):
    """Base class for scalar quantities computed from PIC variables.
    Handles MPI communication within and between clones, but requires subclasses to implement the local update of self.local_value[0]."""
    
    def __init__(self, pic_variable: PICVariable, normalization: float=1.0,):
        assert isinstance(pic_variable, PICVariable), "variable must be an instance of PICVariable"
        super().__init__(pic_variable)
        
        self.velocities = pic_variable.particles.velocities # TODO: velocities need to redefined for Particles5d? Put magnetic moment as COM.
        self.weights = pic_variable.particles.weights
        self.Np = pic_variable.particles.Np
        self.clone_config = pic_variable.particles.clone_config
        self.normalization = normalization

    def _local_update(self):
        raise NotImplementedError("Subclasses of PICScalar must implement _local_update to compute self.local_value[0].")
        
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
        if self.clone_config is not None:
            self.clone_config.inter_comm.Allreduce(
                MPI.IN_PLACE,
                self.value,
                op=MPI.SUM,
            )
    
class KineticEnergyPIC(PICScalar):
    r"""Scalar representing the kinetic energy computed from a PIC variable, according to
    
    :math:
    
        \mathcal E = \frac{\alpha}{2 N_p} \sum_{i=0}^{N_p-1} w_i v_i^2\,,
        
    where :math:`\alpha` is a normalization constant, :math:`N_p` is the total number of particles, 
    and :math:`w_i` and :math:`v_i` are the weight and velocity of particle :math:`i`, respectively.
    """
    def _local_update(self):
        energy = self.normalization * 0.5 / self.Np * xp.sum(self.weights * xp.sum(self.velocities**2, axis=1))
        self.local_value[0] = energy

    # def update_scalar(self, name, value=None):
    #     """Update a scalar during the simulation.

    #     Parameters
    #     ----------
    #         name : str
    #             Dictionary key of the scalar.

    #         value : float, optional
    #             Value to be saved. Required if there are no summands.
    #     """

    #     # Ensure the name is a string
    #     assert isinstance(name, str)

    #     scalars = self.scalar_quantities

    #     variable: PICVariable | SPHVariable = scalars[name]["variable"]
    #     summands = scalars[name]["summands"]
    #     compute = scalars[name]["compute"]

    #     if compute == "from_particles":
    #         compute_operations = [
    #             "sum_within_clone",
    #             "sum_between_clones",
    #             "divide_n_mks",
    #         ]
    #     elif compute == "from_sph":
    #         compute_operations = [
    #             "sum_world",
    #             "divide_n_mks",
    #         ]
    #     elif compute == "from_field":
    #         compute_operations = []
    #     else:
    #         compute_operations = []

    #     if summands is None:
    #         # Ensure the value is a float if there are no summands
    #         assert isinstance(value, float)

    #         # Create a numpy array to hold the scalar value
    #         value_array = xp.array([value], dtype=xp.float64)

    #         # Perform MPI operations based on the compute flags
    #         if "sum_world" in compute_operations and not isinstance(MPI, MockMPI):
    #             MPI.COMM_WORLD.Allreduce(
    #                 MPI.IN_PLACE,
    #                 value_array,
    #                 op=MPI.SUM,
    #             )

    #         if "sum_within_clone" in compute_operations and Propagator.derham.comm is not None:
    #             Propagator.derham.comm.Allreduce(
    #                 MPI.IN_PLACE,
    #                 value_array,
    #                 op=MPI.SUM,
    #             )
    #         if self.clone_config is None:
    #             num_clones = 1
    #         else:
    #             num_clones = self.clone_config.num_clones

    #         if "sum_between_clones" in compute_operations and num_clones > 1:
    #             self.clone_config.inter_comm.Allreduce(
    #                 MPI.IN_PLACE,
    #                 value_array,
    #                 op=MPI.SUM,
    #             )

    #         if "average_between_clones" in compute_operations and num_clones > 1:
    #             self.clone_config.inter_comm.Allreduce(
    #                 MPI.IN_PLACE,
    #                 value_array,
    #                 op=MPI.SUM,
    #             )
    #             value_array /= num_clones

    #         if "divide_n_mks" in compute_operations:
    #             # Initialize the total number of markers
    #             n_mks_tot = xp.array([variable.particles.Np])
    #             value_array /= n_mks_tot

    #         # Update the scalar value
    #         scalars[name]["value"][0] = value_array[0]

    #     else:
    #         # Sum the values of the summands
    #         value = sum(scalars[summand]["value"][0] for summand in summands)
    #         scalars[name]["value"][0] = value
