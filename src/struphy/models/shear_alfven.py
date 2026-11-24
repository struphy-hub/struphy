import copy
from typing import Literal

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, Scalars
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator
from struphy.propagators.shear_alfven_propagator import ShearAlfvenPropagator
from struphy.utils.utils import check_option


class ShearAlfven(StruphyModel):
    """Shear-Alfvén part of :class:`~struphy.models.linear_mhd.LinearMHD` with zero-flow equilibrium.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the ion species (default: 1.0)
    b_space: {"Hdiv", "Hcurl"}
        Space of the magnetic field (default: "Hdiv"). ``"Hdiv"``: strong induction equation, implicit solve for the
        velocity (:class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`). ``"Hcurl"``: weak
        induction equation, implicit curl-curl solve for the magnetic field
        (:class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`).
    projected: bool
        Only with ``b_space="Hcurl"``: the curl-curl term of the implicit step contains the projection of the Lorentz
        force, which makes the step exactly energy conserving at the price of inner mass matrix solves
        (default: False).
    """

    OptsBSpace = Literal["Hdiv", "Hcurl"]

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species
    class EMFields(FieldSpecies):
        def __init__(self, b_space: str = "Hdiv"):
            self.b_field = FEECVariable(space=b_space)
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self, mass_number: float = 1.0):
            self.velocity = FEECVariable(space="Hdiv")
            self.init_variables(mass_number=mass_number)

    class Propagators:
        def __init__(self, b_space: str = "Hdiv", projected: bool = False) -> None:
            if b_space == "Hdiv":
                self.shear_alf = ShearAlfvenPropagator()
            else:
                self.shear_alf = ShearAlfvenCurlCurlPropagator(projected=projected)

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def post_allocate(self):
        # project background magnetic field (2-form or 1-form, same space as the magnetic field)
        if self._b_space == "Hdiv":
            self._b_eq = Propagator.projected_equil.b2
            self._M_b = Propagator.mass_ops.M2
        else:
            self._b_eq = Propagator.projected_equil.b1
            self._M_b = Propagator.mass_ops.M1

        # temporary vectors for scalar quantities
        self._tmp_b1 = self._b_eq.space.zeros()
        self._tmp_b2 = self._b_eq.space.zeros()

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        b_space: OptsBSpace = "Hdiv",
        projected: bool = False,
    ):
        check_option(b_space, self.OptsBSpace)
        if projected and b_space != "Hcurl":
            raise ValueError('projected=True requires b_space="Hcurl".')
        self._b_space = b_space

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields(b_space=b_space)
        self.mhd = self.MHD(mass_number=mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(b_space=b_space, projected=projected)

        # 4. assign variables to propagators
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        background_magnetic = FunctionScalarFEEC(self._compute_en_B_eq)
        total_magnetic = FunctionScalarFEEC(self._compute_en_B_tot)
        self.scalars = Scalars(
            en_tot=kinetic_energy + magnetic_energy,
            en_U=kinetic_energy,
            en_B=magnetic_energy,
            en_B_eq=background_magnetic,
            en_B_tot=total_magnetic,
            en_tot2=kinetic_energy + magnetic_energy + background_magnetic,
        )

    def _compute_en_B_eq(self):
        self._M_b.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        return self._b_eq.inner(self._tmp_b1) / 2

    def _compute_en_B_tot(self):
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.em_fields.b_field.spline.vector
        self._M_b.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        return self._tmp_b1.inner(self._tmp_b2) / 2

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Momentum:

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0

        Induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0) = 0

        Here, :math:`\tilde{\mathbf U}` is the perturbed velocity, :math:`\tilde{\mathbf B}` the perturbed magnetic field,
        and :math:`\rho_0`, :math:`\mathbf B_0` are the mass density and magnetic field of a static MHD equilibrium.

        **Conserved quantities:**

        Perturbed energy (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` and :math:`\mathbf B_0 \cdot \mathbf n = 0`
        on the boundary, or periodic boundaries):

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \left( \frac{\rho_0}{2} |\tilde{\mathbf U}|^2 + \frac 12 |\tilde{\mathbf B}|^2 \right) \textnormal d \mathbf x = 0

        Divergence of the magnetic field:

        .. math::

            \frac{\partial}{\partial t} \nabla \cdot \tilde{\mathbf B} = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the Alfvén speed,

        .. math::

            \hat U = \hat v_A.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Perturbed kinetic energy: ``en_U``
        - Perturbed magnetic energy: ``en_B``
        - Background magnetic energy: ``en_B_eq``
        - Total magnetic energy including background: ``en_B_tot``
        - Perturbation total energy: ``en_tot``. Conserved by the implicit time step (up to the solver tolerances)
          with ``b_space="Hdiv"`` and with ``b_space="Hcurl", projected=True``; with ``b_space="Hcurl"``,
          ``projected=False`` it is not increased (numerical damping that vanishes with the resolution along the
          background field).
        - Total energy including background contribution: ``en_tot2``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator` (``b_space="Hdiv"``) or
           :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`
           (``b_space="Hcurl"``)
        """
        doc = rf"""**1. ShearAlfvenPropagator** (``b_space="Hdiv"``):

{ShearAlfvenPropagator.__doc__}

**1. ShearAlfvenCurlCurlPropagator** (``b_space="Hcurl"``):

{ShearAlfvenCurlCurlPropagator.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""ShearAlfven is the incompressible magnetic-tension subsystem extracted
        from linear MHD. It is useful when one wants to isolate Alfvénic
        propagation from the full linear MHD model. Two discretizations are
        available through the flag ``b_space``. With ``"Hdiv"`` (default), the
        magnetic field is a 2-form, the induction equation holds strongly and the
        implicit step solves for the velocity. With ``"Hcurl"``, the magnetic
        field is a 1-form, the induction equation is weak and the implicit step
        solves a field-aligned curl-curl system for the magnetic field,
        preconditioned by the Hiptmair-Xu auxiliary space preconditioner, which
        allows large time steps. This step damps the waves numerically, with a
        damping that vanishes quickly with the resolution along the background
        field; ``projected=True`` gives the exactly energy conserving version, at
        the price of an inner mass matrix solve in every application of the Schur
        operator (and no effective curl-curl preconditioner). With
        ``b_space="Hcurl"``, essential boundary conditions in the de Rham
        sequence set the tangential components of the magnetic field to zero on
        that boundary (stronger than :math:`\mathbf B \cdot \mathbf n = 0`), so
        free boundary conditions are recommended for bounded domains."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a shear-Alfvén model, with the magnetic field in :math:`H(\textnormal{div})` or in
        :math:`H(\textnormal{curl})` (exactly energy conserving step):

        .. code-block:: python

            from struphy.models import ShearAlfven

            model = ShearAlfven()
            model.em_fields.b_field
            model.mhd.velocity

            model_cc = ShearAlfven(b_space="Hcurl", projected=True)
            model_cc.propagators.shear_alf.options = model_cc.propagators.shear_alf.Options(
                precond="MassMatrixPreconditioner"
            )
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - isolated shear-Alfvén wave propagation
        - large time steps with the curl-curl discretization (``b_space="Hcurl"``)
        - verification of the shear-Alfvén propagators
        - reduced linear-MHD studies without compressibility"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - magnetosonic or compressible wave physics
        - nonlinear MHD dynamics
        - kinetic particle coupling
        - resistive, viscous, or Hall-MHD effects
        - polar splines with the Hiptmair-Xu preconditioner (``b_space="Hcurl"``)"""
