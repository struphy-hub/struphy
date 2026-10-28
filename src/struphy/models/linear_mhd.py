import copy
from typing import Literal

from feectools.linalg.block import BlockVector

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, Scalars, VolumeFormEnergyFEEC
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator
from struphy.propagators.shear_alfven_propagator import ShearAlfvenPropagator
from struphy.utils.utils import check_option


class LinearMHD(StruphyModel):
    """Linear ideal MHD with zero-flow equilibrium for magnetohydrodynamic wave propagation.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the ion species (default: 1.0)
    b_space: {"Hdiv", "Hcurl"}
        Space of the magnetic field (default: "Hdiv"). ``"Hdiv"``: strong induction equation, implicit shear-Alfvén
        solve for the velocity (:class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`).
        ``"Hcurl"``: weak induction equation, implicit curl-curl solve for the magnetic field
        (:class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`).
    projected: bool
        Only with ``b_space="Hcurl"``: the curl-curl term of the implicit shear-Alfvén step contains the projection of
        the Lorentz force, which makes this step exactly energy conserving at the price of inner mass matrix solves
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
        def __init__(
            self,
            mass_number: float = 1.0,
        ):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.pressure = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    ## propagators

    class Propagators:
        def __init__(self, b_field: FEECVariable = None, b_space: str = "Hdiv", projected: bool = False):
            if b_space == "Hdiv":
                self.shear_alf = ShearAlfvenPropagator()
            else:
                self.shear_alf = ShearAlfvenCurlCurlPropagator(projected=projected)
            self.mag_sonic = Magnetosonic(b_field=b_field)

    ## abstract methods

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

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields(b_space=b_space)
        self.mhd = self.MHD(mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(b_field=self.em_fields.b_field, b_space=b_space, projected=projected)

        # 4. assign variables to propagators
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        self.propagators.mag_sonic.variables.n = self.mhd.density
        self.propagators.mag_sonic.variables.u = self.mhd.velocity
        self.propagators.mag_sonic.variables.p = self.mhd.pressure

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        # The volume integral is retained as a diagnostic of the mean pressure
        # perturbation.  It is linear and is not part of the linearized energy.
        pressure_energy = VolumeFormEnergyFEEC(self.mhd.pressure, normalization=1.0 / (5 / 3 - 1))
        thermal_energy = BilinearEnergyFEEC(
            self.mhd.pressure,
            bilinear_form_name="M3p_inv",
            normalization=1.0 / (5 / 3),
        )
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        background_pressure = FunctionScalarFEEC(self._compute_en_p_eq)
        background_magnetic = FunctionScalarFEEC(self._compute_en_B_eq)
        total_magnetic = FunctionScalarFEEC(self._compute_en_B_tot)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_p=pressure_energy,
            en_thermal=thermal_energy,
            en_B=magnetic_energy,
            en_p_eq=background_pressure,
            en_B_eq=background_magnetic,
            en_B_tot=total_magnetic,
            en_tot=kinetic_energy + thermal_energy + magnetic_energy,
        )

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def post_allocate(self):
        self._ones = Propagator.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        # background magnetic field and mass matrix in the space of the perturbation (2-form, or 1-form with
        # b_space="Hcurl")
        if self.em_fields.b_field.space == "Hcurl":
            self._b_eq = Propagator.projected_equil.b1
            self._M_b = Propagator.mass_ops.M1
        else:
            self._b_eq = Propagator.projected_equil.b2
            self._M_b = Propagator.mass_ops.M2

        self._tmp_b1: BlockVector = self._b_eq.space.zeros()
        self._tmp_b2: BlockVector = self._b_eq.space.zeros()

    def _compute_en_B_eq(self):
        self._M_b.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        return self._b_eq.inner(self._tmp_b1) / 2

    def _compute_en_p_eq(self):
        return Propagator.projected_equil.p3.inner(self._ones) / (5 / 3 - 1)

    def _compute_en_B_tot(self):
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.em_fields.b_field.spline.vector

        self._M_b.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        return self._tmp_b1.inner(self._tmp_b2) / 2

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity (mass conservation):

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        Momentum (Lorentz force):

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}

        Energy (adiabatic process):

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        Induction (Faraday's law):

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0) = 0.

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde p`, :math:`\tilde{\mathbf B}` are the perturbed
        density, velocity, pressure and magnetic field, and :math:`\rho_0`, :math:`p_0`, :math:`\mathbf B_0` those of a
        static MHD equilibrium; :math:`\gamma = 5/3`.

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` and :math:`\mathbf B_0 \cdot \mathbf n = 0`
        on the boundary, or periodic boundaries):

        Mass perturbation and divergence of the magnetic field:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\partial}{\partial t} \nabla \cdot \tilde{\mathbf B} = 0

        Perturbed energy, for an equilibrium with :math:`\nabla \times \mathbf B_0 = 0` and :math:`\nabla p_0 = 0`:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \left( \frac{\rho_0}{2} |\tilde{\mathbf U}|^2 + \frac 12 |\tilde{\mathbf B}|^2
            + \frac{\tilde p^2}{2 \gamma p_0} \right) \textnormal d \mathbf x = 0

        For general equilibria and perturbations generated by a plasma displacement :math:`\boldsymbol \xi`
        (:math:`\partial_t \boldsymbol \xi = \tilde{\mathbf U}`,
        :math:`\tilde{\mathbf B} = \nabla \times (\boldsymbol \xi \times \mathbf B_0)`,
        :math:`\tilde p = - \boldsymbol \xi \cdot \nabla p_0 - \gamma p_0 \nabla \cdot \boldsymbol \xi`, e.g. for a pure
        velocity perturbation at :math:`t = 0`), the conserved energy is the sum of the kinetic energy and the potential
        energy :math:`\delta W(\boldsymbol \xi)` (energy principle of ideal MHD).
        """

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the Alfvén velocity:

        .. math::

            \hat{U} = \hat{v}_\mathrm{A} = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}

        The model therefore uses the Alfvén velocity as its characteristic speed scale."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy (perturbation): :math:`E_U = \frac{1}{2} \int \rho_0 |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V`
        - Magnetic energy (perturbation): :math:`E_B = \frac{1}{2} \int \frac{|\tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`
        - Mean pressure perturbation: :math:`E_p = \int \frac{\tilde{p}}{\gamma - 1} \, \mathrm{d}V` (``en_p``). This is a linear diagnostic, not an energy.
        - Thermal perturbation energy: :math:`E_{\mathrm{thermal}} = \frac{1}{2}\int \frac{\tilde{p}^2}{\gamma p_0} \, \mathrm{d}V` (``en_thermal``), with :math:`\gamma = 5/3`
        - Total perturbed energy: :math:`E_{\mathrm{tot}} = E_U + E_B + E_{\mathrm{thermal}}` (``en_tot``). Its
          change by the shear-Alfvén step is zero (up to the solver tolerances) with ``b_space="Hdiv"`` and with
          ``b_space="Hcurl", projected=True``; with ``b_space="Hcurl"``, ``projected=False`` this step damps it
          numerically (the damping vanishes with the resolution along the background field).
        - Equilibrium magnetic energy: :math:`E_{B0} = \frac{1}{2} \int \frac{|\mathbf{B}_0|^2}{\mu_0} \, \mathrm{d}V`
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V`
        - Total magnetic energy: :math:`E_{B,\mathrm{tot}} = \frac{1}{2} \int \frac{|\mathbf{B}_0 + \tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator` (``b_space="Hdiv"``) or
           :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`
           (``b_space="Hcurl"``)
        2. :class:`~struphy.propagators.magnetosonic.Magnetosonic`"""
        doc = rf"""**1. ShearAlfvenPropagator** (``b_space="Hdiv"``):

{ShearAlfvenPropagator.__doc__}

**1. ShearAlfvenCurlCurlPropagator** (``b_space="Hcurl"``):

{ShearAlfvenCurlCurlPropagator.__doc__}

**2. Magnetosonic:**

{Magnetosonic.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This model simulates small-amplitude perturbations in a magnetized plasma with a static
        equilibrium magnetic field :math:`\mathbf{B}_0` and zero background flow. The model solves the linearized
        ideal magnetohydrodynamic equations, which couple fluid dynamics (density, velocity, pressure)
        with magnetic field evolution.

        The linear MHD system supports three wave families:

        - **Alfvén waves:** incompressible shear waves propagating along :math:`\mathbf{B}_0`
        - **Fast magnetosonic waves:** compressible waves with phase velocity above the Alfvén speed
        - **Slow magnetosonic waves:** compressible waves with phase velocity below the Alfvén speed

        All evolved quantities are perturbations around a stationary equilibrium:

        - :math:`\tilde{\rho}` - density perturbation
        - :math:`\tilde{\mathbf{U}}` - velocity perturbation
        - :math:`\tilde{p}` - pressure perturbation
        - :math:`\tilde{\mathbf{B}}` - magnetic field perturbation

        The corresponding equilibrium quantities are:

        - :math:`\rho_0` - background density
        - :math:`p_0` - background pressure
        - :math:`\mathbf{B}_0` - background magnetic field

        Two discretizations of the magnetic field are available through the flag ``b_space``. With ``"Hdiv"``
        (default), :math:`\tilde{\mathbf B}` is a 2-form and the induction equation holds strongly. With ``"Hcurl"``,
        :math:`\tilde{\mathbf B}` is a 1-form, the induction equation is weak and the implicit shear-Alfvén step solves a
        field-aligned curl-curl system for the magnetic field, preconditioned by Hiptmair-Xu, which allows large time
        steps; the force :math:`(\nabla \times \mathbf B_0) \times \tilde{\mathbf B}` of the magnetosonic step then
        enters through the mixed mass matrix :attr:`~struphy.feec.mass.WeightedMassOperators.M21J`. This shear-Alfvén
        step damps the waves numerically, with a damping that vanishes quickly with the resolution along the
        background field; ``projected=True`` gives the exactly energy conserving version, at the price of an inner mass
        matrix solve in every application of the Schur operator.

        Boundary conditions with ``b_space="Hcurl"``: essential boundary conditions in the de Rham sequence set the
        tangential components of the magnetic field to zero on that boundary, which is unphysical at a wall with
        :math:`\mathbf B_0 \cdot \mathbf n = 0`. Free boundary conditions are therefore recommended: they impose
        :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` weakly in the shear-Alfvén step and :math:`\tilde p = 0` weakly in
        the magnetosonic step (natural boundary condition of the pressure term)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a linear MHD model:

        .. code-block:: python

            from struphy.models import LinearMHD

            model = LinearMHD()

            # Access fields
            model.em_fields.b_field
            model.mhd.density
            model.mhd.velocity
            model.mhd.pressure

            # Access tracked scalar quantities
            model.scalars["en_U"]
            model.scalars["en_B"]
            model.scalars["en_p"]

            # magnetic field in H(curl), exactly energy conserving shear-Alfvén step
            model_cc = LinearMHD(b_space="Hcurl", projected=True)
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for studying linear wave propagation in a static magnetized plasma,
        including Alfvén and magnetosonic dynamics around a prescribed equilibrium.

        Typical use cases include:

        - verification of linear MHD wave dispersion and mode structure
        - perturbative studies around static equilibria
        - benchmark problems for ideal-MHD field-fluid coupling
        - large time steps with the curl-curl discretization (``b_space="Hcurl"``)"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear MHD dynamics with finite-amplitude perturbations
        - equilibria with non-zero background flow
        - dissipative effects such as resistivity or viscosity
        - kinetic or particle-field effects beyond the fluid MHD description
        - polar splines with the Hiptmair-Xu preconditioner (``b_space="Hcurl"``)"""
