import copy
import logging

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import (
    BilinearEnergyFEEC,
    FunctionScalarPIC,
    KineticEnergyPIC,
    LostMarkersPIC,
    Scalars,
    VolumeFormEnergyFEEC,
)
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.propagators.current_coupling_5d_curlb import CurrentCoupling5DCurlb
from struphy.propagators.current_coupling_5d_density import CurrentCoupling5DDensity
from struphy.propagators.current_coupling_5d_gradb import CurrentCoupling5DGradB
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel
from struphy.propagators.shear_alfven_current_coupling_5d import ShearAlfvenCurrentCoupling5D

logger = logging.getLogger("struphy")
rank = MPI.COMM_WORLD.Get_rank()


class LinearMHDDriftkineticCC(StruphyModel):
    """Hybrid linear ideal MHD coupled with energetic ions (5D drift-kinetic) via the current-coupling scheme.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mhd_mass_number: float
        Mass number (in units of Proton mass) of the MHD bulk species (default: 1.0)
    hot_charge_number: int
        Charge number (in units of the positive elementary charge) of the energetic ion species (default: 1)
    hot_mass_number: float
        Mass number (in units of Proton mass) of the energetic ion species (default: 1.0)
    hot_epsilon: float, optional
        Normalized cyclotron period of the energetic ion species. If None, computed from units and charge/mass numbers.
    turn_off: tuple[str, ...]
        Names of coupling terms to turn off (default: (None,))
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Hybrid"

    ## species
    class EnergeticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.pressure = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.init_variables(mass_number=mass_number)

    ## propagators

    class Propagators:
        def __init__(
            self,
            turn_off: tuple[str, ...] = (None,),
            b_field: FEECVariable = None,
            energetic_ions_var: PICVariable = None,
        ):
            if "PushGuidingCenterBxEstar" not in turn_off:
                self.push_bxe = PushGuidingCenterBxEstar(b_tilde=b_field)
            if "PushGuidingCenterParallel" not in turn_off:
                self.push_parallel = PushGuidingCenterParallel(b_tilde=b_field)
            if "ShearAlfvenCurrentCoupling5D" not in turn_off:
                self.shearalfen_cc5d = ShearAlfvenCurrentCoupling5D(energetic_ions=energetic_ions_var)
            if "Magnetosonic" not in turn_off:
                self.magnetosonic = Magnetosonic(b_field=b_field)
            if "CurrentCoupling5DDensity" not in turn_off:
                self.cc5d_density = CurrentCoupling5DDensity(energetic_ions=energetic_ions_var, b_tilde=b_field)
            if "CurrentCoupling5DGradB" not in turn_off:
                self.cc5d_gradb = CurrentCoupling5DGradB(b_tilde=b_field)
            if "CurrentCoupling5DCurlb" not in turn_off:
                self.cc5d_curlb = CurrentCoupling5DCurlb(b_tilde=b_field)

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mhd_mass_number: float = 1.0,
        hot_charge_number: int = 1,
        hot_mass_number: float = 1.0,
        hot_epsilon: float = None,
        turn_off: tuple[str, ...] = (None,),
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(mhd_mass_number)
        self.energetic_ions = self.EnergeticIons(
            hot_charge_number,
            hot_mass_number,
            hot_epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(
            turn_off,
            b_field=self.em_fields.b_field,
            energetic_ions_var=self.energetic_ions.var,
        )

        # 4. assign variables to propagators
        if "ShearAlfvenCurrentCoupling5D" not in turn_off:
            self.propagators.shearalfen_cc5d.variables.u = self.mhd.velocity
            self.propagators.shearalfen_cc5d.variables.b = self.em_fields.b_field
        if "Magnetosonic" not in turn_off:
            self.propagators.magnetosonic.variables.n = self.mhd.density
            self.propagators.magnetosonic.variables.u = self.mhd.velocity
            self.propagators.magnetosonic.variables.p = self.mhd.pressure
        if "CurrentCoupling5DDensity" not in turn_off:
            self.propagators.cc5d_density.variables.u = self.mhd.velocity
        if "CurrentCoupling5DGradB" not in turn_off:
            self.propagators.cc5d_gradb.variables.u = self.mhd.velocity
            self.propagators.cc5d_gradb.variables.energetic_ions = self.energetic_ions.var
        if "CurrentCoupling5DCurlb" not in turn_off:
            self.propagators.cc5d_curlb.variables.u = self.mhd.velocity
            self.propagators.cc5d_curlb.variables.energetic_ions = self.energetic_ions.var
        if "PushGuidingCenterBxEstar" not in turn_off:
            self.propagators.push_bxe.variables.ions = self.energetic_ions.var
        if "PushGuidingCenterParallel" not in turn_off:
            self.propagators.push_parallel.variables.ions = self.energetic_ions.var

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        pressure_energy = VolumeFormEnergyFEEC(self.mhd.pressure, normalization=1.0 / (5 / 3 - 1))
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number
        particle_parallel = KineticEnergyPIC(self.energetic_ions.var, normalization=Ah / Ab)
        particle_magnetic = FunctionScalarPIC(self._compute_en_fB, self.energetic_ions.var)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_p=pressure_energy,
            en_B=magnetic_energy,
            en_fv=particle_parallel,
            en_fB=particle_magnetic,
            en_tot=kinetic_energy + pressure_energy + magnetic_energy + particle_parallel + particle_magnetic,
            n_lost_particles=LostMarkersPIC(self.energetic_ions.var),
        )

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        self._ones = Propagator.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._en_fv = xp.empty(1, dtype=float)
        self._en_fB = xp.empty(1, dtype=float)
        self._en_tot = xp.empty(1, dtype=float)

        self._PB = getattr(Propagator.basis_ops, "PB")
        self._PBb = self._PB.codomain.zeros()

    def _compute_en_fB(self):
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number
        particles = self.energetic_ions.var.particles
        self._PBb = self._PB.dot(self.em_fields.b_field.spline.vector)
        particles.save_magnetic_energy(self._PBb)

        return (
            particles.markers[~particles.holes, 5].dot(
                particles.markers[~particles.holes, 8],
            )
            * Ah
            / Ab
        )

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        MHD continuity:

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        MHD momentum:

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B} + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \left[ \frac{1}{\epsilon} n_\textnormal{gc} \tilde{\mathbf{U}} - \frac{1}{\epsilon} \mathbf{J}_\textnormal{gc} - \nabla \times \mathbf{M}_\textnormal{gc} \right] \times \mathbf{B}

        MHD pressure:

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        MHD induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}) = 0

        Energetic-particle drift-kinetic equation:

        .. math::

            \frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{B_\parallel^*} \left( v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^* \right) \cdot \nabla f_\textnormal{h} + \frac{1}{\epsilon} \frac{1}{B_\parallel^*} (\mathbf{B}^* \cdot \mathbf{E}^*) \frac{\partial f_\textnormal{h}}{\partial v_\parallel} = 0

        Energetic-particle moments:

        .. math::

            n_\textnormal{gc} = \int f_\textnormal{h} B_\parallel^* \, \textnormal{d} v_\parallel \textnormal{d} \mu

        .. math::

            \mathbf{J}_\textnormal{gc} = \int \frac{f_\textnormal{h}}{B_\parallel^*} \left( v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^* \right) \, \textnormal{d} v_\parallel \textnormal{d} \mu

        .. math::

            \mathbf{M}_\textnormal{gc} = -\int f_\textnormal{h} B_\parallel^* \mu \mathbf{b}_0 \, \textnormal{d} v_\parallel \textnormal{d} \mu

        where

        .. math::

            B^*_\parallel = \mathbf{b}_0 \cdot \mathbf{B}^*
            \\[2mm]
            \mathbf{B}^* &= \mathbf{B} + \epsilon v_\parallel \nabla \times \mathbf{b}_0
            \\[2mm]
            \mathbf{E}^* &= -\tilde{\mathbf{U}} \times \mathbf{B} - \epsilon \mu \nabla (\mathbf{b}_0 \cdot \mathbf{B})

        """

    @classmethod
    def doc_normalization(cls):
        r"""The bulk and energetic-particle flow scales are normalized with the
        bulk Alfvén speed, while the magnetic moment carries its own unit:

        .. math::

            \hat U = \hat v = \hat v_{A,\mathrm{bulk}},\qquad
            \hat\mu = A_h m_H \hat v_h^2 / \hat B.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - MHD kinetic energy: ``en_U``
        - Thermal pressure energy: ``en_p``
        - Magnetic energy: ``en_B``
        - Parallel energetic-particle energy: ``en_fv``
        - Magnetic-moment energetic-particle energy: ``en_fB``
        - Total energy: ``en_tot``
        - Lost particles: ``n_lost_particles``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.push_guiding_center_bx_estar.PushGuidingCenterBxEstar`
        2. :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel`
        3. :class:`~struphy.propagators.current_coupling_5d_gradb.CurrentCoupling5DGradB`
        4. :class:`~struphy.propagators.current_coupling_5d_curlb.CurrentCoupling5DCurlb`
        5. :class:`~struphy.propagators.current_coupling_5d_density.CurrentCoupling5DDensity`
        6. :class:`~struphy.propagators.shear_alfven_current_coupling_5d.ShearAlfvenCurrentCoupling5D`
        7. :class:`~struphy.propagators.magnetosonic.Magnetosonic`
        """
        doc = rf"""**1. push_guiding_center_bx_estar.PushGuidingCenterBxEstar:**

    {PushGuidingCenterBxEstar.__doc__}

    **2. push_guiding_center_parallel.PushGuidingCenterParallel:**

    {PushGuidingCenterParallel.__doc__}

**3. current_coupling_5d_gradb.CurrentCoupling5DGradB:**

{CurrentCoupling5DGradB.__doc__}

**4. current_coupling_5d_curlb.CurrentCoupling5DCurlb:**

{CurrentCoupling5DCurlb.__doc__}

**5. CurrentCoupling5DDensity:**

{CurrentCoupling5DDensity.__doc__}

**6. ShearAlfvenCurrentCoupling5D:**

{ShearAlfvenCurrentCoupling5D.__doc__}

**7. Magnetosonic:**

{Magnetosonic.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""LinearMHDDriftkineticCC is the reduced-kinetic hybrid current-coupling
        model for energetic ions. It is useful when gyrophase averaging is
        acceptable but energetic-particle feedback on linear MHD still has to be
        retained."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the linear MHD plus drift-kinetic CC model:

        .. code-block:: python

            from struphy.models import LinearMHDDriftkineticCC

            model = LinearMHDDriftkineticCC()
            model.em_fields.b_field
            model.mhd.velocity
            model.energetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear energetic-ion effects with guiding-center reduction
        - current-coupling hybrid mode studies in strong magnetic fields
        - verification of 5D hybrid coupling operators"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - full 6D energetic-particle dynamics
        - nonlinear hybrid turbulence
        - resistive or viscous MHD closures
        - regimes where gyrophase resolution is required"""
