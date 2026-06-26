import copy
import logging

from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, KineticEnergyPIC, LostMarkersPIC, Scalars, VolumeFormEnergyFEEC
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.propagators.pressure_coupling_6d import PressureCoupling6D
from struphy.propagators.push_eta_pc import PushEtaPC
from struphy.propagators.push_vxb import PushVxB
from struphy.propagators.shear_alfven_propagator import ShearAlfvenPropagator

logger = logging.getLogger("struphy")
rank = MPI.COMM_WORLD.Get_rank()


class LinearMHDVlasovPC(StruphyModel):
    r"""
    Hybrid linear MHD + energetic ions (6D Vlasov) with **pressure coupling scheme**.

    :ref:`normalization`:

    .. math::

        \hat U = \hat v =: \hat v_\textnormal{A, bulk} \,, \qquad
        \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,,\qquad 
        \hat{\mathbb{P}}_\textnormal{h} = A_\textnormal{h}m_\textnormal{H}\hat n \hat v_\textnormal{A}^2\,,

    Implemented equations:

    .. math::

        \begin{align}
        \textnormal{MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde{\rho}}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\
        \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p + \frac{A_\textnormal{h}}{A_\textnormal{b}} \nabla\cdot \tilde{\mathbb{P}}_{\textnormal{h},\perp}
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,, 
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        &\quad\,\,\frac{\partial f_\textnormal{h}}{\partial t} + (\mathbf{v} + \tilde{\mathbf{U}}_\perp)\cdot \nabla f_\textnormal{h}
        + \left[\frac{1}{\epsilon}\, \mathbf{v}\times(\mathbf{B}_0 + \tilde{\mathbf{B}}) - \nabla \tilde{\mathbf{U}}_\perp\cdot \mathbf{v} \right]\cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}}
        = 0\,,
        \\
        &\quad\,\,\tilde{\mathbb{P}}_{\textnormal{h},\perp} = \int \mathbf{v}_\perp\mathbf{v}^\top_\perp f_\textnormal{h} d\mathbf{v} \,,
        \end{aligned}
        \right.
        \end{align}

    where 

    .. math::

        \epsilon = \frac{\hat \omega}{2 \pi \, \hat \Omega_{\textnormal{c,hot}}} \,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c,hot}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.push_eta_pc.PushEtaPC`
    2. :class:`~struphy.propagators.push_vxb.PushVxB`
    3. :class:`~struphy.propagators.pressure_coupling_6d.PressureCoupling6D`
    4. :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`
    5. :class:`~struphy.propagators.magnetosonic.Magnetosonic`

    :ref:`Model info <add_model>`:
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
            self.var = PICVariable(space="Particles6D")
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
        def __init__(self, turn_off: tuple[str, ...] = (None,),
                     b2_var: FEECVariable = None,
                     b_field: FEECVariable = None,
                     u_tilde: FEECVariable = None,
                     ):
            if "PushEtaPC" not in turn_off:
                self.push_eta_pc = PushEtaPC(u_tilde=u_tilde)
            if "PushVxB" not in turn_off:
                self.push_vxb = PushVxB(b2_var=b2_var)
            if "PressureCoupling6D" not in turn_off:
                self.pc6d = PressureCoupling6D()
            if "ShearAlfven" not in turn_off:
                self.shearalfven = ShearAlfvenPropagator()
            if "Magnetosonic" not in turn_off:
                self.magnetosonic = Magnetosonic(b_field=b_field)

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
        self.propagators = self.Propagators(turn_off, 
                                            b2_var=self.em_fields.b_field,
                                            b_field=self.em_fields.b_field, 
                                            u_tilde=self.mhd.velocity,
                                            )

        # 4. assign variables to propagators
        if "ShearAlfven" not in turn_off:
            self.propagators.shearalfven.variables.u = self.mhd.velocity
            self.propagators.shearalfven.variables.b = self.em_fields.b_field
        if "Magnetosonic" not in turn_off:
            self.propagators.magnetosonic.variables.n = self.mhd.density
            self.propagators.magnetosonic.variables.u = self.mhd.velocity
            self.propagators.magnetosonic.variables.p = self.mhd.pressure
        if "PressureCoupling6D" not in turn_off:
            self.propagators.pc6d.variables.u = self.mhd.velocity
            self.propagators.pc6d.variables.energetic_ions = self.energetic_ions.var
        if "PushEtaPC" not in turn_off:
            self.propagators.push_eta_pc.variables.var = self.energetic_ions.var
        if "PushVxB" not in turn_off:
            self.propagators.push_vxb.variables.ions = self.energetic_ions.var

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        pressure_energy = VolumeFormEnergyFEEC(self.mhd.pressure, normalization=1.0 / (5 / 3 - 1))
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number
        particle_energy = KineticEnergyPIC(self.energetic_ions.var, normalization=Ah / Ab)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_p=pressure_energy,
            en_B=magnetic_energy,
            en_f=particle_energy,
            en_tot=kinetic_energy + pressure_energy + magnetic_energy + particle_energy,
            n_lost_particles=LostMarkersPIC(self.energetic_ions.var),
        )

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        MHD continuity:

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        MHD momentum:

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \nabla \cdot \tilde{\mathbb{P}}_{\textnormal{h},\perp} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + \mathbf{J}_0 \times \tilde{\mathbf{B}}

        .. math::

            \mathbf{J}_0 = \nabla \times \mathbf{B}_0

        MHD pressure:

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        MHD induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0) = 0

        Energetic-particle Vlasov equation:

        .. math::

            \frac{\partial f_\textnormal{h}}{\partial t} + (\mathbf{v} + \tilde{\mathbf{U}}_\perp) \cdot \nabla f_\textnormal{h} + \left[ \frac{1}{\epsilon} \mathbf{v} \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) - \nabla \tilde{\mathbf{U}}_\perp \cdot \mathbf{v} \right] \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} = 0

        Perpendicular pressure tensor:

        .. math::

            \tilde{\mathbb{P}}_{\textnormal{h},\perp} = \int \mathbf{v}_\perp \mathbf{v}_\perp^\top f_\textnormal{h} \, d \mathbf{v}

        """

    @classmethod
    def doc_normalization(cls):
        r"""Fluid and hot-particle velocities are normalized with the bulk Alfvén
        speed. The kinetic pressure tensor is scaled consistently with
        :math:`A_h m_H \hat n \hat v_A^2`, and the hot cyclotron parameter is
        :math:`\varepsilon`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - MHD kinetic energy: ``en_U``
        - Thermal pressure energy: ``en_p``
        - Magnetic energy: ``en_B``
        - Energetic-particle kinetic energy: ``en_f``
        - Total energy: ``en_tot``
        - Lost particles: ``n_lost_particles``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.push_eta_pc.PushEtaPC`
        2. :class:`~struphy.propagators.push_vxb.PushVxB`
        3. :class:`~struphy.propagators.pressure_coupling_6d.PressureCoupling6D`
        4. :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`
        5. :class:`~struphy.propagators.magnetosonic.Magnetosonic`
        """
        doc = rf"""**1. push_eta_pc.PushEtaPC:**

    {PushEtaPC.__doc__}

    **2. push_vxb.PushVxB:**

    {PushVxB.__doc__}

**3. pressure_coupling_6d.PressureCoupling6D:**

{PressureCoupling6D.__doc__}

**4. ShearAlfvenPropagator:**

{ShearAlfvenPropagator.__doc__}

**5. Magnetosonic:**

{Magnetosonic.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""LinearMHDVlasovPC is the pressure-coupling counterpart to the
        current-coupling hybrid model. It is targeted at linear problems where
        energetic-particle pressure anisotropy is the relevant feedback channel
        on the bulk MHD dynamics."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the linear MHD-Vlasov pressure-coupling model:

        .. code-block:: python

            from struphy.models import LinearMHDVlasovPC

            model = LinearMHDVlasovPC()
            model.em_fields.b_field
            model.mhd.velocity
            model.energetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear pressure-coupling hybrid studies
        - energetic-particle pressure feedback on MHD modes
        - verification of PushEtaPC and pressure-coupling operators"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - current-coupling closure studies
        - nonlinear hybrid turbulence
        - dissipative/resistive MHD
        - fully kinetic treatment of the bulk plasma"""

    def allocate_helpers(self):
        self._ones = Propagator.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0