import copy
import logging

import cunumpy as xp
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
from struphy.propagators.current_coupling_6d_current import CurrentCoupling6DCurrent
from struphy.propagators.current_coupling_6d_density import CurrentCoupling6DDensity
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB
from struphy.propagators.shear_alfven_propagator import ShearAlfvenPropagator

logger = logging.getLogger("struphy")
rank = MPI.COMM_WORLD.Get_rank()


class LinearMHDVlasovCC(StruphyModel):
    """Hybrid linear MHD coupled with energetic ions (6D Vlasov) via the current-coupling scheme.

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
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Hybrid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.pressure = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

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

    ## propagators

    class Propagators:
        def __init__(
            self,
            b_field: FEECVariable = None,
            energetic_ions_var: PICVariable = None,
        ):
            self.couple_dens = CurrentCoupling6DDensity(energetic_ions=energetic_ions_var, b_tilde=b_field)
            self.shear_alf = ShearAlfvenPropagator()
            self.couple_curr = CurrentCoupling6DCurrent(b_tilde=b_field)
            self.push_eta = PushEta()
            self.push_vxb = PushVxB()
            self.mag_sonic = Magnetosonic(b_field=b_field)

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mhd_mass_number: float = 1.0,
        hot_charge_number: int = 1,
        hot_mass_number: float = 1.0,
        hot_epsilon: float = None,
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
            b_field=self.em_fields.b_field,
            energetic_ions_var=self.energetic_ions.var,
        )

        # 4. assign variables to propagators
        self.propagators.couple_dens.variables.u = self.mhd.velocity

        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        self.propagators.couple_curr.variables.ions = self.energetic_ions.var
        self.propagators.couple_curr.variables.u = self.mhd.velocity

        self.propagators.push_eta.variables.var = self.energetic_ions.var
        self.propagators.push_vxb.variables.ions = self.energetic_ions.var

        self.propagators.mag_sonic.variables.n = self.mhd.density
        self.propagators.mag_sonic.variables.u = self.mhd.velocity
        self.propagators.mag_sonic.variables.p = self.mhd.pressure

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        pressure_energy = VolumeFormEnergyFEEC(self.mhd.pressure, normalization=1.0 / (5 / 3 - 1))
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number
        particle_energy = KineticEnergyPIC(self.energetic_ions.var, normalization=Ah / Ab)
        lost_particles = LostMarkersPIC(self.energetic_ions.var)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_p=pressure_energy,
            en_B=magnetic_energy,
            en_f=particle_energy,
            en_tot=kinetic_energy + pressure_energy + magnetic_energy + particle_energy,
            n_lost_particles=lost_particles,
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

        self._tmp = xp.empty(1, dtype=float)

        # add control variate to mass_ops object
        if self.energetic_ions.var.particles.control_variate:
            Propagator.mass_ops.weights["f0"] = self.energetic_ions.var.particles.f0

        self._Ah = self.energetic_ions.mass_number
        self._Ab = self.mhd.mass_number

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "saving_params = " in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["saving_params = SavingParameters(binning_plots=(binplot,))\n\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        MHD continuity:

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        MHD momentum:

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + \mathbf{J}_0 \times \tilde{\mathbf{B}} \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \left( n_\textnormal{h} \tilde{\mathbf{U}} - n_\textnormal{h} \mathbf{u}_\textnormal{h} \right) \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{black}

        MHD pressure:

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + (\gamma - 1) \nabla \cdot (p_0 \tilde{\mathbf{U}}) + p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        MHD induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} = \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0), \qquad \nabla \cdot \tilde{\mathbf{B}} = 0

        Energetic-particle Vlasov equation:

        .. math::

            \frac{\partial f_\textnormal{h}}{\partial t} + \mathbf{v} \cdot \nabla f_\textnormal{h} + \frac{1}{\varepsilon} \left[ \color{blue} (\mathbf{B}_0 + \tilde{\mathbf{B}}) \times \tilde{\mathbf{U}} \color{black} + \mathbf{v} \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \right] \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} = 0

        Energetic-particle moments:

        .. math::

            n_\textnormal{h} = \int_{\mathbb{R}^3} f_\textnormal{h} \, \textnormal{d}^3 \mathbf{v}, \qquad n_\textnormal{h} \mathbf{u}_\textnormal{h} = \int_{\mathbb{R}^3} f_\textnormal{h} \mathbf{v} \, \textnormal{d}^3 \mathbf{v}

        where :math:`\mathbf{J}_0 = \nabla\times\mathbf{B}_0`.
        """

    @classmethod
    def doc_normalization(cls):
        r"""Both fluid and particle velocities are normalized with the Alfvén
        speed,

        .. math::

            \hat U = \hat v = \hat v_A,\qquad \hat f_h = \hat n / \hat v_A^3.

        The hot-species cyclotron parameter is :math:`\varepsilon =
        1/(\hat\Omega_{c,\mathrm{hot}}\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - MHD kinetic energy: ``en_U``
        - Thermal pressure energy: ``en_p``
        - Magnetic energy: ``en_B``
        - Energetic-particle energy: ``en_f``
        - Total energy: ``en_tot``
        - Lost particles: ``n_lost_particles``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.current_coupling_6d_density.CurrentCoupling6DDensity`
        2. :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`
        3. :class:`~struphy.propagators.current_coupling_6d_current.CurrentCoupling6DCurrent`
        4. :class:`~struphy.propagators.push_eta.PushEta`
        5. :class:`~struphy.propagators.push_vxb.PushVxB`
        6. :class:`~struphy.propagators.magnetosonic.Magnetosonic`
        """
        doc = rf"""**1. CurrentCoupling6DDensity:**

{CurrentCoupling6DDensity.__doc__}

**2. ShearAlfvenPropagator:**

{ShearAlfvenPropagator.__doc__}

**3. current_coupling_6d_current.CurrentCoupling6DCurrent:**

{CurrentCoupling6DCurrent.__doc__}

**4. push_eta.PushEta:**

{PushEta.__doc__}

**5. push_vxb.PushVxB:**

{PushVxB.__doc__}

**6. Magnetosonic:**

{Magnetosonic.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""LinearMHDVlasovCC is a current-coupling hybrid model for small
        perturbations around an MHD equilibrium. It is intended for energetic
        particle effects on linear MHD waves and instabilities without the cost
        of a fully kinetic background plasma."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the linear MHD-Vlasov current-coupling model:

        .. code-block:: python

            from struphy.models import LinearMHDVlasovCC

            model = LinearMHDVlasovCC()
            model.em_fields.b_field
            model.mhd.velocity
            model.energetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear energetic-particle coupling to MHD modes
        - current-coupling hybrid verification
        - reduced-cost studies of fast-ion effects on wave propagation"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - strongly nonlinear dynamics far from equilibrium
        - full multi-species kinetic plasma evolution
        - pressure-coupling closures
        - collisional or dissipative MHD effects not present in the equations"""
