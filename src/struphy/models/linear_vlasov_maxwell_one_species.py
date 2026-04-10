from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.linear_vlasov_ampere_one_species import LinearVlasovAmpereOneSpecies
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalar, Scalars
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators import (
    propagators_coupling,
    propagators_fields,
    propagators_markers,
)

rank = MPI.COMM_WORLD.Get_rank()


class LinearVlasovMaxwellOneSpecies(LinearVlasovAmpereOneSpecies):
    r"""Linearized Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
            & \frac{\partial \tilde{\mathbf E}}{\partial t} = \nabla \times \tilde{\mathbf B} - \frac{\alpha^2}{\varepsilon} \int_{\mathbb R^3}\mathbf{v} \tilde f\, \textrm d^3 \mathbf v \,,
            \\[2mm]
            & \frac{\partial \tilde{\mathbf B}}{\partial t} = - \nabla \times \tilde{\mathbf E} \,,
            \\[2mm]
            & \frac{\partial \tilde f}{\partial t} + \mathbf{v} \cdot \, \nabla \tilde f + \frac{1}{\varepsilon} \left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial \tilde f}{\partial \mathbf{v}} = \frac{1}{v_{\text{th}}^2 \varepsilon} \, \tilde{\mathbf E} \cdot \mathbf{v} f_0 \,,
        \end{align}

    with the normalization parameter

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons. The background distribution function :math:`f_0` is a uniform Maxwellian

    .. math::

        f_0 = \frac{n_0(\mathbf{x})}{\left( \sqrt{2 \pi} v_{\text{th}} \right)^3}
        \exp \left( - \frac{|\mathbf{v}|^2}{2 v_{\text{th}}^2} \right) \,,

    and the background electric field has to verify the following compatibility condition between with background density

    .. math::

        \nabla_{\mathbf{x}} \ln (n_0(\mathbf{x})) = \frac{1}{v_{\text{th}}^2 \varepsilon} \mathbf{E}_0 \,.

    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \psi\, \tilde f \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \tilde{\mathbf{E}(t=0)} &= -\nabla \phi(t=0) \,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v} = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_markers.PushVinEfield`
    3. :class:`~struphy.propagators.propagators_coupling.EfieldWeights`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`
    5. :class:`~struphy.propagators.propagators_fields.Maxwell`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="DeltaFParticles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        def __init__(
            self,
            with_B0: bool = True,
            with_E0: bool = True,
        ):
            self.push_eta = propagators_markers.PushEta()
            if with_E0:
                self.push_vinE = propagators_markers.PushVinEfield()
            self.coupling_Eweights = propagators_coupling.EfieldWeights()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()
            self.maxwell = propagators_fields.Maxwell()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
        with_B0: bool = True,
        with_E0: bool = True,
    ):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons(
            charge_number,
            mass_number,
            alpha,
            epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0, with_E0=with_E0)

        # 4. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        if with_E0:
            self.propagators.push_vinE.variables.var = self.kinetic_ions.var
        self.propagators.coupling_Eweights.variables.e = self.em_fields.e_field
        self.propagators.coupling_Eweights.variables.ions = self.kinetic_ions.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        electric_energy = BilinearEnergyFEEC(self.em_fields.e_field, bilinear_form_name="M1", normalization=0.5)
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field, bilinear_form_name="M2", normalization=0.5)
        particle_energy = FunctionScalar(
            self._compute_en_w,
            self.kinetic_ions.var,
        )
        self.scalars = Scalars(
            en_E=electric_energy,
            en_B=magnetic_energy,
            en_w=particle_energy,
            en_tot=electric_energy + magnetic_energy + particle_energy,
        )

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi
