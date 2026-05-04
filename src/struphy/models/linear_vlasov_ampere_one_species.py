import logging
from struphy.propagators.poisson_field_solve import PoissonFieldSolve

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarPIC, Scalars
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators import (
    propagators_coupling,
    propagators_fields,
    propagators_markers,
)
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel

logger = logging.getLogger("struphy")
rank = MPI.COMM_WORLD.Get_rank()

class LinearVlasovAmpereOneSpecies(StruphyModel):
    r"""Linearized Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
            & \frac{\partial \tilde{\mathbf E}}{\partial t} = - \frac{\alpha^2}{\varepsilon} \int_{\mathbb R^3} \mathbf{v} \tilde f\, \textrm d^3 \mathbf v \,,
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
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon}  \int_\Omega \int_{\mathbb{R}^3} \psi\, \tilde f \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \tilde{\mathbf{E}}(t=0) &= -\nabla \phi(t=0) \,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v} = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_markers.PushVinEfield`
    3. :class:`~struphy.propagators.propagators_coupling.EfieldWeights`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
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

        # 5. define scalars to be tracked during simulation
        electric_energy = BilinearEnergyFEEC(self.em_fields.e_field)
        particle_energy = FunctionScalarPIC(
            self._compute_en_w,
            self.kinetic_ions.var,
        )
        self.scalars = Scalars(
            en_E=electric_energy,
            en_w=particle_energy,
            en_tot=electric_energy + particle_energy,
        )

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = PoissonFieldSolve()
        self.initial_poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Linearized Ampère's law:

        .. math::

            \frac{\partial \tilde{\mathbf{E}}}{\partial t} = -\frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} \tilde{f} \, \textrm{d}^3 \mathbf{v}

        Linearized Vlasov equation:

        .. math::

            \frac{\partial \tilde{f}}{\partial t} + \mathbf{v} \cdot \nabla \tilde{f} + \frac{1}{\varepsilon} \left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right) \cdot \frac{\partial \tilde{f}}{\partial \mathbf{v}} = \frac{1}{v_{\text{th}}^2 \varepsilon} \tilde{\mathbf{E}} \cdot \mathbf{v} f_0

        where :math:`Z=-1` and :math:`A=1/1836` for electrons. The background distribution function :math:`f_0` is a uniform Maxwellian

        .. math::

            f_0 = \frac{n_0(\mathbf{x})}{\left( \sqrt{2 \pi} v_{\text{th}} \right)^3} \exp \left( - \frac{|\mathbf{v}|^2}{2 v_{\text{th}}^2} \right)

        and the background electric field has to verify the following compatibility condition between with background density

        .. math::

            \nabla_{\mathbf{x}} \ln (n_0(\mathbf{x})) = \frac{1}{v_{\text{th}}^2 \varepsilon} \mathbf{E}_0

        At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

        .. math::

            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \, \textrm{d} \mathbf{x}
            &= \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \psi \, \tilde{f} \, \text{d}^3 \mathbf{v} \, \textrm{d} \mathbf{x}
            \qquad \forall \ \psi \in H^1
            \\[2mm]
            \tilde{\mathbf{E}}(t=0) &= -\nabla \phi(t=0)

        Moreover, it is assumed that

        .. math::

            \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v} = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""The light speed defines the velocity scale:

        .. math::

            \hat v = c,\qquad \hat E = \hat B \hat v,\qquad \hat\phi = \hat E \hat x.

        The species parameters are :math:`\alpha=\hat\Omega_p/\hat\Omega_c` and
        :math:`\varepsilon=1/(\hat\Omega_c\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy: ``en_E``
        - Perturbation particle energy: ``en_w``
        - Total energy: ``en_tot``"""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. propagators_markers.PushEta:**

{propagators_markers.PushEta.__doc__}

**2. propagators_markers.PushVinEfield:**

{propagators_markers.PushVinEfield.__doc__}

**3. propagators_coupling.EfieldWeights:**

{propagators_coupling.EfieldWeights.__doc__}

**4. propagators_markers.PushVxB:**

{propagators_markers.PushVxB.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""LinearVlasovAmpereOneSpecies is the linearized delta-f counterpart of
        VlasovAmpereOneSpecies. It is intended for weakly perturbed kinetic
        problems around a prescribed Maxwellian equilibrium and is especially
        useful for linear instability or damping studies."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the linear Vlasov-Ampère model:

        .. code-block:: python

            from struphy.models import LinearVlasovAmpereOneSpecies

            model = LinearVlasovAmpereOneSpecies()
            model.em_fields.e_field
            model.em_fields.phi
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear delta-f kinetic instabilities
        - weakly perturbed Landau-like damping studies with electrostatic fields
        - verification of linearized field-particle coupling"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - strongly nonlinear departures from the chosen equilibrium
        - multi-species kinetic coupling
        - fully electromagnetic magnetic-field evolution
        - equilibria that are not compatible with the built-in Maxwellian assumptions"""

    def allocate_helpers(self, verbose: bool = False):
        """Solve initial Poisson equation.

        :meta private:
        """
        self._tmp = xp.empty(1, dtype=float)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info("\nINITIAL POISSON SOLVE:")

        # use control variate method
        particles = self.kinetic_ions.var.particles
        particles.update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [xp.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            Propagator.mass_ops,
            Propagator.domain.args_domain,
        )

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(Propagator.mass_ops)

        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

        self.initial_poisson.options.rho = charge_accum
        self.initial_poisson.options.rho_coeffs = alpha**2 / epsilon
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info("\nSolving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        Propagator.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            logger.info("... Done.")

    def _compute_en_w(self):
        particles = self.kinetic_ions.var.particles

        # evaluate f0
        if not hasattr(self, "_f0"):
            backgrounds = self.kinetic_ions.var.backgrounds
            if isinstance(backgrounds, list):
                self._f0 = backgrounds[0]
            else:
                self._f0 = backgrounds
            self._f0_values = xp.zeros(
                self.kinetic_ions.var.particles.markers.shape[0],
                dtype=float,
            )
            assert isinstance(self._f0, Maxwellian3D)

        self._f0_values[particles.valid_mks] = self._f0(*particles.phasespace_coords.T)

        # alpha^2 * v_th^2 / (2*N) * sum_p s_0 * w_p^2 / f_{0,p}
        alpha = self.kinetic_ions.equation_params.alpha
        vth = self._f0.maxw_params["vth1"][0]

        self._tmp[0] = (
            alpha**2
            * vth**2
            / (2 * particles.Np)
            * xp.dot(
                particles.weights**2,  # w_p^2
                particles.sampling_density / self._f0_values[particles.valid_mks],  # s_{0,p} / f_{0,p}
            )
        )
        return self._tmp[0]

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "maxwellian_1 + maxwellian_2" in line:
                    new_file += ["background = maxwellian_1\n"]
                elif "maxwellian_1pt =" in line:
                    new_file += ["maxwellian_1pt = maxwellians.Maxwellian3D(n=(0.0, perturbation))\n"]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
