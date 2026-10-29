import logging

import cunumpy as xp

from struphy import BaseUnits
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarPIC
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable
from struphy.models.vlasov import Vlasov
from struphy.propagators.efield_weights_coupling import EfieldWeightsCoupling
from struphy.propagators.push_v_in_force_field import PushVinForceField

logger = logging.getLogger("struphy")


class LinearVlasovAmpereOneSpecies(Vlasov):
    """Linearized Vlasov-Ampère equations for one kinetic species around a Maxwellian background.

    Extends :class:`~struphy.models.Vlasov`, written for the perturbation :math:`\tilde f` (delta-f markers), by the
    force of the background electric field and the coupling of the perturbed electric field to the background
    (linearized Ampère's law and the source term of the linearized Vlasov equation).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    charge_number: int
        Charge number (in units of the positive elementary charge) of the species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the species (default: 1.0)
    alpha: float, optional
        Dimensionless parameter: plasma frequency / cyclotron frequency. If None, computed from units and charge/mass numbers.
    epsilon: float, optional
        Normalized cyclotron period: 1 / (cyclotron frequency × time unit). If None, computed from units and charge/mass numbers.
    """

    ## additional species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    ## changed species

    class KineticIons(Vlasov.KineticIons):
        space = "DeltaFParticles6D"

    ## additional propagators

    class Propagators(Vlasov.Propagators):
        """Propagators of :class:`~struphy.models.Vlasov` plus ``push_vinE`` (force of the background electric field)
        and ``coupling_Eweights`` (coupling of the perturbed electric field and the delta-f weights).

        Parameters
        ----------
        **kwargs
            Passed to :class:`Vlasov.Propagators <struphy.models.Vlasov.Propagators>`.
        """

        splitting_order = ("push_eta", "push_vinE", "coupling_Eweights", "push_vxb")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.push_vinE = PushVinForceField()
            self.coupling_Eweights = EfieldWeightsCoupling()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    em_fields: EMFields
    kinetic_ions: KineticIons
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see Vlasov
        super().__init__(base_units, charge_number, mass_number)

        # initial Poisson (not a propagator used in time stepping)
        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon
        self._create_initial_poisson(rho_coeffs=alpha**2 / epsilon, delta_f_source=False)

    def _create_species(self):
        """:meta private:"""
        self.em_fields = self.EMFields()
        super()._create_species()

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.push_vinE.variables.var = self.kinetic_ions.var
        self.propagators.coupling_Eweights.variables.e = self.em_fields.e_field
        self.propagators.coupling_Eweights.variables.ions = self.kinetic_ions.var

    def _create_scalars(self):
        """:meta private:"""
        en_E = BilinearEnergyFEEC(self.em_fields.e_field)
        en_f = FunctionScalarPIC(self._compute_en_f, self.kinetic_ions.var)
        return {"en_E": en_E, "en_f": en_f, "en_tot": en_E + en_f}

    @property
    def velocity_scale(self):
        return "light"

    def post_allocate(self):
        """Solve initial Poisson equation.

        :meta private:
        """
        self._tmp = xp.empty(1, dtype=float)

        # No control variate here: the markers of a DeltaFParticles6D species already carry
        # delta_f alone (w = delta_f / s0 / Np, set by initialize_weights), so subtracting the
        # background f0 a second time -- which is what update_weights does -- would leave every
        # weight offset by the constant -f0 / s0 / Np. That constant is a net charge, and it makes
        # the periodic Poisson problem below singular.
        self._solve_initial_poisson()

    def _compute_en_f(self):
        particles = self.kinetic_ions.var.particles

        # evaluate f0
        if not hasattr(self, "_f0"):
            backgrounds = self.kinetic_ions.var.backgrounds
            if isinstance(backgrounds, list):
                self._f0 = backgrounds[0]
            else:
                self._f0 = backgrounds
            self._f0_values = xp.zeros(
                self.kinetic_ions.var.particles.n_markers_max,
                dtype=float,
            )
            assert isinstance(self._f0, Maxwellian3D)

        self._f0_values[particles.valid_mks] = self._f0(*particles.phasespace_coords)

        # alpha^2 * v_th^2 * N / 2 * sum_p s_0 * w_p^2 / f_{0,p}  (weights are w_p = delta f_p / (N * s_{0,p}))
        alpha = self.kinetic_ions.equation_params.alpha
        vth = self._f0.params["vth1"][0]

        self._tmp[0] = (
            alpha**2
            * vth**2
            * particles.Np
            / 2
            * xp.dot(
                particles.weights**2,  # w_p^2
                particles.sampling_density_values / self._f0_values[particles.valid_mks],  # s_{0,p} / f_{0,p}
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
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
        return params_path

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
        - Perturbation particle energy: ``en_f``
        - Total energy: ``en_tot``"""

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
