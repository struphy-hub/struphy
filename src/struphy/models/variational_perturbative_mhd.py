from struphy.io.options import BaseUnits
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC
from struphy.models.variables import FEECVariable
from struphy.models.variational_linear_mhd import VariationalLinearMHD
from struphy.propagators.base import Propagator
from struphy.propagators.utils import VelocityMass
from struphy.propagators.variational_linear_alfven_evolve import VariationalLinearAlfvenEvolve
from struphy.propagators.variational_linear_sound_evolve import VariationalLinearSoundEvolve
from struphy.propagators.variational_momentum_advection import VariationalMomentumAdvection
from struphy.propagators.variational_nonlinear_pb_evolve import VariationalNonlinearPBEvolve
from struphy.propagators.variational_resistivity import VariationalResistivity
from struphy.propagators.variational_viscosity import VariationalViscosity


class VariationalPerturbativeMHD(VariationalLinearMHD):
    """Perturbative (delta-f) visco-resistive MHD equations, with the pressure variable, discretized with a variational
    method.

    Extends :class:`~struphy.models.variational_linear_mhd.VariationalLinearMHD` by the terms that are nonlinear in
    the perturbations: the advection of the momentum and the transport of the magnetic field and of the pressure with the
    perturbations as weights.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
    with_viscosity: bool
        Whether to include viscous dissipation (default: True)
    with_resistivity: bool
        Whether to include resistive dissipation (default: True)
    u_space: str
        Space of the velocity, ``"H1vec"`` (vector field) or ``"Hdiv"`` (2-form) (default: ``"H1vec"``)
    """

    ## propagators

    class Propagators(VariationalLinearMHD.Propagators):
        """Propagators of :class:`~struphy.models.variational_linear_mhd.VariationalLinearMHD`, with the velocity mass
        matrix of the total density and the sound step with the total pressure as transport weight
        (``total_weight=True``, which closes the nonlinear sound coupling inside one implicit step), plus
        ``variat_mom`` (advection of the momentum) and ``variat_nonlinear`` (the nonlinear terms of the magnetic
        field). The linear steps and the nonlinear one are integrated symmetrically (``strang_groups``), the nonlinear
        one in the middle.

        Parameters
        ----------
        rho : FEECVariable, optional
            The density perturbation, for the velocity mass matrix of the total density.
        with_viscosity, with_resistivity : bool
            Whether to include the dissipative propagators.
        **kwargs
            Passed to :class:`VariationalLinearMHD.Propagators <struphy.models.VariationalLinearMHD.Propagators>`.
        """

        strang_groups = (("variat_lin_alfven", "variat_lin_sound", "variat_nonlinear"),)

        def __init__(
            self,
            rho: FEECVariable = None,
            with_viscosity: bool = True,
            with_resistivity: bool = True,
            bt2: FEECVariable = None,
            div_u: FEECVariable = None,
            u2: FEECVariable = None,
            pt3: FEECVariable = None,
            **kwargs,
        ):
            super().__init__(
                rho=rho,
                with_viscosity=with_viscosity,
                with_resistivity=with_resistivity,
                bt2=bt2,
                div_u=div_u,
                u2=u2,
                pt3=pt3,
                **kwargs,
            )
            # the linear steps with the velocity mass matrix of the total density
            self.variat_lin_alfven = VariationalLinearAlfvenEvolve(bt2=bt2, rho=rho)
            self.variat_lin_sound = VariationalLinearSoundEvolve(
                div_u=div_u, u2=u2, pt3=pt3, rho=rho, total_weight=True
            )
            self.variat_mom = VariationalMomentumAdvection(rho=rho)
            self.variat_nonlinear = VariationalNonlinearPBEvolve(rho=rho)
            # the nonlinear model heats the pressure (full_p)
            if with_viscosity:
                self.variat_viscous = VariationalViscosity(rho=rho)
            if with_resistivity:
                self.variat_resist = VariationalResistivity(rho=rho)
            order = ("variat_dens", "variat_mom", "variat_lin_alfven", "variat_lin_sound", "variat_nonlinear")
            order += tuple(n for n in ("variat_viscous", "variat_resist") if hasattr(self, n))
            self.splitting_order = order

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        with_viscosity: bool = True,
        with_resistivity: bool = True,
        u_space: str = "H1vec",
    ):
        # 0. store input parameters (before super().__init__, the first call wins)
        self._store_params(locals())
        super().__init__(
            base_units=base_units,
            mass_number=mass_number,
            with_viscosity=with_viscosity,
            with_resistivity=with_resistivity,
            u_space=u_space,
        )

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        props = self.propagators
        props.variat_mom.variables.u = self.mhd.velocity
        props.variat_nonlinear.variables.u = self.mhd.velocity
        props.variat_nonlinear.variables.b = self.em_fields.b_field

    def _kinetic_energy(self):
        """The kinetic energy with the mass matrix of the total density (rho_0 + rho)."""
        if self._u_space == "Hdiv":
            return FunctionScalarFEEC(self._compute_en_U_hdiv)
        return BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="WMMnew")

    @property
    def propagator_variants(self) -> dict:
        """The values of the option ``model`` of the variational propagators of this model."""
        return {
            "variat_dens": "perturbative",
            "variat_lin_sound": "linear_p",
            "variat_viscous": "full_p",
            "variat_resist": "full_p",
        }

    def post_allocate(self):
        """:meta private:"""
        super().post_allocate()
        # velocity in Hdiv: mass matrix of the total density for the kinetic energy
        if self._u_space == "Hdiv":
            self._velocity_mass = VelocityMass(
                Propagator.derham,
                Propagator.domain,
                Propagator.mass_ops,
                Propagator.projected_equil,
                rho_field=self.mhd.density,
            )
            self._tmp_M_u = Propagator.derham.V2pol.zeros()

    def _compute_en_U_hdiv(self):
        """Kinetic energy 1/2 u^T M^2_rho u with the total density rho_0 + rho (velocity in Hdiv)."""
        self._velocity_mass.update()
        u = self.mhd.velocity.spline.vector
        return 0.5 * u.inner(self._velocity_mass.M_rho.dot(u, out=self._tmp_M_u))

    def _edit_default_parameter_lines(self, lines: list) -> list:
        """As the linear model, without the density background, and with the degree >= 2 needed by the momentum
        advection of a velocity in Hdiv."""
        new_lines = []
        for line in super()._edit_default_parameter_lines(lines):
            if line.startswith("model.mhd.density.add_background"):
                continue
            if line.startswith("derham_opts = DerhamOptions()") and self._u_space == "Hdiv":
                # the bracket of the momentum advection with a velocity in Hdiv needs the degree p >= 2 (except in
                # periodic directions with one element)
                new_lines += [
                    "# velocity in Hdiv: the momentum advection needs degree >= 2 (except in periodic directions "
                    "with one element)\n",
                    "derham_opts = DerhamOptions(degree=(2, 2, 1))\n",
                ]
            else:
                new_lines += [line]
        return new_lines

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        The unknowns are the perturbations :math:`(\tilde\rho, \tilde p, \tilde{\mathbf B})` with respect to a
        zero-flow MHD equilibrium :math:`(\rho_0, p_0, \mathbf B_0)`, and the velocity :math:`\tilde{\mathbf u}`.

        Continuity:

        .. math::

            \partial_t \tilde{\rho} + \nabla \cdot \left( (\tilde{\rho} + \rho_0) \tilde{\mathbf{u}} \right) = 0

        Momentum:

        .. math::

            \partial_t \left( (\tilde{\rho} + \rho_0) \tilde{\mathbf{u}} \right) + \nabla \cdot \left( (\tilde{\rho} + \rho_0) \tilde{\mathbf{u}} \otimes \tilde{\mathbf{u}} \right) + \nabla \tilde{p} + \mathbf{B}_0 \times \nabla \times \tilde{\mathbf{B}} + \tilde{\mathbf{B}} \times \nabla \times \mathbf{B}_0 + \tilde{\mathbf{B}} \times \nabla \times \tilde{\mathbf{B}} - \nabla \cdot \left( (\mu + \mu_a(\mathbf{x})) \nabla \tilde{\mathbf{u}} \right) = 0

        Pressure:

        .. math::

            \partial_t \tilde{p} + \tilde{\mathbf{u}} \cdot \nabla (\tilde{p} + p_0) + \gamma (\tilde{p} + p_0) \nabla \cdot \tilde{\mathbf{u}} = (\gamma - 1) \left( (\mu + \mu_a(\mathbf{x})) |\nabla \tilde{\mathbf{u}}|^2 + (\eta + \eta_a(\mathbf{x})) |\nabla \times \tilde{\mathbf{B}}|^2 \right)

        Induction:

        .. math::

            \partial_t \tilde{\mathbf{B}} + \nabla \times \left( (\tilde{\mathbf{B}} + \mathbf{B}_0) \times \tilde{\mathbf{u}} \right) + \nabla \times (\eta + \eta_a(\mathbf{x})) \nabla \times \tilde{\mathbf{B}} = 0

        Here :math:`\mu_a(\mathbf{x})` and :math:`\eta_a(\mathbf{x})` are artificial viscosity and resistivity
        coefficients. These are the equations of
        :class:`~struphy.models.variational_linear_mhd.VariationalLinearMHD` plus the terms that are nonlinear in the
        perturbations.

        **Conserved quantities:** in the ideal case the energy

        .. math::

            \frac12 \int (\rho_0 + \tilde\rho) |\tilde{\mathbf u}|^2 + \frac12 \int |\tilde{\mathbf B}|^2
            + \int \mathbf B_0 \cdot \hat{\mathbf B} + \int \frac{\hat p}{\gamma - 1} \,,

        with the second-order magnetic field :math:`\hat{\mathbf B}` and pressure :math:`\hat p` of the linear model,
        for :math:`\tilde{\mathbf u} \cdot \mathbf n = 0` and :math:`\mathbf B_0 \cdot \mathbf n = 0` on the boundary
        (the energy of the full model without its terms linear in the perturbations, whose forces balance in the
        equilibrium), and the mass :math:`\int \tilde\rho`.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy (total density): ``en_U``
        - Thermal energies: ``en_thermo`` (of the second-order pressure), ``en_thermo_l1``
        - Magnetic energies: ``en_mag_1``, ``en_mag_2`` (second-order term), ``en_mag_l1``
        - Total energies: ``en_tot``, ``en_tot_l1``"""

    @classmethod
    def doc_long_description(cls):
        r"""VariationalPerturbativeMHD is the dissipative perturbative (delta-f) MHD model using the pressure as
        thermodynamic variable. It is intended for departures from an equilibrium that are not small enough for the
        nonlinearities to be negligible, but still naturally formulated as perturbations. The step is that of the
        linear model (:class:`~struphy.models.variational_linear_mhd.VariationalLinearMHD`, one linear solve per
        substep, with the velocity mass matrix of the total density) plus the advection of the momentum and the
        nonlinear terms of the magnetic field and the pressure (Picard iteration), integrated symmetrically; it
        reduces to the linear model at small amplitude by construction."""

    @classmethod
    def doc_examples(cls):
        r"""Create the perturbative model, with the velocity in :math:`(H^1)^3` or in :math:`H(\textnormal{div})`:

        .. code-block:: python

            from struphy.models import VariationalPerturbativeMHD

            model = VariationalPerturbativeMHD()
            model_hdiv = VariationalPerturbativeMHD(u_space="Hdiv")
            model.mhd.pressure
            model.em_fields.b_field
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - perturbative nonlinear MHD around a prescribed equilibrium (e.g. saturation of Alfvén eigenmodes)
        - comparing the nonlinear evolution with the linear model
          (:class:`~struphy.models.variational_linear_mhd.VariationalLinearMHD`)
        - perturbative nonlinear MHD in toroidal geometries with the velocity in :math:`H(\textnormal{div})`
          (``u_space="Hdiv"``), whose wall condition is only :math:`\mathbf u \cdot \mathbf n = 0`"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - full-f MHD evolution without a reference equilibrium
        - perturbations comparable to the equilibrium (the Picard iteration of the nonlinear step then needs small time
          steps)
        - kinetic or particle-resolved plasma effects
        - entropy- or q-based thermodynamic formulations
        - exact energy balance of the dissipation with polar splines (the heating of the pressure is exact only if the
          constants are in the space of 3-forms, which is not the case for polar splines)"""
