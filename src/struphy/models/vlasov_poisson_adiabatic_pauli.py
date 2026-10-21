import copy
import logging

from maybempi import MPI

from struphy import BaseUnits
from struphy.models.scalars import FunctionScalarFEEC, FunctionScalarPIC, KineticEnergyPIC, Scalars
from struphy.models.variables import FEECVariable, PICVariable
from struphy.models.vlasov_poisson_adiabatic import VlasovPoissonAdiabatic
from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.adiabatic_poisson_solve import AdiabaticPoissonSolve
from struphy.propagators.base import Propagator
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_v_in_force_field import PushVinForceField
from struphy.propagators.push_vxb import PushVxB

logger = logging.getLogger("struphy")


class VlasovPoissonAdiabaticPauli(VlasovPoissonAdiabatic):
    r"""Guiding-center ion dynamics with adiabatic electrons, computed from classical Pauli particles.

    The ions are full-orbit (6D) markers with a magnetic moment (classical Pauli particles, Xiao and Qin 2021),
    loaded on the slow manifold without gyration. They follow guiding-center orbits with the full-orbit pushers of
    :class:`~struphy.models.VlasovPoissonAdiabatic`, with time steps larger than the cyclotron period.
    With the control variate (delta-f), the polarization density must be added in the Poisson equation
    (``AdiabaticPoissonSolve.Options(polarization=True)``); without the Poisson solve (``with_poisson=False``)
    the model computes guiding-center orbits in the equilibrium magnetic field.

    Parameters
    ----------
    base_units : BaseUnits
        Base units for normalization (default: ``BaseUnits(kBT=1.0)``).
    charge_number : int
        Charge number (in units of the positive elementary charge) of the ion species (default: 1).
    mass_number : float
        Mass number (in units of the proton mass) of the ion species (default: 1.0).
    alpha : float, optional
        Ratio of plasma to cyclotron frequency :math:`\hat \omega_p / \hat \Omega_c`.
        If ``None``, computed from ``base_units`` and the charge/mass numbers.
    epsilon : float, optional
        Normalized inverse cyclotron frequency :math:`1 / (\hat{\Omega}_\mathrm{c} \hat{t})`.
        If ``None``, computed from ``base_units`` and the charge/mass numbers.
    with_poisson : bool
        Whether to solve the Poisson equation for the self-consistent potential (default: True);
        if False, the markers move in the equilibrium magnetic field only (orbit studies).
    """

    ## species

    class KineticIons(VlasovPoissonAdiabatic.KineticIons):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="PauliParticles6D")
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
            phi: FEECVariable = None,
            rho: ParticlesToGrid = None,
            epsilon: float = 1.0,
            alpha: float = 1.0,
            Z: int = 1,
            with_poisson: bool = True,
        ):
            # as in VlasovPoissonAdiabatic; with Strang splitting the last propagator (push_vxb) makes one full
            # step between the two half kicks, i.e. the Boris scheme for push_vxb with algo="implicit"
            self.push_eta = PushEta()
            if with_poisson:
                self.poisson = AdiabaticPoissonSolve(rho=rho, epsilon=epsilon, alpha=alpha, Z=Z)
            self.push_v_E = PushVinForceField(potential=phi if with_poisson else None, magnetic_moment=True)
            self.push_vxb = PushVxB()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
        with_poisson: bool = True,
    ):
        # the magnetic moment needs the background magnetic field
        self.with_B0 = True
        self.with_poisson = with_poisson

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

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
        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

        # 3. instantiate all propagators
        rho = ParticlesToGrid(
            self.kinetic_ions.var,
            "H1",
            charge_density_0form,
        )
        self.propagators = self.Propagators(
            phi=self.em_fields.phi,
            rho=rho,
            epsilon=epsilon,
            alpha=alpha,
            Z=charge_number,
            with_poisson=with_poisson,
        )

        # 4. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        if with_poisson:
            self.propagators.poisson.variables.phi = self.em_fields.phi
        self.propagators.push_v_E.variables.var = self.kinetic_ions.var
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var

        # 5. define scalars to be tracked during simulation
        en_particles = KineticEnergyPIC(self.kinetic_ions.var)
        en_mu = FunctionScalarPIC(self._compute_en_mu, self.kinetic_ions.var)
        mu_prime_max = FunctionScalarFEEC(self._compute_mu_prime_max)
        if with_poisson:
            en_phi = FunctionScalarFEEC(self._compute_en_phi)
            self.scalars = Scalars(
                en_phi=en_phi,
                en_particles=en_particles,
                en_mu=en_mu,
                en_tot=en_phi + en_particles + en_mu,
                mu_prime_max=mu_prime_max,
            )
        else:
            self.scalars = Scalars(
                en_particles=en_particles,
                en_mu=en_mu,
                en_tot=en_particles + en_mu,
                mu_prime_max=mu_prime_max,
            )

    def post_allocate(self):
        """Solve the initial Poisson equation (if ``with_poisson``) and allocate helper arrays.

        :meta private:
        """
        if self.with_poisson:
            super().post_allocate()
            control_variate = self.kinetic_ions.var.particles.control_variate
            polarization = self.propagators.poisson.options.polarization
            if control_variate and not polarization:
                logger.warning(
                    "VlasovPoissonAdiabaticPauli with the control variate but without the polarization term: the "
                    "control variate removes the polarization density, set poisson.Options(polarization=True)."
                )
            elif polarization and not control_variate:
                logger.warning(
                    "VlasovPoissonAdiabaticPauli with full-f weights and the polarization term: the full-f marker "
                    "density already contains the polarization density, which is then counted twice."
                )

    def _compute_en_phi(self):
        r"""Field energy of :class:`~struphy.models.VlasovPoissonAdiabatic`, plus the polarization energy
        :math:`\frac{1}{2} \int \frac{n_0}{|\mathbf B_0|^2} |\nabla_\perp \phi|^2` with ``polarization=True``."""
        en = super()._compute_en_phi()
        if self.propagators.poisson.options.polarization:
            e = Propagator.derham.grad.dot(self.em_fields.phi.spline.vector, out=self._grad_phi)
            en += 0.5 * Propagator.mass_ops.M1gyro.dot_inner(e, e)
        return en

    def _compute_en_mu(self):
        r"""Local part of the magnetic-moment energy :math:`\sum_p w_p \mu_p |\mathbf B_0(\mathbf x_p)|`."""
        particles = self.kinetic_ions.var.particles
        if particles.n_mks_loc == 0:
            return 0.0
        _, absB = particles._unit_b_and_absB(particles.positions)
        return float((particles.weights * particles.magnetic_moments * absB).sum())

    def _compute_mu_prime_max(self):
        r"""Maximum over all markers of :math:`\mu'_p = |\mathbf v_p \times \mathbf b_0|^2 / (2 |\mathbf B_0|)`."""
        particles = self.kinetic_ions.var.particles
        mu_prime = particles.mu_prime()
        value = float(mu_prime.max()) if mu_prime.size > 0 else 0.0
        # maximum within the clone and between clones
        comm = Propagator.derham.comm
        if comm is not None and comm.Get_size() > 1:
            value = comm.allreduce(value, op=MPI.MAX)
        if particles.clone_config is not None:
            value = particles.clone_config.inter_comm.allreduce(value, op=MPI.MAX)
        return value

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "if model.with_B0:" in line:
                    continue
                elif "poisson.Options" in line:
                    new_file += ["if model.with_poisson:\n"]
                    new_file += ["    " + line]
                elif "push_vxb.Options" in line:
                    new_file += [
                        "model.propagators.push_vxb.options = model.propagators.push_vxb.Options(algo='implicit')\n"
                    ]
                elif "time_opts = Time(" in line:
                    new_file += ["time_opts = Time(split_algo='Strang')\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
        return params_path

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation for classical Pauli particles, with the magnetic moment :math:`\mu \geq 0`
        as a parameter of the distribution function :math:`f(\mathbf{x}, \mathbf{v}, \mu, t)`:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \left[ \frac{1}{\varepsilon} \left( \mathbf{E}^\dagger + \mathbf{v} \times \mathbf{B}_0 \right) \right] \cdot \frac{\partial f}{\partial \mathbf{v}} = 0\,,
            \qquad \mathbf{E}^\dagger = -\nabla \left( \phi + \varepsilon \mu |\mathbf{B}_0| \right)\,,

        with :math:`f` concentrated on the slow manifold :math:`\mathbf v \times \mathbf b_0 \approx 0`.
        Poisson equation with adiabatic electrons:

        .. math::

            -\frac{\varepsilon}{\alpha^2} \Delta \phi + \frac{1}{Z \varepsilon} \frac{n_0}{T_0} \left( \phi - \langle \phi \rangle \right) = \int_0^\infty \int_{\mathbb{R}^3} f \, \mathrm{d}^3 \mathbf{v}\, \mathrm{d} \mu - n_0\,,

        where :math:`n_0` is the equilibrium ion density, :math:`T_0` the electron temperature and :math:`Z` the ion
        charge number. On the slow manifold the marker positions are guiding-center positions, so the charge density
        is the guiding-center density (no gyro-average). With full-f weights the polarization density comes out of
        the marker orbits (compression by the polarization drift). With the control variate it is removed from the
        deposited density (see the long description) and is added in the quasi-neutrality equation, as for guiding
        centers (option ``polarization`` of the Poisson solve):

        .. math::

            -\varepsilon \nabla \cdot \left( \frac{n_0}{|\mathbf B_0|^2} \nabla_\perp \phi \right) + \frac{1}{Z \varepsilon} \frac{n_0}{T_0} \left( \phi - \langle \phi \rangle \right) = \int_0^\infty \int_{\mathbb{R}^3} (f - f_0) \, \mathrm{d}^3 \mathbf{v}\, \mathrm{d} \mu\,.

        As in :class:`~struphy.models.VlasovPoissonAdiabatic`, the vacuum term is dropped in the quasi-neutral limit
        (option ``quasi_neutral``) and the flux-surface average :math:`\langle \phi \rangle` is optional.
        With ``with_poisson=False`` the Poisson equation is not solved and :math:`\phi = 0`.
        """

    @classmethod
    def doc_normalization(cls):
        r"""As in :class:`~struphy.models.VlasovPoissonAdiabatic`: the reference speed is the ion thermal speed,

        .. math::

            \hat v = \sqrt{k_B \hat T / m_i},\qquad \hat E = \hat v \hat B,\qquad \hat\phi = \hat E \hat x,\qquad \hat \mu = \frac{m_i \hat v^2}{\hat B}\,,

        with the dimensionless parameters :math:`\varepsilon = 1/(\hat\Omega_c\hat t)` and
        :math:`\alpha = \hat{\omega}_\mathrm{p}/\hat{\Omega}_\mathrm{c}`. The normalized magnetic moment is
        :math:`\mu = v_\perp^2 / (2 |\mathbf B_0|)`, as in the guiding-center models."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation** (the sum of the energies is conserved by the model equations):

        - Field energy (only if ``with_poisson``): ``en_phi``
          :math:`= \frac{1}{2\alpha^2} \int |\nabla \phi|^2 \, \mathrm d \mathbf x + \frac{1}{2 Z \varepsilon^2} \int \frac{n_0}{T_0} \phi (\phi - \langle \phi \rangle) \, \mathrm d \mathbf x`
          (the first term only if ``quasi_neutral=False``), plus the polarization energy
          :math:`\frac{1}{2} \int \frac{n_0}{|\mathbf B_0|^2} |\nabla_\perp \phi|^2 \, \mathrm d \mathbf x` with ``polarization=True``
        - Kinetic energy: ``en_particles`` :math:`= \frac{1}{2} \sum_p w_p |\mathbf{v}_p|^2`
        - Magnetic-moment energy: ``en_mu`` :math:`= \sum_p w_p \mu_p |\mathbf B_0(\mathbf x_p)|`
        - Total energy: ``en_tot`` = ``en_phi + en_particles + en_mu``; with the control variate it is conserved
          only approximately (the E x B kinetic energy of the background is represented by the polarization energy,
          that of the delta-f weights remains in ``en_particles``)
        - Distance from the slow manifold: ``mu_prime_max``
          :math:`= \max_p |\mathbf v_p \times \mathbf b_0|^2 / (2 |\mathbf B_0|)`, Eq. (7) of Xiao and Qin;
          it must stay small compared to the magnetic moments."""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.push_eta.PushEta`
        2. :class:`~struphy.propagators.adiabatic_poisson_solve.AdiabaticPoissonSolve`
        3. :class:`~struphy.propagators.push_v_in_force_field.PushVinForceField` (with ``magnetic_moment=True``)
        4. :class:`~struphy.propagators.push_vxb.PushVxB`

        Without the Poisson solve (``with_poisson=False``), propagator 2 is left out.
        """
        doc = rf"""Time integration is performed by the following propagators (in sequence).
With Strang splitting, a time step is a drift by dt/2, the Poisson solve, a kick by dt/2 with the modified field
E† = -grad(phi + epsilon mu |B0|), the rotation by v x B0 over the full dt, a second kick by dt/2 and a drift by dt/2.
With ``PushVxB(algo="implicit")`` (Crank-Nicolson, i.e. the Cayley rotation) this is the Boris scheme of the
classical Pauli particle (BAP2 in Xiao and Qin 2021), which gives the E x B and grad-B drifts for any
dt / epsilon. The exact rotation (``algo="analytic"``) and Lie-Trotter splitting do not; they need dt < epsilon.

The markers are loaded with :class:`~struphy.pic.particles.PauliParticles6D`: the magnetic moment of each marker
is computed from its drawn perpendicular velocity, which is then set to zero. Without the Poisson solve
(``with_poisson=False``), propagator 2 is left out and the kick has only the magnetic-moment force.

**1. PushEta:**

{PushEta.__doc__}

**2. AdiabaticPoissonSolve:**

{AdiabaticPoissonSolve.__doc__}

**3. PushVinForceField:**

{PushVinForceField.__doc__}

**4. PushVxB:**

{PushVxB.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""Guiding-center dynamics computed with the slow manifold of the classical Pauli particle
        (J. Xiao and H. Qin, "Slow manifolds of classical Pauli particle enable structure-preserving geometric
        algorithms for guiding center dynamics", Comput. Phys. Commun. 265, 107981 (2021)).
        A classical Pauli particle is a charged particle with an intrinsic magnetic moment :math:`\mu`, with the
        Lagrangian :math:`L = \frac{1}{2}|\dot{\mathbf x}|^2 + \dot{\mathbf x} \cdot \mathbf A - \phi - \mu B`.
        When its perpendicular velocity vanishes initially, its effective magnetic moment
        :math:`\mu' = |\dot{\mathbf x} \times \mathbf b|^2 / (2B)` is an adiabatic invariant and stays close to zero:
        the particle moves on a slow manifold, along which its position follows the guiding-center dynamics of
        a particle with magnetic moment :math:`\mu`. Unlike the guiding-center Lagrangian, the Lagrangian of the
        classical Pauli particle is regular, so the full-orbit Boris scheme can be used with time steps larger
        than the cyclotron period.

        The model is the counterpart of :class:`~struphy.models.VlasovPoissonAdiabatic` (full orbits, with
        finite-Larmor-radius effects) and :class:`~struphy.models.DriftKineticElectrostaticAdiabatic`
        (guiding centers with the polarization density in the Poisson equation): the ions have no gyration, hence
        no finite-Larmor-radius effects, and the polarization drift comes out of the orbits. The magnetic moment of
        each marker is taken from its drawn perpendicular velocity (:class:`~struphy.pic.particles.PauliParticles6D`),
        such that the markers represent the gyro-average of the initial condition, given in the guiding-center
        position.

        **Polarization and delta-f.** The polarization drift on the slow manifold is compressible, so the
        phase-space density of the markers is not constant along their orbits. With full-f weights this compression
        gives the polarization density. The control variate subtracts the background :math:`f_0` along the
        markers, which removes the compression part :math:`n_0 \nabla \cdot \boldsymbol \xi` of the polarization density
        (:math:`\boldsymbol \xi` the polarization displacement) from the deposited charge; the advection part
        :math:`\boldsymbol \xi \cdot \nabla n_0` remains in the weights. Without correction, a slab drift wave then has
        the frequency without polarization and becomes unstable at grid scale. With the guiding-center polarization
        term in the Poisson equation (``polarization=True``, the symmetric operator
        :math:`-\nabla \cdot (n_0 |\mathbf B_0|^{-2} \nabla_\perp \phi)` of the drift-kinetic model, which also
        contains the advection part), the delta-f slab drift wave has the drift-kinetic frequency, also with time
        steps of 8 and 16 cyclotron periods. A weak instability remains, which needs the density gradient and is
        independent of the time step: in the slab example :math:`\mu'` stays below :math:`5 \cdot 10^{-3}` up to
        t = 25 and grows by a factor of about 5 per 5 time units
        and the field energy grows from t = 25 on (about one wave period). A likely cause is that
        the compression part of the polarization is now instantaneous (field equation), while its advection part
        follows the marker orbits, including their cyclotron response. Removing only the advection part from the
        polarization term does not change it. Full-f simulations of small perturbations are dominated by the noise,
        since the perpendicular ion response is that of cold ions."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the model:

        .. code-block:: python

            from struphy.models import VlasovPoissonAdiabaticPauli

            model = VlasovPoissonAdiabaticPauli(epsilon=0.1)
            model.em_fields.phi
            model.kinetic_ions.var

            # with the control variate (delta-f): polarization density in the quasi-neutrality equation
            model.propagators.poisson.options = model.propagators.poisson.Options(polarization=True)

            # guiding-center orbits in the equilibrium magnetic field, without the Poisson solve
            model = VlasovPoissonAdiabaticPauli(epsilon=0.01, with_poisson=False)

            # Boris rotation (Cayley transform) for time steps larger than the cyclotron period
            model.propagators.push_vxb.options = model.propagators.push_vxb.Options(algo="implicit")
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electrostatic guiding-center ion dynamics with adiabatic electrons (delta-f with ``polarization=True``),
          with time steps larger than the cyclotron period (drift waves, slab ITG modes, ...)
        - guiding-center orbits (banana orbits, grad-B and curvature drifts) in general magnetic fields, with a
          full-orbit pusher and time steps larger than the cyclotron period (``with_poisson=False``)
        - verification of guiding-center pushers"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - small perturbations with full-f weights, which are dominated by the noise (use the control variate)
        - finite-Larmor-radius effects (use :class:`~struphy.models.VlasovPoissonAdiabatic`)
        - magnetic fields without scale separation, where the magnetic moment is not an adiabatic invariant
        - time-dependent magnetic fields and electromagnetic dynamics"""
