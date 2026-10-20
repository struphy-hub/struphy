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
    loaded on the slow manifold without gyration. They follow the guiding-center dynamics of
    :class:`~struphy.models.DriftKineticElectrostaticAdiabatic` with the full-orbit pushers of
    :class:`~struphy.models.VlasovPoissonAdiabatic`, with time steps larger than the cyclotron period.

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
        ):
            # as in VlasovPoissonAdiabatic; with Strang splitting the last propagator (push_vxb) makes one full
            # step between the two half kicks, i.e. the Boris scheme for push_vxb with algo="implicit"
            self.push_eta = PushEta()
            self.poisson = AdiabaticPoissonSolve(rho=rho, epsilon=epsilon, alpha=alpha, Z=Z)
            self.push_v_E = PushVinForceField(potential=phi, magnetic_moment=True)
            self.push_vxb = PushVxB()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
    ):
        # the magnetic moment needs the background magnetic field
        self.with_B0 = True

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
        )

        # 4. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        self.propagators.poisson.variables.phi = self.em_fields.phi
        self.propagators.push_v_E.variables.var = self.kinetic_ions.var
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var

        # 5. define scalars to be tracked during simulation
        en_phi = FunctionScalarFEEC(self._compute_en_phi)
        en_particles = KineticEnergyPIC(self.kinetic_ions.var)
        en_mu = FunctionScalarPIC(self._compute_en_mu, self.kinetic_ions.var)
        self.scalars = Scalars(
            en_phi=en_phi,
            en_particles=en_particles,
            en_mu=en_mu,
            en_tot=en_phi + en_particles + en_mu,
            mu_prime_max=FunctionScalarFEEC(self._compute_mu_prime_max),
        )

    def _compute_en_mu(self):
        r"""Local part of the magnetic-moment energy :math:`\sum_p w_p \mu_p |\mathbf B_0(\mathbf x_p)|`."""
        particles = self.kinetic_ions.var.particles
        if particles.n_mks_loc == 0:
            return 0.0
        _, absB = particles._unit_b_and_absB(particles.positions)
        return float((particles.weights * particles.magnetic_moments * absB).sum())

    def _compute_mu_prime_max(self):
        r"""Maximum over all markers of :math:`\mu'_p = |\mathbf v_p \times \mathbf b_0|^2 / (2 |\mathbf B_0|)`."""
        mu_prime = self.kinetic_ions.var.particles.mu_prime()
        value = float(mu_prime.max()) if mu_prime.size > 0 else 0.0
        comm = Propagator.derham.comm
        if comm is not None and comm.Get_size() > 1:
            value = comm.allreduce(value, op=MPI.MAX)
        return value

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "if model.with_B0:" in line:
                    continue
                elif "push_vxb.Options" in line:
                    new_file += [
                        "model.propagators.push_vxb.options = model.propagators.push_vxb.Options(algo='implicit')\n"
                    ]
                elif "split_algo=" in line:
                    new_file += [line.replace("LieTrotter", "Strang")]
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
        is the guiding-center density (no gyro-average); the polarization density comes out of the marker orbits.
        As in :class:`~struphy.models.VlasovPoissonAdiabatic`, the vacuum term is dropped in the quasi-neutral limit
        (option ``quasi_neutral``) and the flux-surface average :math:`\langle \phi \rangle` is optional.
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

        - Field energy: ``en_phi``
          :math:`= \frac{1}{2\alpha^2} \int |\nabla \phi|^2 \, \mathrm d \mathbf x + \frac{1}{2 Z \varepsilon^2} \int \frac{n_0}{T_0} \phi (\phi - \langle \phi \rangle) \, \mathrm d \mathbf x`
          (the first term only if ``quasi_neutral=False``)
        - Kinetic energy: ``en_particles`` :math:`= \frac{1}{2} \sum_p w_p |\mathbf{v}_p|^2`
        - Magnetic-moment energy: ``en_mu`` :math:`= \sum_p w_p \mu_p |\mathbf B_0(\mathbf x_p)|`
        - Total energy: ``en_tot`` = ``en_phi + en_particles + en_mu``
          (of :math:`f - f_0` with the control variate)
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
        """
        doc = rf"""Time integration is performed by the following propagators (in sequence).
With Strang splitting, a time step is a drift by dt/2, the Poisson solve, a kick by dt/2 with the modified field
E† = -grad(phi + epsilon mu |B0|), the rotation by v x B0 over the full dt, a second kick by dt/2 and a drift by dt/2.
With ``PushVxB(algo="implicit")`` (Crank-Nicolson, i.e. the Cayley rotation) this is the Boris scheme of the
classical Pauli particle (BAP2 in Xiao and Qin 2021), which gives the E x B and grad-B drifts for any
dt / epsilon. The exact rotation (``algo="analytic"``) and Lie-Trotter splitting do not; they need dt < epsilon.

The markers are loaded with :class:`~struphy.pic.particles.PauliParticles6D`: the magnetic moment of each marker
is computed from its drawn perpendicular velocity, which is then set to zero.

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
        each marker is taken from its drawn perpendicular velocity (:class:`~struphy.pic.particles.PauliParticles6D`);
        the kinetic backgrounds must be gyrotropic Maxwellians (isotropic thermal velocities, no mean perpendicular
        velocity) given in the guiding-center position."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the model:

        .. code-block:: python

            from struphy.models import VlasovPoissonAdiabaticPauli

            model = VlasovPoissonAdiabaticPauli(epsilon=0.1)
            model.em_fields.phi
            model.kinetic_ions.var

            # Boris rotation (Cayley transform) for time steps larger than the cyclotron period
            model.propagators.push_vxb.options = model.propagators.push_vxb.Options(algo="implicit")
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electrostatic guiding-center ion dynamics with adiabatic electrons, with time steps larger than the
          cyclotron period (drift waves, slab ITG modes, ...)
        - guiding-center orbits (banana orbits, grad-B and curvature drifts) in general magnetic fields, with a
          full-orbit pusher
        - verification of guiding-center pushers"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - finite-Larmor-radius effects (use :class:`~struphy.models.VlasovPoissonAdiabatic`)
        - magnetic fields without scale separation, where the magnetic moment is not an adiabatic invariant
        - time-dependent magnetic fields and electromagnetic dynamics"""
