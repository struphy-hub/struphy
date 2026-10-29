import logging

from maybempi import MPI

from struphy import BaseUnits
from struphy.models.scalars import FunctionScalarFEEC, FunctionScalarPIC
from struphy.models.vlasov import Vlasov
from struphy.propagators.base import Propagator
from struphy.propagators.push_v_in_force_field import PushVinForceField

logger = logging.getLogger("struphy")


class PauliMixin:
    """Scalars and parameter-file defaults shared by the models with classical Pauli particles
    (:class:`~struphy.models.VlasovPauli`, :class:`~struphy.models.VlasovPoissonAdiabaticPauli`)."""

    def _pauli_scalars(self, scalars: dict) -> dict:
        """Add the magnetic-moment energy ``en_mu`` (also to ``en_tot``) and ``mu_prime_max`` to the scalars of the
        parent model."""
        scalars.pop("en_tot", None)
        scalars["en_mu"] = FunctionScalarPIC(self._compute_en_mu, self.kinetic_ions.var)
        terms = [scalars[name] for name in ("en_phi", "en_f", "en_mu") if name in scalars]
        en_tot = terms[0]
        for term in terms[1:]:
            en_tot = en_tot + term
        scalars["en_tot"] = en_tot
        scalars["mu_prime_max"] = FunctionScalarFEEC(self._compute_mu_prime_max)
        return scalars

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

    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "BaseUnits(" in line:
                    new_file += ["base_units = BaseUnits(kBT=1.0)\n"]
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


class VlasovPauli(PauliMixin, Vlasov):
    r"""Guiding-center orbits in a static magnetic field, computed from classical Pauli particles.

    The particles are full-orbit (6D) markers with a magnetic moment (classical Pauli particles, Xiao and Qin 2021),
    loaded on the slow manifold without gyration. Extends :class:`~struphy.models.Vlasov` by the magnetic-moment
    (mirror) force :math:`-\mu \nabla |\mathbf B_0|`. The markers follow guiding-center orbits with full-orbit pushers,
    with time steps larger than the cyclotron period. With the self-consistent electrostatic potential and adiabatic
    electrons, see :class:`~struphy.models.VlasovPoissonAdiabaticPauli`.

    Parameters
    ----------
    base_units : BaseUnits
        Base units for normalization (default: ``BaseUnits(kBT=1.0)``).
    charge_number : int
        Charge number (in units of the positive elementary charge) of the species (default: 1).
    mass_number : float
        Mass number (in units of the proton mass) of the species (default: 1.0).
    epsilon : float, optional
        Normalized inverse cyclotron frequency :math:`1 / (\hat{\Omega}_\mathrm{c} \hat{t})`.
        If ``None``, computed from ``base_units`` and the charge/mass numbers.
    """

    ## changed species

    class KineticIons(Vlasov.KineticIons):
        space = "PauliParticles6D"

    ## additional propagators

    class Propagators(Vlasov.Propagators):
        """Propagators of :class:`~struphy.models.Vlasov` plus ``push_v_mu`` (magnetic-moment force).

        Parameters
        ----------
        **kwargs
            Passed to :class:`Vlasov.Propagators <struphy.models.Vlasov.Propagators>`.
        """

        # with Strang splitting the last propagator (push_vxb) makes one full step between the two half kicks,
        # i.e. the Boris scheme for push_vxb with algo="implicit"
        splitting_order = ("push_eta", "push_v_mu", "push_vxb")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.push_v_mu = PushVinForceField(magnetic_moment=True)

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    kinetic_ions: KineticIons
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see Vlasov
        super().__init__(base_units, charge_number, mass_number)

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.push_v_mu.variables.var = self.kinetic_ions.var

    def _create_scalars(self):
        """:meta private:"""
        return self._pauli_scalars(super()._create_scalars())

    @property
    def velocity_scale(self):
        return "thermal"

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation for classical Pauli particles in a static magnetic field :math:`\mathbf B_0`, with the
        magnetic moment :math:`\mu \geq 0` as a parameter of the distribution function
        :math:`f(\mathbf{x}, \mathbf{v}, \mu, t)`:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \left( \frac{1}{\varepsilon} \mathbf{v} \times \mathbf{B}_0 - \mu \nabla |\mathbf{B}_0| \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0\,,

        with :math:`f` concentrated on the slow manifold :math:`\mathbf v \times \mathbf b_0 \approx 0`.

        **Conserved quantities:** the number of particles and the energy

        .. math::

            \frac{\textrm d}{\textrm d t} \int_0^\infty \int_{\mathbb{R}^3} \int_\Omega \left( \frac{1}{2} |\mathbf v|^2 + \mu |\mathbf B_0| \right) f \, \textrm d \mathbf x \, \textrm d^3 \mathbf v \, \textrm d \mu = 0\,,

        for periodic boundary conditions or particles that do not leave the domain.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The reference speed is the thermal speed,

        .. math::

            \hat v = \sqrt{k_B \hat T / m}\,,\qquad \hat \mu = \frac{m \hat v^2}{\hat B}\,,

        with the dimensionless parameter :math:`\varepsilon = 1/(\hat\Omega_c\hat t)`. The normalized magnetic moment is
        :math:`\mu = v_\perp^2 / (2 |\mathbf B_0|)`, as in the guiding-center models."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation** (the sum of the energies is conserved by the model equations):

        - Kinetic energy: ``en_f`` :math:`= \frac{1}{2} \sum_p w_p |\mathbf{v}_p|^2`
        - Magnetic-moment energy: ``en_mu`` :math:`= \sum_p w_p \mu_p |\mathbf B_0(\mathbf x_p)|`
        - Total energy: ``en_tot`` = ``en_f + en_mu``
        - Distance from the slow manifold: ``mu_prime_max``
          :math:`= \max_p |\mathbf v_p \times \mathbf b_0|^2 / (2 |\mathbf B_0|)`, Eq. (7) of Xiao and Qin;
          it must stay small compared to the magnetic moments."""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence).
        With Strang splitting, a time step is a drift by dt/2, a kick by dt/2 with the magnetic-moment force
        -mu grad |B0|, the rotation by v x B0 over the full dt, a second kick by dt/2 and a drift by dt/2.
        With ``PushVxB(algo="implicit")`` (Crank-Nicolson, i.e. the Cayley rotation) this is the Boris scheme of the
        classical Pauli particle (BAP2 in Xiao and Qin 2021), which gives the grad-B and curvature drifts for any
        dt / epsilon. The exact rotation (``algo="analytic"``) and Lie-Trotter splitting do not; they need dt < epsilon.

        The markers are loaded with :class:`~struphy.pic.particles.PauliParticles6D`: the magnetic moment of each
        marker is computed from its drawn perpendicular velocity, which is then set to zero."""
        return cls.propagators_doc()

    @classmethod
    def doc_long_description(cls):
        r"""Guiding-center orbits computed with the slow manifold of the classical Pauli particle
        (J. Xiao and H. Qin, "Slow manifolds of classical Pauli particle enable structure-preserving geometric
        algorithms for guiding center dynamics", Comput. Phys. Commun. 265, 107981 (2021)).
        A classical Pauli particle is a charged particle with an intrinsic magnetic moment :math:`\mu`, with the
        Lagrangian :math:`L = \frac{1}{2}|\dot{\mathbf x}|^2 + \dot{\mathbf x} \cdot \mathbf A - \mu B`.
        When its perpendicular velocity vanishes initially, its effective magnetic moment
        :math:`\mu' = |\dot{\mathbf x} \times \mathbf b|^2 / (2B)` is an adiabatic invariant and stays close to zero:
        the particle moves on a slow manifold, along which its position follows the guiding-center dynamics of
        a particle with magnetic moment :math:`\mu`. Unlike the guiding-center Lagrangian, the Lagrangian of the
        classical Pauli particle is regular, so the full-orbit Boris scheme can be used with time steps larger
        than the cyclotron period."""

    @classmethod
    def doc_examples(cls):
        r"""Guiding-center orbits in the equilibrium magnetic field:

        .. code-block:: python

            from struphy.models import VlasovPauli

            model = VlasovPauli(epsilon=0.01)
            model.kinetic_ions.var

            # Boris rotation (Cayley transform) for time steps larger than the cyclotron period
            model.propagators.push_vxb.options = model.propagators.push_vxb.Options(algo="implicit")
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - guiding-center orbits (banana orbits, grad-B and curvature drifts) in general magnetic fields, with a
          full-orbit pusher and time steps larger than the cyclotron period
        - verification of guiding-center pushers"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - self-consistent fields (use :class:`~struphy.models.VlasovPoissonAdiabaticPauli` for the electrostatic case)
        - finite-Larmor-radius effects (use :class:`~struphy.models.Vlasov`)
        - magnetic fields without scale separation, where the magnetic moment is not an adiabatic invariant
        - time-dependent magnetic fields"""
