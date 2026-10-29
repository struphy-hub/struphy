import logging

import cunumpy as xp
from maybempi import MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergyPIC
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.base import Propagator
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB

logger = logging.getLogger("struphy")


class Vlasov(StruphyModel):
    """Vlasov equation for a single species in a static background magnetic field.

    The models of the Vlasov family inherit from this model: they call ``super().__init__()`` and extend the hooks of
    :meth:`~struphy.models.base.StruphyModel._setup` (species, propagators, variable links, scalars).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    charge_number: int
        Charge number (in units of the positive elementary charge) of the species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the species (default: 1.0)
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## species

    class KineticIons(ParticleSpecies):
        """Kinetic species of the Vlasov and guiding-center models. Models with other markers override ``space``;
        the equation parameters ``alpha``, ``epsilon``, ``lam`` and ``kappa`` are passed if the model has them."""

        space = "Particles6D"

        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
            lam: float = None,
            kappa: float = None,
        ):
            self.var = PICVariable(space=self.space)
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
                lam=lam,
                kappa=kappa,
            )

    ## propagators

    class Propagators:
        """Propagators of the Vlasov equation: ``push_vxb`` (Lorentz force) and ``push_eta`` (streaming).

        Parameters
        ----------
        b2_var : FEECVariable, optional
            Magnetic field (2-form) added to the equilibrium field in ``push_vxb``, for models with an evolving
            magnetic field (default: None, the equilibrium field alone).
        """

        def __init__(self, b2_var: FEECVariable = None):
            self.push_vxb = PushVxB(b2_var=b2_var)
            self.push_eta = PushEta()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    kinetic_ions: KineticIons
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
    ):
        # 0. store input parameters (a model inheriting from Vlasov stores its own before calling super().__init__)
        self._store_params(locals())

        # 1.-5. species, units, propagators, variable links and scalars, see StruphyModel._setup
        self._setup(base_units)

    def _create_species(self):
        """:meta private:"""
        self.kinetic_ions = self.KineticIons(**self._particle_species_args())

    def _link_propagator_variables(self):
        """:meta private:"""
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.push_eta.variables.var = self.kinetic_ions.var

    def _create_scalars(self):
        """:meta private:"""
        en_f = KineticEnergyPIC(self.kinetic_ions.var)
        return {"en_f": en_f}

    ## initial Poisson solve and Gauss-law diagnostic, for the models with an electric field

    # Models of the Vlasov family whose electric field is evolved in time (Ampère's law) solve Poisson's equation once
    # at t=0, E(0) = -grad phi(0), such that Gauss' law holds initially. They need the species ``em_fields``
    # (variables ``e_field`` in Hcurl and ``phi`` in H1); Vlasov itself and the electrostatic models do not use this.
    # The initial Poisson solve is not a propagator of the time stepping; its options are set in the parameter file
    # with ``model.initial_poisson.options``.

    def _create_initial_poisson(self, rho_coeffs: float = None, delta_f_source: bool = True):
        """Create the initial Poisson solve as attribute ``initial_poisson`` (call in ``__init__``).

        Parameters
        ----------
        rho_coeffs : float, optional
            Coefficient of the charge density on the right-hand side (can be set later, before ``post_allocate``).
        delta_f_source : bool
            Whether to deposit the charge of :math:`f - f_0` with the delta-f weights, also for full-f markers
            (see :class:`~struphy.propagators.poisson_solve.PoissonSolve`).
        """
        particles_to_grid = ParticlesToGrid(
            self.kinetic_ions.var,
            "H1",
            charge_density_0form,
        )
        self.initial_poisson = PoissonSolve(
            rho=particles_to_grid,
            rho_coeffs=rho_coeffs,
            delta_f_source=delta_f_source,
        )
        self.initial_poisson.variables.phi = self.em_fields.phi

    def _solve_initial_poisson(self):
        """Allocate and solve the initial Poisson equation and set :math:`\\mathbf e = -\\mathbb G \\boldsymbol \\phi`
        (call in ``post_allocate``)."""
        logger.info("\nINITIAL POISSON SOLVE:")
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        logger.info("Solving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        Propagator.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        # The pushers evaluate e at the markers, including in the ghost cells: grad.dot does not fill them,
        # so without this sync the first time step pushes with a wrong field (and breaks energy conservation).
        self.em_fields.e_field.spline.vector.update_ghost_regions()
        logger.info("... Done.")

    def _setup_gauss_law(self):
        """Prepare the Gauss-law diagnostic :meth:`calculate_gauss_error` (call in ``post_allocate``, after
        :meth:`_solve_initial_poisson`)."""
        self._tmp_gauss = xp.empty(1, dtype=float)
        self.op = Propagator.derham.grad.T @ Propagator.mass_ops.M1

        # keep the AccumulatorVector built by the propagator
        self.charge_accum = self.initial_poisson.sources[0]

        # Full-f markers: the time stepping conserves the charge of the full-f weights, while the initial solve used
        # the delta-f weights. The Gauss-law diagnostic therefore deposits the full-f weights and subtracts the
        # constant offset alpha^2/eps (rho_full - rho_delta_f) at t=0 (the initial f0 deposit minus n0).
        # On restart, post_allocate runs before the markers are restored, so the offset belongs to freshly loaded
        # t=0 markers: consistent with the restored run only for reproducible loading (fixed seed, same decomposition).
        self._gauss_offset = None
        particles = self.kinetic_ions.var.particles
        if not particles.control_variate:
            self.charge_accum()
            rho_full = self.charge_accum.vectors[0].copy()
            particles.update_weights()
            self.charge_accum()
            particles.weights = particles.weights0.copy()
            rho_full -= self.charge_accum.vectors[0]
            self._gauss_offset = self.initial_poisson.coeffs[0] * rho_full

    def calculate_gauss_error(self):
        r"""Maximum norm of the weak Gauss-law residual

        .. math::

            \mathbb G^\top \mathbb M^1 \mathbf e + c_\rho \boldsymbol \rho\,,

        where :math:`c_\rho` is the charge coefficient of the initial Poisson solve and :math:`\boldsymbol \rho` is
        the charge density of :math:`f - f_0`: with the control variate deposited with the delta-f weights, as in the
        initial Poisson solve; for full-f markers deposited with the full-f weights (whose charge the time stepping
        conserves) minus the constant offset :math:`\boldsymbol \rho_\textnormal{full}(0) - \boldsymbol \rho_{\delta f}(0)`
        of the initial solve, which used the delta-f weights. Since :math:`\mathbf e = -\mathbb G \boldsymbol \phi` it
        vanishes at :math:`t=0`, up to the net charge (the kernel of the periodic Poisson problem)."""
        if not hasattr(self, "charge_accum"):
            raise RuntimeError(
                f"{type(self).__name__}: the Gauss-law diagnostic is not prepared, call _setup_gauss_law() in "
                "post_allocate (e.g. with measure_gauss_law=True)."
            )
        particles = self.kinetic_ions.var.particles
        if particles.control_variate:
            particles.update_weights()
        self.charge_accum()
        rho = self.charge_accum.vectors[0]

        e = self.em_fields.e_field.spline.vector
        residual = self.op.dot(e)
        residual += self.initial_poisson.coeffs[0] * rho
        if self._gauss_offset is not None:
            residual -= self._gauss_offset

        # maximum residual over the local MPI rank, then over all ranks of the domain decomposition
        # (all clones hold the same accumulated charge density)
        self._tmp_gauss[0] = xp.max(xp.abs(residual.toarray()))
        if Propagator.derham.comm is not None:
            Propagator.derham.comm.Allreduce(MPI.IN_PLACE, self._tmp_gauss, op=MPI.MAX)

        return self._tmp_gauss[0]

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        if not hasattr(self, "initial_poisson"):
            return params_path
        # models with an initial Poisson solve: its options go after those of the last propagator
        last_prop = list(self.propagators.__dict__)[-1]
        model_name = type(self).__name__
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if f"{last_prop}.Options" in line:
                    new_file += [line]
                    new_file += ["model.initial_poisson.options = model.initial_poisson.Options()\n"]
                elif "saving_params = " in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["saving_params = SavingParameters(binning_plots=(binplot,))\n\n"]
                elif "measure_gauss_law" in self.params and f"{model_name}()" in line:
                    new_file += [f"\nmodel = {model_name}(measure_gauss_law=True)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
        return params_path

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "cyclotron"

    def post_allocate(self):
        pass

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \left( \mathbf{v} \times \mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""The characteristic speed is the cyclotron scale

        .. math::

            \hat v = \hat\Omega_c \hat x.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Particle kinetic energy: ``en_f``"""

    @classmethod
    def doc_long_description(cls):
        r"""Vlasov is the simplest kinetic test-particle model in the 6D hierarchy.
        It evolves particles in a static magnetic background without electric or
        magnetic self-consistency."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a Vlasov test-particle model:

        .. code-block:: python

            from struphy.models import Vlasov

            model = Vlasov()
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - test-particle motion in prescribed magnetic fields
        - verification of the Boris-like VxB and PushEta splitting
        - reduced kinetic transport studies without field feedback"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - self-consistent electrostatic or electromagnetic coupling
        - collisional kinetic dynamics
        - guiding-center reduction studies
        - fluid or MHD-scale closures"""
