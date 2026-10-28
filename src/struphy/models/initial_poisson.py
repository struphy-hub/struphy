import logging

import cunumpy as xp
from maybempi import MPI

from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.base import Propagator
from struphy.propagators.poisson_solve import PoissonSolve

logger = logging.getLogger("struphy")


class InitialPoissonMixin:
    r"""Initial Poisson solve :math:`\mathbf E(t=0) = -\nabla \phi(t=0)` and Gauss-law diagnostic, shared by the models
    whose electric field is evolved in time (Ampère's law), such that Gauss' law holds only at :math:`t=0`.

    The model must have the species ``em_fields`` (with the variables ``e_field`` in ``Hcurl`` and ``phi`` in ``H1``)
    and ``kinetic_ions`` (with one PIC variable ``var``). The initial Poisson solve is not a propagator of the time
    stepping; its options are set in the parameter file with ``model.initial_poisson.options``.
    """

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

    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        # the options of the initial Poisson solve go after those of the last propagator
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
