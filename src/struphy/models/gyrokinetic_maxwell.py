import copy
import logging

import cunumpy as xp
from feectools.linalg.solvers import inverse
from maybempi import MPI

from struphy import BaseUnits
from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import (
    BilinearEnergyFEEC,
    FunctionScalarFEEC,
    FunctionScalarPIC,
    KineticEnergyPIC,
    LostMarkersPIC,
    Scalars,
)
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.kernels.gc_density_0form import gc_density_0form
from struphy.pic.accumulation.kernels.gk_density_gyro import gk_density_gyro
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.pic.diagnostics.kernels.gk_energy_gyro import gk_energy_gyro
from struphy.propagators.base import Propagator
from struphy.propagators.gyrokinetic_maxwell import (
    BackgroundWeights,
    GyroAverage,
    GyroDriftCoupling,
    GyroEparCoupling,
    GyroParallelPush,
    MaxwellGyroMagnetisation,
    PushGuidingCenterParallelFullB,
    SpeciesBackground,
    dielectric_matrix,
    magnetizing_matrix,
    parallel_flow_from_equilibrium,
)

logger = logging.getLogger("struphy")


class GyrokineticMaxwell(StruphyModel):
    """Gauge-invariant electromagnetic gyrokinetic Maxwell model of Remmerswaal, Hatzky and Sonnendrücker (2025):
    gyrocentre species (5D) coupled to Maxwell's equations with the polarization, magnetisation and the
    gyro-averaged currents of the gyrocentres.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    species: tuple[tuple[str, int, float]]
        The gyrocentre species as ``(name, charge_number, mass_number)``; the species are the attributes
        ``kinetic_<name>`` of the model (default: ions and electrons)
    with_drifts: bool
        Whether to include the perpendicular (E x B and grad-B) drifts and their currents
        (:class:`~struphy.propagators.gyrokinetic_maxwell.GyroDriftCoupling`), default: True
    kinetic_backgrounds: bool
        Whether the background moments of the field equations (polarization, parallel flow, pressure anisotropy)
        are taken from the kinetic backgrounds of the species (default), or the polarization from the fluid
        equilibrium density for every species (no flow, isotropic)
    background_current: bool
        Whether the equilibrium current :math:`\\nabla \\times \\mathbf B_0` enters Ampère's law, to be balanced by the
        gyrocentre currents of the backgrounds (see :meth:`parallel_flow`), default: False
    n_gyro: int
        Number of points of the gyro-average (ring) of the fields at the gyrocentres; 0 is the zero-Larmor-radius
        limit (default)
    n_disc: int
        Number of radial Gauss-Legendre points of the disc average (default: 2)
    solve_initial_poisson: bool
        Whether to solve the Gauss law with the dielectric matrix for the initial electric field (default: True)
    measure_gauss_law: bool
        Whether to track the Gauss-law error as a scalar quantity (default: False)
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class Gyrocentres(ParticleSpecies):
        def __init__(self, charge_number: int = 1, mass_number: float = 1.0):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(charge_number=charge_number, mass_number=mass_number)

    DEFAULT_SPECIES = (("ions", 1, 1.0), ("electrons", -1, 1.0 / 1836))

    ## propagators

    class Propagators:
        def __init__(
            self,
            species: dict[str, PICVariable],
            b_field: FEECVariable = None,
            background: BackgroundWeights = None,
            polarization=0.0,
            background_current: bool = False,
            gyro: GyroAverage = None,
            with_drifts: bool = True,
        ):
            variables = tuple(species.values())
            self.maxwell = MaxwellGyroMagnetisation(
                species=variables,
                background=background,
                polarization=None if background is not None else polarization,
                background_current=background_current,
                gyro=gyro,
            )
            for name in species:
                setattr(self, f"epar_{name}", GyroEparCoupling(b_tilde=b_field, polarization=polarization, gyro=gyro))
            for name in species:
                if gyro is not None and gyro.flr:
                    setattr(self, f"push_para_{name}", GyroParallelPush(b_tilde=b_field, gyro=gyro))
                else:
                    setattr(self, f"push_para_{name}", PushGuidingCenterParallelFullB(b_tilde=b_field))
            if with_drifts:
                self.drifts = GyroDriftCoupling(
                    species=variables, b_tilde=b_field, polarization=polarization, gyro=gyro
                )

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        species: tuple = DEFAULT_SPECIES,
        with_drifts: bool = True,
        kinetic_backgrounds: bool = True,
        background_current: bool = False,
        n_gyro: int = 0,
        n_disc: int = 2,
        solve_initial_poisson: bool = True,
        measure_gauss_law: bool = False,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())
        self.params["species"] = tuple(tuple(s) for s in species)

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self._species_names = []
        for name, charge_number, mass_number in species:
            setattr(self, f"kinetic_{name}", self.Gyrocentres(int(charge_number), float(mass_number)))
            self._species_names += [name]
        assert len(self._species_names) > 0, "At least one gyrocentre species is needed."

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        self._alpha2 = {
            name: self.gyrocentres(name).var.species.equation_params.alpha ** 2 for name in self._species_names
        }
        self._polarization_constant = float(sum(self._alpha2.values()))
        self.kinetic_backgrounds = kinetic_backgrounds
        self.background_current = background_current
        self.gyro = GyroAverage(n_gyro=n_gyro, n_disc=n_disc)
        self.solve_initial_poisson = solve_initial_poisson
        self.measure_gauss_law = measure_gauss_law

        # the background weights are built in post_allocate (they need the kinetic backgrounds and the equilibrium);
        # until then the propagators hold a placeholder that is resolved at allocation
        self._background = None
        self._polarization = _LazyPolarization(self)

        # 3. instantiate all propagators
        variables = {name: self.gyrocentres(name).var for name in self._species_names}
        self.with_drifts = with_drifts
        self._make_propagators(variables)

        # 4. assign variables to propagators
        self._assign_variables(variables)

        # 5. define scalars to be tracked during simulation
        scalars_dict = dict(self._field_energies())
        total = None
        for scalar in scalars_dict.values():
            total = scalar if total is None else total + scalar
        for name, var in variables.items():
            energy = KineticEnergyPIC(var, normalization=self._alpha2[name]) + FunctionScalarPIC(
                lambda name=name: self._compute_en_muB(name), var
            )
            scalars_dict[f"en_{name}"] = energy
            total = energy if total is None else total + energy
        scalars_dict["en_tot"] = total
        for name, var in variables.items():
            scalars_dict[f"n_lost_{name}"] = LostMarkersPIC(var)
        if measure_gauss_law:
            scalars_dict["gauss_error"] = FunctionScalarFEEC(self.calculate_gauss_error)
        self.scalars = Scalars(**scalars_dict)

    def _make_propagators(self, variables: dict[str, PICVariable]):
        """Instantiate the propagators (the field operators' backgrounds are resolved lazily at allocation)."""
        self.propagators = self.Propagators(
            species=variables,
            b_field=self.em_fields.b_field,
            background=_LazyBackground(self),
            polarization=self._polarization,
            background_current=self.background_current,
            gyro=self.gyro,
            with_drifts=self.with_drifts,
        )

    def _assign_variables(self, variables: dict[str, PICVariable]):
        """Assign the model variables to the propagators."""
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field
        for name, var in variables.items():
            epar = getattr(self.propagators, f"epar_{name}")
            epar.variables.e = self.em_fields.e_field
            epar.variables.var = var
            getattr(self.propagators, f"push_para_{name}").variables.ions = var
        if self.with_drifts:
            self.propagators.drifts.variables.e = self.em_fields.e_field

    def _field_energies(self) -> dict:
        """The field energy scalars (name -> Scalar)."""
        return {
            "en_E": FunctionScalarFEEC(self._compute_en_E),
            "en_B": FunctionScalarFEEC(self._compute_en_B),
        }

    @property
    def species_names(self) -> list[str]:
        """The names of the gyrocentre species (attributes ``kinetic_<name>``)."""
        return self._species_names

    def gyrocentres(self, name: str) -> ParticleSpecies:
        """The gyrocentre species ``kinetic_<name>``."""
        return getattr(self, f"kinetic_{name}")

    @property
    def bulk_species(self):
        return self.gyrocentres(self._species_names[0])

    @property
    def velocity_scale(self):
        return "light"

    @property
    def background(self) -> BackgroundWeights:
        """The background weights of the field operators (available after :meth:`post_allocate`)."""
        return self._background

    def species_background(self, name: str) -> SpeciesBackground:
        """The background moments of the species ``kinetic_<name>``."""
        return SpeciesBackground(self.gyrocentres(name).var, kinetic=self.kinetic_backgrounds)

    def parallel_flow(self, equil, name: str, fraction: float = 1.0):
        r"""The parallel flow of the species ``kinetic_<name>`` that carries the ``fraction`` of the equilibrium current
        :math:`\mathbf b_0 \cdot \nabla \times \mathbf B_0` (Eq. (5.62) of the paper, see
        :func:`~struphy.propagators.gyrokinetic_maxwell.parallel_flow_from_equilibrium`), to be used as
        ``u_para=(..., None)`` of the species' ``GyroMaxwellian2D`` with ``background_current=True``.

        The density of the flow is taken from the kinetic background added so far (or the fluid equilibrium if none)."""
        var = self.gyrocentres(name).var
        params = var.species.equation_params
        background = SpeciesBackground(var, kinetic=var.backgrounds is not None)

        def density(e1, e2, e3):
            return background.n(e1, e2, e3, equil=equil)

        return parallel_flow_from_equilibrium(equil, params.alpha, params.epsilon, density, fraction=fraction)

    def _setup_backgrounds(self):
        """Resolve the background weights of the field operators from the kinetic backgrounds and the equilibrium."""
        backgrounds = tuple(self.species_background(name) for name in self._species_names)
        self._background = BackgroundWeights(backgrounds, Propagator.projected_equil.equil)

    def post_allocate(self):
        for name in self._species_names:
            species = self.gyrocentres(name)
            if species.var.species.weights_params.control_variate:
                raise NotImplementedError(
                    f"{self.__class__.__name__} is a full-f model: the control variate would drop the "
                    "field-dependent drifts of f_0 (e.g. the grad-B current of the perturbation)."
                )

        # the field operators' backgrounds are resolved before the propagators are allocated (sim.allocate calls
        # post_allocate after the propagators' allocate; the lazy placeholders resolved them already)
        if self._background is None:
            self._setup_backgrounds()

        self._PB = getattr(Propagator.basis_ops, "PB")
        self._PBb = self._PB.codomain.zeros()
        self._E0T = Propagator.derham.extraction_ops["0"].transpose()
        self._B_tilde_par = Propagator.derham.V0.zeros()
        self._mass_ops = Propagator.mass_ops
        self._A = dielectric_matrix(self._mass_ops, self._background.polarization)
        aniso = self._background.anisotropy if self._background.has_anisotropy else None
        self._M2H = magnetizing_matrix(self._mass_ops, aniso)
        self._tmp = xp.empty(1, dtype=float)

        self._setup_gauss_law()

    def _setup_gauss_law_accumulators(self):
        """Charge accumulators of all species (gyro-averaged with finite Larmor radius)."""
        self._charge_accums = {}
        self._charge_coeffs = {}
        for name in self._species_names:
            var = self.gyrocentres(name).var
            params = var.species.equation_params
            self._charge_coeffs[name] = params.alpha**2 / params.epsilon
            if self.gyro.flr:
                kernel = gk_density_gyro
                pe = Propagator.projected_equil
                args = (
                    params.epsilon,
                    self.gyro.n_gyro,
                    self.gyro.ring_s,
                    self.gyro.ring_w,
                    pe.unit_b1[0]._data,
                    pe.unit_b1[1]._data,
                    pe.unit_b1[2]._data,
                    pe.absB0._data,
                )
            else:
                kernel, args = gc_density_0form, ()
            accum = AccumulatorVector(var.particles, "H1", kernel, self._mass_ops, Propagator.domain.args_domain)
            self._charge_accums[name] = (accum, args)
        grad = Propagator.derham.grad
        self._rho = grad.domain.zeros()
        self._gauss_rhs = grad.domain.zeros()

    def _setup_gauss_law(self):
        """Charge accumulators of all species, the weak Gauss operator and the initial Poisson solve."""
        grad = Propagator.derham.grad
        self._setup_gauss_law_accumulators()
        self._gauss_op = grad.T @ self._A
        self._residual = grad.domain.zeros()

        if self.solve_initial_poisson:
            self._initial_poisson()

    def _charge_density(self, out):
        """The free charge density sum_s alpha_s^2/eps_s rho_s (weak, in V1 ... as a V0 vector), into ``out``."""
        out *= 0.0
        for name, (accum, args) in self._charge_accums.items():
            accum(*args)
            self._rho *= 0.0
            self._rho += accum.vectors[0]
            self._rho *= self._charge_coeffs[name]
            out += self._rho
        return out

    def _initial_poisson(self):
        r"""Solve the weak Gauss law :math:`\mathbb G^\top \mathbb A \mathbb G \phi = \sum_s \frac{\alpha_s^2}{\varepsilon_s} \boldsymbol \rho_s`
        with the dielectric matrix and set :math:`\mathbf e = -\mathbb G \phi` (Eq. (5.44a) of the paper)."""
        logger.info("\nINITIAL POISSON SOLVE (Gauss's law with the dielectric matrix):")
        grad = Propagator.derham.grad
        rhs = self._charge_density(self._gauss_rhs)

        # the periodic problem needs a zero-mean right-hand side (the net charge is in the kernel of the Laplacian)
        ones = rhs.space.zeros()
        ones[:] = 1.0
        ones.update_ghost_regions()
        M0 = self._mass_ops.M0
        mean = rhs.inner(ones) / M0.dot_inner(ones, ones)
        rhs -= mean * M0.dot(ones)

        operator = grad.T @ self._A @ grad
        pc = preconditioner.StiffnessPreconditioner(self._mass_ops, "grad", mid=self._A)
        solver = inverse(operator, "pcg", pc=pc, tol=1e-12, maxiter=3000)
        phi = solver.solve(rhs)
        logger.info(f"... {solver._info}")

        e = self.em_fields.e_field.spline.vector
        grad.dot(-phi, out=e)
        e.update_ghost_regions()

    def calculate_gauss_error(self):
        r"""Maximum norm of the weak Gauss-law residual

        .. math::

            \mathbb G^\top \mathbb A \mathbf e + \sum_s \frac{\alpha_s^2}{\varepsilon_s} \boldsymbol \rho_s\,,

        where :math:`\boldsymbol \rho_s = \sum_p w_p \langle \vec \Lambda^0 \rangle_p` is the (gyro-averaged) gyrocentre
        density of species :math:`s`."""
        rhs = self._charge_density(self._gauss_rhs)
        e = self.em_fields.e_field.spline.vector
        residual = self._gauss_op.dot(e, out=self._residual)
        residual += rhs

        self._tmp[0] = xp.max(xp.abs(residual.toarray()))
        if Propagator.derham.comm is not None:
            Propagator.derham.comm.Allreduce(MPI.IN_PLACE, self._tmp, op=MPI.MAX)
        return self._tmp[0]

    def _compute_en_E(self):
        r"""Electric energy :math:`\frac12 \mathbf e^\top \mathbb A \mathbf e` with the dielectric matrix."""
        e = self.em_fields.e_field.spline.vector
        return 0.5 * self._A.dot_inner(e, e)

    def _compute_en_B(self):
        r"""Magnetic energy :math:`\frac12 \mathbf b^\top \mathbb M^2_H \mathbf b` with the magnetizing matrix."""
        b = self.em_fields.b_field.spline.vector
        return 0.5 * self._M2H.dot_inner(b, b)

    def _compute_en_muB(self, name: str):
        r"""Magnetic-moment energy :math:`\alpha_s^2 \sum_p w_p \mu_p B_\parallel(\boldsymbol \eta_p)` of one species,
        with :math:`B_\parallel = |B_0| + \langle\langle \Pi^0(\mathbf b_0 \cdot \tilde{\mathbf B}) \rangle\rangle`."""
        var = self.gyrocentres(name).var
        particles = var.particles
        PBb = self._PB.dot(self.em_fields.b_field.spline.vector, out=self._PBb)
        if self.gyro.flr:
            B_tilde_par = self._E0T.dot(PBb, out=self._B_tilde_par)
            B_tilde_par.update_ghost_regions()
            pe = Propagator.projected_equil
            gk_energy_gyro(
                particles.markers,
                Propagator.derham.args_derham,
                Propagator.domain.args_domain,
                particles.first_diagnostics_idx,
                particles.mu_idx,
                var.species.equation_params.epsilon,
                self.gyro.n_gyro,
                self.gyro.disc_s,
                self.gyro.disc_w,
                pe.unit_b1[0]._data,
                pe.unit_b1[1]._data,
                pe.unit_b1[2]._data,
                pe.absB0._data,
                B_tilde_par._data,
            )
        else:
            particles.save_magnetic_energy(PBb)
        valid = particles.valid_mks
        # The weights (column 5) already carry the 1/N_p of the Monte-Carlo estimate, see KineticEnergyPIC.
        energy = particles.markers[valid, 5].dot(particles.markers[valid, particles.first_diagnostics_idx])
        return self._alpha2[name] * energy

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                # b_field is the perturbation of the equilibrium field: no background
                if "model.em_fields.b_field.add_background(" in line:
                    continue
                new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

        return params_path

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Gyrocentre equations of motion (per species :math:`s`, transformation parameters :math:`(\xi_R, \xi_\Theta) = (1, 0)`):

        .. math::

            \dot{\mathbf R} = v_\parallel \frac{\mathbf B^*}{B^*_\parallel} + \frac{\mathbf E^* \times \mathbf b_0}{B^*_\parallel}\,,
            \qquad \dot v_\parallel = \frac{1}{\varepsilon_s} \frac{\mathbf B^* \cdot \mathbf E^*}{B^*_\parallel}\,,
            \qquad \dot \mu = 0\,,

        where

        .. math::

            \mathbf E^* = \langle \mathbf E \rangle - \varepsilon_s \mu \nabla \left(B_0 + \langle\langle \tilde B_\parallel \rangle\rangle\right)\,, \qquad
            \mathbf B^* = \mathbf B_0 + \langle \tilde{\mathbf B} \rangle + \varepsilon_s v_\parallel \nabla \times \mathbf b_0\,, \qquad
            B^*_\parallel = \mathbf b_0 \cdot \mathbf B^*\,, \qquad
            \tilde B_\parallel = \Pi^0(\mathbf b_0 \cdot \tilde{\mathbf B})\,,

        with the gyro-average :math:`\langle \cdot \rangle` and the disc average :math:`\langle\langle \cdot \rangle\rangle`
        (point evaluations in the zero-Larmor-radius limit ``n_gyro = 0``).

        Field equations (centred background with parallel flow :math:`u_{0,\parallel,s}` and pressures
        :math:`p_{0,\parallel}, p_{0,\perp}`, :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`):

        .. math::

            \left(\mathbf I + \sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2} \Pi_\perp\right) \frac{\partial \mathbf E}{\partial t}
            &= \nabla \times \left(\mathbf B - \frac{p_{0,\parallel} - p_{0,\perp}}{B_0^2} \mathbf B_\perp
            + \sum_s \alpha_s^2 \frac{n_{0,s} u_{0,\parallel,s}}{B_0^2} \mathbf b_0 \times \mathbf E
            + \sum_s \alpha_s^2 \int f_s\, \mu\, \mathbf b_0 \,\textrm d v_\parallel \textrm d \mu\right)
            \\
            &\quad + \sum_s \alpha_s^2 \frac{n_{0,s} u_{0,\parallel,s}}{B_0^2} \mathbf b_0 \times (\nabla \times \mathbf E)
            - \sum_s \frac{\alpha_s^2}{\varepsilon_s} \int f_s\, \dot{\mathbf R} \,\textrm d v_\parallel \textrm d \mu\,,
            \\[2mm]
            \frac{\partial \mathbf B}{\partial t} &= - \nabla \times \mathbf E\,,

        with the gyro-average adjoints in the currents. The left-hand side operator is the vacuum permittivity
        plus the polarization, the curl terms on the right are the curl of the magnetisation. Gauss's law
        :math:`\nabla \cdot \mathbf D = \rho_f` is solved for the initial electric field (``solve_initial_poisson``)
        and monitored (``measure_gauss_law``); :math:`\nabla \cdot \mathbf B = 0` holds exactly in the discrete de Rham complex.
        Since the model is gauge invariant, it is formulated in :math:`\mathbf E \in H(\textnormal{curl})` and
        :math:`\mathbf B \in H(\textnormal{div})` rather than in the potentials.

        Conserved energy:

        .. math::

            \mathcal H = \frac12 \mathbf e^\top \mathbb A \mathbf e + \frac12 \mathbf b^\top \mathbb M^2_H \mathbf b
            + \sum_s \alpha_s^2 \sum_p w_p \left(\frac{v_{\parallel,p}^2}{2} + \mu_p \left(B_0 + \langle\langle \tilde B_\parallel \rangle\rangle\right)(\boldsymbol \eta_p)\right)\,.

        Reference: R. Remmerswaal, R. Hatzky, E. Sonnendrücker, *Gauge-invariant variational formulations of
        electromagnetic gyrokinetic theory*, J. Plasma Phys. 91, E127 (2025), `doi:10.1017/S0022377825100688 <https://doi.org/10.1017/S0022377825100688>`_.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses the light speed as reference velocity:

        .. math::

            \hat v = c,\qquad \hat E = \hat B \hat v,\qquad \hat t = \hat x / c\,.

        The species parameters :math:`\alpha_s=\hat\Omega_{p,s}/\hat\Omega_{c,s}` and
        :math:`\varepsilon_s=1/(\hat\Omega_{c,s}\hat t)` are signed (negative for negative charges).
        The background moments :math:`n_{0,s}`, :math:`u_{0,\parallel,s}` and the pressures
        :math:`p_{\parallel,s} = \alpha_s^2 n_{0,s} v_{\textnormal{th},\parallel,s}^2`,
        :math:`p_{\perp,s} = \alpha_s^2 n_{0,s} v_{\textnormal{th},\perp,s}^2` are those of the kinetic backgrounds
        (``kinetic_backgrounds=True``), else :math:`n_{0,s} = n_0` of the fluid equilibrium."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy (with the dielectric matrix): ``en_E``
        - Magnetic field energy (with the magnetizing matrix): ``en_B``
        - Species energies :math:`\alpha_s^2 \sum_p w_p (v_\parallel^2/2 + \mu B_\parallel)`: ``en_<name>``
        - Total energy: ``en_tot``
        - Lost markers: ``n_lost_<name>``
        - Optional Gauss-law diagnostic: ``gauss_error``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.gyrokinetic_maxwell.MaxwellGyroMagnetisation`
        2. :class:`~struphy.propagators.gyrokinetic_maxwell.GyroEparCoupling` (one per species)
        3. :class:`~struphy.propagators.gyrokinetic_maxwell.PushGuidingCenterParallelFullB` (one per species, ZLR) or
           :class:`~struphy.propagators.gyrokinetic_maxwell.GyroParallelPush` (finite Larmor radius)
        4. :class:`~struphy.propagators.gyrokinetic_maxwell.GyroDriftCoupling` (all species, if ``with_drifts``)
        """
        doc = rf"""**1. MaxwellGyroMagnetisation:**

{MaxwellGyroMagnetisation.__doc__}

**2. GyroEparCoupling (one per species):**

{GyroEparCoupling.__doc__}

**3. PushGuidingCenterParallelFullB (one per species, zero Larmor radius):**

{PushGuidingCenterParallelFullB.__doc__}

**3'. GyroParallelPush (one per species, finite Larmor radius):**

{GyroParallelPush.__doc__}

**4. GyroDriftCoupling (all species):**

{GyroDriftCoupling.__doc__}

All field solves are implicit, so the light wave, the plasma oscillation and the compressional Alfvén wave
do not limit the time step. The common dielectric matrix is assembled once by
:func:`~struphy.propagators.gyrokinetic_maxwell.dielectric_matrix`; with finite Larmor radius the particle
couplings are applied matrix-free (:class:`~struphy.propagators.gyrokinetic_maxwell.GyroParticleOperator`).
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""GyrokineticMaxwell is the gauge-invariant electromagnetic gyrokinetic model of
        Remmerswaal, Hatzky and Sonnendrücker (J. Plasma Phys. 91, E127, 2025), Sec. 5 ("gyrokinetic Maxwell model"),
        with the transformation parameters :math:`(\xi_R, \xi_\Theta) = (1, 0)`.
        Any number of gyrocentre species is coupled to the electric and magnetic fields, with the polarization
        entering as a dielectric matrix, the pressure anisotropy and the parallel flow of the backgrounds as
        magnetizing and flow operators, and the gyrocentre magnetisation and the gyro-averaged currents as sources of
        Ampère's law. The equilibrium current can be carried by the species backgrounds (``background_current``),
        Gauss's law is solved initially and can be monitored, and finite Larmor radius effects enter through the
        gyro- and disc averages (``n_gyro``, ``n_disc``) with matrix-free particle couplings.
        It is full-f (no control variate) and uses tensor-product coefficients (no polar splines)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the model:

        .. code-block:: python

            from struphy.models import GyrokineticMaxwell

            model = GyrokineticMaxwell(species=(("ions", 1, 1.0), ("electrons", -1, 1 / 100)), n_gyro=4)
            model.em_fields.e_field
            model.em_fields.b_field
            model.kinetic_ions.var
            model.kinetic_electrons.var

        ``b_field`` is the perturbation of the equilibrium field (do not add a ``FieldsBackground`` to it).
        The loading parameters should match the thermal speeds in light units, e.g.
        ``LoadingParameters(moments=(0, 0, vth, vth), B0=...)``, and the kinetic backgrounds
        (``GyroMaxwellian2D``) provide the background density, flow and pressures of the field equations.
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electromagnetic gyrokinetic simulations with finite Larmor radius (shear, kinetic and compressional Alfvén waves)
        - energy-conserving coupling of several gyrocentre species to Maxwell's equations
        - backgrounds with parallel flow (equilibrium current) and pressure anisotropy
        - prototyping of the gauge-invariant gyrokinetic formulation in Struphy"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - delta-f (control variate) simulations
        - polar-spline (pole) domains
        - the quasi-neutral limit without the fast waves (see the Darwin variant,
          :class:`~struphy.models.gyrokinetic_darwin.GyrokineticDarwin`)
        - gyro-averages beyond the leading order in the background gradient length (the gyro-points are placed with
          the inverse Jacobian at the gyrocentre)"""


class _LazyPolarization:
    """Resolves to the polarization weight of the model's backgrounds when the propagators are allocated."""

    def __init__(self, model: GyrokineticMaxwell):
        self._model = model

    def _resolve(self):
        if self._model._background is None:
            self._model._setup_backgrounds()
        return self._model._background.polarization

    def __call__(self, e1, e2, e3):
        return self._resolve()(e1, e2, e3)


class _LazyBackground:
    """Resolves to the model's :class:`BackgroundWeights` when the propagators are allocated."""

    def __init__(self, model: GyrokineticMaxwell):
        self._model = model

    def _resolve(self) -> BackgroundWeights:
        if self._model._background is None:
            self._model._setup_backgrounds()
        return self._model._background

    def __getattr__(self, name):
        return getattr(self._resolve(), name)
