import copy
import logging

import cunumpy as xp

from struphy import BaseUnits
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
from struphy.propagators.base import Propagator
from struphy.propagators.gyrokinetic_maxwell_zlr import (
    GyroDriftCoupling,
    GyroEparCoupling,
    MaxwellGyroMagnetisation,
    PushGuidingCenterParallelFullB,
    dielectric_matrix,
)

logger = logging.getLogger("struphy")


class GyrokineticMaxwellZLR(StruphyModel):
    """Gauge-invariant electromagnetic gyrokinetic model of Remmerswaal, Hatzky and Sonnendrücker (2025) in the
    zero-Larmor-radius (ZLR) limit: kinetic ions and electrons (gyrocentres, 5D) coupled to Maxwell's equations
    with the ZLR polarization and magnetisation.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    ion_mass_number: float
        Mass number (in units of Proton mass) of the ion species (default: 1.0)
    electron_mass_number: float
        Mass number (in units of Proton mass) of the electron species (default: 1/1836)
    with_drifts: bool
        Whether to include the perpendicular (E x B and grad-B) drifts and their currents
        (:class:`~struphy.propagators.gyrokinetic_maxwell_zlr.GyroDriftCoupling`), default: True
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

    class KineticIons(ParticleSpecies):
        def __init__(self, mass_number: float = 1.0):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(charge_number=1, mass_number=mass_number)

    class KineticElectrons(ParticleSpecies):
        def __init__(self, mass_number: float = 1.0 / 1836):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(charge_number=-1, mass_number=mass_number)

    ## propagators

    class Propagators:
        def __init__(
            self,
            species: tuple[PICVariable, ...] = (),
            b_field: FEECVariable = None,
            polarization: float = 0.0,
            with_drifts: bool = True,
        ):
            self.maxwell = MaxwellGyroMagnetisation(species=species)
            self.epar_ions = GyroEparCoupling(b_tilde=b_field, polarization=polarization)
            self.epar_electrons = GyroEparCoupling(b_tilde=b_field, polarization=polarization)
            self.push_para_ions = PushGuidingCenterParallelFullB(b_tilde=b_field)
            self.push_para_electrons = PushGuidingCenterParallelFullB(b_tilde=b_field)
            if with_drifts:
                self.drifts = GyroDriftCoupling(species=species, b_tilde=b_field, polarization=polarization)

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        ion_mass_number: float = 1.0,
        electron_mass_number: float = 1.0 / 1836,
        with_drifts: bool = True,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons(ion_mass_number)
        self.kinetic_electrons = self.KineticElectrons(electron_mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        species = (self.kinetic_ions.var, self.kinetic_electrons.var)
        self._alpha2 = {var: var.species.equation_params.alpha**2 for var in species}
        self._polarization = float(sum(self._alpha2.values()))

        # 3. instantiate all propagators
        self.propagators = self.Propagators(
            species=species,
            b_field=self.em_fields.b_field,
            polarization=self._polarization,
            with_drifts=with_drifts,
        )

        # 4. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field
        self.propagators.epar_ions.variables.e = self.em_fields.e_field
        self.propagators.epar_ions.variables.var = self.kinetic_ions.var
        self.propagators.epar_electrons.variables.e = self.em_fields.e_field
        self.propagators.epar_electrons.variables.var = self.kinetic_electrons.var
        self.propagators.push_para_ions.variables.ions = self.kinetic_ions.var
        self.propagators.push_para_electrons.variables.ions = self.kinetic_electrons.var
        if with_drifts:
            self.propagators.drifts.variables.e = self.em_fields.e_field

        # 5. define scalars to be tracked during simulation
        electric_energy = FunctionScalarFEEC(self._compute_en_E)
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        ion_energy = KineticEnergyPIC(
            self.kinetic_ions.var,
            normalization=self._alpha2[self.kinetic_ions.var],
        ) + FunctionScalarPIC(self._compute_en_muB_ions, self.kinetic_ions.var)
        electron_energy = KineticEnergyPIC(
            self.kinetic_electrons.var,
            normalization=self._alpha2[self.kinetic_electrons.var],
        ) + FunctionScalarPIC(self._compute_en_muB_electrons, self.kinetic_electrons.var)
        self.scalars = Scalars(
            en_E=electric_energy,
            en_B=magnetic_energy,
            en_ions=ion_energy,
            en_electrons=electron_energy,
            en_tot=electric_energy + magnetic_energy + ion_energy + electron_energy,
            n_lost_ions=LostMarkersPIC(self.kinetic_ions.var),
            n_lost_electrons=LostMarkersPIC(self.kinetic_electrons.var),
        )

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def post_allocate(self):
        for species in (self.kinetic_ions, self.kinetic_electrons):
            if species.var.species.weights_params.control_variate:
                raise NotImplementedError(
                    f"{self.__class__.__name__} is a full-f model: the control variate would drop the "
                    "field-dependent drifts of f_0 (e.g. the grad-B current of the perturbation)."
                )

        self._PB = getattr(Propagator.basis_ops, "PB")
        self._PBb = self._PB.codomain.zeros()

    def _compute_en_E(self):
        r"""Electric energy :math:`\frac12 \mathbf e^\top \mathbb A \mathbf e` with the dielectric matrix."""
        A = dielectric_matrix(Propagator.mass_ops, self._polarization)
        e = self.em_fields.e_field.spline.vector
        return 0.5 * A.dot_inner(e, e)

    def _compute_en_muB(self, var: PICVariable):
        r"""Magnetic-moment energy :math:`\alpha_s^2 \sum_p w_p \mu_p B_\parallel(\boldsymbol \eta_p)` of one species,
        with :math:`B_\parallel = |B_0| + \Pi^0(\mathbf b_0 \cdot \tilde{\mathbf B})`."""
        particles = var.particles
        PBb = self._PB.dot(self.em_fields.b_field.spline.vector, out=self._PBb)
        particles.save_magnetic_energy(PBb)
        valid = particles.valid_mks
        # The weights (column 5) already carry the 1/N_p of the Monte-Carlo estimate, see KineticEnergyPIC.
        energy = particles.markers[valid, 5].dot(particles.markers[valid, particles.first_diagnostics_idx])
        return self._alpha2[var] * energy

    def _compute_en_muB_ions(self):
        return self._compute_en_muB(self.kinetic_ions.var)

    def _compute_en_muB_electrons(self):
        return self._compute_en_muB(self.kinetic_electrons.var)

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

            \mathbf E^* = \mathbf E - \varepsilon_s \mu \nabla B_\parallel\,, \qquad
            \mathbf B^* = \mathbf B_0 + \tilde{\mathbf B} + \varepsilon_s v_\parallel \nabla \times \mathbf b_0\,, \qquad
            B^*_\parallel = \mathbf b_0 \cdot \mathbf B^*\,, \qquad
            B_\parallel = |\mathbf B_0| + \Pi^0(\mathbf b_0 \cdot \tilde{\mathbf B})\,.

        Field equations (ZLR limit, centred isotropic background, :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`):

        .. math::

            \left(\mathbf I + \sum_s \alpha_s^2 \frac{n_0}{|B_0|^2} \Pi_\perp\right) \frac{\partial \mathbf E}{\partial t}
            &= \nabla \times \left(\mathbf B + \sum_s \alpha_s^2 \int f_s\, \mu\, \mathbf b_0 \,\textrm d v_\parallel \textrm d \mu\right)
            - \sum_s \frac{\alpha_s^2}{\varepsilon_s} \int f_s\, \dot{\mathbf R} \,\textrm d v_\parallel \textrm d \mu\,,
            \\[2mm]
            \frac{\partial \mathbf B}{\partial t} &= - \nabla \times \mathbf E\,.

        The left-hand side operator is the vacuum permittivity plus the ZLR polarization, the curl term on the right
        is the curl of the intrinsic magnetisation :math:`-\mu \mathbf b_0`. Gauss's law and :math:`\nabla \cdot \mathbf B = 0`
        are not imposed; :math:`\nabla \cdot \mathbf B = 0` holds exactly in the discrete de Rham complex.
        Since the model is gauge invariant, it is formulated in :math:`\mathbf E \in H(\textnormal{curl})` and
        :math:`\mathbf B \in H(\textnormal{div})` rather than in the potentials.

        Conserved energy:

        .. math::

            \mathcal H = \frac12 \mathbf e^\top \mathbb A \mathbf e + \frac12 \mathbf b^\top \mathbb M^2 \mathbf b
            + \sum_s \alpha_s^2 \sum_p w_p \left(\frac{v_{\parallel,p}^2}{2} + \mu_p B_\parallel(\boldsymbol \eta_p)\right)\,.

        Reference: R. Remmerswaal, R. Hatzky, E. Sonnendrücker, *Gauge-invariant variational formulations of
        electromagnetic gyrokinetic theory*, J. Plasma Phys. 91, E127 (2025), `doi:10.1017/S0022377825100688 <https://doi.org/10.1017/S0022377825100688>`_.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses the light speed as reference velocity:

        .. math::

            \hat v = c,\qquad \hat E = \hat B \hat v,\qquad \hat t = \hat x / c\,.

        The species parameters :math:`\alpha_s=\hat\Omega_{p,s}/\hat\Omega_{c,s}` and
        :math:`\varepsilon_s=1/(\hat\Omega_{c,s}\hat t)` are signed (negative for electrons).
        The background density of all species is the fluid equilibrium density :math:`n_0`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy (with the dielectric matrix): ``en_E``
        - Magnetic field energy: ``en_B``
        - Ion energy :math:`\alpha_i^2 \sum_p w_p (v_\parallel^2/2 + \mu B_\parallel)`: ``en_ions``
        - Electron energy :math:`\alpha_e^2 \sum_p w_p (v_\parallel^2/2 + \mu B_\parallel)`: ``en_electrons``
        - Total energy: ``en_tot``
        - Lost markers: ``n_lost_ions``, ``n_lost_electrons``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.gyrokinetic_maxwell_zlr.MaxwellGyroMagnetisation`
        2. :class:`~struphy.propagators.gyrokinetic_maxwell_zlr.GyroEparCoupling` (ions)
        3. :class:`~struphy.propagators.gyrokinetic_maxwell_zlr.GyroEparCoupling` (electrons)
        4. :class:`~struphy.propagators.gyrokinetic_maxwell_zlr.PushGuidingCenterParallelFullB` (ions)
        5. :class:`~struphy.propagators.gyrokinetic_maxwell_zlr.PushGuidingCenterParallelFullB` (electrons)
        6. :class:`~struphy.propagators.gyrokinetic_maxwell_zlr.GyroDriftCoupling` (both species, if ``with_drifts``)
        """
        doc = rf"""**1. MaxwellGyroMagnetisation:**

{MaxwellGyroMagnetisation.__doc__}

**2. and 3. GyroEparCoupling (one per species):**

{GyroEparCoupling.__doc__}

**4. and 5. PushGuidingCenterParallelFullB (one per species):**

{PushGuidingCenterParallelFullB.__doc__}

**6. GyroDriftCoupling (both species):**

{GyroDriftCoupling.__doc__}

All field solves are implicit, so the light wave, the plasma oscillation and the compressional Alfvén wave
do not limit the time step. The common dielectric matrix is assembled once by
:func:`~struphy.propagators.gyrokinetic_maxwell_zlr.dielectric_matrix`.
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""GyrokineticMaxwellZLR is a prototype of the gauge-invariant electromagnetic gyrokinetic model of
        Remmerswaal, Hatzky and Sonnendrücker (J. Plasma Phys. 91, E127, 2025) in the zero-Larmor-radius limit.
        Two gyrocentre species (ions and electrons) are coupled to the electric and magnetic fields,
        with the polarization entering as a dielectric matrix and the magnetisation as a source of Ampère's law.
        It is full-f (no control variate) and uses tensor-product coefficients (no polar splines).
        The background must be centred and isotropic (no parallel flow, no pressure anisotropy)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the model:

        .. code-block:: python

            from struphy.models import GyrokineticMaxwellZLR

            model = GyrokineticMaxwellZLR(ion_mass_number=1.0, electron_mass_number=1 / 100)
            model.em_fields.e_field
            model.em_fields.b_field
            model.kinetic_ions.var
            model.kinetic_electrons.var

        ``b_field`` is the perturbation of the equilibrium field (do not add a ``FieldsBackground`` to it).
        The loading parameters should match the thermal speeds in light units, e.g.
        ``LoadingParameters(moments=(0, 0, vth, vth), B0=...)``.
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electromagnetic gyrokinetic simulations in the zero-Larmor-radius limit (shear and kinetic Alfvén waves)
        - energy-conserving coupling of several gyrocentre species to Maxwell's equations
        - prototyping of the gauge-invariant gyrokinetic formulation in Struphy"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - finite-Larmor-radius effects beyond the polarization (no disc or radial gyro-averages)
        - backgrounds with parallel flow or pressure anisotropy
        - delta-f (control variate) simulations
        - polar-spline (pole) domains
        - quasi-neutral (Darwin) limits, which need a separate variant"""
