import logging

import cunumpy as xp

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.linear_mhd import LinearMHD
from struphy.models.scalars import FunctionScalarPIC, KineticEnergyPIC, LostMarkersPIC
from struphy.models.species import ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.base import Propagator
from struphy.propagators.current_coupling_5d_curlb import CurrentCoupling5DCurlb
from struphy.propagators.current_coupling_5d_density import CurrentCoupling5DDensity
from struphy.propagators.current_coupling_5d_gradb import CurrentCoupling5DGradB
from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel
from struphy.propagators.shear_alfven_current_coupling_5d import ShearAlfvenCurrentCoupling5D

logger = logging.getLogger("struphy")

# algorithms of the guiding-center pushers in the default parameter file: the energy-conserving discrete gradient
# methods of the reference discretization, second order (implicit mid-point) for the drift perpendicular to b0, first
# order (Itoh-Abe) with Newton iterations for the parallel motion
PUSH_BXE_ALGO = "discrete_gradient_2nd_order"
PUSH_PARALLEL_ALGO = "discrete_gradient_1st_order_newton"


class LinearMHDDriftkineticCC(LinearMHD):
    """Hybrid linear ideal MHD coupled with energetic ions (5D drift-kinetic) via the current-coupling scheme.

    Extends :class:`~struphy.models.LinearMHD` (magnetic field in H(div)) by energetic ions in the guiding-center
    approximation, whose current and magnetization enter the MHD momentum equation. The shear-Alfvén step
    ``shear_alf`` of LinearMHD is replaced by
    :class:`~struphy.propagators.shear_alfven_current_coupling_5d.ShearAlfvenCurrentCoupling5D`, which includes the
    coupling to the guiding-center density.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mhd_mass_number: float
        Mass number (in units of Proton mass) of the MHD bulk species (default: 1.0)
    hot_charge_number: int
        Charge number (in units of the positive elementary charge) of the energetic ion species (default: 1)
    hot_mass_number: float
        Mass number (in units of Proton mass) of the energetic ion species (default: 1.0)
    hot_epsilon: float, optional
        Normalized cyclotron period of the energetic ion species. If None, computed from units and charge/mass numbers.
    """

    # propagators renamed when the model joined the LinearEuler family (options of old run metadata)
    _renamed_propagators = {"shearalfen_cc5d": "shear_alf"}

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Hybrid"

    ## additional species

    class EnergeticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## additional propagators

    class Propagators(LinearMHD.Propagators):
        """Propagators of :class:`~struphy.models.LinearMHD`, with ``shear_alf`` replaced by the shear-Alfvén step
        coupled to the guiding-center density, plus the guiding-center pushers (``push_bxe``, ``push_parallel``) and
        the current coupling (``cc5d_density``, ``cc5d_gradb``, ``cc5d_curlb``).

        Parameters
        ----------
        b_field : FEECVariable, optional
            Magnetic field perturbation, added to the equilibrium field in the pushers and couplings (default: None).
        energetic_ions_var : PICVariable, optional
            Markers of the energetic ions, deposited by ``shear_alf`` and ``cc5d_density`` (default: None).
        **kwargs
            Passed to :class:`LinearMHD.Propagators <struphy.models.LinearMHD.Propagators>`.
        """

        splitting_order = (
            "push_bxe",
            "push_parallel",
            "shear_alf",
            "bg_current",
            "bg_pressure",
            "sound",
            "cc5d_density",
            "cc5d_gradb",
            "cc5d_curlb",
        )

        def __init__(
            self,
            b_field: FEECVariable = None,
            energetic_ions_var: PICVariable = None,
            **kwargs,
        ):
            super().__init__(b_field=b_field, **kwargs)
            self.push_bxe = PushGuidingCenterBxEstar(b_tilde=b_field)
            self.push_parallel = PushGuidingCenterParallel(b_tilde=b_field)
            # replaces the shear-Alfven step of LinearMHD
            self.shear_alf = ShearAlfvenCurrentCoupling5D(energetic_ions=energetic_ions_var)
            self.cc5d_density = CurrentCoupling5DDensity(energetic_ions=energetic_ions_var, b_tilde=b_field)
            self.cc5d_gradb = CurrentCoupling5DGradB(b_tilde=b_field)
            self.cc5d_curlb = CurrentCoupling5DCurlb(b_tilde=b_field)

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    energetic_ions: EnergeticIons
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mhd_mass_number: float = 1.0,
        hot_charge_number: int = 1,
        hot_mass_number: float = 1.0,
        hot_epsilon: float = None,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see LinearMHD (magnetic field in H(div))
        super().__init__(base_units, mass_number=mhd_mass_number)

    def _create_species(self):
        """:meta private:"""
        super()._create_species()
        self.energetic_ions = self.EnergeticIons(
            self.params["hot_charge_number"],
            self.params["hot_mass_number"],
            self.params["hot_epsilon"],
        )

    def _propagator_args(self):
        """:meta private:"""
        return {**super()._propagator_args(), "energetic_ions_var": self.energetic_ions.var}

    def _link_propagator_variables(self):
        """:meta private:"""
        # shear_alf (u, b) is linked by LinearMHD
        super()._link_propagator_variables()
        self.propagators.cc5d_density.variables.u = self.mhd.velocity
        self.propagators.cc5d_gradb.variables.u = self.mhd.velocity
        self.propagators.cc5d_gradb.variables.energetic_ions = self.energetic_ions.var
        self.propagators.cc5d_curlb.variables.u = self.mhd.velocity
        self.propagators.cc5d_curlb.variables.energetic_ions = self.energetic_ions.var
        self.propagators.push_bxe.variables.ions = self.energetic_ions.var
        self.propagators.push_parallel.variables.ions = self.energetic_ions.var

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        en_tot = scalars.pop("en_tot")
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number
        en_fv = KineticEnergyPIC(self.energetic_ions.var, normalization=Ah / Ab)
        en_fB = FunctionScalarPIC(self._compute_en_fB, self.energetic_ions.var)
        scalars["en_fv"] = en_fv
        scalars["en_fB"] = en_fB
        scalars["en_tot"] = en_tot + en_fv + en_fB
        scalars["n_lost_particles"] = LostMarkersPIC(self.energetic_ions.var)
        return scalars

    def generate_default_parameter_file(self, path=None, prompt=True):
        """The default parameter file, with the algorithms of the reference discretization for the guiding-center
        pushers (``PUSH_BXE_ALGO``, ``PUSH_PARALLEL_ALGO``)."""
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        with open(params_path, "r") as f:
            lines = f.readlines()
        algos = {"push_bxe": PUSH_BXE_ALGO, "push_parallel": PUSH_PARALLEL_ALGO}
        for i, line in enumerate(lines):
            for prop, algo in algos.items():
                if f"model.propagators.{prop}.Options()" in line:
                    lines[i] = line.replace(
                        f"model.propagators.{prop}.Options()", f'model.propagators.{prop}.Options(algo="{algo}")'
                    )
        with open(params_path, "w") as f:
            f.writelines(lines)
        return params_path

    def post_allocate(self):
        """:meta private:"""
        super().post_allocate()

        self._PB = getattr(Propagator.basis_ops, "PB")
        self._PBb = self._PB.codomain.zeros()

    def _compute_en_fB(self):
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number
        particles = self.energetic_ions.var.particles
        self._PBb = self._PB.dot(self.em_fields.b_field.spline.vector)
        particles.save_magnetic_energy(self._PBb)

        return (
            particles.markers[5, particles.valid_mks].dot(
                particles.markers[8, particles.valid_mks],
            )
            * Ah
            / Ab
        )

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        MHD continuity:

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        MHD momentum:

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B} + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \left[ \frac{1}{\epsilon} n_\textnormal{gc} \tilde{\mathbf{U}} - \frac{1}{\epsilon} \mathbf{J}_\textnormal{gc} - \nabla \times \mathbf{M}_\textnormal{gc} \right] \times \mathbf{B}

        MHD pressure:

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        MHD induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}) = 0

        Energetic-particle drift-kinetic equation:

        .. math::

            \frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{B_\parallel^*} \left( v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^* \right) \cdot \nabla f_\textnormal{h} + \frac{1}{\epsilon} \frac{1}{B_\parallel^*} (\mathbf{B}^* \cdot \mathbf{E}^*) \frac{\partial f_\textnormal{h}}{\partial v_\parallel} = 0

        Energetic-particle moments:

        .. math::

            n_\textnormal{gc} = \int f_\textnormal{h} B_\parallel^* \, \textnormal{d} v_\parallel \textnormal{d} \mu

        .. math::

            \mathbf{J}_\textnormal{gc} = \int \frac{f_\textnormal{h}}{B_\parallel^*} \left( v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^* \right) \, \textnormal{d} v_\parallel \textnormal{d} \mu

        .. math::

            \mathbf{M}_\textnormal{gc} = -\int f_\textnormal{h} B_\parallel^* \mu \mathbf{b}_0 \, \textnormal{d} v_\parallel \textnormal{d} \mu

        where

        .. math::

            B^*_\parallel = \mathbf{b}_0 \cdot \mathbf{B}^*
            \\[2mm]
            \mathbf{B}^* &= \mathbf{B} + \epsilon v_\parallel \nabla \times \mathbf{b}_0
            \\[2mm]
            \mathbf{E}^* &= -\tilde{\mathbf{U}} \times \mathbf{B} - \epsilon \mu \nabla (\mathbf{b}_0 \cdot \mathbf{B})

        """

    @classmethod
    def doc_normalization(cls):
        r"""The bulk and energetic-particle flow scales are normalized with the
        bulk Alfvén speed, while the magnetic moment carries its own unit:

        .. math::

            \hat U = \hat v = \hat v_{A,\mathrm{bulk}},\qquad
            \hat\mu = A_h m_H \hat v_h^2 / \hat B.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - The scalars of :class:`~struphy.models.LinearMHD`: ``en_U``, ``en_p``, ``en_thermal``, ``en_p_eq``,
          ``en_B``, ``en_B_eq``, ``en_B_tot``
        - Parallel energetic-particle energy: ``en_fv``
        - Magnetic-moment energetic-particle energy: ``en_fB``
        - Total energy: ``en_tot = en_U + en_thermal + en_B + en_fv + en_fB``
        - Lost particles: ``n_lost_particles``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence). The default parameter file of this
        model (``struphy params LinearMHDDriftkineticCC``) sets the energy-conserving discrete gradient methods of the
        reference discretization for the guiding-center pushers: ``algo="discrete_gradient_2nd_order"`` (implicit
        mid-point) for ``push_bxe`` and ``algo="discrete_gradient_1st_order_newton"`` (Itoh-Abe with Newton
        iterations) for ``push_parallel``. The explicit Runge-Kutta methods (``algo="explicit"``) are cheaper but not
        energy conserving."""
        return cls.propagators_doc()

    @classmethod
    def doc_long_description(cls):
        r"""LinearMHDDriftkineticCC is the reduced-kinetic hybrid current-coupling
        model for energetic ions. It is useful when gyrophase averaging is
        acceptable but energetic-particle feedback on linear MHD still has to be
        retained."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the linear MHD plus drift-kinetic CC model:

        .. code-block:: python

            from struphy.models import LinearMHDDriftkineticCC

            model = LinearMHDDriftkineticCC()
            model.em_fields.b_field
            model.mhd.velocity
            model.energetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear energetic-ion effects with guiding-center reduction
        - current-coupling hybrid mode studies in strong magnetic fields
        - verification of 5D hybrid coupling operators"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - full 6D energetic-particle dynamics
        - nonlinear hybrid turbulence
        - resistive or viscous MHD closures
        - regimes where gyrophase resolution is required"""
