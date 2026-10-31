"""Mixins shared by the models of the LinearEuler family with the pressure (:class:`~struphy.models.LinearEuler`,
:class:`~struphy.models.LinearMHD`) and with the variable :math:`q = \\sqrt p` (:class:`~struphy.models.LinearEulerWithQ`,
:class:`~struphy.models.LinearMHDWithQ`). They extend the hooks of :meth:`~struphy.models.base.StruphyModel._setup` and
must come before the model class in the bases (``class LinearMHD(MagneticFieldMixin, LinearEuler)``)."""

from typing import Literal

from feectools.linalg.block import BlockVector

from struphy.feec.density_weighted import VelocityMass
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.background_current import BackgroundCurrent
from struphy.propagators.base import Propagator
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator
from struphy.propagators.shear_alfven_propagator import ShearAlfvenPropagator
from struphy.utils.utils import check_option


class EquilibriumPressureMixin:
    """Equilibrium check and internal energy of the equilibrium, :math:`\\int p_0 / (\\gamma - 1)` (scalar
    ``en_p_eq``), for the models of the LinearEuler family."""

    def post_allocate(self):
        """:meta private:"""
        super().post_allocate()
        if Propagator.projected_equil is None:
            raise ValueError(
                f"{type(self).__name__} needs an equilibrium (Simulation(equil=...)) providing n0 and p0, e.g. "
                "equils.HomogenSlab."
            )
        self._ones = Propagator.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def _compute_en_p_eq(self):
        return Propagator.projected_equil.p3.inner(self._ones) / (5 / 3 - 1)


def add_magnetic_propagators(
    propagators,
    b_field: FEECVariable = None,
    b_space: str = "Hdiv",
    projected: bool = False,
    rho_field: FEECVariable = None,
):
    """Add the propagators of the magnetic field of the linear MHD models to ``propagators``: ``shear_alf``
    (shear-Alfvén waves) and ``bg_current`` (force of the background current).

    Parameters
    ----------
    propagators : object
        The ``Propagators`` instance of the model.
    b_field : FEECVariable, optional
        Magnetic field perturbation, used by ``bg_current`` (default: None).
    b_space : str
        Space of the magnetic field, ``"Hdiv"`` or ``"Hcurl"`` (default: ``"Hdiv"``).
    projected : bool
        Projected curl-curl shear-Alfvén step, only with ``b_space="Hcurl"`` (default: False).
    rho_field : FEECVariable, optional
        Density perturbation, for the inertia of the total density (perturbative models, only with ``b_space="Hdiv"``;
        default: None).
    """
    if b_space == "Hdiv":
        propagators.shear_alf = ShearAlfvenPropagator(rho_field=rho_field)
    else:
        assert rho_field is None, 'rho_field (perturbative models) needs b_space="Hdiv".'
        propagators.shear_alf = ShearAlfvenCurlCurlPropagator(projected=projected)
    propagators.bg_current = BackgroundCurrent(b_field=b_field, rho_field=rho_field)


class MagneticFieldMixin:
    """Magnetic field of the linear MHD models of the LinearEuler family: the species ``em_fields``, the variable links
    of the propagators ``shear_alf`` and ``bg_current`` (created by :func:`add_magnetic_propagators` in the
    ``Propagators`` of the model), the magnetic energies and the Alfvén velocity scale.

    The model calls :meth:`_init_magnetic_field` in its ``__init__`` before ``super().__init__()``."""

    OptsBSpace = Literal["Hdiv", "Hcurl"]

    class EMFields(FieldSpecies):
        def __init__(self, b_space: str = "Hdiv"):
            self.b_field = FEECVariable(space=b_space)
            self.init_variables()

    em_fields: EMFields

    def _init_magnetic_field(self, b_space: str, projected: bool):
        """Check the flags of the magnetic field and keep them for the hooks."""
        check_option(b_space, self.OptsBSpace)
        if projected and b_space != "Hcurl":
            raise ValueError('projected=True requires b_space="Hcurl".')
        self._b_space = b_space
        self._projected = projected

    def _create_species(self):
        """:meta private:"""
        self.em_fields = self.EMFields(b_space=self._b_space)
        super()._create_species()

    def _propagator_args(self):
        """:meta private:"""
        return {
            **super()._propagator_args(),
            "b_field": self.em_fields.b_field,
            "b_space": self._b_space,
            "projected": self._projected,
        }

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        self.propagators.bg_current.variables.u = self.mhd.velocity

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        # en_tot of the parent is replaced by the one including the magnetic energy (and kept last)
        en_tot = scalars.pop("en_tot")
        en_B = BilinearEnergyFEEC(self.em_fields.b_field)
        scalars["en_B"] = en_B
        scalars["en_B_eq"] = FunctionScalarFEEC(self._compute_en_B_eq)
        scalars["en_B_tot"] = FunctionScalarFEEC(self._compute_en_B_tot)
        scalars["en_tot"] = en_tot + en_B
        return scalars

    @property
    def velocity_scale(self):
        return "alfvén"

    def post_allocate(self):
        """:meta private:"""
        super().post_allocate()

        # background magnetic field and mass matrix in the space of the perturbation (2-form, or 1-form with
        # b_space="Hcurl")
        if self.em_fields.b_field.space == "Hcurl":
            self._b_eq = Propagator.projected_equil.b1
            self._M_b = Propagator.mass_ops.M1
        else:
            self._b_eq = Propagator.projected_equil.b2
            self._M_b = Propagator.mass_ops.M2

        self._tmp_b1: BlockVector = self._b_eq.space.zeros()
        self._tmp_b2: BlockVector = self._b_eq.space.zeros()

    def _compute_en_B_eq(self):
        self._M_b.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        return self._b_eq.inner(self._tmp_b1) / 2

    def _compute_en_B_tot(self):
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.em_fields.b_field.spline.vector

        self._M_b.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        return self._tmp_b1.inner(self._tmp_b2) / 2

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence). The shear-Alfvén step
        ``shear_alf`` is :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator` for
        ``b_space="Hdiv"`` (listed below) and
        :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator` for
        ``b_space="Hcurl"`` (at the end):"""
        doc = cls.propagators_doc()
        doc += '**1. shear_alf** with ``b_space="Hcurl"`` '
        doc += "(:class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`):\n\n"
        doc += f"{ShearAlfvenCurlCurlPropagator.__doc__}\n"
        return doc


class PerturbativeQMixin:
    """Nonlinear (perturbative) terms of the models with the variable :math:`q = \\sqrt p`
    (:class:`~struphy.models.PerturbativeEulerWithQ`, :class:`~struphy.models.PerturbativeMHDWithQ`): the inertia of
    the total density in all steps (the density perturbation passed as ``rho_field`` to ``Propagators``), the variable
    links of ``nonlin_sound`` and ``mom_adv`` (created in the ``Propagators`` of the model) and the scalars of the
    nonlinear energy. Must come before the linear model in the bases."""

    GAMMA = 5 / 3

    def _propagator_args(self):
        """:meta private:"""
        return {**super()._propagator_args(), "rho_field": self.mhd.density}

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.nonlin_sound.variables.u = self.mhd.velocity
        self.propagators.nonlin_sound.variables.q = self.mhd.sqrt_p

        self.propagators.mom_adv.variables.n = self.mhd.density
        self.propagators.mom_adv.variables.u = self.mhd.velocity

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        # the energy of the linear model is replaced by the energy of the perturbation of the full system
        scalars.pop("en_tot")
        scalars["en_U"] = FunctionScalarFEEC(self._compute_en_U)
        scalars["en_q2"] = BilinearEnergyFEEC(
            self.mhd.sqrt_p, bilinear_form_name="M3", normalization=2 / (self.GAMMA - 1)
        )
        scalars["en_internal"] = FunctionScalarFEEC(self._compute_en_internal)
        scalars["en_tot"] = FunctionScalarFEEC(self._compute_en_tot)
        return scalars

    def post_allocate(self):
        """:meta private:"""
        super().post_allocate()
        self._kinetic_mass = VelocityMass(
            Propagator.derham,
            Propagator.domain,
            Propagator.mass_ops,
            Propagator.projected_equil,
            rho_field=self.mhd.density,
        )
        self._tmp_u: BlockVector = self.mhd.velocity.spline.vector.space.zeros()
        # linear form of q0 in V3, for int 2 q0 q
        self._M3_q0 = Propagator.mass_ops.M3.dot(Propagator.projected_equil.q3)

    def _compute_en_U(self):
        self._kinetic_mass.update()
        u = self.mhd.velocity.spline.vector
        return u.inner(self._kinetic_mass.M.dot(u, out=self._tmp_u)) / 2

    def _compute_en_internal(self):
        q = self.mhd.sqrt_p.spline.vector
        q2 = q.inner(Propagator.mass_ops.M3.dot(q))
        return (2 * q.inner(self._M3_q0) + q2) / (self.GAMMA - 1)

    def _compute_en_tot(self):
        en_tot = self._compute_en_U() + self._compute_en_internal()
        if hasattr(self, "em_fields"):
            en_tot += self._compute_en_B_tot() - self._compute_en_B_eq()
        return en_tot
