import inspect
import logging

from feectools.linalg.block import BlockVector

import struphy.models as models
from struphy.feec.density_weighted import VelocityMass
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator

logger = logging.getLogger("struphy")


# renamed models (old name -> new name), so that old run metadata and parameter files still find their model
RENAMED_MODELS = {
    "PressureLessSPH": "ColdEulerSPH",
}


# TODO: These utils should be classmethod of the model baseclass!
def get_model_by_name(model_name: str) -> type[StruphyModel]:
    if model_name in RENAMED_MODELS:
        logger.warning(f"The model {model_name} has been renamed to {RENAMED_MODELS[model_name]}.")
        model_name = RENAMED_MODELS[model_name]
    try:
        model_class: StruphyModel = getattr(models, model_name)
        if not issubclass(model_class, StruphyModel):
            raise TypeError(f"{model_name} is not a StruphyModel subclass.")
        else:
            return model_class
    except AttributeError:
        raise ModuleNotFoundError(f"{model_name} not found in models.")


def get_models(model_type: LiteralOptions.ModelTypes | None = None) -> list[type[StruphyModel]]:
    model_classes = []

    for model in StruphyModel:
        if model_type is None or model_type == model.model_type():
            model_classes.append(model)

    return model_classes


def get_model_names(model_type: LiteralOptions.ModelTypes | None = None) -> list[str]:
    model_classes = get_models(model_type=model_type)

    return [model.name() for model in model_classes]


def get_model_type_message(model_type: LiteralOptions.ModelTypes) -> str:
    model_names = get_model_names(model_type=model_type)

    # fluid message
    models_message = f"{model_type} models:\n"
    models_message += "-------------\n"
    models_message += "\n".join(model_names)

    return models_message


def generate_models_message() -> str:
    fluid_message = get_model_type_message(model_type="Fluid")
    kinetic_message = get_model_type_message(model_type="Kinetic")
    hybrid_message = get_model_type_message(model_type="Hybrid")

    # model message
    model_message = "run one of the following models:\n"
    model_message += "\n" + fluid_message
    model_message += "\n\n" + kinetic_message
    model_message += "\n\n" + hybrid_message

    return model_message


# ------------------------------------------------------------------------------------------------------------------
# Utilities shared by the models of a family (or of several families), called explicitly from the hooks of the
# models (no mixin classes). Short code stays in the models.
# ------------------------------------------------------------------------------------------------------------------

## LinearEuler family (with p and with q = sqrt(p))


def ones_l2():
    """The vector of ones in the discrete :math:`L^2` space (FE coefficients of a 3-form), for volume integrals of
    3-forms (allocated, call in ``post_allocate``)."""
    ones = Propagator.projected_equil.p3.space.zeros()
    if isinstance(ones, PolarVector):
        ones.tp[:] = 1.0
    else:
        ones[:] = 1.0
    return ones


## magnetic field


class MagneticEnergies:
    """Energies of the equilibrium magnetic field, :math:`\\frac 12 \\int |\\mathbf B_0|^2`, and of the total magnetic
    field, :math:`\\frac 12 \\int |\\mathbf B_0 + \\tilde{\\mathbf B}|^2`, in the space of the perturbation (2-form, or
    1-form with ``b_space="Hcurl"``).

    Parameters
    ----------
    b_field : FEECVariable
        Magnetic field perturbation.
    """

    def __init__(self, b_field: FEECVariable):
        self._b_field = b_field

    def allocate(self):
        """Background field, mass matrix and buffers (call in ``post_allocate`` of the model)."""
        if self._b_field.space == "Hcurl":
            self._b_eq = Propagator.projected_equil.b1
            self._M_b = Propagator.mass_ops.M1
        else:
            self._b_eq = Propagator.projected_equil.b2
            self._M_b = Propagator.mass_ops.M2

        self._tmp_b1: BlockVector = self._b_eq.space.zeros()
        self._tmp_b2: BlockVector = self._b_eq.space.zeros()

    def en_B_eq(self):
        """Energy of the equilibrium magnetic field."""
        self._M_b.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        return self._b_eq.inner(self._tmp_b1) / 2

    def en_B_tot(self):
        """Energy of the total magnetic field."""
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self._b_field.spline.vector

        self._M_b.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        return self._tmp_b1.inner(self._tmp_b2) / 2


def add_magnetic_scalars(scalars: dict, b_field: FEECVariable, energies: MagneticEnergies):
    """Add the magnetic energies to the scalars of the parent model: ``en_B``, ``en_B_eq``, ``en_B_tot``, and replace
    ``en_tot`` by the one including the magnetic energy (kept last).

    Parameters
    ----------
    scalars : dict
        Scalars of the parent model, modified in place.
    b_field : FEECVariable
        Magnetic field perturbation.
    energies : MagneticEnergies
        Energies of the equilibrium and the total magnetic field.
    """
    en_tot = scalars.pop("en_tot")
    en_B = BilinearEnergyFEEC(b_field)
    scalars["en_B"] = en_B
    scalars["en_B_eq"] = FunctionScalarFEEC(energies.en_B_eq)
    scalars["en_B_tot"] = FunctionScalarFEEC(energies.en_B_tot)
    scalars["en_tot"] = en_tot + en_B


def magnetic_discretization_doc(model_cls) -> str:
    """Body of ``doc_discretization`` of the linear MHD models: the propagators of the model (with the shear-Alfvén
    step of ``b_space="Hdiv"``), followed by the shear-Alfvén step of ``b_space="Hcurl"``.

    Parameters
    ----------
    model_cls : type
        The model class.
    """
    doc = model_cls.propagators_doc()
    doc += '**1. shear_alf** with ``b_space="Hcurl"`` '
    doc += "(:class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`):\n\n"
    doc += f"{ShearAlfvenCurlCurlPropagator.__doc__}\n"
    return doc


## nonlinear (perturbative) terms of the models with q = sqrt(p)

GAMMA = 5 / 3


def link_perturbative_propagators(propagators, mhd):
    """Assign the variables of the propagators ``nonlin_sound`` and ``mom_adv`` of the perturbative models.

    Parameters
    ----------
    propagators : object
        The ``Propagators`` instance of the model.
    mhd : FluidSpecies
        The fluid species ``mhd`` of the model (``density``, ``velocity``, ``sqrt_p``).
    """
    propagators.nonlin_sound.variables.u = mhd.velocity
    propagators.nonlin_sound.variables.q = mhd.sqrt_p

    propagators.mom_adv.variables.n = mhd.density
    propagators.mom_adv.variables.u = mhd.velocity


class PerturbativeEnergies:
    """Energies of the perturbation of the full (nonlinear) system in the perturbative models: the kinetic energy with
    the total density, :math:`\\frac 12 \\int (\\rho_0 + \\tilde \\rho) |\\tilde{\\mathbf U}|^2`, the internal energy
    :math:`\\int (2 q_0 \\tilde q + \\tilde q^2) / (\\gamma - 1)` and their sum, plus the change of the magnetic energy
    :math:`\\frac 12 \\int (|\\mathbf B_0 + \\tilde{\\mathbf B}|^2 - |\\mathbf B_0|^2)` if ``magnetic_energies`` is given.

    Parameters
    ----------
    mhd : FluidSpecies
        The fluid species ``mhd`` of the model (``density``, ``velocity``, ``sqrt_p``).
    magnetic_energies : MagneticEnergies, optional
        Magnetic energies of the model with a magnetic field (default: None, no magnetic field).
    """

    def __init__(self, mhd, magnetic_energies: MagneticEnergies = None):
        self._mhd = mhd
        self._magnetic_energies = magnetic_energies

    def allocate(self):
        """Mass matrix of the total density and buffers (call in ``post_allocate`` of the model)."""
        self._kinetic_mass = VelocityMass(
            Propagator.derham,
            Propagator.domain,
            Propagator.mass_ops,
            Propagator.projected_equil,
            rho_field=self._mhd.density,
        )
        self._tmp_u: BlockVector = self._mhd.velocity.spline.vector.space.zeros()
        # linear form of q0 in V3, for int 2 q0 q
        self._M3_q0 = Propagator.mass_ops.M3.dot(Propagator.projected_equil.q3)

    def en_U(self):
        """Kinetic energy with the total density."""
        self._kinetic_mass.update()
        u = self._mhd.velocity.spline.vector
        return u.inner(self._kinetic_mass.M.dot(u, out=self._tmp_u)) / 2

    def en_internal(self):
        """Perturbation of the internal energy."""
        q = self._mhd.sqrt_p.spline.vector
        q2 = q.inner(Propagator.mass_ops.M3.dot(q))
        return (2 * q.inner(self._M3_q0) + q2) / (GAMMA - 1)

    def en_tot(self):
        """Total energy of the perturbation."""
        en_tot = self.en_U() + self.en_internal()
        if self._magnetic_energies is not None:
            en_tot += self._magnetic_energies.en_B_tot() - self._magnetic_energies.en_B_eq()
        return en_tot


def add_perturbative_scalars(scalars: dict, mhd, energies: PerturbativeEnergies):
    """Replace the energies of the linear model by those of the perturbation of the full system: ``en_U`` (total
    density), ``en_q2``, ``en_internal`` and ``en_tot`` (kept last).

    Parameters
    ----------
    scalars : dict
        Scalars of the linear model, modified in place.
    mhd : FluidSpecies
        The fluid species ``mhd`` of the model.
    energies : PerturbativeEnergies
        Energies of the perturbation.
    """
    scalars.pop("en_tot")
    scalars["en_U"] = FunctionScalarFEEC(energies.en_U)
    scalars["en_q2"] = BilinearEnergyFEEC(mhd.sqrt_p, bilinear_form_name="M3", normalization=2 / (GAMMA - 1))
    scalars["en_internal"] = FunctionScalarFEEC(energies.en_internal)
    scalars["en_tot"] = FunctionScalarFEEC(energies.en_tot)


if __name__ == "__main__":
    fluid_models = get_models(model_type="Fluid")
    kinetic_models = get_models(model_type="Kinetic")
    hybrid_models = get_models(model_type="Hybrid")

    logger.info(f"Fluid models: {[mod.__name__ for mod in fluid_models]}")
    logger.info(f"Kinetic_models: {[mod.__name__ for mod in kinetic_models]}")
    logger.info(f"Hybrid models: {[mod.__name__ for mod in hybrid_models]}")

    model_message = generate_models_message()

    logger.info(model_message)
