import inspect
import logging

from feectools.linalg.block import BlockVector

import struphy.models as models
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator

logger = logging.getLogger("struphy")


# TODO: These utils should be classmethod of the model baseclass!
def get_model_by_name(model_name: str) -> type[StruphyModel]:
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


if __name__ == "__main__":
    fluid_models = get_models(model_type="Fluid")
    kinetic_models = get_models(model_type="Kinetic")
    hybrid_models = get_models(model_type="Hybrid")

    logger.info(f"Fluid models: {[mod.__name__ for mod in fluid_models]}")
    logger.info(f"Kinetic_models: {[mod.__name__ for mod in kinetic_models]}")
    logger.info(f"Hybrid models: {[mod.__name__ for mod in hybrid_models]}")

    model_message = generate_models_message()

    logger.info(model_message)
