import inspect

import struphy.models as models
from struphy.api.options import LiteralOptions
from struphy.models.base import StruphyModel


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

    for name, obj in inspect.getmembers(models):
        # Only include classes that are subclasses of StruphyModel, excluding StruphyModel itself
        if inspect.isclass(obj) and issubclass(obj, StruphyModel) and obj is not StruphyModel:
            if model_type is None or model_type == obj.model_type():
                model_classes.append(obj)

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


if __name__ == "__main__":
    fluid_models = get_models(model_type="Fluid")
    kinetic_models = get_models(model_type="Kinetic")
    hybrid_models = get_models(model_type="Hybrid")

    print(f"Fluid models: {[mod.__name__ for mod in fluid_models]}")
    print(f"Kinetic_models: {[mod.__name__ for mod in kinetic_models]}")
    print(f"Hybrid models: {[mod.__name__ for mod in hybrid_models]}")

    model_message = generate_models_message()

    print(model_message)
