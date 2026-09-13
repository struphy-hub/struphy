"""Class introspection helpers without simulation or MPI dependencies."""

import inspect


def __class_with_params_repr_no_defaults__(cls_instance):
    sig = inspect.signature(cls_instance.__class__.__init__)
    defaults = {k: v.default for k, v in sig.parameters.items() if k != "self"}
    out = f"{cls_instance.__class__.__name__}("
    for k, v in cls_instance.params.items():
        if k in defaults and v != defaults[k]:
            out += f"{k}={v}, "
    out += ")"
    return out


def all_class_params_are_default(cls_instance):
    return cls_instance.__repr_no_defaults__() == cls_instance.__class__.__name__ + "()"


def all_subclasses(cls):
    subclasses = cls.__subclasses__()
    subclasses = subclasses + [g for s in subclasses for g in all_subclasses(s)]
    return subclasses
