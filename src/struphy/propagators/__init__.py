# import struphy.propagators.propagators_coupling as propagators_coupling
# import struphy.propagators.propagators_fields as propagators_fields
# import struphy.propagators.propagators_markers as propagators_markers
# from struphy.propagators.base import Propagator

# model_names = []
# for model_type in [propagators_coupling, propagators_fields, propagators_markers]:
#     for name, cls in model_type.__dict__.items():
#         if isinstance(cls, type) and issubclass(cls, Propagator) and cls != Propagator:
#             model_names.append(cls.__name__)
#             print(f"from {model_type.__name__} import {cls.__name__}")
# # print(", ".join(model_names))
# print(model_names)

from struphy.propagators.propagators_coupling import (
    CurrentCoupling5DCurlb,
    CurrentCoupling5DGradB,
    CurrentCoupling6DCurrent,
    EfieldWeights,
    PressureCoupling6D,
    VlasovAmpere,
)
from struphy.propagators.propagators_fields import (
    AdiabaticPhi,
    CurrentCoupling5DDensity,
    CurrentCoupling6DDensity,
    FaradayExtended,
    Hall,
    HasegawaWakatani,
    ImplicitDiffusion,
    JxBCold,
    Magnetosonic,
    MagnetosonicCurrentCoupling5D,
    MagnetosonicUniform,
    Maxwell,
    OhmCold,
    Poisson,
    ShearAlfven,
    ShearAlfvenB1,
    ShearAlfvenCurrentCoupling5D,
    TimeDependentSource,
    TwoFluidQuasiNeutralFull,
    VariationalDensityEvolve,
    VariationalEntropyEvolve,
    VariationalMagFieldEvolve,
    VariationalMomentumAdvection,
    VariationalPBEvolve,
    VariationalQBEvolve,
    VariationalResistivity,
    VariationalViscosity,
)
from struphy.propagators.propagators_markers import (
    PushDeterministicDiffusion,
    PushEta,
    PushEtaPC,
    PushGuidingCenterBxEstar,
    PushGuidingCenterParallel,
    PushRandomDiffusion,
    PushVinEfield,
    PushVinSPHpressure,
    PushVxB,
    StepStaticEfield,
)

__all__ = [
    "VlasovAmpere",
    "EfieldWeights",
    "PressureCoupling6D",
    "CurrentCoupling6DCurrent",
    "CurrentCoupling5DCurlb",
    "CurrentCoupling5DGradB",
    "Maxwell",
    "OhmCold",
    "JxBCold",
    "ShearAlfven",
    "ShearAlfvenB1",
    "Hall",
    "Magnetosonic",
    "MagnetosonicUniform",
    "FaradayExtended",
    "CurrentCoupling6DDensity",
    "ShearAlfvenCurrentCoupling5D",
    "MagnetosonicCurrentCoupling5D",
    "CurrentCoupling5DDensity",
    "ImplicitDiffusion",
    "Poisson",
    "VariationalMomentumAdvection",
    "VariationalDensityEvolve",
    "VariationalEntropyEvolve",
    "VariationalMagFieldEvolve",
    "VariationalPBEvolve",
    "VariationalQBEvolve",
    "VariationalViscosity",
    "VariationalResistivity",
    "TimeDependentSource",
    "AdiabaticPhi",
    "HasegawaWakatani",
    "TwoFluidQuasiNeutralFull",
    "PushEta",
    "PushVxB",
    "PushVinEfield",
    "PushEtaPC",
    "PushGuidingCenterBxEstar",
    "PushGuidingCenterParallel",
    "StepStaticEfield",
    "PushDeterministicDiffusion",
    "PushRandomDiffusion",
    "PushVinSPHpressure",
]
