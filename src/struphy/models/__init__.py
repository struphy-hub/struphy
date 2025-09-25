# import struphy.models.fluid as fluid
# import struphy.models.hybrid as hybrid
# import struphy.models.kinetic as kinetic
# import struphy.models.toy as toy
# from struphy.models.base import StruphyModel

# model_names = []
# for model_type in [toy, fluid, hybrid, kinetic]:
#     for name, cls in model_type.__dict__.items():
#         if (
#             isinstance(cls, type)
#             and issubclass(cls, StruphyModel)
#             and cls != StruphyModel
#         ):
#             model_names.append(cls.__name__)
#             print(f"from {model_type.__name__} import {cls.__name__}")
# print(model_names)


from struphy.models.fluid import (
    ColdPlasma,
    HasegawaWakatani,
    IsothermalEulerSPH,
    LinearExtendedMHDuniform,
    LinearMHD,
    ViscoresistiveDeltafMHD,
    ViscoresistiveDeltafMHD_with_q,
    ViscoresistiveLinearMHD,
    ViscoresistiveLinearMHD_with_q,
    ViscoresistiveMHD,
    ViscoresistiveMHD_with_p,
    ViscoresistiveMHD_with_q,
    ViscousFluid,
)
from struphy.models.hybrid import (
    ColdPlasmaVlasov,
    LinearMHDDriftkineticCC,
    LinearMHDVlasovCC,
    LinearMHDVlasovPC,
)
from struphy.models.kinetic import (
    DriftKineticElectrostaticAdiabatic,
    LinearVlasovAmpereOneSpecies,
    LinearVlasovMaxwellOneSpecies,
    VlasovAmpereOneSpecies,
    VlasovMaxwellOneSpecies,
)
from struphy.models.toy import (
    DeterministicParticleDiffusion,
    GuidingCenter,
    Maxwell,
    Poisson,
    PressureLessSPH,
    RandomParticleDiffusion,
    ShearAlfven,
    TwoFluidQuasiNeutralToy,
    VariationalBarotropicFluid,
    VariationalCompressibleFluid,
    VariationalPressurelessFluid,
    Vlasov,
)

__all__ = [
    "Maxwell",
    "Vlasov",
    "GuidingCenter",
    "ShearAlfven",
    "VariationalPressurelessFluid",
    "VariationalBarotropicFluid",
    "VariationalCompressibleFluid",
    "Poisson",
    "DeterministicParticleDiffusion",
    "RandomParticleDiffusion",
    "PressureLessSPH",
    "TwoFluidQuasiNeutralToy",
    "LinearMHD",
    "LinearExtendedMHDuniform",
    "ColdPlasma",
    "ViscoresistiveMHD",
    "ViscousFluid",
    "ViscoresistiveMHD_with_p",
    "ViscoresistiveLinearMHD",
    "ViscoresistiveDeltafMHD",
    "ViscoresistiveMHD_with_q",
    "ViscoresistiveLinearMHD_with_q",
    "ViscoresistiveDeltafMHD_with_q",
    "IsothermalEulerSPH",
    "HasegawaWakatani",
    "LinearMHDVlasovCC",
    "LinearMHDVlasovPC",
    "LinearMHDDriftkineticCC",
    "ColdPlasmaVlasov",
    "VlasovAmpereOneSpecies",
    "VlasovMaxwellOneSpecies",
    "LinearVlasovAmpereOneSpecies",
    "LinearVlasovMaxwellOneSpecies",
    "DriftKineticElectrostaticAdiabatic",
]
