"""This module contains all the models implemented in Struphy.
Each model is defined in its own submodule, and this __init__.py file
makes all the models available as ``from struphy.models import <Model>``.
Models are imported lazily: only the submodule of the requested model is loaded.
"""

import importlib
from typing import TYPE_CHECKING

# class name -> module defining it, resolved on first access by __getattr__ below
_LAZY_IMPORTS = {
    "ColdPlasma": "struphy.models.cold_plasma",
    "ColdPlasmaVlasov": "struphy.models.cold_plasma_vlasov",
    "DeterministicParticleDiffusion": "struphy.models.deterministic_particle_diffusion",
    "DriftKineticElectrostaticAdiabatic": "struphy.models.drift_kinetic_electrostatic_adiabatic",
    "GuidingCenter": "struphy.models.guiding_center",
    "HasegawaWakatani": "struphy.models.hasegawa_wakatani",
    "IncompressibleNavierStokesSPH": "struphy.models.incompressible_navier_stokes_sph",
    "LinearExtendedMHDuniform": "struphy.models.linear_extended_mh_duniform",
    "LinearMHD": "struphy.models.linear_mhd",
    "LinearMHDDriftkineticCC": "struphy.models.linear_mhd_driftkinetic_cc",
    "LinearMHDVlasovCC": "struphy.models.linear_mhd_vlasov_cc",
    "LinearMHDVlasovPC": "struphy.models.linear_mhd_vlasov_pc",
    "LinearVlasovAmpereOneSpecies": "struphy.models.linear_vlasov_ampere_one_species",
    "LinearVlasovMaxwellOneSpecies": "struphy.models.linear_vlasov_maxwell_one_species",
    "Maxwell": "struphy.models.maxwell",
    "Poisson": "struphy.models.poisson",
    "PressureLessSPH": "struphy.models.pressure_less_sph",
    "RandomParticleDiffusion": "struphy.models.random_particle_diffusion",
    "ShearAlfven": "struphy.models.shear_alfven",
    "ToyDrift": "struphy.models.toy_drift",
    "TwoFluidQuasiNeutralToy": "struphy.models.two_fluid_quasi_neutral_toy",
    "VariationalBarotropicFluid": "struphy.models.variational_barotropic_fluid",
    "VariationalCompressibleFluid": "struphy.models.variational_compressible_fluid",
    "VariationalPressurelessFluid": "struphy.models.variational_pressureless_fluid",
    "ViscoResistiveDeltafMHD": "struphy.models.visco_resistive_deltaf_mhd",
    "ViscoResistiveDeltafMHD_with_q": "struphy.models.visco_resistive_deltaf_mhd_with_q",
    "ViscoResistiveLinearMHD": "struphy.models.visco_resistive_linear_mhd",
    "ViscoResistiveLinearMHD_with_q": "struphy.models.visco_resistive_linear_mhd_with_q",
    "ViscoResistiveMHD": "struphy.models.visco_resistive_mhd",
    "ViscoResistiveMHD_with_p": "struphy.models.visco_resistive_mhd_with_p",
    "ViscoResistiveMHD_with_q": "struphy.models.visco_resistive_mhd_with_q",
    "ViscousEulerSPH": "struphy.models.viscous_euler_sph",
    "ViscousFluid": "struphy.models.viscous_fluid",
    "Vlasov": "struphy.models.vlasov",
    "VlasovAmpereOneSpecies": "struphy.models.vlasov_ampere_one_species",
    "VlasovMaxwellOneSpecies": "struphy.models.vlasov_maxwell_one_species",
}

if TYPE_CHECKING:  # static analysis and IDEs see the eager imports
    from struphy.models.cold_plasma import ColdPlasma
    from struphy.models.cold_plasma_vlasov import ColdPlasmaVlasov
    from struphy.models.deterministic_particle_diffusion import DeterministicParticleDiffusion
    from struphy.models.drift_kinetic_electrostatic_adiabatic import DriftKineticElectrostaticAdiabatic
    from struphy.models.guiding_center import GuidingCenter
    from struphy.models.hasegawa_wakatani import HasegawaWakatani
    from struphy.models.incompressible_navier_stokes_sph import IncompressibleNavierStokesSPH
    from struphy.models.linear_extended_mh_duniform import LinearExtendedMHDuniform
    from struphy.models.linear_mhd import LinearMHD
    from struphy.models.linear_mhd_driftkinetic_cc import LinearMHDDriftkineticCC
    from struphy.models.linear_mhd_vlasov_cc import LinearMHDVlasovCC
    from struphy.models.linear_mhd_vlasov_pc import LinearMHDVlasovPC
    from struphy.models.linear_vlasov_ampere_one_species import LinearVlasovAmpereOneSpecies
    from struphy.models.linear_vlasov_maxwell_one_species import LinearVlasovMaxwellOneSpecies
    from struphy.models.maxwell import Maxwell
    from struphy.models.poisson import Poisson
    from struphy.models.pressure_less_sph import PressureLessSPH
    from struphy.models.random_particle_diffusion import RandomParticleDiffusion
    from struphy.models.shear_alfven import ShearAlfven
    from struphy.models.toy_drift import ToyDrift
    from struphy.models.two_fluid_quasi_neutral_toy import TwoFluidQuasiNeutralToy
    from struphy.models.variational_barotropic_fluid import VariationalBarotropicFluid
    from struphy.models.variational_compressible_fluid import VariationalCompressibleFluid
    from struphy.models.variational_pressureless_fluid import VariationalPressurelessFluid
    from struphy.models.visco_resistive_deltaf_mhd import ViscoResistiveDeltafMHD
    from struphy.models.visco_resistive_deltaf_mhd_with_q import ViscoResistiveDeltafMHD_with_q
    from struphy.models.visco_resistive_linear_mhd import ViscoResistiveLinearMHD
    from struphy.models.visco_resistive_linear_mhd_with_q import ViscoResistiveLinearMHD_with_q
    from struphy.models.visco_resistive_mhd import ViscoResistiveMHD
    from struphy.models.visco_resistive_mhd_with_p import ViscoResistiveMHD_with_p
    from struphy.models.visco_resistive_mhd_with_q import ViscoResistiveMHD_with_q
    from struphy.models.viscous_euler_sph import ViscousEulerSPH
    from struphy.models.viscous_fluid import ViscousFluid
    from struphy.models.vlasov import Vlasov
    from struphy.models.vlasov_ampere_one_species import VlasovAmpereOneSpecies
    from struphy.models.vlasov_maxwell_one_species import VlasovMaxwellOneSpecies

__all__ = [
    "ColdPlasma",
    "ColdPlasmaVlasov",
    "DeterministicParticleDiffusion",
    "DriftKineticElectrostaticAdiabatic",
    "GuidingCenter",
    "HasegawaWakatani",
    "IncompressibleNavierStokesSPH",
    "LinearExtendedMHDuniform",
    "LinearMHD",
    "LinearMHDDriftkineticCC",
    "LinearMHDVlasovCC",
    "LinearMHDVlasovPC",
    "LinearVlasovAmpereOneSpecies",
    "LinearVlasovMaxwellOneSpecies",
    "Maxwell",
    "Poisson",
    "PressureLessSPH",
    "RandomParticleDiffusion",
    "ShearAlfven",
    "ToyDrift",
    "TwoFluidQuasiNeutralToy",
    "VariationalBarotropicFluid",
    "VariationalCompressibleFluid",
    "VariationalPressurelessFluid",
    "ViscoResistiveDeltafMHD",
    "ViscoResistiveDeltafMHD_with_q",
    "ViscoResistiveLinearMHD",
    "ViscoResistiveLinearMHD_with_q",
    "ViscoResistiveMHD",
    "ViscoResistiveMHD_with_p",
    "ViscoResistiveMHD_with_q",
    "ViscousEulerSPH",
    "ViscousFluid",
    "Vlasov",
    "VlasovAmpereOneSpecies",
    "VlasovMaxwellOneSpecies",
]


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache: later lookups bypass __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
