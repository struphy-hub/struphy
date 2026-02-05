"""This module contains all the models implemented in Struphy. 
Each model is defined in its own submodule, and this __init__.py file 
imports all the models and makes them available for use when 
the struphy.models package is imported.
"""

from struphy.models.cold_plasma import ColdPlasma
from struphy.models.cold_plasma_vlasov import ColdPlasmaVlasov
from struphy.models.deterministic_particle_diffusion import DeterministicParticleDiffusion
from struphy.models.drift_kinetic_electrostatic_adiabatic import DriftKineticElectrostaticAdiabatic
from struphy.models.euler_sph import EulerSPH
from struphy.models.guiding_center import GuidingCenter
from struphy.models.hasegawa_wakatani import HasegawaWakatani
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
    "EulerSPH",
    "GuidingCenter",
    "HasegawaWakatani",
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
