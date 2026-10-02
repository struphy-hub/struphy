import importlib
from typing import TYPE_CHECKING

# class name -> module defining it, resolved on first access by __getattr__ below
_LAZY_IMPORTS = {
    "AdiabaticPhi": "struphy.propagators.adiabatic_phi",
    "CurlCurlSolve": "struphy.propagators.curl_curl_solve",
    "CurrentCoupling5DCurlb": "struphy.propagators.current_coupling_5d_curlb",
    "CurrentCoupling5DDensity": "struphy.propagators.current_coupling_5d_density",
    "CurrentCoupling5DGradB": "struphy.propagators.current_coupling_5d_gradb",
    "CurrentCoupling6DCurrent": "struphy.propagators.current_coupling_6d_current",
    "CurrentCoupling6DDensity": "struphy.propagators.current_coupling_6d_density",
    "EfieldWeightsCoupling": "struphy.propagators.efield_weights_coupling",
    "FaradayExtended": "struphy.propagators.faraday_extended",
    "Hall": "struphy.propagators.hall",
    "HasegawaWakataniStep": "struphy.propagators.hasegawa_wakatani_step",
    "ImplicitDiffusion": "struphy.propagators.implicit_diffusion",
    "JxBCold": "struphy.propagators.jxb_cold",
    "Magnetosonic": "struphy.propagators.magnetosonic",
    "MagnetosonicUniform": "struphy.propagators.magnetosonic_uniform",
    "MaxwellWeakAmpere": "struphy.propagators.maxwell_weak_ampere",
    "OhmCold": "struphy.propagators.ohm_cold",
    "PoissonAdiabaticGyrokinetic": "struphy.propagators.poisson_adiabatic_gyrokinetic",
    "PoissonSolve": "struphy.propagators.poisson_solve",
    "PressureCoupling6D": "struphy.propagators.pressure_coupling_6d",
    "PushDeterministicDiffusion": "struphy.propagators.push_deterministic_diffusion",
    "PushEta": "struphy.propagators.push_eta",
    "PushEtaPC": "struphy.propagators.push_eta_pc",
    "PushGuidingCenterBxEstar": "struphy.propagators.push_guiding_center_bx_estar",
    "PushGuidingCenterParallel": "struphy.propagators.push_guiding_center_parallel",
    "PushRandomDiffusion": "struphy.propagators.push_random_diffusion",
    "PushVinForceField": "struphy.propagators.push_v_in_force_field",
    "PushVinSPHpressure": "struphy.propagators.push_v_in_sph_pressure",
    "PushVinViscousPotential": "struphy.propagators.push_v_in_viscous_potential",
    "PushVxB": "struphy.propagators.push_vxb",
    "ShearAlfvenB1": "struphy.propagators.shear_alfven_b1",
    "ShearAlfvenCurrentCoupling5D": "struphy.propagators.shear_alfven_current_coupling_5d",
    "ShearAlfvenPropagator": "struphy.propagators.shear_alfven_propagator",
    "TimeDependentSource": "struphy.propagators.time_dependent_source",
    "TwoFluidQuasiNeutralFull": "struphy.propagators.two_fluid_quasi_neutral_full",
    "VariationalDensityEvolve": "struphy.propagators.variational_density_evolve",
    "VariationalEntropyEvolve": "struphy.propagators.variational_entropy_evolve",
    "VariationalMagFieldEvolve": "struphy.propagators.variational_mag_field_evolve",
    "VariationalMomentumAdvection": "struphy.propagators.variational_momentum_advection",
    "VariationalPBEvolve": "struphy.propagators.variational_pb_evolve",
    "VariationalQBEvolve": "struphy.propagators.variational_qb_evolve",
    "VariationalResistivity": "struphy.propagators.variational_resistivity",
    "VariationalViscosity": "struphy.propagators.variational_viscosity",
    "VlasovAmpereCoupling": "struphy.propagators.vlasov_ampere_coupling",
}

if TYPE_CHECKING:  # static analysis and IDEs see the eager imports
    from struphy.propagators.adiabatic_phi import AdiabaticPhi
    from struphy.propagators.curl_curl_solve import CurlCurlSolve
    from struphy.propagators.current_coupling_5d_curlb import CurrentCoupling5DCurlb
    from struphy.propagators.current_coupling_5d_density import CurrentCoupling5DDensity
    from struphy.propagators.current_coupling_5d_gradb import CurrentCoupling5DGradB
    from struphy.propagators.current_coupling_6d_current import CurrentCoupling6DCurrent
    from struphy.propagators.current_coupling_6d_density import CurrentCoupling6DDensity
    from struphy.propagators.efield_weights_coupling import EfieldWeightsCoupling
    from struphy.propagators.faraday_extended import FaradayExtended
    from struphy.propagators.hall import Hall
    from struphy.propagators.hasegawa_wakatani_step import HasegawaWakataniStep
    from struphy.propagators.implicit_diffusion import ImplicitDiffusion
    from struphy.propagators.jxb_cold import JxBCold
    from struphy.propagators.magnetosonic import Magnetosonic
    from struphy.propagators.magnetosonic_uniform import MagnetosonicUniform
    from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere
    from struphy.propagators.ohm_cold import OhmCold
    from struphy.propagators.poisson_adiabatic_gyrokinetic import PoissonAdiabaticGyrokinetic
    from struphy.propagators.poisson_solve import PoissonSolve
    from struphy.propagators.pressure_coupling_6d import PressureCoupling6D
    from struphy.propagators.push_deterministic_diffusion import PushDeterministicDiffusion
    from struphy.propagators.push_eta import PushEta
    from struphy.propagators.push_eta_pc import PushEtaPC
    from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
    from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel
    from struphy.propagators.push_random_diffusion import PushRandomDiffusion
    from struphy.propagators.push_v_in_force_field import PushVinForceField
    from struphy.propagators.push_v_in_sph_pressure import PushVinSPHpressure
    from struphy.propagators.push_v_in_viscous_potential import PushVinViscousPotential
    from struphy.propagators.push_vxb import PushVxB
    from struphy.propagators.shear_alfven_b1 import ShearAlfvenB1
    from struphy.propagators.shear_alfven_current_coupling_5d import ShearAlfvenCurrentCoupling5D
    from struphy.propagators.shear_alfven_propagator import ShearAlfvenPropagator
    from struphy.propagators.time_dependent_source import TimeDependentSource
    from struphy.propagators.two_fluid_quasi_neutral_full import TwoFluidQuasiNeutralFull
    from struphy.propagators.variational_density_evolve import VariationalDensityEvolve
    from struphy.propagators.variational_entropy_evolve import VariationalEntropyEvolve
    from struphy.propagators.variational_mag_field_evolve import VariationalMagFieldEvolve
    from struphy.propagators.variational_momentum_advection import VariationalMomentumAdvection
    from struphy.propagators.variational_pb_evolve import VariationalPBEvolve
    from struphy.propagators.variational_qb_evolve import VariationalQBEvolve
    from struphy.propagators.variational_resistivity import VariationalResistivity
    from struphy.propagators.variational_viscosity import VariationalViscosity
    from struphy.propagators.vlasov_ampere_coupling import VlasovAmpereCoupling

__all__ = [
    "AdiabaticPhi",
    "CurlCurlSolve",
    "CurrentCoupling5DCurlb",
    "CurrentCoupling5DDensity",
    "CurrentCoupling5DGradB",
    "CurrentCoupling6DCurrent",
    "CurrentCoupling6DDensity",
    "EfieldWeightsCoupling",
    "FaradayExtended",
    "Hall",
    "HasegawaWakataniStep",
    "ImplicitDiffusion",
    "JxBCold",
    "Magnetosonic",
    "MagnetosonicUniform",
    "MaxwellWeakAmpere",
    "OhmCold",
    "PoissonAdiabaticGyrokinetic",
    "PoissonSolve",
    "PressureCoupling6D",
    "PushDeterministicDiffusion",
    "PushEta",
    "PushEtaPC",
    "PushGuidingCenterBxEstar",
    "PushGuidingCenterParallel",
    "PushRandomDiffusion",
    "PushVinForceField",
    "PushVinSPHpressure",
    "PushVinViscousPotential",
    "PushVxB",
    "ShearAlfvenB1",
    "ShearAlfvenCurrentCoupling5D",
    "ShearAlfvenPropagator",
    "TimeDependentSource",
    "TwoFluidQuasiNeutralFull",
    "VariationalDensityEvolve",
    "VariationalEntropyEvolve",
    "VariationalMagFieldEvolve",
    "VariationalMomentumAdvection",
    "VariationalPBEvolve",
    "VariationalQBEvolve",
    "VariationalResistivity",
    "VariationalViscosity",
    "VlasovAmpereCoupling",
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
