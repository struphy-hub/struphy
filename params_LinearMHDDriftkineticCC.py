from struphy.io.options import EnvironmentOptions, BaseUnits, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.io.options import FieldsBackground
from struphy.initial import perturbations
from struphy.kinetic_background import maxwellians
from struphy.pic.utilities import (LoadingParameters,
                                   WeightsParameters,
                                   BoundaryParameters,
                                   BinningPlot,
                                   KernelDensityPlot,
                                   )
from struphy import main

# import model, set verbosity
from struphy.models.hybrid import LinearMHDDriftkineticCC

# environment options
env = EnvironmentOptions()

# units
base_units = BaseUnits()

# time stepping
time_opts = Time()

# geometry
domain = domains.Cuboid()

# fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# grid
grid = grids.TensorProductGrid(Nel = (16,16,16))

# derham options
derham_opts = DerhamOptions()

# light-weight model instance
model = LinearMHDDriftkineticCC()

# species parameters
model.mhd.set_phys_params()
model.energetic_ions.set_phys_params()

loading_params = LoadingParameters(ppc=1000)
weights_params = WeightsParameters()
boundary_params = BoundaryParameters()
model.energetic_ions.set_markers(loading_params=loading_params,
                                 weights_params=weights_params,
                                 boundary_params=boundary_params,
                                 )
model.energetic_ions.set_sorting_boxes()
model.energetic_ions.set_save_data()

# propagator options
model.propagators.push_bxe.options = model.propagators.push_bxe.Options(
                        b_tilde = model.em_fields.b_field,)
model.propagators.push_parallel.options = model.propagators.push_parallel.Options(
                        b_tilde = model.em_fields.b_field,)
model.propagators.shearalfen_cc5d.options = model.propagators.shearalfen_cc5d.Options(
                        energetic_ions = model.energetic_ions.var,)
model.propagators.magnetosonic.options = model.propagators.magnetosonic.Options(
                        b_field=model.em_fields.b_field,)
model.propagators.cc5d_density.options = model.propagators.cc5d_density.Options(
                        energetic_ions = model.energetic_ions.var,
                        b_tilde = model.em_fields.b_field,)
model.propagators.cc5d_gradb.options = model.propagators.cc5d_gradb.Options(
                        b_tilde = model.em_fields.b_field,)
model.propagators.cc5d_curlb.options = model.propagators.cc5d_curlb.Options(
                        b_tilde = model.em_fields.b_field,)

# background, perturbations and initial conditions
model.mhd.velocity.add_background(FieldsBackground())
model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=0))
model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=1))
model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=2))
maxwellian_1 = maxwellians.GyroMaxwellian2D(n=(1.0, None), equil=equil)
maxwellian_2 = maxwellians.GyroMaxwellian2D(n=(0.1, None), equil=equil)
background = maxwellian_1 + maxwellian_2
model.energetic_ions.var.add_background(background)

# if .add_initial_condition is not called, the background is the kinetic initial condition
perturbation = perturbations.TorusModesCos()
maxwellian_1pt = maxwellians.GyroMaxwellian2D(n=(1.0, perturbation), equil=equil)
init = maxwellian_1pt + maxwellian_2
model.energetic_ions.var.add_initial_condition(init)

# optional: exclude variables from saving
# model.energetic_ions.var.save_data = False

if __name__ == "__main__":
    # start run
    verbose = True

    main.run(model,
             params_path=__file__,
             env=env,
             base_units=base_units,
             time_opts=time_opts,
             domain=domain,
             equil=equil,
             grid=grid,
             derham_opts=derham_opts,
             verbose=verbose,
             )