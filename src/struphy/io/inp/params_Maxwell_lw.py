from struphy.io.options import Units, Time, DerhamOptions, FieldsBackground
from struphy.fields_background import equils
from struphy.geometry import domains
from struphy.initial import perturbations
from struphy.kinetic_background import maxwellians
from struphy.topology import grids

# import model 
from struphy.models.toy import Maxwell as Model
verbose = False

# units
units = Units()

# geometry
domain = domains.Cuboid()

# fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# time
time = Time()

# grid
grid = grids.TensorProductGrid(
    Nel=(12, 14, 1),
    p=(2, 3, 1),
    spl_kind=(False, True, True),
)

# derham options
derham = DerhamOptions()

# light-weight instance of model
model = Model(units, domain, equil, verbose=verbose)
# model.fluid.set_phys_params("mhd", options.PhysParams())
# model.kinetic.set_phys_params("mhd", options.PhysParams())

# propagator options
model.propagators.maxwell.set_options(algo="explicit")

# initial conditions for model variables (background + perturbation)
model.em_fields.e_field.add_background(
    FieldsBackground(
        type="LogicalConst",
        values=(0.3, 0.15, None),
    ),
    verbose=verbose,
)
model.em_fields.e_field.add_perturbation(
    perturbations.TorusModesCos(
        ms=[1, 3],
        given_in_basis="v",
        comp=1,
    ),
    verbose=verbose,
)

model.em_fields.b_field.add_background(
    FieldsBackground(
        type="LogicalConst",
        values=(0.3, 0.15, None),
    ),
    verbose=verbose,
)
model.em_fields.b_field.add_perturbation(
    perturbations.TorusModesCos(
        ms=[1, 3],
        given_in_basis="v",
        comp=1,
    ),
    verbose=verbose,
)

# exclude variable from saving
model.em_fields.e_field.save_data = False