from struphy.io import options

# import model 
from struphy.models.toy import Maxwell as Model
verbose = True

# units
units = options.Units(x=2.0,)

# geometry
domain = options.domains.Cuboid()

# fluid equilibrium (can be used as part of initial conditions)
equil = options.equils.HomogenSlab()

# time
time = options.Time()

# grid
grid = options.grids.TensorProductGrid(
    Nel=(12, 14, 1),
    p=(2, 3, 1),
    spl_kind=(False, True, True),
)

# derham options
derham = options.DerhamOptions()

# light-weight instance of model
model = Model(verbose=verbose)
species = model.species
propagators = model.propagators

model.species2.em_fields.e_field.
model.fluids.ions.add_variable()
# model.fluid.set_phys_params("mhd", options.PhysParams())
# model.kinetic.set_phys_params("mhd", options.PhysParams())

# propagator options
propagators.Maxwell.set_options(verbose=verbose)

# initial conditions for model variables (background + perturbation)
species.em_fields.add_background(
    "e_field",
    options.FieldsBackground(
        kind="LogicalConst",
        values=(0.3, 0.15, None),
    ),
    verbose=verbose,
)
species.em_fields.add_perturbation(
    "e_field",
    options.perturbations.TorusModesCos(
        ms=[[None], [1, 3], [None]],
    ),
    given_in_basis=(None, "v", None),
    verbose=verbose,
)

species.em_fields.add_background(
    "b_field",
    options.FieldsBackground(
        kind="LogicalConst",
        values=(0.3, 0.15, None),
    ),
    verbose=verbose,
)
species.em_fields.add_perturbation(
    "b_field",
    options.perturbations.TorusModesCos(
        ms=[[None], [1, 3], [None]],
    ),
    given_in_basis=(None, "v", None),
    verbose=verbose,
)