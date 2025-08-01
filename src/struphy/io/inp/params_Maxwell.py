from struphy.io.options import Units, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.initial import perturbations
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.io.options import FieldsBackground

# import model
from struphy.models.toy import Maxwell as Model

# light-weight model instance
model = Model()

# units
units = Units()

# time stepping
time = Time()

# geometry
domain = domains.Cuboid()

# fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# grid
grid = grids.TensorProductGrid()

# derham options
derham = DerhamOptions()

# propagator options
model.propagators.maxwell.set_options()

# initial conditions (background + perturbation)
model.em_fields.b_field.add_background(FieldsBackground())
model.em_fields.b_field.add_perturbation(perturbations.TorusModesCos())

# optional: exclude variables from saving
# model.em_fields.b_field.save_data = False
