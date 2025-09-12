from struphy.fields_background import equils
from struphy.geometry import domains
from struphy.initial import perturbations
from struphy.io.options import DerhamOptions, FieldsBackground, Time, Units
from struphy.kinetic_background import maxwellians

# import model
from struphy.models.toy import Maxwell as Model
from struphy.topology import grids

verbose = True

# units
units = Units()

# geometry
a1 = 2.326744
a2 = 3.686839
domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=2.0)

# fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# time
time = Time(dt=0.05, Tend=10.0)

# grid
grid = grids.TensorProductGrid(
    Nel=(32, 64, 1),
    p=(3, 3, 1),
    spl_kind=(False, True, True),
    dirichlet_bc=[[True, True], [False, False], [False, False]],
)

# derham options
derham = DerhamOptions()

# light-weight instance of model
model = Model(units, domain, equil, verbose=verbose)
# model.fluid.set_phys_params("mhd", options.PhysParams())
# model.kinetic.set_phys_params("mhd", options.PhysParams())

# propagator options
model.propagators.maxwell.set_options(algo="implicit")

# initial conditions for model variables (background + perturbation)
model.em_fields.e_field.add_perturbation(
    perturbations.CoaxialWaveguideElectric_r(m=3, a1=a1, a2=a2),
    verbose=verbose,
)

model.em_fields.e_field.add_perturbation(
    perturbations.CoaxialWaveguideElectric_theta(m=3, a1=a1, a2=a2),
    verbose=verbose,
)

model.em_fields.b_field.add_perturbation(
    perturbations.CoaxialWaveguideMagnetic(m=3, a1=a1, a2=a2),
    verbose=verbose,
)
