from struphy.models import (toy, fluid, kinetic, hybrid,)
from struphy.io import options

# model class (do not instantiate)
Model = toy.Maxwell

# units
units = options.Units(
    x=1.0,
    B=1.0,
    n=1.0,
    kBT=1.0,
)

# time
time = options.Time(split_algo="LieTrotter")

# geometry
domain = options.domains.Cuboid()

# grid
grid = options.grids.TensorProductGrid(
    Nel=(12, 14, 1),
    p=(2, 3, 1),
    spl_kind=(False, True, True),
)

# derham options
derham = options.DerhamOptions()

# fluid equilibrium (can be used as part of initial conditions)
equil = options.equils.HomogenSlab()

# light-weight instance of model
model = Model()
print(f'{model.__em_fields__ = }')

# species parameters
species = model.species()
print(f'{species.em_fields = }')
print(f'{species.fluid = }')
print(f'{species.kinetic = }')
# species.em_fields.set_options(prop, prop.options())
# model.fluid.set_phys_params("mhd", options.PhysParams())
# model.fluid.set_propagator_options("mhd", prop, prop.options())
# model.kinetic.set_phys_params("mhd", options.PhysParams())

# initial conditions for model variables (background + perturbation)
species.em_fields.add_background("e_ield", options.FieldsBackground(kind="LogicalConst", 
                                                        values=(0.3, 0.15, None),),)
species.em_fields.add_perturbation("e_field", options.perturbations.TorusModesCos(ms=[[None], [1, 3], [None]],),
                                 given_in_basis=(None, "v", None),)

species.em_fields.add_background("b_field", options.FieldsBackground(kind="LogicalConst", 
                                                        values=(0.3, 0.15, None),),)
species.em_fields.add_perturbation("b_field", options.perturbations.TorusModesCos(ms=[[None], [1, 3], [None]],),
                                 given_in_basis=(None, "v", None),)

# FOR NOW: initial conditions and options
em_fields = {}
em_fields["background"] = {}
em_fields["perturbation"] = {}
em_fields["options"] = {}

em_fields["background"]["e_field"] = {"LogicalConst": {"values": [0.3, 0.15, None]}}
em_fields["background"]["b_field"] = {"LogicalConst": {"values": [0.3, 0.15, None]}}

em_fields["perturbation"]["e_field"] = {}
em_fields["perturbation"]["b_field"] = {}
em_fields["perturbation"]["e_field"]["TorusModesCos"] = {
    "given_in_basis": [None, "v", None],
    "ms": [[None], [1, 3], [None]],
}
em_fields["perturbation"]["b_field"]["TorusModesCos"] = {
    "given_in_basis": [None, "v", None],
    "ms": [[None], [1, 3], [None]],
}

solver = {
    "type": ["pcg", "MassMatrixPreconditioner"],
    "tol": 1.0e-08,
    "maxiter": 3000,
    "info": False,
    "verbose": False,
    "recycle": True,
}

em_fields["options"]["Maxwell"] = {"algo": "implicit", "solver": solver}
