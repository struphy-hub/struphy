from struphy.io import options

# model
model = "Maxwell"

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

# fluid equilibrium
equil = options.equils.HomogenSlab()

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
