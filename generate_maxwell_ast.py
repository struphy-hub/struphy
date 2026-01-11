import ast
from struphy.io.options import (
    EnvironmentOptions,
    BaseUnits,
    Time,
    DerhamOptions,
    FieldsBackground,
)
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.initial import perturbations
from struphy.kinetic_background import maxwellians
from struphy.pic.utilities import (
    LoadingParameters,
    WeightsParameters,
    BoundaryParameters,
    BinningPlot,
    KernelDensityPlot,
)
from struphy import main
from struphy.models.toy import Maxwell
from ast_helpers import (
    assign_constructor_with_defaults,
    attr_chain,
    call_attr,
    import_from,
)

# Imports
imports = [
    import_from(
        "struphy.io.options",
        [
            "EnvironmentOptions",
            "BaseUnits",
            "Time",
            "DerhamOptions",
            "FieldsBackground",
        ],
    ),
    import_from("struphy.geometry", ["domains"]),
    import_from("struphy.fields_background", ["equils"]),
    import_from("struphy.topology", ["grids"]),
    import_from("struphy.initial", ["perturbations"]),
    import_from("struphy.kinetic_background", ["maxwellians"]),
    import_from(
        "struphy.pic.utilities",
        [
            "LoadingParameters",
            "WeightsParameters",
            "BoundaryParameters",
            "BinningPlot",
            "KernelDensityPlot",
        ],
    ),
    import_from("struphy", ["main"]),
    import_from("struphy.models.toy", ["Maxwell"]),
]

# Assignments
assignments = [
    assign_constructor_with_defaults("env", EnvironmentOptions),
    assign_constructor_with_defaults("base_units", BaseUnits),
    assign_constructor_with_defaults("time_opts", Time, None, dt=0.01, Tend=0.10),
    assign_constructor_with_defaults("domain", domains.Cuboid, "domains.Cuboid"),
    assign_constructor_with_defaults("equil", equils.HomogenSlab, "equils.HomogenSlab"),
    assign_constructor_with_defaults(
        "grid", grids.TensorProductGrid, "grids.TensorProductGrid"
    ),
    assign_constructor_with_defaults("derham_opts", DerhamOptions),
    assign_constructor_with_defaults("model", Maxwell),
]

# propagator options
prop_options_assign = ast.Assign(
    targets=[
        attr_chain(["model", "propagators", "maxwell", "options"], ctx=ast.Store())
    ],
    value=ast.Call(
        func=attr_chain(["model", "propagators", "maxwell", "Options"]),
        args=[],
        keywords=[],
    ),
)
assignments.append(prop_options_assign)

# Perturbations
perturb_calls = []
for comp in range(3):
    perturb_calls.append(
        call_attr(
            attr_chain(["model", "em_fields", "b_field"]),
            "add_perturbation",
            args=[
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="perturbations", ctx=ast.Load()),
                        attr="TorusModesCos",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[
                        ast.keyword(arg="given_in_basis", value=ast.Constant("v")),
                        ast.keyword(arg="comp", value=ast.Constant(comp)),
                    ],
                )
            ],
        )
    )

# main
main_guard = ast.If(
    test=ast.Compare(
        left=ast.Name(id="__name__", ctx=ast.Load()),
        ops=[ast.Eq()],
        comparators=[ast.Constant("__main__")],
    ),
    body=[
        ast.Assign(
            targets=[ast.Name(id="verbose", ctx=ast.Store())],
            value=ast.Constant(True),
        ),
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="main", ctx=ast.Load()),
                    attr="run",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id="model", ctx=ast.Load())],
                keywords=[
                    ast.keyword(
                        arg="params_path", value=ast.Name(id="__file__", ctx=ast.Load())
                    ),
                    ast.keyword(arg="env", value=ast.Name(id="env", ctx=ast.Load())),
                    ast.keyword(
                        arg="base_units",
                        value=ast.Name(id="base_units", ctx=ast.Load()),
                    ),
                    ast.keyword(
                        arg="time_opts", value=ast.Name(id="time_opts", ctx=ast.Load())
                    ),
                    ast.keyword(
                        arg="domain", value=ast.Name(id="domain", ctx=ast.Load())
                    ),
                    ast.keyword(
                        arg="equil", value=ast.Name(id="equil", ctx=ast.Load())
                    ),
                    ast.keyword(arg="grid", value=ast.Name(id="grid", ctx=ast.Load())),
                    ast.keyword(
                        arg="derham_opts",
                        value=ast.Name(id="derham_opts", ctx=ast.Load()),
                    ),
                    ast.keyword(
                        arg="verbose", value=ast.Name(id="verbose", ctx=ast.Load())
                    ),
                ],
            )
        ),
    ],
    orelse=[],
)

# Assemble module
module = ast.Module(
    body=imports + assignments + perturb_calls + [main_guard], type_ignores=[]
)

ast.fix_missing_locations(module)

# print source code
source = ast.unparse(module)
print(source)
