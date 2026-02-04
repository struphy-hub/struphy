import inspect
import os
import re

import struphy.plasma_models.fluid as fluid
import struphy.plasma_models.hybrid as hybrid
import struphy.plasma_models.kinetic as kinetic
import struphy.plasma_models.toy as toy

from struphy.plasma_models.base import StruphyModel


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


imports = """
import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector

from struphy.feec.projectors import L2Projector
from struphy.feec.variational_utilities import (
    H1vecMassMatrix_density,
    InternalEnergyEvaluator,
)
from struphy.kinetic_background.base import KineticBackground
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.models.base import StruphyModel
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
    FluidSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.polar.basic import PolarVector
from struphy.propagators import (
    propagators_coupling,
    propagators_fields,
    propagators_markers,
)
from struphy.utils.pyccel import Pyccelkernel

rank = MPI.COMM_WORLD.Get_rank()
"""

# Output directory
out_dir = "src/struphy/models"
os.makedirs(out_dir, exist_ok=True)

model_dict = {}

# Iterate over all modules and discover subclasses of StruphyModel
for model_type in [toy, fluid, hybrid, kinetic]:
    for _, cls in model_type.__dict__.items():
        if isinstance(cls, type) and issubclass(cls, StruphyModel) and cls != StruphyModel:
            model_name = cls.__name__
            try:
                # Get the source code of the class
                model_code = inspect.getsource(cls)
                model_dict[model_name] = model_code
            except Exception as e:
                print(f"Could not get source for {model_name}: {e}")

# Write each model to its own file
for model_name, model_code in model_dict.items():
    file_name = camel_to_snake(model_name) + ".py"
    file_path = os.path.join(out_dir, file_name)
    with open(file_path, "w") as f:
        f.write(imports)
        f.write("\n\n")
        f.write(model_code)

print(f"Written {len(model_dict)} model files to {out_dir}")
