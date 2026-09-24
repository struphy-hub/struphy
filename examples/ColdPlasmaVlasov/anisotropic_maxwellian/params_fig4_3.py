"""
Historical Standard-FEM/PIC particle run for thesis Fig. 4.3.

This wrapper reuses the exact Run-1 integrator already contained in
examples/ColdPlasmaVlasov/anisotropic_maxwellian/params_4_1_4.py, but runs
with CONTROL_VARIATE=0 because the historical Fig. 4.3 particle histogram
uses the physical particle weights directly.

Run from the Struphy repository root:
    python params_fig4_3.py

No geometric ColdPlasmaVlasov code is used here.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CANDIDATES = [
    ROOT / "examples" / "ColdPlasmaVlasov" / "anisotropic_maxwellian",
    ROOT / "papers" / "01_Comp_FEEC_standard" / "01_Figures" / "Python_scripts",
]
for p in CANDIDATES:
    if (p / "Utilitis_HybridCode.py").exists():
        sys.path.insert(0, str(p))

# Import the already-checked historical Run-1 wrapper.
sys.path.insert(0, str(ROOT / "examples" / "ColdPlasmaVlasov" / "anisotropic_maxwellian"))
import params_4_1_4 as base

# Fig. 4.3 uses the direct physical particle weights.
base.CONTROL_VARIATE = 0
base.OUT = ROOT / "thesis_fig4_3_particles"
base.OUT.mkdir(parents=True, exist_ok=True)

# Keep the same deterministic seed as the Run-1 reproduction wrapper.
base.RNG_SEED = 1234

if __name__ == "__main__":
    base.run()
