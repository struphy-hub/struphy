from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "examples" / "ColdPlasmaVlasov" / "anisotropic_maxwellian"
sys.path.insert(0, str(LEGACY))

import params_fig4_7 as base


# ============================================================
# THESIS FIGURES 4.5 / 4.6 — HISTORICAL STANDARD FEM, RUN 2
# ============================================================
#
# Exact thesis parameters:
#   v_th,k   = 0.2 c
#   v_th,perp= 0.6 c
#   nu_h     = 0.05
#   Omega_pe = 2 |Omega_ce|
#   k        = 1.50 ... 3.00 in steps of 0.25
#   L        = 2 pi / k
#   N_el     = 200
#   p        = 3
#   Np       = 5e4
#   dt       = 0.05
#   Tend     = 400
#   control variate = ON
#   initial Bx = 1e-4 sin(k z)
#
# ============================================================

K_VALUES = np.arange(1.50, 3.0001, 0.25)

base.nuh = 0.05
base.nh = base.nuh * base.wpe**2

base.wpar = 0.20
base.wperp = 0.60

base.Nz = 200
base.p = 3
base.Np = 50_000

base.T = 400.0
base.dt = 0.05

base.control = 1
base.saving_step = 1

# Single-mode magnetic perturbation.
base.ini = 3
base.amp = 1.0e-4
base.eps = 0.0

base.RNG_SEED = 1234


for k in K_VALUES:

    base.k = float(k)
    base.Lz = 2.0 * np.pi / base.k

    tag = f"{base.k:.2f}".replace(".", "p")
    base.OUT = ROOT / f"thesis_run2_k{tag}"
    base.OUT.mkdir(parents=True, exist_ok=True)

    output = base.OUT / "run3_fig4_7.npz"

    if output.exists():
        print(f"\nSKIPPING existing run: k={base.k:.2f}")
        print(f"  {output}")
        continue

    print("\n" + "=" * 72)
    print(f"THESIS RUN 2 — k = {base.k:.2f}")
    print("=" * 72)
    print(f"Lz  = {base.Lz}")
    print(f"Nz  = {base.Nz}")
    print(f"p   = {base.p}")
    print(f"Np  = {base.Np}")
    print(f"dt  = {base.dt}")
    print(f"T   = {base.T}")
    print(f"CV  = {base.control}")
    print("=" * 72)

    base.run()

print("\n" + "=" * 72)
print("ALL FIGURE 4.5 / 4.6 RUNS COMPLETE")
print("=" * 72)
