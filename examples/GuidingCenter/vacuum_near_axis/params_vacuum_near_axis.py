"""Guiding-centre orbits near the magnetic axis of a helical vacuum field with high-order (homotopy) polar splines of
regularity r: energy conservation of the explicit RK4 pusher with the projected background (the orbit study of the
polar FEEC talk of R. Remmerswaal)."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--regularity", type=int, default=2, help="Regularity r of the polar splines at the axis.")
parser.add_argument("--degree", type=int, default=4, help="Spline degree in eta1 and eta2.")
parser.add_argument("--num-elements", type=int, nargs=3, default=[8, 12, 16], help="Number of elements.")
parser.add_argument("--dt", type=float, default=0.04, help="Time step of RK4.")
parser.add_argument("--periods", type=float, default=20, help="Number of toroidal periods (transits of the cylinder).")
parser.add_argument("--epsilon", type=float, default=0.01, help="Normalized cyclotron period (size of the drifts).")
parser.add_argument("--save-step", type=int, default=5, help="Save the orbits every save_step time steps.")
parser.add_argument("--test-run", action="store_true", help="Small resolution and a few time steps, for the tests.")
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements, args.degree, args.regularity, args.periods, args.save_step = [4, 9, 4], 3, 2, 0.02, 1

import numpy as np

LZ = 2 * np.pi
A = 0.3
tend = args.periods * LZ

name = f"Guiding-centre orbits near the axis of a helical vacuum field, polar splines r = {args.regularity}"
description = f"""
Guiding-centre orbits near the magnetic axis of a straight helical vacuum field B = B0 grad(z + delta/k I_2(k r)
sin(2 theta - k z)) (current-free, elliptical rotating flux surfaces, three periods along the cylinder), traced with
the explicit RK4 pusher of the GuidingCenter model in the background projected into the polar spline spaces.

Physics. The guiding centres move along the field lines and drift with the grad-B and curvature drifts; the energy
E = v_par^2 / 2 + mu |B| and the magnetic moment mu are invariants. The model evaluates the background (|B|, the unit
vector b, grad |B| and (curl b) . b) from its commuting projections into the spline spaces. Near the axis (the pole
of the polar coordinates of the cross-section), the regularity of these splines at the axis determines how smooth
the projected field is, and thereby how well RK4 conserves the energy of orbits that pass close to or through the
axis. The talk shows that the energy error of RK4 decreases like dt^4 only for sufficiently regular (and
r-compatible) splines.

Setup. Cylinder of radius {A} and length 2 pi (HollowCylinder(a1=0)), {args.num_elements} elements, degree
{args.degree}, DerhamOptions(polar_splines=True, polar_regularity={args.regularity}), HelicalVacuumField(B0=1,
delta=0.5, m=2, k=3). Eight markers at distances 0.01 ... 0.04 from the axis, co- and counter-passing (v_par = +-1)
with mu = 1e-3, epsilon = {args.epsilon}. RK4 with dt = {args.dt} for {args.periods} transits (t = {tend:.4g}).

Expected result. The pproc plots the orbits projected onto the cross-section (overview and close-up) and the
relative energy error max_t |E(t) - E(0)| / (mu |B|) of each orbit. For the dt-convergence of the talk run
convergence_vacuum_near_axis.py (dt = 0.04 ... 0.0025, r = 0 ... 4; a long study).

Not stated in the talk and chosen here: the vacuum field, the geometry and the markers.

References:
- R. Remmerswaal, High-order FEEC discretization on 2D and 3D domains with a disk-like coordinate singularity
  (draft); talk at the NMPP seminar 2026; PolarFEEC.jl (https://gitlab.mpcdf.mpg.de/rwr/polarfeec.jl).
- R. G. Littlejohn, "Variational principles of guiding centre motion", J. Plasma Phys. 29, 111 (1983).
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
)
from struphy.models import GuidingCenter
from struphy.ode.utils import ButcherTableau

model = GuidingCenter(base_units=BaseUnits(), epsilon=args.epsilon)
model.kinetic_ions.var.save_data = True

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, save_step=args.save_step)
time_opts = Time(dt=args.dt, Tend=tend, split_algo="LieTrotter")
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements), mpi_dims_mask=(False, True, True))
domain = domains.HollowCylinder(a1=0.0, a2=A, Lz=LZ)
derham_opts = DerhamOptions(
    degree=(args.degree, args.degree, 3),
    bcs=(("free", "free"), None, None),
    polar_splines=True,
    polar_regularity=args.regularity,
)
equil = equils.HelicalVacuumField(B0=1.0, delta=0.5, m=2, k=3.0)

sim = Simulation(
    model=model,
    name=name,
    description=description,
    params_path=__file__,
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)

# eight markers (eta1, eta2, eta3, v_par, mu) at distances 0.01 ... 0.04 from the axis
MU = 1e-3
initial = tuple(
    (r / A, angle, 0.0, sign, MU)
    for r, angle in ((0.01, 0.0), (0.02, 0.125), (0.03, 0.25), (0.04, 0.375))
    for sign in (1.0, -1.0)
)

model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(Np=len(initial), seed=1234, specific_markers=initial),
    weights_params=WeightsParameters(),
    boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
    saving_params=SavingParameters(n_markers=len(initial)),
    bufsize=2.0,
)

butcher = ButcherTableau(algo="rk4")
model.propagators.push_bxe.options = model.propagators.push_bxe.Options(algo="explicit", butcher=butcher)
model.propagators.push_parallel.options = model.propagators.push_parallel.Options(algo="explicit", butcher=butcher)

model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(1.0, None), B0=1.0))

if __name__ == "__main__":
    # post-processing: pproc_vacuum_near_axis.py
    sim.run()
