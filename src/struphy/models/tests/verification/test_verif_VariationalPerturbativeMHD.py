"""Verification of the nonlinear VariationalPerturbativeMHD with the velocity in H(div) (u_space="Hdiv"): energy
conservation, the energy balance of the viscous step, and agreement with the velocity in H1vec."""

import numpy as np
import pytest
from maybempi import MPI

from struphy import DerhamOptions, Simulation, Time, domains, equils, grids, perturbations
from struphy.models import VariationalPerturbativeMHD


def _model(u_space, map_name, num_elements=(8, 8, 1), degree=2, mu=0.0, eta=0.0, dt=0.05):
    """The nonlinear model with a velocity perturbation (normal to the wall on the torus, where it vanishes),
    allocated (no output)."""
    model = VariationalPerturbativeMHD(u_space=u_space)
    if map_name == "HollowTorus":
        # the geometry of the TAE test cases
        domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=5.0, sfl=False, pol_period=1, tor_period=1)
        equil = equils.AdhocTorus(
            a=1.0,
            R0=5.0,
            B0=1.0,
            q_kind=0,
            p_kind=1,
            q0=1.25,
            q1=2.25,
            n1=0.0,
            n2=0.0,
            na=1.0,
            p1=0.95,
            p2=0.05,
            beta=0.0018,
        )
        pert = perturbations.TorusModesCos(ms=(1,), ns=(-1,), amps=(1e-2,), comp=0)
        bcs = (("dirichlet", "dirichlet"), None, None)
    else:
        domain = domains.Cuboid()
        equil = equils.HomogenSlab(B0x=0.3, B0y=0.0, B0z=1.0, beta=0.5, n0=1.0)
        pert = perturbations.ModesCos(ls=(1,), ms=(1,), amps=(1e-2,), comp=0)
        bcs = (None, None, None)
    sim = Simulation(
        model=model,
        time_opts=Time(dt=dt, Tend=dt),
        domain=domain,
        equil=equil,
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=tuple(1 if n == 1 else degree for n in num_elements), bcs=bcs),
    )
    props = model.propagators
    props.variat_dens.options = props.variat_dens.Options(model="perturbative")
    props.variat_pb.options = props.variat_pb.Options(model="perturbative")
    props.variat_viscous.options = props.variat_viscous.Options(model="full_p", mu=mu)
    props.variat_resist.options = props.variat_resist.Options(model="full_p", eta=eta)
    model.mhd.velocity.add_perturbation(pert)
    sim.allocate()
    return model, domain


def _scalars(model, names):
    model.scalars.update()
    model.scalars.to_host()
    return {k: float(model.scalars.host_value(k)[0]) for k in names}


# the Cuboid case is the core test that also runs on pull requests
@pytest.mark.parametrize("map_name", [pytest.param("Cuboid", marks=pytest.mark.verif_pr), "HollowTorus"])
def test_hdiv_energy_conservation(map_name):
    """The total energy of the ideal nonlinear model is conserved, while energy is exchanged between kinetic,
    magnetic and thermal parts."""
    num_elements = (8, 8, 1) if map_name == "Cuboid" else (6, 6, 4)
    dt = 0.05 if map_name == "Cuboid" else 0.5
    model, _ = _model("Hdiv", map_name, num_elements=num_elements, degree=2 if map_name == "Cuboid" else 3, dt=dt)
    names = ["en_U", "en_mag_1", "en_tot"]
    s0 = _scalars(model, names)
    for _ in range(3):
        model.integrate(dt, split_algo="LieTrotter")
    s1 = _scalars(model, names)
    assert abs(s1["en_tot"] - s0["en_tot"]) / s0["en_tot"] < 1e-7, f"{s0 = }, {s1 = }"
    # the dynamics converts kinetic into magnetic energy
    assert s1["en_U"] < 0.99 * s0["en_U"]
    assert s1["en_mag_1"] > 1e-3 * s0["en_U"]


def test_hdiv_viscous_heating():
    """The viscous step dissipates kinetic energy into the pressure, conserving their sum."""
    model, _ = _model("Hdiv", "Cuboid", mu=0.05)
    names = ["en_U", "en_thermo_l1"]
    s0 = _scalars(model, names)
    model.propagators.variat_viscous(0.05)
    s1 = _scalars(model, names)
    dissipated = s0["en_U"] - s1["en_U"]
    assert dissipated > 1e-3 * s0["en_U"]
    assert abs((s1["en_thermo_l1"] - s0["en_thermo_l1"]) - dissipated) < 1e-8 * s0["en_U"], f"{s0 = }, {s1 = }"


def test_hdiv_agrees_with_h1vec():
    """The velocities in H(div) and in H1vec are consistent discretizations: their results agree up to the
    discretization error, which decreases under refinement."""
    pts = [np.linspace(0.05, 0.95, 6), np.linspace(0.0, 1.0, 9)[:-1], np.array([0.0])]
    E = np.meshgrid(*pts, indexing="ij")

    diffs = []
    for n in (6, 12):
        vals = {}
        for u_space in ("Hdiv", "H1vec"):
            model, domain = _model(u_space, "Cuboid", num_elements=(n, n, 1), degree=2, dt=0.05)
            for _ in range(4):
                model.integrate(0.05, split_algo="LieTrotter")
            v = model.mhd.velocity.spline(*E)
            vals[u_space] = np.array([c / domain.jacobian_det(*E) for c in v] if u_space == "Hdiv" else v)
        diffs += [np.linalg.norm(vals["Hdiv"] - vals["H1vec"]) / np.linalg.norm(vals["H1vec"])]
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"{diffs = }")
    assert diffs[1] < 0.5 * diffs[0], f"{diffs = }"
    assert diffs[1] < 0.05, f"{diffs = }"


if __name__ == "__main__":
    test_hdiv_energy_conservation("HollowTorus")
    test_hdiv_viscous_heating()
    test_hdiv_agrees_with_h1vec()
