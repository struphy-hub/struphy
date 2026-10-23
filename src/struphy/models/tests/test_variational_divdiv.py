"""Energy conservation of the variational MHD/fluid models with the div-div regularized kinetic energy."""

import numpy as np
import pytest
from maybempi import MPI

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)
from struphy.models import VariationalCompressibleFluid, VariationalMHD


def _run(model, tmp_path, steps=5):
    # dt = 0.01: with the div-div term, the (default) Picard iteration of VariationalMomentumAdvection
    # converges only for small time steps (see kinetic_metric)
    dt = 0.01
    sim = Simulation(
        model=model,
        # tmp_path differs per rank: all ranks write to the folder of rank 0
        env=EnvironmentOptions(out_folders=MPI.COMM_WORLD.bcast(str(tmp_path), root=0), sim_folder="sim"),
        time_opts=Time(dt=dt, Tend=steps * dt),
        domain=domains.Cuboid(r1=2.0, r2=3.0),
        equil=equils.HomogenSlab(B0x=0.0, B0y=0.5, B0z=1.0, beta=1.0),
        grid=grids.TensorProductGrid(num_elements=(12, 4, 1)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    return sim.run()


def _init(species):
    species.density.add_background(FieldsBackground(type="FluidEquilibrium", variable="n3"))
    species.entropy.add_background(FieldsBackground(type="FluidEquilibrium", variable="s3_monoatomic"))
    # compressive velocity mode u_x = 0.2 sin(2 pi x / Lx): div u != 0
    species.velocity.add_perturbation(perturbations.ModesSin(ls=(1,), amps=(0.2,), Lx=2.0, comp=0))


@pytest.mark.models
@pytest.mark.fluid
@pytest.mark.parametrize("with_viscosity", [False, True])
@pytest.mark.parametrize("divdiv_alpha", [0.0, 0.5])
def test_variational_mhd_divdiv_energy(divdiv_alpha, with_viscosity, tmp_path):
    """The total energy (with the div-div part of the kinetic energy) is conserved, also with viscous heating."""
    model = VariationalMHD(with_viscosity=with_viscosity, with_resistivity=False, divdiv_alpha=divdiv_alpha)
    props = model.propagators
    props.variat_dens.options = props.variat_dens.Options(model="full")
    if with_viscosity:
        props.variat_viscous.options = props.variat_viscous.Options(model="full", mu=0.05)
    _init(model.mhd)
    model.em_fields.b_field.add_background(FieldsBackground(type="FluidEquilibrium", variable="b2"))

    sc = _run(model, tmp_path).scalars
    en_tot = sc["en_tot"].values
    assert np.max(np.abs(en_tot - en_tot[0])) / en_tot[0] < 1e-9

    if divdiv_alpha > 0.0:
        en_divdiv = sc["en_divdiv"].values
        assert en_divdiv[0] > 1e-3 * en_tot[0]
        np.testing.assert_allclose(
            en_tot, sc["en_U"] + sc["en_mag"] + sc["en_thermo"] + en_divdiv, rtol=1e-12, atol=1e-12
        )
    else:
        assert "en_divdiv" not in sc


@pytest.mark.models
@pytest.mark.fluid
def test_compressible_fluid_divdiv_energy(tmp_path):
    """VariationalCompressibleFluid conserves the regularized energy, and the regularization changes the flow."""
    results = {}
    for divdiv_alpha in (0.0, 0.5):
        model = VariationalCompressibleFluid(divdiv_alpha=divdiv_alpha)
        model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model="full")
        _init(model.fluid)
        sc = _run(model, tmp_path / f"alpha_{divdiv_alpha}").scalars
        en_tot = sc["en_tot"].values
        assert np.max(np.abs(en_tot - en_tot[0])) / en_tot[0] < 1e-9
        results[divdiv_alpha] = sc["en_U"].values

    # the div-div term resists compression: less kinetic energy goes into the internal energy
    assert not np.allclose(results[0.0], results[0.5], rtol=1e-6)
