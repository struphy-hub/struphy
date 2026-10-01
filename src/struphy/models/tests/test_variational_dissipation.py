import pytest

from struphy import EnvironmentOptions, FieldsBackground, Simulation, Time, domains, equils, grids, perturbations
from struphy.models import ViscoResistiveLinearMHD_with_q, ViscoResistiveMHD


def _run(model, tmp_path):
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim"),
        time_opts=Time(dt=0.01, Tend=0.01),
        domain=domains.Cuboid(),
        equil=equils.HomogenSlab(),
        grid=grids.TensorProductGrid(num_elements=(8, 4, 1)),
    )
    sim.run()


@pytest.mark.models
@pytest.mark.fluid
@pytest.mark.parametrize("fast", [False, True])
def test_linear_q_viscosity_resistivity(fast, tmp_path):
    """Non-zero mu/eta enters the Newton iteration of VariationalViscosity/Resistivity (issue #422)."""
    model = ViscoResistiveLinearMHD_with_q()
    props = model.propagators
    props.variat_dens.options = props.variat_dens.Options(model="linear_q")
    props.variat_qb.options = props.variat_qb.Options(model="linear_q")
    props.variat_viscous.options = props.variat_viscous.Options(model="linear_q", mu=0.1, fast=fast)
    props.variat_resist.options = props.variat_resist.Options(model="linear_q", eta=0.1, fast=fast)
    model.mhd.density.add_background(FieldsBackground())
    model.mhd.sqrt_p.add_background(FieldsBackground())
    model.mhd.sqrt_p.add_perturbation(perturbations.TorusModesCos())
    _run(model, tmp_path)


@pytest.mark.models
@pytest.mark.fluid
def test_full_viscosity_resistivity(tmp_path):
    """model="full" evaluates the density 3-form in VariationalViscosity (issue #446)."""
    model = ViscoResistiveMHD()
    props = model.propagators
    props.variat_dens.options = props.variat_dens.Options(model="full")
    props.variat_viscous.options = props.variat_viscous.Options(model="full", mu=0.1, mu_a=0.1)
    props.variat_resist.options = props.variat_resist.Options(model="full", eta=0.1, eta_a=0.1)
    model.mhd.density.add_background(FieldsBackground())
    model.mhd.entropy.add_background(FieldsBackground())
    model.mhd.entropy.add_perturbation(perturbations.TorusModesCos())
    _run(model, tmp_path)
