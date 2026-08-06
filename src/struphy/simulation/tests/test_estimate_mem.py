import os
import shutil

import pytest

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
    perturbations,
)
from struphy.models import Maxwell, VlasovAmpereOneSpecies


def _real_vector_nbytes(vector):
    """Real (backing-array, incl. ghost/padding regions) memory footprint of a psydac
    StencilVector or BlockVector, i.e. what actually got allocated in RAM."""
    if hasattr(vector, "_data"):
        return vector._data.nbytes
    return sum(_real_vector_nbytes(block) for block in vector.blocks)


@pytest.fixture
def out_folders(tmp_path):
    out = os.path.join(str(tmp_path), "struphy_estimate_mem_tests")
    yield out
    shutil.rmtree(out, ignore_errors=True)


def test_estimate_mem_feec_only_matches_allocation(out_folders):
    """estimate_mem() must be callable before allocate(), and its FEEC estimates must
    exactly match the real (backing-array) memory footprint after allocate()."""
    model = Maxwell()

    env = EnvironmentOptions(out_folders=out_folders, sim_folder="light_wave_1d")
    time_opts = Time(dt=0.05, Tend=50.0)
    domain = domains.Cuboid(r3=20.0)
    grid = grids.TensorProductGrid(num_elements=(1, 1, 32))
    derham_opts = DerhamOptions(degree=(1, 1, 3))

    model.propagators.maxwell.options = model.propagators.maxwell.Options(algo="explicit")
    model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=0, seed=123))

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    # estimate_mem() must not require (or trigger) allocate() first
    assert not hasattr(sim, "_derham")
    mem_before = sim.estimate_mem(print_report=False)
    # and must not leave the simulation in an allocated state
    assert not hasattr(sim, "_derham")

    sim.allocate()

    for species, spec in sim.model.field_species.items():
        for k, v in spec.variables.items():
            estimated = mem_before[f"{species}.{k}"]
            actual = _real_vector_nbytes(v.spline.vector)
            assert estimated == actual, f"{species}.{k}: estimated {estimated} != actual {actual}"

    assert mem_before["total"] == sum(v for k, v in mem_before.items() if k != "total")

    # the FEEC matrices (much bigger than the coefficient vectors) are part of the estimate ...
    matrices = {k: v for k, v in mem_before.items() if k.startswith("matrices.")}
    assert set(matrices) == {f"matrices.{name}" for name in ("derivatives", "M0", "M1", "M2", "M3", "Mv")}
    assert matrices["matrices.M1"] > mem_before["em_fields.e_field"]

    # ... and the mass matrices this model really uses are estimated exactly
    allocated = sim.mass_ops.allocated_mem()
    assert set(allocated) == {"M1", "M2"}
    for name, nbytes in allocated.items():
        assert mem_before[f"matrices.{name}"] == nbytes

    # report_mem() sees all allocated stencil matrices, i.e. at least the mass matrices
    report = sim.report_mem(print_report=False)
    assert report["feec_matrices"] >= sum(allocated.values())
    assert report["spline_coeffs"] == sum(
        _real_vector_nbytes(v.spline.vector)
        for spec in sim.model.field_species.values()
        for v in spec.variables.values()
    )
    assert report["markers"] == 0
    assert report["total"] == sum(v for k, v in report.items() if k != "total")


def test_estimate_mem_hybrid_feec_and_pic(out_folders):
    """estimate_mem() on a model with both FEEC and PIC variables: FEEC estimates match
    exactly, and the PIC estimate matches the real marker-array footprint plus the
    (separately estimated) marker-saving buffer."""
    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)

    env = EnvironmentOptions(out_folders=out_folders, sim_folder="weak_Landau")
    time_opts = Time(dt=0.05, Tend=15)
    domain = domains.Cuboid(r1=12.56)
    grid = grids.TensorProductGrid(num_elements=(16, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))

    loading_params = LoadingParameters(ppc=200, seed=1234)
    weights_params = WeightsParameters(control_variate=True)
    boundary_params = BoundaryParameters()
    sorting_params = SortingParameters(boxes_per_dim=(8, 1, 1), do_sort=True)
    saving_params = SavingParameters(n_markers=50)

    model.kinetic_ions.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
        bufsize=0.4,
    )

    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

    background = maxwellians.Maxwellian3D(n=(1.0, None))
    model.kinetic_ions.var.add_background(background)

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    mem_before = sim.estimate_mem(print_report=False)
    sim.allocate()

    for species, spec in sim.model.field_species.items():
        for k, v in spec.variables.items():
            estimated = mem_before[f"{species}.{k}"]
            actual = _real_vector_nbytes(v.spline.vector)
            assert estimated == actual, f"{species}.{k}: estimated {estimated} != actual {actual}"

    markers = 0
    for species, spec in sim.model.particle_species.items():
        for k, v in spec.variables.items():
            estimated = mem_before[f"{species}.{k}"]
            actual = v.particles.nbytes_local
            if v.n_to_save > 0:
                actual += v.saved_markers.nbytes
            assert estimated == actual, f"{species}.{k}: estimated {estimated} != actual {actual}"
            markers += actual

    report = sim.report_mem(print_report=False)
    assert report["markers"] == markers
    assert report["feec_matrices"] >= sum(sim.mass_ops.allocated_mem().values()) > 0
