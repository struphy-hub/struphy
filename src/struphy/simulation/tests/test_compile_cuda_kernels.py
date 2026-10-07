"""Tests for ``Simulation.kernels`` and ``Simulation.compile_cuda_kernels`` (struphy#704).

The CUDA kernels are compiled once at setup instead of at their first call in the time loop. Without a
GPU, the CuPy path is tested with the backend switched by monkeypatching and with cunumpy's fake CuPy
(in a subprocess); the test marked ``requires_cupy`` compiles the kernels of a small model on a GPU.
"""

import subprocess
import sys

import cunumpy
import pytest
from cunumpy.kernels import CudaKernel, Kernel

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids, maxwellians
from struphy.geometry.tests.test_domain import _cupy_installed, serial_child_env
from struphy.models import Vlasov, VlasovAmpereOneSpecies
from struphy.particles.parameters import BoundaryParameters, LoadingParameters
from struphy.pic.pushing.kernels.push_bxu_Hdiv import push_bxu_Hdiv
from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage
from struphy.pic.pushing.kernels.reflect import reflect
from struphy.utils.kernel_compilation import collect_kernels, missing_cuda

# simulations write to tmp_path, which differs per rank
pytestmark = pytest.mark.mpi_skip

requires_cupy = pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")

# kernels of the domain and the Derham complex, listed for every model with a Derham complex
DOMAIN_KERNELS = {"kernel_evaluate", "kernel_evaluate_pic", "kernel_pullpush", "kernel_pullpush_pic"}
DERHAM_KERNELS = {
    "eval_spline_mpi_markers",
    "eval_spline_mpi_matrix",
    "eval_spline_mpi_sparse_meshgrid",
    "stencil_dot_3d",
    "stencil_transpose_3d",
    "stencil_inner_3d",
    "stencil_axpy_3d",
}


def make_vlasov_ampere(tmp_path, bc=("periodic", "periodic", "periodic")) -> Simulation:
    """VlasovAmpereOneSpecies on 8 x 1 x 1 elements with 5 markers per cell, one time step."""
    model = VlasovAmpereOneSpecies(with_B0=False)
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=5, seed=1234),
        boundary_params=BoundaryParameters(bc=list(bc)),
    )
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim_1"),
        time_opts=Time(dt=0.05, Tend=0.05),
        domain=domains.Cuboid(r1=12.56),
        grid=grids.TensorProductGrid(num_elements=(8, 1, 1)),
        derham_opts=DerhamOptions(degree=(2, 1, 1)),
    )


def make_vlasov(tmp_path) -> Simulation:
    """Vlasov (PushVxB and PushEta) on 4 x 4 x 1 elements, whose kernels all have CUDA versions."""
    model = Vlasov()
    model.kinetic_ions.set_markers(loading_params=LoadingParameters(ppc=5, seed=1234))
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim_1"),
        time_opts=Time(dt=0.05, Tend=0.05),
        domain=domains.Cuboid(),
        equil=equils.HomogenSlab(),
        grid=grids.TensorProductGrid(num_elements=(4, 4, 1)),
        derham_opts=DerhamOptions(degree=(1, 1, 1)),
    )


def names(kernels) -> set[str]:
    return {kernel.name for kernel in kernels}


def test_collect_kernels_walks_owners_and_containers():
    """Kernels, ``kernels()`` methods and containers are walked; each kernel is listed once, in order."""

    class Owner:
        def __init__(self, *kernels):
            self._kernels = kernels

        def kernels(self):
            return self._kernels

    collected = collect_kernels(
        None,
        "not an owner",
        push_eta_stage,
        Owner(push_eta_stage, reflect),
        {"a": [Owner(push_bxu_Hdiv)], "b": (reflect,)},
        Owner,  # a class with a kernels() method is not an owner
    )
    assert collected == (push_eta_stage, reflect, push_bxu_Hdiv)
    assert missing_cuda(collected) == [push_bxu_Hdiv]


def test_kernels_of_a_small_model(tmp_path):
    """The kernels of VlasovAmpereOneSpecies: domain, Derham and its two propagators."""
    sim = make_vlasov_ampere(tmp_path)
    sim.allocate()

    kernels = sim.kernels()
    assert all(isinstance(kernel, Kernel) for kernel in kernels)
    assert len(set(map(id, kernels))) == len(kernels)
    assert names(kernels) == DOMAIN_KERNELS | DERHAM_KERNELS | {
        "push_eta_stage",
        "vlasov_maxwell",
        "push_v_with_efield",
    }
    assert names(sim.model.propagators.push_eta.kernels()) == {"push_eta_stage"}
    assert names(sim.model.propagators.coupling_va.kernels()) == {"vlasov_maxwell", "push_v_with_efield"}
    # every kernel of this model has a CUDA version (vlasov_maxwell since #713)
    assert missing_cuda(kernels) == []


def test_kernels_of_reflecting_particles(tmp_path):
    """Particles with a reflecting boundary list the reflection kernel."""
    sim = make_vlasov_ampere(tmp_path, bc=("reflect", "periodic", "periodic"))
    sim.allocate()
    assert sim.model.kinetic_ions.var.particles.kernels() == (reflect,)
    assert reflect in sim.kernels()


def test_kernels_cover_the_time_loop(tmp_path, monkeypatch):
    """Every kernel called in the time loop of a run is listed by ``Simulation.kernels()``."""
    sim = make_vlasov_ampere(tmp_path)
    called = {}
    in_loop = [False]
    call = Kernel.__call__

    def recording_call(self, *args, **kwargs):
        if in_loop[0]:
            called[id(self)] = self
        return call(self, *args, **kwargs)

    integrate = sim.model.integrate

    def integrate_in_loop(*args, **kwargs):
        in_loop[0] = True
        try:
            return integrate(*args, **kwargs)
        finally:
            in_loop[0] = False

    monkeypatch.setattr(Kernel, "__call__", recording_call)
    monkeypatch.setattr(sim.model, "integrate", integrate_in_loop)
    sim.run(one_time_step=True)

    assert called, "no kernel was called in the time loop"
    listed = {id(kernel) for kernel in sim.kernels()}
    assert names(k for i, k in called.items() if i not in listed) == set()


def test_compile_cuda_kernels_is_a_no_op_on_numpy(tmp_path, monkeypatch):
    """On NumPy nothing is compiled or checked, also when the model has kernels without CUDA version."""

    def fail(self):
        raise AssertionError(f"{self.name} compiled on the NumPy backend")

    monkeypatch.setattr(Kernel, "compile", fail)
    sim = make_vlasov_ampere(tmp_path)
    with cunumpy.use_backend("numpy"):
        sim.allocate()
        assert sim.compile_cuda_kernels() == ()


def test_compile_cuda_kernels_on_cupy_backend(tmp_path, monkeypatch):
    """On CuPy (backend switched by monkeypatching) every kernel is compiled; missing CUDA versions raise first."""
    sim = make_vlasov_ampere(tmp_path)
    compiled = []
    monkeypatch.setattr(CudaKernel, "compile", lambda self, **kwargs: compiled.append(self.name))
    monkeypatch.setattr(cunumpy, "get_backend", lambda: "cupy")

    monkeypatch.setattr(Simulation, "kernels", lambda self: (push_eta_stage, reflect))
    assert sim.compile_cuda_kernels() == (push_eta_stage, reflect)
    assert compiled == ["push_eta_stage", "reflect"]

    compiled.clear()
    monkeypatch.setattr(Simulation, "kernels", lambda self: (push_eta_stage, push_bxu_Hdiv, reflect))
    with pytest.raises(NotImplementedError, match=r"1 of the\s+3 kernels.*\n  - push_bxu_Hdiv \(expected at .*"):
        sim.compile_cuda_kernels()
    assert compiled == []


FAKE_CUPY_CHECK = """
import cunumpy
from cunumpy.kernels import CudaKernel

from struphy import Simulation
from struphy.models import Maxwell
from struphy.pic.pushing.kernels.push_bxu_Hdiv import push_bxu_Hdiv
from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage
from struphy.pic.pushing.kernels.reflect import reflect

compiled = []
CudaKernel.compile = lambda self, **kwargs: compiled.append(self.name)
sim = Simulation(model=Maxwell())
with cunumpy.use_backend("numpy"):
    Simulation.kernels = lambda self: (push_eta_stage, push_bxu_Hdiv)
    assert sim.compile_cuda_kernels() == ()
with cunumpy.use_backend("cupy"):
    try:
        sim.compile_cuda_kernels()
    except NotImplementedError as error:
        assert "push_bxu_Hdiv" in str(error) and "push_eta_stage" not in str(error), error
    else:
        raise AssertionError("a kernel without CUDA version did not raise")
    assert compiled == []
    Simulation.kernels = lambda self: (push_eta_stage, reflect)
    assert sim.compile_cuda_kernels() == (push_eta_stage, reflect)
    assert compiled == ["push_eta_stage", "reflect"]
"""


@pytest.mark.skipif(_cupy_installed(), reason="the fake CuPy cannot replace an installed CuPy")
def test_compile_cuda_kernels_fake_cupy():
    """With cunumpy's fake CuPy the CuPy backend is active: kernels are checked and compiled (compile recorded)."""
    result = subprocess.run(
        [sys.executable, "-c", FAKE_CUPY_CHECK],
        env=serial_child_env(CUNUMPY_FAKE_CUPY="1"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]


@requires_cupy
def test_compile_cuda_kernels_on_gpu(tmp_path):
    """On a GPU: the kernels of Vlasov and VlasovAmpereOneSpecies are compiled at setup, a kernel without CUDA
    version fails fast."""
    with cunumpy.use_backend("cupy"):
        sim = make_vlasov(tmp_path)
        sim.allocate()
        compiled = sim.compile_cuda_kernels()
        assert compiled == sim.kernels()
        assert {"push_vxb_analytic", "push_eta_stage"} <= names(compiled)

        sim = make_vlasov_ampere(tmp_path / "va")
        sim.allocate()
        compiled = sim.compile_cuda_kernels()
        assert compiled == sim.kernels()
        assert "vlasov_maxwell" in names(compiled)

        sim.kernels = lambda: (push_eta_stage, push_bxu_Hdiv)
        with pytest.raises(NotImplementedError, match="push_bxu_Hdiv"):
            sim.compile_cuda_kernels()
