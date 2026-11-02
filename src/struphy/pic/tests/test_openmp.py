"""Compare threaded compiled kernels with serial Python loops in fresh processes.

Run after ``struphy compile --openmp``. Separate processes ensure that the
OpenMP runtime reads each requested thread count before extensions are loaded.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.mpi_skip
def test_openmp_matches_serial_loops(tmp_path):
    from struphy.utils.utils import read_state

    if not (read_state().get("last_used_omp") or "").strip():
        pytest.skip("compile kernels with --openmp to exercise threaded execution")

    for mode in ("reference", "1", "2", "4"):
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="1" if mode == "reference" else mode, OMP_DYNAMIC="FALSE", OMP_STACKSIZE="64M")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode, str(tmp_path / f"{mode}.npz")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

    with np.load(tmp_path / "reference.npz") as expected:
        for threads in (1, 2, 4):
            with np.load(tmp_path / f"{threads}.npz") as actual:
                assert actual.files == expected.files
                for name in expected.files:
                    np.testing.assert_allclose(
                        actual[name],
                        expected[name],
                        rtol=2e-12,
                        atol=2e-12,
                        err_msg=f"{name}, OMP_NUM_THREADS={threads}",
                    )


def _run_cases(reference, output):
    import ast
    import importlib
    import importlib.util

    from struphy.pic.tests.cuda_parity_cases import PARITY_CASES
    from struphy.pic.tests.kernel_test_args import analytic_domains, derham_arguments, marker_arguments

    results = {}
    root = Path(__file__).resolve().parents[2]

    def execute(package, name, args, case):
        source = root / package / name / f"{name}_kernels.py"
        module_name = f"struphy.{package.replace('/', '.')}.{name}.{name}_kernels"
        if reference:
            spec = importlib.util.spec_from_file_location(f"_serial_{name}", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
            assert Path(module.__file__).suffix != ".py", f"{name} is not compiled"
        getattr(module, name)(*args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                results[f"{name}-{case}-{i}"] = arg.copy()
            elif hasattr(arg, "markers"):
                results[f"{name}-{case}-markers"] = arg.markers.copy()

    # Spatially varying fields, heterogeneous degrees, holes, and kinetic boundaries.
    for name in (
        "push_eta_stage",
        "push_v_with_efield",
        "push_vxb_analytic",
        "push_vxb_implicit",
        "push_weights_with_efield_lin_va",
        "charge_density_0form",
    ):
        spec = PARITY_CASES[name]
        package = "pic/accumulation/kernels" if name == "charge_density_0form" else "pic/pushing/kernels"
        for case_id, case in enumerate(spec.cases):
            execute(package, name, spec.build(case), case_id)

    # Exercise all restored deposition reductions, including nonzero initial outputs.
    for name in (
        "gc_mag_density_0form",
        "gc_density_0form",
        "cc_lin_mhd_6d_1",
        "cc_lin_mhd_6d_2",
        "cc_lin_mhd_5d_M",
        "cc_lin_mhd_5d_D",
        "vlasov_maxwell",
        "linear_vlasov_ampere",
        "div_u_weak_1form",
    ):
        source = root / "pic/accumulation/kernels" / name / f"{name}_kernels.py"
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef))
        for basis in (0, 1, 2) if any(a.arg == "basis_u" for a in fn.args.args) else (0,):
            markers, _ = marker_arguments((0, 0, 0))
            markers.markers[0, 0] = -1.0
            derham = derham_arguments()
            domain = analytic_domains()[2].args_domain  # nonconstant metric exposes scratch-array races
            rng = np.random.default_rng(42)
            shape = (14, 16, 12)
            args = []
            for param in fn.args.args:
                key = param.arg
                annotation = ast.unparse(param.annotation)
                if key == "args_markers":
                    value = markers
                elif key == "args_derham":
                    value = derham
                elif key == "args_domain":
                    value = domain
                elif key.startswith("mat"):
                    value = np.full((*shape, 5, 7, 3), 0.125)
                elif key.startswith("vec"):
                    value = np.full(shape, 0.125)
                elif key == "f0_values":
                    value = rng.uniform(0.1, 1.0, markers.n_markers)
                elif ":" in annotation:
                    value = rng.uniform(0.5, 1.5, shape)
                elif key == "basis_u":
                    value = basis
                elif key == "boundary_cut":
                    value = 0.0
                else:
                    value = 0.7
                args.append(value)
            execute("pic/accumulation/kernels", name, args, basis)

    # Reduction and per-particle diagnostic kernels also evaluate shared spline data.
    for name in ("canonical_kinetic_particles", "eval_magnetic_background_energy"):
        markers, domain = marker_arguments((0, 0, 0))
        markers.markers[0, 0] = -1.0
        derham = derham_arguments()
        coeffs = tuple(np.random.default_rng(i).uniform(0.5, 1.5, (18, 20, 16)) for i in range(3))
        if name == "canonical_kinetic_particles":
            args = (np.zeros(1), markers.markers, derham, domain, *coeffs)
        else:
            args = (markers.markers, derham, domain, markers.first_diagnostics_idx, markers.mu_idx, coeffs[0])
        execute("pic/diagnostics/kernels", name, args, 0)

    np.savez_compressed(output, **results)


if __name__ == "__main__":
    _run_cases(sys.argv[1] == "reference", sys.argv[2])
