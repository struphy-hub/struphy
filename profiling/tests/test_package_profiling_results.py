import json
from pathlib import Path

from profiling.package_profiling_results import package_results


def _write_profiling_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_package_results_reads_sim_metadata_from_attribute_constructor(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results-root"
    testcase_dir = results_root / "toy_case"
    _write_profiling_file(testcase_dir / "sim_ranks1" / "profiling_data.h5")
    (testcase_dir / "parameters.py").write_text(
        "\n".join(
            [
                "import struphy as sp",
                'name = "Diocotron instability"',
                'description = "Shear-driven non-neutral plasma instability."',
                "sim = sp.Simulation(model=None, name=name, description=description)",
            ]
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "packaged"
    created_dirs = package_results(
        results_root=results_root,
        language="fortran",
        commit="19c82323312d9f83e995f2bdd8dcec2df18820c7",
        output_root=output_root,
    )

    metadata_path = created_dirs[0] / "case_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["name"] == "Diocotron instability"
    assert metadata["description"] == "Shear-driven non-neutral plasma instability."


def test_package_results_promotes_nested_parameters_and_uses_metadata(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results-root"
    testcase_dir = results_root / "toy_case"
    _write_profiling_file(testcase_dir / "sim_ranks2" / "profiling_data.h5")
    (testcase_dir / "sim_ranks2" / "parameters.py").write_text(
        "\n".join(
            [
                'name = "Nested params name"',
                'description = "Nested params description"',
                "from struphy import Simulation",
                "sim = Simulation(model=None, name=name, description=description)",
            ]
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "packaged"
    created_dirs = package_results(
        results_root=results_root,
        language="fortran",
        commit="19c82323312d9f83e995f2bdd8dcec2df18820c7",
        output_root=output_root,
    )

    promoted_parameters = testcase_dir / "parameters.py"
    assert promoted_parameters.exists()

    metadata_path = created_dirs[0] / "case_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["name"] == "Nested params name"
    assert metadata["description"] == "Nested params description"
    assert Path(metadata["source_parameters_file"]) == promoted_parameters


def test_package_results_uses_whereami_variables_for_hardware_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    results_root = tmp_path / "results-root"
    testcase_dir = results_root / "toy_case"
    _write_profiling_file(testcase_dir / "sim_ranks2" / "profiling_data.h5")
    (testcase_dir / "parameters.py").write_text(
        "\n".join(
            [
                'name = "Whereami case"',
                "from struphy import Simulation",
                "sim = Simulation(model=None, name=name)",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MACHINE_NAME", "cluster-alpha")
    monkeypatch.setenv("MACHINE_HOST", "alpha-host")
    monkeypatch.setenv("CPU_VENDOR", "ExampleCPU")
    monkeypatch.setenv("CHIP", "ExampleChip")
    monkeypatch.setenv("GPU_VENDOR", "ExampleGPUVendor")
    monkeypatch.setenv("GPU_NAME", "ExampleGPU")
    monkeypatch.setenv("GPUS_FOUND", "4")
    monkeypatch.setenv("MACHINE_HOSTNAME", "alpha-node-001")

    output_root = tmp_path / "packaged"
    created_dirs = package_results(
        results_root=results_root,
        language="fortran",
        commit="19c82323312d9f83e995f2bdd8dcec2df18820c7",
        output_root=output_root,
    )

    metadata_path = created_dirs[0] / "case_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hardware = metadata["hardware_information"]

    assert hardware["cluster_name"] == "cluster-alpha"
    assert hardware["whereami"]["MACHINE_NAME"] == "cluster-alpha"
    assert hardware["machine_information"]["machine_host"] == "alpha-host"
    assert hardware["machine_information"]["cpu_vendor"] == "ExampleCPU"
    assert hardware["machine_information"]["gpus_found"] == "4"
