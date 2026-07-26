import json
from pathlib import Path

import pytest

from profiling import package_profiling_results
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
    general = metadata["general_information"]
    assert general["simulation_name"] == "Diocotron instability"
    assert general["simulation_description"] == "Shear-driven non-neutral plasma instability."


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
    general = metadata["general_information"]
    assert general["simulation_name"] == "Nested params name"
    assert general["simulation_description"] == "Nested params description"
    assert Path(metadata["software_information"]["parameter_file"]) == promoted_parameters


def _write_toy_case(testcase_dir: Path) -> None:
    _write_profiling_file(testcase_dir / "sim_ranks2" / "profiling_data.h5")
    (testcase_dir / "parameters.py").write_text(
        "\n".join(
            [
                'name = "Machine case"',
                "from struphy import Simulation",
                "sim = Simulation(model=None, name=name)",
            ]
        ),
        encoding="utf-8",
    )


def test_package_results_copies_whereami_json_verbatim(tmp_path: Path, monkeypatch) -> None:
    results_root = tmp_path / "results-root"
    testcase_dir = results_root / "toy_case"
    _write_toy_case(testcase_dir)

    # The job ran `whereami --output <case dir>/machine_params.json` on the compute node.
    machine_params = """{
  "MACHINE_NAME": "Pitagora (DCGP)",
  "MACHINE_HOST": "CINECA",
  "CPU_VENDOR": "AMD",
  "CHIP": "Genoa",
  "GPU_VENDOR": "none",
  "GPU_NAME": "none",
  "GPUS_FOUND": false,
  "MACHINE_HOSTNAME": "r350c06s02"
}"""
    (testcase_dir / "machine_params.json").write_text(machine_params, encoding="utf-8")
    monkeypatch.setenv("HPC_SYSTEM", "pitagora")

    output_root = tmp_path / "packaged"
    created_dirs = package_results(
        results_root=results_root,
        language="fortran",
        commit="19c82323312d9f83e995f2bdd8dcec2df18820c7",
        output_root=output_root,
    )

    # The whereami export is packaged unparsed, next to the metadata.
    packaged_params = created_dirs[0] / "machine_params.json"
    assert packaged_params.read_text(encoding="utf-8") == machine_params

    metadata = json.loads((created_dirs[0] / "case_metadata.json").read_text(encoding="utf-8"))
    hardware = metadata["hardware_information"]
    assert hardware["machine_params_file"] == "machine_params.json"
    assert hardware["cluster_name"] == "Pitagora (DCGP)"

    # CPU/GPU details are not duplicated into the metadata, and the github block is gone.
    assert "cpu_vendor" not in hardware
    assert "whereami" not in hardware
    assert "machine_information" not in hardware
    assert "github" not in metadata


def test_package_results_without_whereami_export(tmp_path: Path, monkeypatch) -> None:
    results_root = tmp_path / "results-root"
    testcase_dir = results_root / "toy_case"
    _write_toy_case(testcase_dir)

    monkeypatch.setattr(package_profiling_results.shutil, "which", lambda name: None)
    for variable in ("HOST", "HOSTNAME", "LMOD_ADMIN_FILE", "HPC_SYSTEM", "NERSC_HOST"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "some-cluster")

    output_root = tmp_path / "packaged"
    created_dirs = package_results(
        results_root=results_root,
        language="fortran",
        commit="19c82323312d9f83e995f2bdd8dcec2df18820c7",
        output_root=output_root,
    )

    assert not (created_dirs[0] / "machine_params.json").exists()
    metadata = json.loads((created_dirs[0] / "case_metadata.json").read_text(encoding="utf-8"))
    hardware = metadata["hardware_information"]
    assert hardware["machine_params_file"] is None
    assert hardware["cluster_name"] == "some-cluster"


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"HOST": "ravenlogin1"}, "Raven"),
        ({"HOSTNAME": "viper12-gpu"}, "Viper-GPU"),
        ({"HOSTNAME": "viper-login1"}, "Viper-CPU"),
        ({"HOST": "cobra01"}, "Cobra"),
        ({"LMOD_ADMIN_FILE": "/etc/lumi/admin"}, "LUMI-G"),
        ({"LMOD_ADMIN_FILE": "/etc/lumi/admin", "PARTITION": "LUMI-C"}, "LUMI-C"),
        ({"HPC_SYSTEM": "leonardo"}, "Leonardo (Booster)"),
        ({"HPC_SYSTEM": "leonardo", "PARTITION": "DCGP"}, "Leonardo (DCGP)"),
        ({"HPC_SYSTEM": "marconi"}, "Marconi"),
        ({"HPC_SYSTEM": "pitagora"}, "Pitagora (DCGP)"),
        ({"HOST": "toki01"}, "TOK"),
        ({"HOSTNAME": "vega-login"}, "Vega (GPU)"),
        ({"HOSTNAME": "vega-login", "PARTITION": "CPU"}, "Vega (CPU)"),
        ({"NERSC_HOST": "perlmutter"}, "Perlmutter"),
        ({"HOSTNAME": "runner-1", "CI_RUNNER_TAGS": "nvidia-cc80"}, "Shared GPU Runner (NVIDIA)"),
        ({"HOSTNAME": "runner-1", "CI_RUNNER_TAGS": "amd-mi200"}, "Shared GPU Runner (AMD)"),
        ({"HOSTNAME": "runner-1"}, "Shared Runner"),
        ({"HOSTNAME": "Maxs-MacBook-Air.local"}, None),
    ],
)
def test_detect_machine_name(environment: dict[str, str], expected: str | None, monkeypatch) -> None:
    for variable in ("HOST", "HOSTNAME", "LMOD_ADMIN_FILE", "HPC_SYSTEM", "NERSC_HOST", "CI_RUNNER_TAGS", "PARTITION"):
        monkeypatch.delenv(variable, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    # `whereami` falls back to `hostname` when HOSTNAME is unset.
    monkeypatch.setattr(package_profiling_results.socket, "gethostname", lambda: "unknown-host")

    assert package_profiling_results.detect_machine_name() == expected
