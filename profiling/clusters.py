"""Static per-cluster SLURM settings and cluster detection, shared across profiling cases.

`job_name` and `ntasks_per_node` are filled in per rank count at submission time by
each `ProfilingCase`.
"""

import getpass
import os
import socket
from pathlib import Path


def detect_machine_name() -> str | None:
    """Name of the current HPC machine

    Returns None on an unrecognised machine (e.g. a laptop).
    """
    host = os.environ.get("HOST", "")
    hostname = os.environ.get("HOSTNAME") or socket.gethostname()
    lmod_admin_file = os.environ.get("LMOD_ADMIN_FILE", "")
    hpc_system = os.environ.get("HPC_SYSTEM", "")
    nersc_host = os.environ.get("NERSC_HOST", "")
    runner_tags = os.environ.get("CI_RUNNER_TAGS", "")
    partition = os.environ.get("PARTITION", "")

    if "raven" in host:
        return "raven"
    if "viper12" in hostname:
        return "viper_gpu"
    if "viper" in hostname:
        return "viper_cpu"
    if "cobra" in host:
        return "cobra"
    if "lumi" in lmod_admin_file:
        lumi_partition = partition or "LUMI-G"
        if lumi_partition in ("LUMI-G", "LUMI-C", "LUMI-D"):
            return lumi_partition.lower().replace("-", "_")
        print(f"Unsupported LUMI partition: {lumi_partition}")
        return None
    if "leonardo" in hpc_system:
        return "leonardo_booster" if (partition or "Booster") == "Booster" else "leonardo_dcgp"
    if "marconi" in hpc_system:
        return "marconi"
    if "pitagora" in hpc_system:
        return "pitagora_dcgp"
    if "toki" in host:
        return "toki"
    if "vega" in hostname:
        return "vega_gpu" if (partition or "GPU") == "GPU" else "vega_cpu"
    if "perlmutter" in nersc_host:
        return "perlmutter"
    if "runner" in hostname:
        if "nvidia-cc80" in runner_tags:
            return "shared_gpu_runner_nvidia"
        if "amd-mi200" in runner_tags:
            return "shared_gpu_runner_amd"
        return "shared_runner"
    return None


SLURM_PRESETS: dict[str, dict] = {
    "pitagora_dcgp": {
        # "nodes": 1, # Should be set by ProfilingCase.launch()
        # "ntasks_per_node": 1,  # Should be set by ProfilingCase.launch()
        "cpus_per_task": 1,
        "mem": "480GB",
        "partition": "dcgp_fua_dbg",
        "account": "FUSIO_HLST_7",
        "output": "./%x.%j.out",
        "error": "./%x.%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
    # Pitagora's GPU (booster) partition, one node with its 4 GPUs. Not detected by `detect_machine_name`
    # (the login nodes are shared with the DCGP partition), so a run picks it with `launch(slurm_preset=...)`.
    "pitagora_booster": {
        "cpus_per_task": 1,
        "mem": "480GB",
        "gres": "gpu:4,tmpfs:10g",
        "partition": "boost_fua_dbg",
        "account": "FUSIO_HLST_6",
        "output": "./%x.%j.out",
        "error": "./%x.%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
    "tok": {
        "cpus_per_task": 1,
        "mem_per_cpu": "1GB",
        "partition": "s.tok",
        "qos": "tok.debug",
        "chdir": "./",
        "output": "./%x.%j.out",
        "error": "./%x.%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
    # MPCDF Raven, standard CPU nodes (72 cores, 256 GB). Jobs on at most half a node (36 cores) share the node with
    # other jobs and can request at most 120000 MB; larger jobs get their nodes exclusively, with up to 240000 MB per
    # node. `mem` is the default of the shared jobs (64000 MB, enough for most profiling runs, queues shorter than a
    # whole node); the exclusive jobs request the memory of the whole node (`SHARED_NODES`). A run can request other
    # memory with `slurm_options={"mem": ...}` (all jobs) or `{"mem_shared": ...}` (only the shared jobs; above 120000
    # MB they allocate the whole node). No account or partition needed.
    "raven": {
        "cpus_per_task": 1,
        "mem": "64000",
        "output": "./%x.%j.out",
        "error": "./%x.%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
    # MPCDF Viper, CPU nodes (128 cores, 512 GB; nodes with more memory are not used). As on Raven: jobs on at most half
    # a node (64 cores) share the node with other jobs and can request at most 250000 MB; larger jobs get their nodes
    # exclusively, with up to 480000 MB per node (`SHARED_NODES`). `mem` is the default of the shared jobs. No account
    # or partition needed (a submit filter chooses the partition).
    "viper_cpu": {
        "cpus_per_task": 1,
        "mem": "64000",
        "output": "./%x.%j.out",
        "error": "./%x.%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
}

# Clusters whose nodes are shared by small jobs: a job with at most `max_cores_per_node` cores per node (tasks times
# cpus_per_task) may share its node with other jobs and request at most `max_mem_shared` (MB) per node (the `mem` of
# the preset, or the `mem_shared` run option of `ProfilingCase.launch`); a job with more cores per node gets its nodes
# exclusively and requests `mem_exclusive` (MB, the memory of a whole node), unless the run sets `mem`. A small job
# that needs more than `max_mem_shared` allocates all cores of its node (and runs only its own ranks in it), since
# `#SBATCH --exclusive` is not accepted on Raven.
SHARED_NODES: dict[str, dict] = {
    "raven": {"max_cores_per_node": 36, "max_mem_shared": 120000, "mem_exclusive": "240000"},
    "viper_cpu": {"max_cores_per_node": 64, "max_mem_shared": 250000, "mem_exclusive": "480000"},
}


def mem_in_mb(mem: str) -> float:
    """A SLURM memory value (`--mem`: a number in MB or with a unit K, M, G or T) in MB."""
    mem = str(mem).strip().upper().removesuffix("B")
    factor = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}.get(mem[-1:], None)
    return float(mem) if factor is None else float(mem[:-1]) * factor


# Folder of the profiling output (job scripts, SLURM logs, simulation runs, packaged cases) per cluster, in place of
# `<repo>/_profiling_jobs`: on clusters whose home file system is small, the runs go to the scratch file system.
# `{user}` is replaced by the user name. `$STRUPHY_PROFILING_DIR` overrides the parent folder on any machine.
SCRATCH_DIRS: dict[str, str] = {
    "raven": "/ptmp/{user}/struphy_profiling",
    "viper_cpu": "/ptmp/{user}/struphy_profiling",
}


def profiling_base_dir(repo_root: Path) -> Path:
    """Folder of the profiling output: `$STRUPHY_PROFILING_DIR/struphy_profiling` if the variable is set, else the
    `SCRATCH_DIRS` entry of the detected cluster, else `<repo_root>/_profiling_jobs`.

    The driver deletes and recreates `profiling-data/` in this folder, so it is always a dedicated subfolder (never
    the folder given by the user itself)."""
    if env_dir := os.environ.get("STRUPHY_PROFILING_DIR"):
        return Path(env_dir).expanduser().resolve() / "struphy_profiling"
    scratch = SCRATCH_DIRS.get(detect_machine_name())
    if scratch is not None:
        return Path(scratch.format(user=getpass.getuser()))
    return repo_root / "_profiling_jobs"


HARDWARE_INFO: dict[str, dict] = {
    "pitagora_dcgp": {
        "cpus_per_node": 256,
        "gpus_per_node": 0,
    },
    "tok": {
        "cpus_per_node": 256,
        "gpus_per_node": 4,
    },
    "raven": {
        "cpus_per_node": 72,
        "gpus_per_node": 4,
    },
    "viper_cpu": {
        "cpus_per_node": 128,
        "gpus_per_node": 0,
    },
}
