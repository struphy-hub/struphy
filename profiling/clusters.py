"""Static per-cluster SLURM settings and cluster detection, shared across profiling cases.

`job_name` and `ntasks_per_node` are filled in per rank count at submission time by
each `ProfilingCase`.
"""

import os
import socket


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
    "pitagora_boost_fua_dbg": {
        # "nodes": 1, # Should be set by ProfilingCase.launch()
        # "ntasks_per_node": 1,  # Should be set by ProfilingCase.launch()
        "cpus_per_task": 16,
        "mem": "480GB",
        "gres": "gpu:4,tmpfs:10g",
        "partition": "boost_fua_dbg",
        "account": "FUSIO_HLST_6",
        "output": "myJob_%j.out",
        "error": "myJob_%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
    "pitagora_boost_fua_prod": {
        # "nodes": 1, # Should be set by ProfilingCase.launch()
        # "ntasks_per_node": 1,  # Should be set by ProfilingCase.launch()
        "cpus_per_task": 1,
        "mem": "480GB",
        "gres": "gpu:4,tmpfs:10g",
        "partition": "boost_fua_prod",
        "account": "FUSIO_HLST_6",
        "output": "myJob_%j.out",
        "error": "myJob_%j.err",
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
}

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
