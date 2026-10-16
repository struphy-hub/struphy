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
}

HARDWARE_INFO: dict[str, dict] = {
    "pitagora_dcgp": {
        "cpus_per_node": 256,
        "gpus_per_node": 0,
        # 2x AMD EPYC 9745: 8 CCDs (chiplets, one L3 each) of 16 cores per socket, numbered consecutively
        "topology": {"sockets": 2, "ccds_per_socket": 8, "cores_per_ccd": 16},
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


PINNINGS = ("compact", "spread", "balanced")


def pinning_cpu_map(pinning: str, ntasks: int, topology: dict) -> list[int]:
    """Core of each of `ntasks` ranks on one node, for `srun --cpu-bind=map_cpu:...`.

    Cores are numbered consecutively: CCD by CCD within a socket, socket by socket.

    - compact: rank r on core r, so the ranks fill one CCD (and socket) after the other.
    - spread: round robin over the sockets, then over the CCDs of a socket, then over the cores
      of a CCD, so the ranks are as far apart as possible and neighbouring ranks share nothing.
    - balanced: the same cores as spread (the same number of ranks per CCD and socket), but
      assigned in core order, so neighbouring ranks share a CCD.

    Args:
        pinning: One of `PINNINGS`.
        ntasks: Number of ranks on the node.
        topology: `sockets`, `ccds_per_socket` and `cores_per_ccd` of the node.

    Returns:
        The core of each rank, indexed by the node-local rank.
    """
    sockets = topology["sockets"]
    ccds_per_socket = topology["ccds_per_socket"]
    cores_per_ccd = topology["cores_per_ccd"]
    num_cores = sockets * ccds_per_socket * cores_per_ccd
    if not 1 <= ntasks <= num_cores:
        raise ValueError(f"ntasks ({ntasks}) must be between 1 and the {num_cores} cores of the node.")

    if pinning == "compact":
        return list(range(ntasks))

    scatter_order = [
        (socket * ccds_per_socket + ccd) * cores_per_ccd + core
        for core in range(cores_per_ccd)
        for ccd in range(ccds_per_socket)
        for socket in range(sockets)
    ]
    if pinning == "spread":
        return scatter_order[:ntasks]
    if pinning == "balanced":
        return sorted(scatter_order[:ntasks])
    raise ValueError(f"pinning must be one of {PINNINGS}, got {pinning!r}")
