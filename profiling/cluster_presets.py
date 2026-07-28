"""Static per-cluster SLURM settings, shared across profiling cases.

`job_name` and `ntasks_per_node` are filled in per rank count at submission time by
each `ProfilingCase`.
"""

CLUSTER_PRESETS: dict[str, dict] = {
    "pitagora": {
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
