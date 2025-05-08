"""
Hardware dictionaries
"""

# Dictionary to store hardware configurations
hardware = {}

# Cobra work hardware configuration
hardware["cobra_work"] = {
    "clock_speed_GHz": 2.4,
    "num_cores": 40,
    "num_fpus": 8,  # https://en.wikipedia.org/wiki/Floating-point_unit
    "num_flops_fma": 2,
    "num_fma_per_cycle": 2,
    "linpack_fullnode_GFLOPps": 1706,
}

# Raven work hardware configuration
hardware["raven_work"] = {
    "clock_speed_GHz": 2.4,
    "num_cores": 72,
    "num_fpus": 8,  # https://en.wikipedia.org/wiki/Floating-point_unit
    "num_flops_fma": 2,
    "num_fma_per_cycle": 2,
    "linpack_fullnode_GFLOPps": 3062.4889,
    "stream_dict_path": "data/stream_data/raven/benchmark_10251584_scatter.pkl",
}

# Calculate theoretical maximum GFLOPs for each hardware configuration
for name, dict in hardware.items():
    dict["theoretical_max_gflops"] = (
        dict["clock_speed_GHz"]
        * dict["num_cores"]
        * dict["num_fpus"]
        * dict["num_flops_fma"]
        * dict["num_fma_per_cycle"]
    )

# Dictionary to store node configurations
node_dict = {
    "cobra_login": {
        "socket0": "12 36 13 37 14 38 15 39 16 40 17 41 18 42 19 43 20 44 21 45 22 46 23 47",
        "socket1": "0 24 1 25 2 26 3 27 4 28 5 29 6 30 7 31 8 32 9 33 10 34 11 35",
    },
    "cobra_work": {
        "socket0": "20 60 21 61 22 62 23 63 24 64 25 65 26 66 27 67 28 68 29 69 30 70 31 71 32 72 33 73 34 74 35 75 36 76 37 77 38 78 39 79",
        "socket1": "0 40 1 41 2 42 3 43 4 44 5 45 6 46 7 47 8 48 9 49 10 50 11 51 12 52 13 53 14 54 15 55 16 56 17 57 18 58 19 59",
    },
    "tok_login": {
        "socket0": "16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31",
        "socket1": "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
    },
    "raven_login": {
        "socket0": "0 72 1 73 2 74 3 75 4 76 5 77 6 78 7 79 8 80 9 81 10 82 11 83 12 84 13 85 14 86 15 87 16 88 17 89 18 90 19 91 20 92 21 93 22 94 23 95 24 96 25 97 26 98 27 99 28 100 29 101 30 102 31 103 32 104 33 105 34 106 35 107",
        "socket1": "36 108 37 109 38 110 39 111 40 112 41 113 42 114 43 115 44 116 45 117 46 118 47 119 48 120 49 121 50 122 51 123 52 124 53 125 54 126 55 127 56 128 57 129 58 130 59 131 60 132 61 133 62 134 63 135 64 136 65 137 66 138 67 139 68 140 69 141 70 142 71 143",
    },
    "raven_work": {
        "socket0": "0 72 1 73 2 74 3 75 4 76 5 77 6 78 7 79 8 80 9 81 10 82 11 83 12 84 13 85 14 86 15 87 16 88 17 89 18 90 19 91 20 92 21 93 22 94 23 95 24 96 25 97 26 98 27 99 28 100 29 101 30 102 31 103 32 104 33 105 34 106 35 107",
        "socket1": "36 108 37 109 38 110 39 111 40 112 41 113 42 114 43 115 44 116 45 117 46 118 47 119 48 120 49 121 50 122 51 123 52 124 53 125 54 126 55 127 56 128 57 129 58 130 59 131 60 132 61 133 62 134 63 135 64 136 65 137 66 138 67 139 68 140 69 141 70 142 71 143",
    },
}
