"""Fixed parameters for the Orszag--Tang NumPy-vs-CuPy profiling case."""

import argparse
import os
import sys
import time
from pathlib import Path


NUM_ELEMENTS = 96
NUM_STEPS = 5
name = "Orszag--Tang: NumPy vs CuPy"
description = "Fixed 96x96x1, five-step, one-rank backend comparison."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    args = parser.parse_args()

    # cunumpy chooses its implementation during import.
    os.environ["ARRAY_BACKEND"] = args.backend

    repo_dir = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_dir))

    from src.struphy.models.tests.Test_Full_MHD import OrszagTang as ot

    ot.NUM_ELEMENTS = (NUM_ELEMENTS, NUM_ELEMENTS, 1)
    ot.T_END = NUM_STEPS * ot.DT
    ot.SIMULATION_FOLDER_NAME = f"sim_{args.id:02d}"
    ot.OUTPUT_DIRECTORY = Path.cwd() / ot.SIMULATION_FOLDER_NAME

    print(
        "BENCHMARK_CONFIG"
        f" backend={args.backend} ranks=1 nel={NUM_ELEMENTS}"
        f" steps={NUM_STEPS} dt={ot.DT}",
    )
    started = time.perf_counter()
    ot.execute(run=True, overwrite=True, post_process=False, do_plot=False)
    print(f"BENCHMARK_WALL_SECONDS {time.perf_counter() - started:.9f}")


if __name__ == "__main__":
    main()
