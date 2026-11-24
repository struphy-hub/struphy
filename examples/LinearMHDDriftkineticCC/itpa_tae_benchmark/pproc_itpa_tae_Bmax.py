"""Post-process the run of params_itpa_tae_Bmax.py, see pproc_itpa_tae.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import params_itpa_tae_Bmax as params  # noqa: E402
import pproc_itpa_tae as pproc  # noqa: E402

pproc.configure(params)
main = pproc.main

if __name__ == "__main__":
    pproc.cli()
