import inspect

import pytest

from struphy.models.tests.util import wrapper_for_testing


@pytest.mark.parametrize(
    "map_and_equil", [("Cuboid", "HomogenSlab"), ("HollowTorus", "AdhocTorus"), ("Tokamak", "EQDSKequilibrium")]
)
def test_hybrid(
    map_and_equil: tuple | list,
    fast: bool,
    vrbose: bool,
    verification: bool,
    nclones: int,
    show_plots: bool,
):
    """Tests all models in models/hybrid.py."""
    wrapper_for_testing(
        mtype="hybrid",
        map_and_equil=map_and_equil,
        fast=fast,
        vrbose=vrbose,
        verification=verification,
        nclones=nclones,
        show_plots=show_plots,
    )
