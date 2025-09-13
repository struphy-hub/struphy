import inspect

import pytest

from struphy.models.tests.util import wrapper_for_testing


@pytest.mark.parametrize(
    "map_and_equil", [("Cuboid", "HomogenSlab"), ("HollowTorus", "AdhocTorus"), ("Tokamak", "EQDSKequilibrium")]
)
def test_kinetic(
    map_and_equil: tuple | list,
    fast: bool,
    vrbose: bool,
    verification: bool,
    nclones: int,
    show_plots: bool,
):
    """Tests models in models/kinetic.py."""
    wrapper_for_testing(
        mtype="kinetic",
        map_and_equil=map_and_equil,
        fast=fast,
        vrbose=vrbose,
        verification=verification,
        nclones=nclones,
        show_plots=show_plots,
    )
