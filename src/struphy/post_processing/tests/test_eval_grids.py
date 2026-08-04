"""Unit tests for the evaluation-grid construction used in post-processing.

These tests do not run a simulation and do not need MPI: ``_create_eval_grids`` only
reads ``derham.num_elements`` and ``derham.domain_array``, so a small fake Derham is
enough. The domain arrays below are written out by hand so that the expected result
can be checked by eye.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from struphy.post_processing.post_processing_tools import PostProcessor


def make_pproc(num_elements, domain_array):
    """A PostProcessor stub that only knows about its Derham decomposition.

    ``__init__`` is bypassed on purpose (it creates output folders and reads meta.yml).
    """
    pproc = PostProcessor.__new__(PostProcessor)
    pproc.derham = SimpleNamespace(
        num_elements=num_elements,
        domain_array=np.array(domain_array, dtype=float),
    )
    return pproc


# one rank owning the whole unit cube: [left, right, n_cells] for eta1, eta2, eta3
DOM_ARR_1_RANK = [
    [0.0, 1.0, 4, 0.0, 1.0, 2, 0.0, 1.0, 2],
]

# two ranks, split in eta1 at 0.5
DOM_ARR_2_RANKS = [
    [0.0, 0.5, 2, 0.0, 1.0, 2, 0.0, 1.0, 2],
    [0.5, 1.0, 2, 0.0, 1.0, 2, 0.0, 1.0, 2],
]

# four ranks, split in eta1 and eta2 at 0.5 (2 x 2 process grid)
DOM_ARR_4_RANKS = [
    [0.0, 0.5, 2, 0.0, 0.5, 1, 0.0, 1.0, 2],
    [0.0, 0.5, 2, 0.5, 1.0, 1, 0.0, 1.0, 2],
    [0.5, 1.0, 2, 0.0, 0.5, 1, 0.0, 1.0, 2],
    [0.5, 1.0, 2, 0.5, 1.0, 1, 0.0, 1.0, 2],
]

NUM_ELEMENTS = (4, 2, 2)


def test_grids_are_uniform_from_0_to_1():
    """Without refinement there is one grid point per cell boundary."""
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_1_RANK)

    grids_log, _ = pproc._create_eval_grids()

    assert np.allclose(grids_log[0], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(grids_log[1], [0.0, 0.5, 1.0])
    assert np.allclose(grids_log[2], [0.0, 0.5, 1.0])


def test_celldivide_refines_each_direction():
    """``celldivide=[n1, n2, n3]`` puts n_i points per cell in direction i."""
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_1_RANK)

    grids_log, _ = pproc._create_eval_grids(celldivide=[2, 3, 1])

    # num_elements * celldivide + 1 points
    assert grids_log[0].size == 4 * 2 + 1
    assert grids_log[1].size == 2 * 3 + 1
    assert grids_log[2].size == 2 * 1 + 1


def test_scalar_celldivide_applies_to_all_directions():
    """``Simulation.pproc`` passes a plain int, which is broadcast to all three directions."""
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_1_RANK)

    grids_scalar, slices_scalar = pproc._create_eval_grids(celldivide=2)
    grids_seq, slices_seq = pproc._create_eval_grids(celldivide=[2, 2, 2])

    for g_scalar, g_seq in zip(grids_scalar, grids_seq):
        assert np.array_equal(g_scalar, g_seq)
    assert slices_scalar == slices_seq


def test_single_rank_owns_the_whole_grid():
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_1_RANK)

    grids_log, grid_slices = pproc._create_eval_grids()

    assert len(grid_slices) == 1
    assert grid_slices[0] == (slice(0, 5), slice(0, 3), slice(0, 3))


def test_boundary_point_goes_to_the_right_neighbour():
    """The shared point eta1=0.5 is owned by rank 1, not by both ranks."""
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_2_RANKS)

    grids_log, grid_slices = pproc._create_eval_grids()

    # eta1 grid is [0, 0.25, 0.5, 0.75, 1]; index 2 is the point at 0.5
    assert grid_slices[0][0] == slice(0, 2)  # 0.0, 0.25
    assert grid_slices[1][0] == slice(2, 5)  # 0.5, 0.75, 1.0

    # the un-split directions are owned completely by both ranks
    for rank in (0, 1):
        assert grid_slices[rank][1] == slice(0, 3)
        assert grid_slices[rank][2] == slice(0, 3)


@pytest.mark.parametrize("dom_arr", [DOM_ARR_1_RANK, DOM_ARR_2_RANKS, DOM_ARR_4_RANKS])
@pytest.mark.parametrize("celldivide", [[1, 1, 1], [2, 2, 2], [3, 1, 2]])
def test_slices_tile_the_grid_exactly(dom_arr, celldivide):
    """Every grid point is owned by exactly one rank - no gaps, no duplicates.

    This is the property the parallel post-processing relies on when the local
    results are gathered on rank 0.
    """
    pproc = make_pproc(NUM_ELEMENTS, dom_arr)

    grids_log, grid_slices = pproc._create_eval_grids(celldivide=celldivide)

    shape = tuple(grid.size for grid in grids_log)

    # count how often each point is claimed
    owners = np.zeros(shape, dtype=int)
    for sl in grid_slices:
        owners[sl] += 1

    assert np.all(owners == 1)


def test_gather_reproduces_the_global_array():
    """Cutting a global array with the slices and pasting it back is the identity.

    This mimics what ``_collect_on_root`` does, but without MPI.
    """
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_4_RANKS)

    grids_log, grid_slices = pproc._create_eval_grids(celldivide=[2, 2, 2])

    shape = tuple(grid.size for grid in grids_log)
    glob_val = np.arange(np.prod(shape), dtype=float).reshape(shape)

    gathered = np.zeros(shape, dtype=float)
    for sl in grid_slices:
        loc_val = glob_val[sl]  # what a rank computes locally
        gathered[sl] = loc_val  # what rank 0 assembles

    assert np.array_equal(gathered, glob_val)


def test_default_celldivide_is_not_mutable_state():
    """Calling twice with the default gives the same grids (no shared default object)."""
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_2_RANKS)

    grids_1, slices_1 = pproc._create_eval_grids()
    grids_2, slices_2 = pproc._create_eval_grids()

    for g1, g2 in zip(grids_1, grids_2):
        assert np.array_equal(g1, g2)
    assert slices_1 == slices_2


def test_celldivide_must_have_three_entries():
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_1_RANK)

    with pytest.raises(AssertionError):
        pproc._create_eval_grids(celldivide=[1, 1])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
