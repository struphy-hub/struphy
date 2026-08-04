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

# two ranks, split in eta3 (the direction that is left whole everywhere else)
DOM_ARR_2_RANKS_ETA3 = [
    [0.0, 1.0, 4, 0.0, 1.0, 2, 0.0, 0.5, 1],
    [0.0, 1.0, 4, 0.0, 1.0, 2, 0.5, 1.0, 1],
]

# three ranks, uneven split of eta1 at 0.25 and 0.75
DOM_ARR_3_RANKS = [
    [0.0, 0.25, 1, 0.0, 1.0, 2, 0.0, 1.0, 2],
    [0.25, 0.75, 2, 0.0, 1.0, 2, 0.0, 1.0, 2],
    [0.75, 1.0, 1, 0.0, 1.0, 2, 0.0, 1.0, 2],
]

NUM_ELEMENTS = (4, 2, 2)

ALL_DOM_ARRS = [
    DOM_ARR_1_RANK,
    DOM_ARR_2_RANKS,
    DOM_ARR_4_RANKS,
    DOM_ARR_2_RANKS_ETA3,
    DOM_ARR_3_RANKS,
]


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

    _, grid_slices = pproc._create_eval_grids()

    assert len(grid_slices) == 1
    assert grid_slices[0] == (slice(0, 5), slice(0, 3), slice(0, 3))


def test_boundary_point_goes_to_the_right_neighbour():
    """The shared point eta1=0.5 is owned by rank 1, not by both ranks."""
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_2_RANKS)

    _, grid_slices = pproc._create_eval_grids()

    # eta1 grid is [0, 0.25, 0.5, 0.75, 1]; index 2 is the point at 0.5
    assert grid_slices[0][0] == slice(0, 2)  # 0.0, 0.25
    assert grid_slices[1][0] == slice(2, 5)  # 0.5, 0.75, 1.0

    # the un-split directions are owned completely by both ranks
    for rank in (0, 1):
        assert grid_slices[rank][1] == slice(0, 3)
        assert grid_slices[rank][2] == slice(0, 3)


def test_split_in_third_direction():
    """A decomposition along eta3 is sliced in eta3 only."""
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_2_RANKS_ETA3)

    _, grid_slices = pproc._create_eval_grids()

    # eta3 grid is [0, 0.5, 1]; the shared point at 0.5 goes to rank 1
    assert grid_slices[0][2] == slice(0, 1)  # 0.0
    assert grid_slices[1][2] == slice(1, 3)  # 0.5, 1.0

    for rank in (0, 1):
        assert grid_slices[rank][0] == slice(0, 5)
        assert grid_slices[rank][1] == slice(0, 3)


def test_uneven_split_over_three_ranks():
    """Ranks may own a different number of points; the slices still tile the grid."""
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_3_RANKS)

    _, grid_slices = pproc._create_eval_grids()

    # eta1 grid is [0, 0.25, 0.5, 0.75, 1]; interior boundaries 0.25 and 0.75
    # are shifted into the rank to their right
    assert grid_slices[0][0] == slice(0, 1)  # 0.0
    assert grid_slices[1][0] == slice(1, 3)  # 0.25, 0.5
    assert grid_slices[2][0] == slice(3, 5)  # 0.75, 1.0


@pytest.mark.parametrize("dom_arr", ALL_DOM_ARRS)
@pytest.mark.parametrize("celldivide", [[1, 1, 1], [2, 2, 2], [3, 1, 2]])
def test_owned_points_lie_inside_the_rank_domain(dom_arr, celldivide):
    """Each rank only owns evaluation points that lie in its own MPI subdomain.

    Ties the slices back to ``domain_array``: without this, a decomposition could
    tile the grid exactly but hand points to the wrong rank.
    """
    pproc = make_pproc(NUM_ELEMENTS, dom_arr)

    grids_log, grid_slices = pproc._create_eval_grids(celldivide=celldivide)

    for rank, sl in enumerate(grid_slices):
        for n, grid in enumerate(grids_log):
            left = dom_arr[rank][3 * n + 0]
            right = dom_arr[rank][3 * n + 1]
            owned = grid[sl[n]]

            assert np.all(owned >= left)
            assert np.all(owned <= right)


@pytest.mark.parametrize("dom_arr", ALL_DOM_ARRS)
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


def test_collect_on_root_is_a_no_op_in_serial():
    """With ``parallel_pproc=False`` the local array is already the global one."""
    pproc = make_pproc(NUM_ELEMENTS, DOM_ARR_1_RANK)
    pproc.parallel_pproc = False

    grids_log, grid_slices = pproc._create_eval_grids()
    shape = tuple(grid.size for grid in grids_log)
    loc_val = np.arange(np.prod(shape), dtype=float).reshape(shape)

    glob_val = pproc._collect_on_root(loc_val, grid_slices, shape)

    assert glob_val is loc_val


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
