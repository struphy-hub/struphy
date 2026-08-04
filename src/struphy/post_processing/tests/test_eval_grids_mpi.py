"""MPI tests for the evaluation-grid decomposition used in parallel post-processing.

The pure-python properties of ``_create_eval_grids`` are covered in
``test_eval_grids.py`` with a fake Derham. Here the same code is driven with a real
communicator, together with ``_collect_on_root``, which is the MPI counterpart that
turns the per-rank slices back into a global array on rank 0.

Run with::

    mpirun -n 2 pytest --with-mpi src/struphy/post_processing/tests/test_eval_grids_mpi.py

The tests adapt to the number of ranks they are launched with, so ``-n 3``/``-n 4``
exercise uneven and multi-dimensional decompositions as well.
"""

from types import SimpleNamespace

import cunumpy as xp
import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy.post_processing.post_processing_tools import PostProcessor

# divisible by 1, 2, 3 and 4, so eta1 can be split evenly over the usual rank counts
NUM_ELEMENTS = (12, 2, 2)


def split_eta1(num_elements, n_parts):
    """Split eta1 into ``n_parts`` contiguous chunks, as ``Derham.domain_array`` does.

    The chunk boundaries are taken from the same ``linspace`` that ``_create_eval_grids``
    builds its grid from, so they compare exactly equal to grid points. ``_create_eval_grids``
    decides ownership of interior boundaries with ``==``, and picking the boundaries any
    other way would make the test depend on floating-point round-off rather than on the
    decomposition logic.
    """
    edges = xp.linspace(0.0, 1.0, num_elements[0] + 1)
    cuts = [round(i * num_elements[0] / n_parts) for i in range(n_parts + 1)]

    rows = []
    for i in range(n_parts):
        rows += [
            [
                edges[cuts[i]],
                edges[cuts[i + 1]],
                cuts[i + 1] - cuts[i],
                0.0,
                1.0,
                num_elements[1],
                0.0,
                1.0,
                num_elements[2],
            ]
        ]
    return rows


def make_mpi_pproc(comm, num_elements=NUM_ELEMENTS):
    """A PostProcessor stub in parallel mode, decomposed over ``comm``.

    ``__init__`` is bypassed on purpose (it creates output folders and reads meta.yml);
    ``_create_eval_grids`` and ``_collect_on_root`` only need the attributes set here.
    """
    pproc = PostProcessor.__new__(PostProcessor)
    pproc.derham = SimpleNamespace(
        num_elements=num_elements,
        domain_array=xp.array(split_eta1(num_elements, comm.Get_size()), dtype=float),
    )
    pproc.parallel_pproc = True
    pproc.comm = comm
    pproc.comm_size = comm.Get_size()
    pproc.rank = comm.Get_rank()
    return pproc


def global_array(shape, dtype=float, offset=0):
    """A deterministic array every rank can compute, so no reference has to be sent."""
    values = xp.arange(xp.prod(xp.array(shape)), dtype=float) + offset
    if dtype == complex:
        values = values + 1j * values
    return values.reshape(shape).astype(dtype)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("celldivide", [1, [1, 1, 1], [2, 2, 2], [3, 1, 2]])
def test_all_ranks_agree_on_the_decomposition(celldivide):
    """Every rank computes the full slice list redundantly - they must all agree.

    ``_collect_on_root`` uses rank 0's slices to place data that the other ranks sized
    with their own copy, so a divergence here would silently corrupt the gathered array.
    """
    comm = MPI.COMM_WORLD
    pproc = make_mpi_pproc(comm)

    grids_log, grid_slices = pproc._create_eval_grids(celldivide=celldivide)

    all_slices = comm.allgather(grid_slices)
    all_shapes = comm.allgather([grid.size for grid in grids_log])

    assert all(slices == all_slices[0] for slices in all_slices)
    assert all(shapes == all_shapes[0] for shapes in all_shapes)

    # and there is exactly one slice per rank
    assert len(grid_slices) == comm.Get_size()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("celldivide", [[1, 1, 1], [2, 2, 2], [3, 1, 2]])
def test_slices_tile_the_grid_under_a_real_decomposition(celldivide):
    """Every evaluation point is owned by exactly one rank - no gaps, no duplicates."""
    comm = MPI.COMM_WORLD
    pproc = make_mpi_pproc(comm)

    grids_log, grid_slices = pproc._create_eval_grids(celldivide=celldivide)

    shape = tuple(grid.size for grid in grids_log)
    owners = xp.zeros(shape, dtype=int)
    for sl in grid_slices:
        owners[sl] += 1

    assert xp.all(owners == 1)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("celldivide", [[1, 1, 1], [2, 2, 2]])
def test_collect_on_root_reproduces_the_global_array(celldivide):
    """The round trip global -> local per rank -> gathered on root is the identity."""
    comm = MPI.COMM_WORLD
    pproc = make_mpi_pproc(comm)

    grids_log, grid_slices = pproc._create_eval_grids(celldivide=celldivide)
    shape = tuple(grid.size for grid in grids_log)

    glob_val = global_array(shape)
    loc_val = glob_val[grid_slices[pproc.rank]]

    gathered = pproc._collect_on_root(loc_val, grid_slices, shape)

    if pproc.rank == 0:
        assert gathered.shape == shape
        assert xp.array_equal(gathered, glob_val)
    else:
        assert gathered is None


@pytest.mark.mpi(min_size=2)
def test_collect_on_root_handles_non_contiguous_local_arrays():
    """Field components reach ``_collect_on_root`` as views, not as fresh arrays.

    Slicing a stacked component array gives a non-contiguous local block; the send
    buffer has to be made contiguous or the received bytes are misinterpreted.
    """
    comm = MPI.COMM_WORLD
    pproc = make_mpi_pproc(comm)

    grids_log, grid_slices = pproc._create_eval_grids()
    shape = tuple(grid.size for grid in grids_log)

    glob_val = global_array(shape)

    # mimic taking one component out of a (3, *shape) array laid out component-last
    stacked = xp.stack([glob_val, glob_val + 1000.0, glob_val + 2000.0], axis=-1)
    loc_val = stacked[grid_slices[pproc.rank]][..., 0]
    assert not loc_val.flags["C_CONTIGUOUS"]

    gathered = pproc._collect_on_root(loc_val, grid_slices, shape)

    if pproc.rank == 0:
        assert xp.array_equal(gathered, glob_val)
    else:
        assert gathered is None


@pytest.mark.mpi(min_size=2)
def test_collect_on_root_repeated_calls_stay_correct():
    """The cached receive buffers must not leak data between time steps.

    ``_collect_on_root`` reuses its receive buffers across calls; if a buffer were
    picked up under a wrong key, a later step would silently show earlier data.
    """
    comm = MPI.COMM_WORLD
    pproc = make_mpi_pproc(comm)

    grids_log, grid_slices = pproc._create_eval_grids()
    shape = tuple(grid.size for grid in grids_log)

    for step in range(3):
        glob_val = global_array(shape, offset=1000 * step)
        loc_val = glob_val[grid_slices[pproc.rank]]

        gathered = pproc._collect_on_root(loc_val, grid_slices, shape)

        if pproc.rank == 0:
            assert xp.array_equal(gathered, glob_val), f"wrong data gathered in step {step}"

    if pproc.rank == 0:
        # buffers were cached, i.e. the reuse path above was actually taken
        assert len(pproc._collect_recv_bufs) == comm.Get_size() - 1


@pytest.mark.mpi(min_size=2)
def test_collect_on_root_keeps_dtypes_apart():
    """The receive-buffer cache is keyed on dtype; a real and a complex field must not mix."""
    comm = MPI.COMM_WORLD
    pproc = make_mpi_pproc(comm)

    grids_log, grid_slices = pproc._create_eval_grids()
    shape = tuple(grid.size for grid in grids_log)

    for dtype in (float, complex, float):
        glob_val = global_array(shape, dtype=dtype)
        loc_val = glob_val[grid_slices[pproc.rank]]

        gathered = pproc._collect_on_root(loc_val, grid_slices, shape)

        if pproc.rank == 0:
            assert gathered.dtype == glob_val.dtype
            assert xp.array_equal(gathered, glob_val)


@pytest.mark.mpi(min_size=2)
def test_only_root_allocates_the_global_array():
    """The point of the gather is that the global grid never exists off rank 0."""
    comm = MPI.COMM_WORLD
    pproc = make_mpi_pproc(comm)

    grids_log, grid_slices = pproc._create_eval_grids()
    shape = tuple(grid.size for grid in grids_log)

    loc_val = global_array(shape)[grid_slices[pproc.rank]]
    gathered = pproc._collect_on_root(loc_val, grid_slices, shape)

    is_root = pproc.rank == 0
    assert (gathered is not None) == is_root

    # the local block is strictly smaller than the global grid on at least one rank
    local_sizes = comm.allgather(loc_val.size)
    assert sum(local_sizes) == int(xp.prod(xp.array(shape)))
    assert max(local_sizes) < sum(local_sizes)


if __name__ == "__main__":
    pytest.main(["-v", "--with-mpi", __file__])
