from types import SimpleNamespace

import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy.pic.base import Particles


def make_stub(comm, periodic_axes=()):
    """A Particles stub exposing only the attributes touched by
    ``_get_domain_decomp`` / ``_compute_neighbor_ranks``, decomposed over ``comm``."""
    stub = SimpleNamespace(
        mpi_size=comm.Get_size(),
        mpi_rank=comm.Get_rank(),
        boxes_per_dim=None,
        _periodic_axes=list(periodic_axes),
    )
    domain_array, nprocs = Particles._get_domain_decomp(stub, mpi_dims_mask=None)
    stub.domain_array = domain_array
    return stub, nprocs


def rank_ijk(rank, nprocs):
    """Inverse of the (i, j, k) -> rank flattening used in ``_get_domain_decomp``."""
    i = rank // (nprocs[1] * nprocs[2])
    nn = rank % (nprocs[1] * nprocs[2])
    j = nn // nprocs[2]
    k = nn % nprocs[2]
    return i, j, k


def ijk_rank(i, j, k, nprocs):
    return i * (nprocs[1] * nprocs[2]) + j * nprocs[2] + k


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("periodic_axes", [(), (0,), (0, 1, 2)])
def test_neighbor_ranks_partition_all_other_ranks(periodic_axes):
    """Every other rank must end up in exactly one of the two output lists."""
    comm = MPI.COMM_WORLD
    stub, _ = make_stub(comm, periodic_axes=periodic_axes)

    neighbor_ranks, non_neighbor_ranks = Particles._compute_neighbor_ranks(stub)

    rank = comm.Get_rank()
    all_others = set(range(comm.Get_size())) - {rank}
    assert set(neighbor_ranks) | set(non_neighbor_ranks) == all_others
    assert set(neighbor_ranks).isdisjoint(non_neighbor_ranks)
    assert rank not in neighbor_ranks
    assert rank not in non_neighbor_ranks


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("periodic_axes", [(), (0,), (0, 1, 2)])
def test_neighbor_relation_is_symmetric(periodic_axes):
    """If rank j is a neighbour of rank i, rank i must be a neighbour of rank j.

    Both ranks classify the same pair of boxes, so a one-sided result would
    mean the touching check disagrees with itself depending on which side asks.
    """
    comm = MPI.COMM_WORLD
    stub, _ = make_stub(comm, periodic_axes=periodic_axes)

    neighbor_ranks, _ = Particles._compute_neighbor_ranks(stub)

    all_neighbor_sets = comm.allgather(set(neighbor_ranks))
    rank = comm.Get_rank()
    for j in range(comm.Get_size()):
        if j == rank:
            continue
        assert (j in all_neighbor_sets[rank]) == (rank in all_neighbor_sets[j]), (
            f"neighbour relation between rank {rank} and rank {j} is not symmetric (periodic_axes={periodic_axes})"
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("periodic_axes", [(), (0,), (1,), (0, 1, 2)])
def test_face_adjacent_ranks_are_always_neighbors(periodic_axes):
    """Ranks whose process-grid index differs by 1 along a single axis share a
    face, so they must always be classified as neighbours -- including across
    the periodic wrap when that axis is periodic."""
    comm = MPI.COMM_WORLD
    stub, nprocs = make_stub(comm, periodic_axes=periodic_axes)

    neighbor_ranks, _ = Particles._compute_neighbor_ranks(stub)

    rank = comm.Get_rank()
    ijk = rank_ijk(rank, nprocs)
    for axis, n_axis in enumerate(nprocs):
        for step in (-1, 1):
            idx = list(ijk)
            idx[axis] += step
            if idx[axis] < 0 or idx[axis] >= n_axis:
                if axis not in periodic_axes or n_axis == 1:
                    continue
                idx[axis] %= n_axis
            other_rank = ijk_rank(*idx, nprocs)
            if other_rank == rank:
                continue
            assert other_rank in neighbor_ranks, (
                f"rank {rank} at grid index {ijk} expected face-neighbour {other_rank} "
                f"at {tuple(idx)} along axis {axis} (periodic_axes={periodic_axes})"
            )
