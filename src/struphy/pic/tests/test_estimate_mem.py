import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy import BoundaryParameters, LoadingParameters, SortingParameters, WeightsParameters, domains
from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.pic.particles import Particles6D
from struphy.topology.grids import TensorProductGrid


def _make_domain_decomp(mpi_comm, num_elements=(8, 6, 4), degree=(2, 2, 2)):
    domain = domains.Cuboid()
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree), comm=mpi_comm)
    domain_decomp = (derham.domain_array, derham.domain_decomposition.nprocs)
    return domain, domain_decomp


@pytest.mark.parametrize("Np", [1000, 12345])
def test_dry_run_does_not_allocate_markers(Np):
    """dry_run=True must compute the marker array sizing (n_rows, n_cols) without
    allocating the (potentially large) marker/sorting/buffer arrays."""
    mpi_comm = MPI.COMM_WORLD
    domain, domain_decomp = _make_domain_decomp(mpi_comm)

    loading_params = LoadingParameters(Np=Np, seed=1234)
    boundary_params = BoundaryParameters()
    sorting_params = SortingParameters(do_sort=False)

    particles = Particles6D(
        comm_world=mpi_comm,
        loading_params=loading_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        domain_decomp=domain_decomp,
        domain=domain,
        dry_run=True,
    )

    # sizing info is available ...
    assert particles.n_rows > 0
    assert particles.n_cols > 0

    # ... but none of the (large) arrays that allocate() would create exist
    for attr in (
        "_markers",
        "_holes",
        "_ghost_particles",
        "_valid_mks",
        "_is_outside_right",
        "_is_outside_left",
        "_is_outside",
        "_lost_markers",
        "_sorting_etas",
        "_is_on_proc_domain",
        "_can_stay",
    ):
        assert not hasattr(particles, attr), f"dry_run=True should not create '{attr}'"


@pytest.mark.parametrize("Np", [1000, 12345])
def test_nbytes_local_matches_real_allocation(Np):
    """The dry_run estimate (nbytes_local) must equal the real memory footprint once the
    particles are actually allocated (dry_run=False), since both use the exact same
    n_rows/n_cols sizing and the same (fixed) list of marker-related arrays."""
    mpi_comm = MPI.COMM_WORLD
    domain, domain_decomp = _make_domain_decomp(mpi_comm)

    loading_params = LoadingParameters(Np=Np, seed=1234)
    boundary_params = BoundaryParameters()
    sorting_params = SortingParameters(do_sort=False)
    weights_params = WeightsParameters()

    common_kwargs = dict(
        comm_world=mpi_comm,
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        domain_decomp=domain_decomp,
        domain=domain,
    )

    dry = Particles6D(**common_kwargs, dry_run=True)
    real = Particles6D(**common_kwargs, dry_run=False)
    real.draw_markers(sort=False)

    assert dry.n_rows == real.n_rows
    assert dry.n_cols == real.n_cols

    real_nbytes = (
        real.markers.nbytes
        + real._sorting_etas.nbytes
        + real._can_stay.nbytes
        + real._holes.nbytes
        + real._ghost_particles.nbytes
        + real._valid_mks.nbytes
        + real._is_outside_right.nbytes
        + real._is_outside_left.nbytes
        + real._is_outside.nbytes
        + real._lost_markers.nbytes
    )

    assert dry.nbytes_local == real_nbytes
    assert real.nbytes_local == real_nbytes


if __name__ == "__main__":
    test_dry_run_does_not_allocate_markers(Np=1000)
    test_nbytes_local_matches_real_allocation(Np=1000)
