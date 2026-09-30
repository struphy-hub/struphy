import pytest


@pytest.mark.mpi_skip
def test_load_external(tmp_path):
    """Markers loaded with loading="external" must equal the markers written to the file."""
    import cunumpy as xp
    import h5py

    from struphy import BoundaryParameters, LoadingParameters, domains
    from struphy.pic.particles import Particles6D

    domain = domains.Cuboid()

    ref = Particles6D(
        loading_params=LoadingParameters(Np=1000, seed=1234),
        boundary_params=BoundaryParameters(),
        domain=domain,
    )
    ref.draw_markers(sort=False)
    n_mks = ref.n_mks_load[0]

    path = tmp_path / "markers.hdf5"
    with h5py.File(path, "w") as file:
        file.create_dataset("markers", data=ref.markers[:n_mks])

    ext = Particles6D(
        loading_params=LoadingParameters(Np=1000, loading="external", dir_external=str(path)),
        boundary_params=BoundaryParameters(),
        domain=domain,
    )
    ext.draw_markers(sort=False)

    assert xp.array_equal(ext.markers[:n_mks], ref.markers[:n_mks])
    assert xp.all(ext.markers[n_mks:] == -1.0)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_load_external(Path(d))
