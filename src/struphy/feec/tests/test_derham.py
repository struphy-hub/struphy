import pytest


# Test matrix for several valid initialization modes of Derham.
@pytest.mark.parametrize("Nel", [(4, 4, 6)])
@pytest.mark.parametrize(
    "init_kwargs, expected",
    [
        pytest.param(
            {
                "p": (1, 2, 3),
                "bcs": (("hom_dirichlet", "free"), ("free", "hom_dirichlet"), None),
            },
            {
                "p": (1, 2, 3),
                "bcs": (("hom_dirichlet", "free"), ("free", "hom_dirichlet"), None),
                "nquads": (2, 3, 4),
                "nq_pr": (2, 3, 4),
                "mpi_dims_mask": (True, True, True),
                "with_projectors": True,
                "with_local_projectors": False,
                "spl_kind": (False, False, True),
                "dirichlet_bc": ((True, False), (False, True), (False, False)),
            },
            id="mixed-hom-dirichlet-eta12",
        ),
        pytest.param(
            {"p": (1, 2, 3), "bcs": (("free", "free"), None, ("hom_dirichlet", "hom_dirichlet"))},
            {
                "p": (1, 2, 3),
                "bcs": (("free", "free"), None, ("hom_dirichlet", "hom_dirichlet")),
                "nquads": (2, 3, 4),
                "nq_pr": (2, 3, 4),
                "mpi_dims_mask": (True, True, True),
                "with_projectors": True,
                "with_local_projectors": False,
                "spl_kind": (False, True, False),
                "dirichlet_bc": ((False, False), (False, False), (True, True)),
            },
            id="mixed-hom-dirichlet-eta13",
        ),
        pytest.param(
            {},
            {
                "p": (1, 1, 1),
                "bcs": (None, None, None),
                "nquads": (2, 2, 2),
                "nq_pr": (2, 2, 2),
                "mpi_dims_mask": (True, True, True),
                "with_projectors": True,
                "with_local_projectors": False,
                "spl_kind": (True, True, True),
                "dirichlet_bc": ((False, False), (False, False), (False, False)),
            },
            id="defaults-only",
        ),
        pytest.param(
            {
                "p": (1, 2, 3),
                "bcs": (("free", "hom_dirichlet"), None, ("hom_dirichlet", "free")),
                "nquads": (4, 5, 6),
                "nq_pr": (5, 6, 7),
                "mpi_dims_mask": (True, False, True),
                "with_projectors": False,
            },
            {
                "p": (1, 2, 3),
                "bcs": (("free", "hom_dirichlet"), None, ("hom_dirichlet", "free")),
                "nquads": (4, 5, 6),
                "nq_pr": (5, 6, 7),
                "mpi_dims_mask": (True, False, True),
                "with_projectors": False,
                "with_local_projectors": False,
                "spl_kind": (False, True, False),
                "dirichlet_bc": ((False, True), (False, False), (True, False)),
            },
            id="custom-quads-without-projectors",
        ),
        pytest.param(
            {
                "p": (1, 2, 3),
                "bcs": (None, None, ("hom_dirichlet", "free")),
                "local_projectors": True,
            },
            {
                "p": (1, 2, 3),
                "bcs": (None, None, ("hom_dirichlet", "free")),
                "nquads": (2, 3, 4),
                "nq_pr": (2, 3, 4),
                "mpi_dims_mask": (True, True, True),
                "with_projectors": True,
                "with_local_projectors": True,
                "spl_kind": (True, True, False),
                "dirichlet_bc": ((False, False), (False, False), (True, False)),
            },
            id="with-local-projectors",
        ),
    ],
)
def test_psydac_derham(Nel, init_kwargs, expected):
    """Test Derham initialization across multiple valid constructor configurations."""

    from feectools.ddm.mpi import mpi as MPI
    from feectools.linalg.block import BlockVector
    from feectools.linalg.stencil import StencilVector

    from struphy.feec.psydac_derham import Derham

    comm = MPI.COMM_WORLD
    derham = Derham(Nel, comm=comm, **init_kwargs)

    assert derham.Nel == tuple(Nel)
    assert derham.p == expected["p"]
    assert derham.bcs == expected["bcs"]
    assert derham.nquads == expected["nquads"]
    assert derham.nq_pr == expected["nq_pr"]
    assert derham.mpi_dims_mask == expected["mpi_dims_mask"]
    assert derham.with_projectors is expected["with_projectors"]
    assert derham.with_local_projectors is expected["with_local_projectors"]
    assert derham.spl_kind == expected["spl_kind"]
    assert derham.dirichlet_bc == expected["dirichlet_bc"]
    assert derham.comm is comm

    # Discrete differential operators should always be built.
    assert derham.grad is not None
    assert derham.curl is not None
    assert derham.div is not None

    # All coefficient spaces must be available and instantiable.
    for form in ("0", "1", "2", "3", "v"):
        vec = derham.coeff_spaces[form].zeros()
        assert isinstance(vec, (StencilVector, BlockVector))

    if derham.with_projectors:
        assert all(projector is not None for projector in derham.projectors.values())
    else:
        assert all(projector is None for projector in derham.projectors.values())

    if derham.with_local_projectors:
        assert all(projector is not None for projector in derham.projectors_global.values())
        assert all(projector is not None for projector in derham.projectors_local.values())
        assert derham.P0 is derham.P0loc
        assert derham.P1 is derham.P1loc
        assert derham.P2 is derham.P2loc
        assert derham.P3 is derham.P3loc
        assert derham.Pv is derham.Pvloc


if __name__ == "__main__":
    pytest.main([__file__])
