import pytest


@pytest.mark.parametrize("Nel", [[8, 8, 12]])
@pytest.mark.parametrize("p", [[1, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[False, False, True]])
def test_psydac_derham(Nel, p, spl_kind):
    """Remark: p=even projectors yield slightly different results, pass with atol=1e-3."""

    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI
    from psydac.linalg.block import BlockVector
    from psydac.linalg.stencil import StencilVector

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("Nel=", Nel)
    print("p=", p)
    print("spl_kind=", spl_kind)

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    ############################
    ### TEST STENCIL VECTORS ###
    ############################
    # Stencil vectors for Psydac:
    x0_PSY = StencilVector(derham.Vh["0"])
    print(f"rank {rank} | 0-form StencilVector:")
    print(f"rank {rank} | starts:", x0_PSY.starts)
    print(f"rank {rank} | ends  :", x0_PSY.ends)
    print(f"rank {rank} | pads  :", x0_PSY.pads)
    print(f"rank {rank} | shape (=dim):", x0_PSY.shape)
    print(f"rank {rank} | [:].shape (=shape):", x0_PSY[:].shape)

    # Block of StencilVecttors
    x1_PSY = BlockVector(derham.Vh["1"])
    print(f"rank {rank} | \n1-form StencilVector:")
    print(f"rank {rank} | starts:", [component.starts for component in x1_PSY])
    print(f"rank {rank} | ends  :", [component.ends for component in x1_PSY])
    print(f"rank {rank} | pads  :", [component.pads for component in x1_PSY])
    print(f"rank {rank} | shape (=dim):", [component.shape for component in x1_PSY])
    print(f"rank {rank} | [:].shape (=shape):", [component[:].shape for component in x1_PSY])

    s11, s12, s13 = [component.starts for component in x1_PSY]
    e11, e12, e13 = [component.ends for component in x1_PSY]

    x2_PSY = BlockVector(derham.Vh["2"])
    print(f"rank {rank} | \n2-form StencilVector:")
    print(f"rank {rank} | starts:", [component.starts for component in x2_PSY])
    print(f"rank {rank} | ends  :", [component.ends for component in x2_PSY])
    print(f"rank {rank} | pads  :", [component.pads for component in x2_PSY])
    print(f"rank {rank} | shape (=dim):", [component.shape for component in x2_PSY])
    print(f"rank {rank} | [:].shape (=shape):", [component[:].shape for component in x2_PSY])

    s21, s22, s23 = [component.starts for component in x2_PSY]
    e21, e22, e23 = [component.ends for component in x2_PSY]

    x3_PSY = StencilVector(derham.Vh["3"])
    print(f"rank {rank} | \n3-form StencilVector:")
    print(f"rank {rank} | starts:", x3_PSY.starts)
    print(f"rank {rank} | ends  :", x3_PSY.ends)
    print(f"rank {rank} | pads  :", x3_PSY.pads)
    print(f"rank {rank} | shape (=dim):", x3_PSY.shape)
    print(f"rank {rank} | [:].shape (=shape):", x3_PSY[:].shape)


if __name__ == "__main__":
    test_psydac_derham([8, 8, 12], [1, 2, 3], [False, False, True])
