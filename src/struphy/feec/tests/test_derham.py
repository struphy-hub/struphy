import pytest


@pytest.mark.parametrize("Nel", [[8, 8, 12]])
@pytest.mark.parametrize("p", [[1, 2, 3]])
@pytest.mark.parametrize("bcs", [
    (("free", "free"), ("free", "free"), None),
    (("free", "free"), None, ("free", "free")),
    (None, ("free", "free"), ("free", "free")),]
)
def test_psydac_derham(Nel, p, bcs):
    """Remark: p=even projectors yield slightly different results, pass with atol=1e-3."""

    from feectools.ddm.mpi import mpi as MPI
    from feectools.linalg.block import BlockVector
    from feectools.linalg.stencil import StencilVector

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("Nel=", Nel)
    print("p=", p)
    print("bcs=", bcs)

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, bcs=bcs, comm=comm)

    #TODO: test different initializations of Derham

if __name__ == "__main__":
    test_psydac_derham([8, 8, 12], [1, 2, 3], (("free", "free"), ("free", "free"), None))
