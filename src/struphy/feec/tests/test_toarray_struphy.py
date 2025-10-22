import pytest


@pytest.mark.parametrize("Nel", [[12, 5, 2], [8, 12, 4], [5, 4, 12]])
@pytest.mark.parametrize("p", [[3, 2, 1]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [True, False, False]])
@pytest.mark.parametrize(
    "mapping", [["Cuboid", {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}]]
)
def test_toarray_struphy(Nel, p, spl_kind, mapping):
    """
    TODO
    """

    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays, create_equal_random_arrays
    from struphy.geometry import domains
    from struphy.utils.arrays import xp

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # assemble mass matrices in V0 and V1
    mass = WeightedMassOperators(derham, domain)

    M0 = mass.M0
    M1 = mass.M1
    M2 = mass.M2
    M3 = mass.M3

    # random vectors
    v0arr, v0 = create_equal_random_arrays(derham.Vh_fem["0"], seed=4568)
    v1arr1, v1 = create_equal_random_arrays(derham.Vh_fem["1"], seed=4568)
    v2arr1, v2 = create_equal_random_arrays(derham.Vh_fem["2"], seed=4568)
    v3arr, v3 = create_equal_random_arrays(derham.Vh_fem["3"], seed=4568)

    # ========= test toarray_struphy =================
    # Get the matrix form of the linear operators M0 to M3
    M0arr = M0.toarray_struphy()
    print("M0 done.")
    M1arr = M1.toarray_struphy()
    M2arr = M2.toarray_struphy()
    M3arr = M3.toarray_struphy()

    v0arr = v0arr[0].flatten()
    v1arr = []
    for i in v1arr1:
        aux = i.flatten()
        for j in aux:
            v1arr.append(j)
    v2arr = []
    for i in v2arr1:
        aux = i.flatten()
        for j in aux:
            v2arr.append(j)
    v3arr = v3arr[0].flatten()

    # not in-place
    compare_arrays(M0.dot(v0), xp.matmul(M0arr, v0arr), rank)
    compare_arrays(M1.dot(v1), xp.matmul(M1arr, v1arr), rank)
    compare_arrays(M2.dot(v2), xp.matmul(M2arr, v2arr), rank)
    compare_arrays(M3.dot(v3), xp.matmul(M3arr, v3arr), rank)

    # Now we test the in-place version
    IM0 = xp.zeros([M0.codomain.dimension, M0.domain.dimension], dtype=M0.dtype)
    IM1 = xp.zeros([M1.codomain.dimension, M1.domain.dimension], dtype=M1.dtype)
    IM2 = xp.zeros([M2.codomain.dimension, M2.domain.dimension], dtype=M2.dtype)
    IM3 = xp.zeros([M3.codomain.dimension, M3.domain.dimension], dtype=M3.dtype)

    M0.toarray_struphy(out=IM0)
    M1.toarray_struphy(out=IM1)
    M2.toarray_struphy(out=IM2)
    M3.toarray_struphy(out=IM3)

    compare_arrays(M0.dot(v0), xp.matmul(IM0, v0arr), rank)
    compare_arrays(M1.dot(v1), xp.matmul(IM1, v1arr), rank)
    compare_arrays(M2.dot(v2), xp.matmul(IM2, v2arr), rank)
    compare_arrays(M3.dot(v3), xp.matmul(IM3, v3arr), rank)

    print("test_toarray_struphy passed!")

    # assert xp.allclose(out1.toarray(), v1.toarray(), atol=1e-5)


if __name__ == "__main__":
    test_toarray_struphy(
        [32, 2, 2],
        [2, 1, 1],
        [True, True, True],
        ["Cuboid", {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}],
    )
    test_toarray_struphy(
        [2, 32, 2],
        [1, 2, 1],
        [False, True, True],
        ["Cuboid", {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}],
    )
    test_toarray_struphy(
        [2, 2, 32],
        [1, 1, 2],
        [True, False, True],
        ["Cuboid", {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}],
    )
    test_toarray_struphy(
        [2, 2, 32],
        [1, 1, 2],
        [False, False, False],
        ["Cuboid", {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}],
    )
