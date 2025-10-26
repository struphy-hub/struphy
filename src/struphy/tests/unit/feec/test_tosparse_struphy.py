import time

import pytest


@pytest.mark.parametrize("Nel", [[12, 5, 2], [8, 12, 4], [5, 4, 12]])
@pytest.mark.parametrize("p", [[3, 2, 1]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [True, False, False]])
@pytest.mark.parametrize(
    "mapping",
    [["Cuboid", {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}]],
)
def test_tosparse_struphy(Nel, p, spl_kind, mapping):
    """
    TODO
    """

    import cunumpy as xp
    from psydac.ddm.mpi import MockComm
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.geometry import domains

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=MPI.COMM_WORLD)

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

    # ========= test toarray_struphy =================

    M0arr = M0.toarray_struphy(is_sparse=True, format="csr")
    M1arr = M1.toarray_struphy(is_sparse=True, format="csc")
    M2arr = M2.toarray_struphy(is_sparse=True, format="bsr")
    M3arr = M3.toarray_struphy(is_sparse=True, format="lil")
    M0arrad = M0.toarray_struphy(is_sparse=True, format="dok")
    M1arrad = M1.toarray_struphy(is_sparse=True, format="coo")
    M2arrad = M2.toarray_struphy(is_sparse=True, format="dia")

    v0_local = M0.dot(v0).toarray()
    if isinstance(comm, MockComm):
        v0_global = v0_local
    else:
        v0_global = M0.domain.zeros().toarray()
        comm.Allreduce(v0_local, v0_global, op=MPI.SUM)

    v1_local = M1.dot(v1).toarray()
    if isinstance(comm, MockComm):
        v1_global = v1_local
    else:
        v1_global = M1.domain.zeros().toarray()
        comm.Allreduce(v1_local, v1_global, op=MPI.SUM)

    v2_local = M2.dot(v2).toarray()
    if isinstance(comm, MockComm):
        v2_global = v2_local
    else:
        v2_global = M2.domain.zeros().toarray()
        comm.Allreduce(v2_local, v2_global, op=MPI.SUM)

    v3_local = M3.dot(v3).toarray()
    if isinstance(comm, MockComm):
        v3_global = v3_local
    else:
        v3_global = M3.domain.zeros().toarray()
        comm.Allreduce(v3_local, v3_global, op=MPI.SUM)

    # not in-place
    assert xp.allclose(v0_global, M0arr.dot(v0arr))
    assert xp.allclose(v1_global, M1arr.dot(v1arr))
    assert xp.allclose(v2_global, M2arr.dot(v2arr))
    assert xp.allclose(v3_global, M3arr.dot(v3arr))
    assert xp.allclose(v0_global, M0arrad.dot(v0arr))
    assert xp.allclose(v1_global, M1arrad.dot(v1arr))
    assert xp.allclose(v2_global, M2arrad.dot(v2arr))

    print("test_tosparse_struphy passed!")


if __name__ == "__main__":
    test_tosparse_struphy(
        [32, 2, 2],
        [2, 1, 1],
        [True, True, True],
        ["Colella", {"Lx": 1.0, "Ly": 2.0, "alpha": 0.5, "Lz": 3.0}],
    )
    test_tosparse_struphy(
        [2, 32, 2],
        [1, 2, 1],
        [True, True, True],
        ["Colella", {"Lx": 1.0, "Ly": 2.0, "alpha": 0.5, "Lz": 3.0}],
    )
    test_tosparse_struphy(
        [2, 2, 32],
        [1, 1, 2],
        [True, True, True],
        ["Colella", {"Lx": 1.0, "Ly": 2.0, "alpha": 0.5, "Lz": 3.0}],
    )
    test_tosparse_struphy(
        [2, 2, 32],
        [1, 1, 2],
        [False, False, False],
        ["Colella", {"Lx": 1.0, "Ly": 2.0, "alpha": 0.5, "Lz": 3.0}],
    )
