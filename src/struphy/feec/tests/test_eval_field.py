import pytest
from psydac.ddm.mpi import MockComm
from psydac.ddm.mpi import mpi as MPI

from struphy.utils.arrays import xp


@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[3, 2, 4]])
@pytest.mark.parametrize("spl_kind", [[False, False, True], [False, True, False], [True, False, False]])
def test_eval_field(Nel, p, spl_kind):
    """Compares distributed array spline evaluation in Field object with legacy code."""

    from struphy.bsplines.evaluation_kernels_3d import evaluate_matrix
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays
    from struphy.geometry.base import Domain
    from struphy.initial import perturbations

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # fem field objects
    p0 = derham.create_spline_function("pressure", "H1")
    E1 = derham.create_spline_function("e_field", "Hcurl")
    B2 = derham.create_spline_function("b_field", "Hdiv")
    n3 = derham.create_spline_function("density", "L2")
    uv = derham.create_spline_function("velocity", "H1vec")

    # initialize with sin/cos perturbations
    pert_p0 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,))

    pert_E1_1 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,), given_in_basis="1", comp=0)
    pert_E1_2 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,), given_in_basis="1", comp=1)
    pert_E1_3 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,), given_in_basis="1", comp=2)

    pert_B2_1 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,), given_in_basis="2", comp=0)
    pert_B2_2 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,), given_in_basis="2", comp=1)
    pert_B2_3 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,), given_in_basis="2", comp=2)

    pert_n3 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,))

    pert_uv_1 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,), given_in_basis="v", comp=0)
    pert_uv_2 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,), given_in_basis="v", comp=1)
    pert_uv_3 = perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(5.0,), given_in_basis="v", comp=2)

    p0.initialize_coeffs(perturbations=pert_p0)
    E1.initialize_coeffs(perturbations=[pert_E1_1, pert_E1_2, pert_E1_3])
    B2.initialize_coeffs(perturbations=[pert_B2_1, pert_B2_2, pert_B2_3])
    n3.initialize_coeffs(perturbations=pert_n3)
    uv.initialize_coeffs(perturbations=[pert_uv_1, pert_uv_2, pert_uv_3])

    # evaluation points for meshgrid
    eta1 = xp.linspace(0, 1, 11)
    eta2 = xp.linspace(0, 1, 14)
    eta3 = xp.linspace(0, 1, 18)

    # evaluation points for markers
    Np = 33
    markers = xp.random.rand(Np, 3)
    markers_1 = xp.zeros((eta1.size, 3))
    markers_1[:, 0] = eta1
    markers_2 = xp.zeros((eta2.size, 3))
    markers_2[:, 1] = eta2
    markers_3 = xp.zeros((eta3.size, 3))
    markers_3[:, 2] = eta3

    # arrays for legacy evaluation
    arr1, arr2, arr3, is_sparse_meshgrid = Domain.prepare_eval_pts(eta1, eta2, eta3)
    tmp = xp.zeros_like(arr1)

    ######
    # V0 #
    ######
    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(p0.vector.toarray(), p0.nbasis)
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(p0.vector, coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["0"].knots[0],
        derham.Vh_fem["0"].knots[1],
        derham.Vh_fem["0"].knots[2],
        p[0],
        p[1],
        p[2],
        derham.indN[0],
        derham.indN[1],
        derham.indN[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        0,
    )
    val_legacy = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val = p0(eta1, eta2, eta3, squeeze_out=True)
    assert xp.allclose(val, val_legacy)

    # marker evaluation
    m_vals = p0(markers)
    assert m_vals.shape == (Np,)

    m_vals_1 = p0(markers_1)
    m_vals_2 = p0(markers_2)
    m_vals_3 = p0(markers_3)
    m_vals_ref_1 = p0(eta1, 0.0, 0.0, squeeze_out=True)
    m_vals_ref_2 = p0(0.0, eta2, 0.0, squeeze_out=True)
    m_vals_ref_3 = p0(0.0, 0.0, eta3, squeeze_out=True)

    assert xp.allclose(m_vals_1, m_vals_ref_1)
    assert xp.allclose(m_vals_2, m_vals_ref_2)
    assert xp.allclose(m_vals_3, m_vals_ref_3)

    ######
    # V1 #
    ######
    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(E1.vector[0].toarray(), E1.nbasis[0])
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(E1.vector[0], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["3"].knots[0],
        derham.Vh_fem["0"].knots[1],
        derham.Vh_fem["0"].knots[2],
        p[0] - 1,
        p[1],
        p[2],
        derham.indD[0],
        derham.indN[1],
        derham.indN[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        11,
    )
    val_legacy_1 = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(E1.vector[1].toarray(), E1.nbasis[1])
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(E1.vector[1], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["0"].knots[0],
        derham.Vh_fem["3"].knots[1],
        derham.Vh_fem["0"].knots[2],
        p[0],
        p[1] - 1,
        p[2],
        derham.indN[0],
        derham.indD[1],
        derham.indN[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        12,
    )
    val_legacy_2 = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(E1.vector[2].toarray(), E1.nbasis[2])
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(E1.vector[2], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["0"].knots[0],
        derham.Vh_fem["0"].knots[1],
        derham.Vh_fem["3"].knots[2],
        p[0],
        p[1],
        p[2] - 1,
        derham.indN[0],
        derham.indN[1],
        derham.indD[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        13,
    )
    val_legacy_3 = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val1, val2, val3 = E1(eta1, eta2, eta3, squeeze_out=True)
    assert xp.allclose(val1, val_legacy_1)
    assert xp.allclose(val2, val_legacy_2)
    assert xp.allclose(val3, val_legacy_3)

    # marker evaluation
    m_vals = E1(markers)
    assert m_vals[0].shape == m_vals[1].shape == m_vals[2].shape == (Np,)

    m_vals_1 = E1(markers_1)
    m_vals_2 = E1(markers_2)
    m_vals_3 = E1(markers_3)
    m_vals_ref_1 = E1(eta1, 0.0, 0.0, squeeze_out=True)
    m_vals_ref_2 = E1(0.0, eta2, 0.0, squeeze_out=True)
    m_vals_ref_3 = E1(0.0, 0.0, eta3, squeeze_out=True)

    assert xp.all(
        [xp.allclose(m_vals_1_i, m_vals_ref_1_i) for m_vals_1_i, m_vals_ref_1_i in zip(m_vals_1, m_vals_ref_1)]
    )
    assert xp.all(
        [xp.allclose(m_vals_2_i, m_vals_ref_2_i) for m_vals_2_i, m_vals_ref_2_i in zip(m_vals_2, m_vals_ref_2)]
    )
    assert xp.all(
        [xp.allclose(m_vals_3_i, m_vals_ref_3_i) for m_vals_3_i, m_vals_ref_3_i in zip(m_vals_3, m_vals_ref_3)]
    )

    ######
    # V2 #
    ######
    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(B2.vector[0].toarray(), B2.nbasis[0])
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(B2.vector[0], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["0"].knots[0],
        derham.Vh_fem["3"].knots[1],
        derham.Vh_fem["3"].knots[2],
        p[0],
        p[1] - 1,
        p[2] - 1,
        derham.indN[0],
        derham.indD[1],
        derham.indD[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        21,
    )
    val_legacy_1 = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(B2.vector[1].toarray(), B2.nbasis[1])
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(B2.vector[1], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["3"].knots[0],
        derham.Vh_fem["0"].knots[1],
        derham.Vh_fem["3"].knots[2],
        p[0] - 1,
        p[1],
        p[2] - 1,
        derham.indD[0],
        derham.indN[1],
        derham.indD[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        22,
    )
    val_legacy_2 = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(B2.vector[2].toarray(), B2.nbasis[2])
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(B2.vector[2], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["3"].knots[0],
        derham.Vh_fem["3"].knots[1],
        derham.Vh_fem["0"].knots[2],
        p[0] - 1,
        p[1] - 1,
        p[2],
        derham.indD[0],
        derham.indD[1],
        derham.indN[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        23,
    )
    val_legacy_3 = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val1, val2, val3 = B2(eta1, eta2, eta3, squeeze_out=True)
    assert xp.allclose(val1, val_legacy_1)
    assert xp.allclose(val2, val_legacy_2)
    assert xp.allclose(val3, val_legacy_3)

    # marker evaluation
    m_vals = B2(markers)
    assert m_vals[0].shape == m_vals[1].shape == m_vals[2].shape == (Np,)

    m_vals_1 = B2(markers_1)
    m_vals_2 = B2(markers_2)
    m_vals_3 = B2(markers_3)
    m_vals_ref_1 = B2(eta1, 0.0, 0.0, squeeze_out=True)
    m_vals_ref_2 = B2(0.0, eta2, 0.0, squeeze_out=True)
    m_vals_ref_3 = B2(0.0, 0.0, eta3, squeeze_out=True)

    assert xp.all(
        [xp.allclose(m_vals_1_i, m_vals_ref_1_i) for m_vals_1_i, m_vals_ref_1_i in zip(m_vals_1, m_vals_ref_1)]
    )
    assert xp.all(
        [xp.allclose(m_vals_2_i, m_vals_ref_2_i) for m_vals_2_i, m_vals_ref_2_i in zip(m_vals_2, m_vals_ref_2)]
    )
    assert xp.all(
        [xp.allclose(m_vals_3_i, m_vals_ref_3_i) for m_vals_3_i, m_vals_ref_3_i in zip(m_vals_3, m_vals_ref_3)]
    )

    ######
    # V3 #
    ######
    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(n3.vector.toarray(), n3.nbasis)
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(n3.vector, coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["3"].knots[0],
        derham.Vh_fem["3"].knots[1],
        derham.Vh_fem["3"].knots[2],
        p[0] - 1,
        p[1] - 1,
        p[2] - 1,
        derham.indD[0],
        derham.indD[1],
        derham.indD[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        3,
    )
    val_legacy = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val = n3(eta1, eta2, eta3, squeeze_out=True)
    assert xp.allclose(val, val_legacy)

    # marker evaluation
    m_vals = n3(markers)
    assert m_vals.shape == (Np,)

    m_vals_1 = n3(markers_1)
    m_vals_2 = n3(markers_2)
    m_vals_3 = n3(markers_3)
    m_vals_ref_1 = n3(eta1, 0.0, 0.0, squeeze_out=True)
    m_vals_ref_2 = n3(0.0, eta2, 0.0, squeeze_out=True)
    m_vals_ref_3 = n3(0.0, 0.0, eta3, squeeze_out=True)

    assert xp.allclose(m_vals_1, m_vals_ref_1)
    assert xp.allclose(m_vals_2, m_vals_ref_2)
    assert xp.allclose(m_vals_3, m_vals_ref_3)

    #########
    # V0vec #
    #########
    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(uv.vector[0].toarray(), uv.nbasis[0])
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(uv.vector[0], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["0"].knots[0],
        derham.Vh_fem["0"].knots[1],
        derham.Vh_fem["0"].knots[2],
        p[0],
        p[1],
        p[2],
        derham.indN[0],
        derham.indN[1],
        derham.indN[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        0,
    )
    val_legacy_1 = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(uv.vector[1].toarray(), uv.nbasis[1])
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(uv.vector[1], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["0"].knots[0],
        derham.Vh_fem["0"].knots[1],
        derham.Vh_fem["0"].knots[2],
        p[0],
        p[1],
        p[2],
        derham.indN[0],
        derham.indN[1],
        derham.indN[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        0,
    )
    val_legacy_2 = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = xp.reshape(uv.vector[2].toarray(), uv.nbasis[2])
    if isinstance(comm, MockComm):
        coeffs = coeffs_loc
    else:
        coeffs = xp.zeros_like(coeffs_loc)
        comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(uv.vector[2], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(
        derham.Vh_fem["0"].knots[0],
        derham.Vh_fem["0"].knots[1],
        derham.Vh_fem["0"].knots[2],
        p[0],
        p[1],
        p[2],
        derham.indN[0],
        derham.indN[1],
        derham.indN[2],
        coeffs,
        arr1,
        arr2,
        arr3,
        tmp,
        0,
    )
    val_legacy_3 = xp.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val1, val2, val3 = uv(eta1, eta2, eta3, squeeze_out=True)
    assert xp.allclose(val1, val_legacy_1)
    assert xp.allclose(val2, val_legacy_2)
    assert xp.allclose(val3, val_legacy_3)

    # marker evaluation
    m_vals = uv(markers)
    assert m_vals[0].shape == m_vals[1].shape == m_vals[2].shape == (Np,)

    m_vals_1 = uv(markers_1)
    m_vals_2 = uv(markers_2)
    m_vals_3 = uv(markers_3)
    m_vals_ref_1 = uv(eta1, 0.0, 0.0, squeeze_out=True)
    m_vals_ref_2 = uv(0.0, eta2, 0.0, squeeze_out=True)
    m_vals_ref_3 = uv(0.0, 0.0, eta3, squeeze_out=True)

    assert xp.all(
        [xp.allclose(m_vals_1_i, m_vals_ref_1_i) for m_vals_1_i, m_vals_ref_1_i in zip(m_vals_1, m_vals_ref_1)]
    )
    assert xp.all(
        [xp.allclose(m_vals_2_i, m_vals_ref_2_i) for m_vals_2_i, m_vals_ref_2_i in zip(m_vals_2, m_vals_ref_2)]
    )
    assert xp.all(
        [xp.allclose(m_vals_3_i, m_vals_ref_3_i) for m_vals_3_i, m_vals_ref_3_i in zip(m_vals_3, m_vals_ref_3)]
    )

    print("\nAll assertions passed.")


if __name__ == "__main__":
    test_eval_field([8, 9, 10], [3, 2, 4], [False, False, True])
