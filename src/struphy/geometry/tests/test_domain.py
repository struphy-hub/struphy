import pytest


def test_prepare_arg():
    """Tests prepare_arg static method in domain base class."""

    import numpy as np

    from struphy.geometry.base import Domain

    def a1(e1, e2, e3):
        return e1 * e2

    def a2(e1, e2, e3):
        return e2 * e3

    def a3(e1, e2, e3):
        return e3 * e1

    def a_vec(e1, e2, e3):
        a_1 = e1 * e2
        a_2 = e2 * e3
        a_3 = e3 * e1

        return np.stack((a_1, a_2, a_3), axis=0)

    # ========== tensor-product/slice evaluation ===============
    e1 = np.random.rand(4)
    e2 = np.random.rand(5)
    e3 = np.random.rand(6)

    E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(e1, e2, e3, flat_eval=False)

    shape_scalar = (E1.shape[0], E2.shape[1], E3.shape[2], 1)
    shape_vector = (E1.shape[0], E2.shape[1], E3.shape[2], 3)

    # ======== callables ============

    # scalar function
    assert Domain.prepare_arg(a1, E1, E2, E3).shape == shape_scalar
    assert Domain.prepare_arg((a1,), E1, E2, E3).shape == shape_scalar
    assert (
        Domain.prepare_arg(
            [
                a1,
            ],
            E1,
            E2,
            E3,
        ).shape
        == shape_scalar
    )

    # vector-valued function
    assert Domain.prepare_arg(a_vec, E1, E2, E3).shape == shape_vector
    assert Domain.prepare_arg((a1, a2, a3), E1, E2, E3).shape == shape_vector
    assert Domain.prepare_arg([a1, a2, a3], E1, E2, E3).shape == shape_vector

    # ======== arrays ===============

    A1 = a1(E1, E2, E3)
    A2 = a2(E1, E2, E3)
    A3 = a3(E1, E2, E3)

    A = a_vec(E1, E2, E3)

    # scalar function
    assert Domain.prepare_arg(A1, E1, E2, E3).shape == shape_scalar
    assert Domain.prepare_arg((A1,), E1, E2, E3).shape == shape_scalar
    assert (
        Domain.prepare_arg(
            [
                A1,
            ],
            E1,
            E2,
            E3,
        ).shape
        == shape_scalar
    )

    # vector-valued function
    assert Domain.prepare_arg(A, E1, E2, E3).shape == shape_vector
    assert Domain.prepare_arg((A1, A2, A3), E1, E2, E3).shape == shape_vector
    assert Domain.prepare_arg([A1, A2, A3], E1, E2, E3).shape == shape_vector

    # ============== markers evaluation ==========================
    markers = np.random.rand(10, 6)

    shape_scalar = (markers.shape[0], 1)
    shape_vector = (markers.shape[0], 3)

    # ======== callables ============

    # scalar function
    assert Domain.prepare_arg(a1, markers).shape == shape_scalar
    assert Domain.prepare_arg((a1,), markers).shape == shape_scalar
    assert (
        Domain.prepare_arg(
            [
                a1,
            ],
            markers,
        ).shape
        == shape_scalar
    )

    # vector-valued function
    assert Domain.prepare_arg(a_vec, markers).shape == shape_vector
    assert Domain.prepare_arg((a1, a2, a3), markers).shape == shape_vector
    assert Domain.prepare_arg([a1, a2, a3], markers).shape == shape_vector

    # ======== arrays ===============

    A1 = a1(markers[:, 0], markers[:, 1], markers[:, 2])
    A2 = a2(markers[:, 0], markers[:, 1], markers[:, 2])
    A3 = a3(markers[:, 0], markers[:, 1], markers[:, 2])

    A = a_vec(markers[:, 0], markers[:, 1], markers[:, 2])

    # scalar function
    assert Domain.prepare_arg(A1, markers).shape == shape_scalar
    assert Domain.prepare_arg((A1,), markers).shape == shape_scalar
    assert (
        Domain.prepare_arg(
            [
                A1,
            ],
            markers,
        ).shape
        == shape_scalar
    )

    # vector-valued function
    assert Domain.prepare_arg(A, markers).shape == shape_vector
    assert Domain.prepare_arg((A1, A2, A3), markers).shape == shape_vector
    assert Domain.prepare_arg([A1, A2, A3], markers).shape == shape_vector


@pytest.mark.parametrize(
    "mapping",
    [
        "Cuboid",
        "HollowCylinder",
        "Colella",
        "Orthogonal",
        "HollowTorus",
        "PoweredEllipticCylinder",
        "ShafranovShiftCylinder",
        "ShafranovSqrtCylinder",
        "ShafranovDshapedCylinder",
        "GVECunit",
        "IGAPolarCylinder",
        "IGAPolarTorus",
        "Tokamak",
    ],
)
def test_evaluation_mappings(mapping):
    """Tests domain object creation with default parameters and evaluation of metric coefficients."""

    import numpy as np

    from struphy.geometry import domains
    from struphy.geometry.base import Domain

    # arrays:
    arr1 = np.linspace(0.0, 1.0, 4)
    arr2 = np.linspace(0.0, 1.0, 5)
    arr3 = np.linspace(0.0, 1.0, 6)
    arrm = np.random.rand(10, 8)
    print()
    print('Testing "evaluate"...')
    print("array shapes:", arr1.shape, arr2.shape, arr3.shape, arrm.shape)

    domain_class = getattr(domains, mapping)
    domain = domain_class()
    print()
    print("Domain object set.")

    assert isinstance(domain, Domain)
    print("domain's kind_map   :", domain.kind_map)
    print("domain's params :", domain.params)

    # point-wise evaluation:
    print("pointwise evaluation, shape:", domain(0.5, 0.5, 0.5, squeeze_out=True).shape)
    assert domain(0.5, 0.5, 0.5, squeeze_out=True).shape == (3,)
    assert domain.jacobian(0.5, 0.5, 0.5, squeeze_out=True).shape == (3, 3)
    assert isinstance(domain.jacobian_det(0.5, 0.5, 0.5, squeeze_out=True), float)
    assert domain.jacobian_inv(0.5, 0.5, 0.5, squeeze_out=True).shape == (3, 3)
    assert domain.metric(0.5, 0.5, 0.5, squeeze_out=True).shape == (3, 3)
    assert domain.metric_inv(0.5, 0.5, 0.5, squeeze_out=True).shape == (3, 3)

    # markers evaluation:
    print("markers evaluation, shape:", domain(arrm).shape)
    assert domain(arrm).shape == (3, arrm.shape[0])
    assert domain.jacobian(arrm).shape == (3, 3, arrm.shape[0])
    assert domain.jacobian_det(arrm).shape == (arrm.shape[0],)
    assert domain.jacobian_inv(arrm).shape == (3, 3, arrm.shape[0])
    assert domain.metric(arrm).shape == (3, 3, arrm.shape[0])
    assert domain.metric_inv(arrm).shape == (3, 3, arrm.shape[0])

    # eta1-array evaluation:
    print("eta1 array evaluation, shape:", domain(arr1, 0.5, 0.5, squeeze_out=True).shape)
    assert domain(arr1, 0.5, 0.5, squeeze_out=True).shape == (3,) + arr1.shape
    assert domain.jacobian(arr1, 0.5, 0.5, squeeze_out=True).shape == (3, 3) + arr1.shape
    assert domain.jacobian_inv(arr1, 0.5, 0.5, squeeze_out=True).shape == (3, 3) + arr1.shape
    assert domain.jacobian_det(arr1, 0.5, 0.5, squeeze_out=True).shape == () + arr1.shape
    assert domain.metric(arr1, 0.5, 0.5, squeeze_out=True).shape == (3, 3) + arr1.shape
    assert domain.metric_inv(arr1, 0.5, 0.5, squeeze_out=True).shape == (3, 3) + arr1.shape

    # eta2-array evaluation:
    print("eta2 array evaluation, shape:", domain(0.5, arr2, 0.5, squeeze_out=True).shape)
    assert domain(0.5, arr2, 0.5, squeeze_out=True).shape == (3,) + arr2.shape
    assert domain.jacobian(0.5, arr2, 0.5, squeeze_out=True).shape == (3, 3) + arr2.shape
    assert domain.jacobian_inv(0.5, arr2, 0.5, squeeze_out=True).shape == (3, 3) + arr2.shape
    assert domain.jacobian_det(0.5, arr2, 0.5, squeeze_out=True).shape == () + arr2.shape
    assert domain.metric(0.5, arr2, 0.5, squeeze_out=True).shape == (3, 3) + arr2.shape
    assert domain.metric_inv(0.5, arr2, 0.5, squeeze_out=True).shape == (3, 3) + arr2.shape

    # eta3-array evaluation:
    print("eta3 array evaluation, shape:", domain(0.5, 0.5, arr3).shape)
    assert domain(0.5, 0.5, arr3, squeeze_out=True).shape == (3,) + arr3.shape
    assert domain.jacobian(0.5, 0.5, arr3, squeeze_out=True).shape == (3, 3) + arr3.shape
    assert domain.jacobian_inv(0.5, 0.5, arr3, squeeze_out=True).shape == (3, 3) + arr3.shape
    assert domain.jacobian_det(0.5, 0.5, arr3, squeeze_out=True).shape == () + arr3.shape
    assert domain.metric(0.5, 0.5, arr3, squeeze_out=True).shape == (3, 3) + arr3.shape
    assert domain.metric_inv(0.5, 0.5, arr3, squeeze_out=True).shape == (3, 3) + arr3.shape

    # eta1-eta2-array evaluation:
    print("eta1-eta2 array evaluation, shape:", domain(arr1, arr2, 0.5, squeeze_out=True))
    assert domain(arr1, arr2, 0.5, squeeze_out=True).shape == (3,) + arr1.shape + arr2.shape
    assert domain.jacobian(arr1, arr2, 0.5, squeeze_out=True).shape == (3, 3) + arr1.shape + arr2.shape
    assert domain.jacobian_inv(arr1, arr2, 0.5, squeeze_out=True).shape == (3, 3) + arr1.shape + arr2.shape
    assert domain.jacobian_det(arr1, arr2, 0.5, squeeze_out=True).shape == () + arr1.shape + arr2.shape
    assert domain.metric(arr1, arr2, 0.5, squeeze_out=True).shape == (3, 3) + arr1.shape + arr2.shape
    assert domain.metric_inv(arr1, arr2, 0.5, squeeze_out=True).shape == (3, 3) + arr1.shape + arr2.shape

    # eta1-eta3-array evaluation:
    print("eta1-eta3 array evaluation, shape:", domain(arr1, 0.5, arr3, squeeze_out=True))
    assert domain(arr1, 0.5, arr3, squeeze_out=True).shape == (3,) + arr1.shape + arr3.shape
    assert domain.jacobian(arr1, 0.5, arr3, squeeze_out=True).shape == (3, 3) + arr1.shape + arr3.shape
    assert domain.jacobian_inv(arr1, 0.5, arr3, squeeze_out=True).shape == (3, 3) + arr1.shape + arr3.shape
    assert domain.jacobian_det(arr1, 0.5, arr3, squeeze_out=True).shape == () + arr1.shape + arr3.shape
    assert domain.metric(arr1, 0.5, arr3, squeeze_out=True).shape == (3, 3) + arr1.shape + arr3.shape
    assert domain.metric_inv(arr1, 0.5, arr3, squeeze_out=True).shape == (3, 3) + arr1.shape + arr3.shape

    # eta2-eta3-array evaluation:
    print("eta2-eta3 array evaluation, shape:", domain(0.5, arr2, arr3, squeeze_out=True))
    assert domain(0.5, arr2, arr3, squeeze_out=True).shape == (3,) + arr2.shape + arr3.shape
    assert domain.jacobian(0.5, arr2, arr3, squeeze_out=True).shape == (3, 3) + arr2.shape + arr3.shape
    assert domain.jacobian_inv(0.5, arr2, arr3, squeeze_out=True).shape == (3, 3) + arr2.shape + arr3.shape
    assert domain.jacobian_det(0.5, arr2, arr3, squeeze_out=True).shape == () + arr2.shape + arr3.shape
    assert domain.metric(0.5, arr2, arr3, squeeze_out=True).shape == (3, 3) + arr2.shape + arr3.shape
    assert domain.metric_inv(0.5, arr2, arr3, squeeze_out=True).shape == (3, 3) + arr2.shape + arr3.shape

    # eta1-eta2-eta3 array evaluation:
    print("eta1-eta2-eta3-array evaluation, shape:", domain(arr1, arr2, arr3))
    assert domain(arr1, arr2, arr3).shape == (3,) + arr1.shape + arr2.shape + arr3.shape
    assert domain.jacobian(arr1, arr2, arr3).shape == (3, 3) + arr1.shape + arr2.shape + arr3.shape
    assert domain.jacobian_inv(arr1, arr2, arr3).shape == (3, 3) + arr1.shape + arr2.shape + arr3.shape
    assert domain.jacobian_det(arr1, arr2, arr3).shape == () + arr1.shape + arr2.shape + arr3.shape
    assert domain.metric(arr1, arr2, arr3).shape == (3, 3) + arr1.shape + arr2.shape + arr3.shape
    assert domain.metric_inv(arr1, arr2, arr3).shape == (3, 3) + arr1.shape + arr2.shape + arr3.shape

    # matrix evaluations at one point in third direction
    mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing="ij")
    mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing="ij")
    mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing="ij")

    # eta1-eta2 matrix evaluation:
    print("eta1-eta2 matrix evaluation, shape:", domain(mat12_x, mat12_y, 0.5, squeeze_out=True).shape)
    assert domain(mat12_x, mat12_y, 0.5, squeeze_out=True).shape == (3,) + mat12_x.shape
    assert domain.jacobian(mat12_x, mat12_y, 0.5, squeeze_out=True).shape == (3, 3) + mat12_x.shape
    assert domain.jacobian_inv(mat12_x, mat12_y, 0.5, squeeze_out=True).shape == (3, 3) + mat12_x.shape
    assert domain.jacobian_det(mat12_x, mat12_y, 0.5, squeeze_out=True).shape == () + mat12_x.shape
    assert domain.metric(mat12_x, mat12_y, 0.5, squeeze_out=True).shape == (3, 3) + mat12_x.shape
    assert domain.metric_inv(mat12_x, mat12_y, 0.5, squeeze_out=True).shape == (3, 3) + mat12_x.shape

    # eta1-eta3 matrix evaluation:
    print("eta1-eta3 matrix evaluation, shape:", domain(mat13_x, 0.5, mat13_z, squeeze_out=True).shape)
    assert domain(mat13_x, 0.5, mat13_z, squeeze_out=True).shape == (3,) + mat13_x.shape
    assert domain.jacobian(mat13_x, 0.5, mat13_z, squeeze_out=True).shape == (3, 3) + mat13_x.shape
    assert domain.jacobian_inv(mat13_x, 0.5, mat13_z, squeeze_out=True).shape == (3, 3) + mat13_x.shape
    assert domain.jacobian_det(mat13_x, 0.5, mat13_z, squeeze_out=True).shape == () + mat13_x.shape
    assert domain.metric(mat13_x, 0.5, mat13_z, squeeze_out=True).shape == (3, 3) + mat13_x.shape
    assert domain.metric_inv(mat13_x, 0.5, mat13_z, squeeze_out=True).shape == (3, 3) + mat13_x.shape

    # eta2-eta3 matrix evaluation:
    print("eta2-eta3 matrix evaluation, shape:", domain(0.5, mat23_y, mat23_z, squeeze_out=True).shape)
    assert domain(0.5, mat23_y, mat23_z, squeeze_out=True).shape == (3,) + mat23_y.shape
    assert domain.jacobian(0.5, mat23_y, mat23_z, squeeze_out=True).shape == (3, 3) + mat23_y.shape
    assert domain.jacobian_inv(0.5, mat23_y, mat23_z, squeeze_out=True).shape == (3, 3) + mat23_y.shape
    assert domain.jacobian_det(0.5, mat23_y, mat23_z, squeeze_out=True).shape == () + mat23_y.shape
    assert domain.metric(0.5, mat23_y, mat23_z, squeeze_out=True).shape == (3, 3) + mat23_y.shape
    assert domain.metric_inv(0.5, mat23_y, mat23_z, squeeze_out=True).shape == (3, 3) + mat23_y.shape

    # matrix evaluations for sparse meshgrid
    mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing="ij", sparse=True)
    print("sparse meshgrid matrix evaluation, shape:", domain(mat_x, mat_y, mat_z).shape)
    assert domain(mat_x, mat_y, mat_z).shape == (3,) + (mat_x.shape[0], mat_y.shape[1], mat_z.shape[2])
    assert domain.jacobian(mat_x, mat_y, mat_z).shape == (3, 3) + (mat_x.shape[0], mat_y.shape[1], mat_z.shape[2])
    assert domain.jacobian_inv(mat_x, mat_y, mat_z).shape == (3, 3) + (mat_x.shape[0], mat_y.shape[1], mat_z.shape[2])
    assert domain.jacobian_det(mat_x, mat_y, mat_z).shape == () + (mat_x.shape[0], mat_y.shape[1], mat_z.shape[2])
    assert domain.metric(mat_x, mat_y, mat_z).shape == (3, 3) + (mat_x.shape[0], mat_y.shape[1], mat_z.shape[2])
    assert domain.metric_inv(mat_x, mat_y, mat_z).shape == (3, 3) + (mat_x.shape[0], mat_y.shape[1], mat_z.shape[2])

    # matrix evaluations
    mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing="ij")
    print("matrix evaluation, shape:", domain(mat_x, mat_y, mat_z).shape)
    assert domain(mat_x, mat_y, mat_z).shape == (3,) + mat_x.shape
    assert domain.jacobian(mat_x, mat_y, mat_z).shape == (3, 3) + mat_x.shape
    assert domain.jacobian_inv(mat_x, mat_y, mat_z).shape == (3, 3) + mat_x.shape
    assert domain.jacobian_det(mat_x, mat_y, mat_z).shape == () + mat_x.shape
    assert domain.metric(mat_x, mat_y, mat_z).shape == (3, 3) + mat_x.shape
    assert domain.metric_inv(mat_x, mat_y, mat_z).shape == (3, 3) + mat_x.shape


def test_pullback():
    """Tests pullbacks to p-forms."""

    import numpy as np

    from struphy.geometry import domains
    from struphy.geometry.base import Domain

    # arrays:
    arr1 = np.linspace(0.0, 1.0, 4)
    arr2 = np.linspace(0.0, 1.0, 5)
    arr3 = np.linspace(0.0, 1.0, 6)
    print()
    print('Testing "pull"...')
    print("array shapes:", arr1.shape, arr2.shape, arr3.shape)

    markers = np.random.rand(13, 6)

    # physical function to pull back (used as components of forms too):
    def fun(x, y, z):
        return np.exp(x) * np.sin(y) * np.cos(z)

    domain_class = getattr(domains, "Colella")
    domain = domain_class()
    print()
    print("Domain object set.")

    assert isinstance(domain, Domain)
    print("domain's kind_map   :", domain.kind_map)
    print("domain's params :", domain.params)

    for p_str in domain.dict_transformations["pull"]:
        print("component:", p_str)

        if p_str == "0" or p_str == "3":
            fun_form = fun
        else:
            fun_form = [fun, fun, fun]

        # point-wise pullback:
        if p_str == "0" or p_str == "3":
            assert isinstance(domain.pull(fun_form, 0.5, 0.5, 0.5, kind=p_str, squeeze_out=True), float)
        else:
            assert domain.pull(fun_form, 0.5, 0.5, 0.5, kind=p_str, squeeze_out=True).shape == (3,)

        # markers pullback:
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, markers, kind=p_str, squeeze_out=True).shape == (markers.shape[0],)
        else:
            assert domain.pull(fun_form, markers, kind=p_str, squeeze_out=True).shape == (3, markers.shape[0])

        # eta1-array pullback:
        # print('eta1 array pullback, shape:', domain.pull(fun_form, arr1, .5, .5, p_str).shape)
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, arr1, 0.5, 0.5, kind=p_str, squeeze_out=True).shape == arr1.shape
        else:
            assert domain.pull(fun_form, arr1, 0.5, 0.5, kind=p_str, squeeze_out=True).shape == (3,) + arr1.shape

        # eta2-array pullback:
        # print('eta2 array pullback, shape:', domain.pull(fun_form, .5, arr2, .5, p_str).shape)
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, 0.5, arr2, 0.5, kind=p_str, squeeze_out=True).shape == arr2.shape
        else:
            assert domain.pull(fun_form, 0.5, arr2, 0.5, kind=p_str, squeeze_out=True).shape == (3,) + arr2.shape

        # eta3-array pullback:
        # print('eta3 array pullback, shape:', domain.pull(fun_form, .5, .5, arr3, p_str).shape)
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, 0.5, 0.5, arr3, kind=p_str, squeeze_out=True).shape == arr3.shape
        else:
            assert domain.pull(fun_form, 0.5, 0.5, arr3, kind=p_str, squeeze_out=True).shape == (3,) + arr3.shape

        # eta1-eta2-array pullback:
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, arr1, arr2, 0.5, kind=p_str, squeeze_out=True).shape == arr1.shape + arr2.shape
        else:
            assert (
                domain.pull(fun_form, arr1, arr2, 0.5, kind=p_str, squeeze_out=True).shape
                == (3,) + arr1.shape + arr2.shape
            )

        # eta1-eta3-array pullback:
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, arr1, 0.5, arr3, kind=p_str, squeeze_out=True).shape == arr1.shape + arr3.shape
        else:
            assert (
                domain.pull(fun_form, arr1, 0.5, arr3, kind=p_str, squeeze_out=True).shape
                == (3,) + arr1.shape + arr3.shape
            )

        # eta2-eta3-array pullback:
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, 0.5, arr2, arr3, kind=p_str, squeeze_out=True).shape == arr2.shape + arr3.shape
        else:
            assert (
                domain.pull(fun_form, 0.5, arr2, arr3, kind=p_str, squeeze_out=True).shape
                == (3,) + arr2.shape + arr3.shape
            )

        # eta1-eta2-eta3 array pullback:
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, arr1, arr2, arr3, kind=p_str).shape == arr1.shape + arr2.shape + arr3.shape
        else:
            assert (
                domain.pull(fun_form, arr1, arr2, arr3, kind=p_str).shape == (3,) + arr1.shape + arr2.shape + arr3.shape
            )

        # matrix pullbacks at one point in third direction
        mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing="ij")
        mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing="ij")
        mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing="ij")

        # eta1-eta2 matrix pullback:
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, mat12_x, mat12_y, 0.5, kind=p_str, squeeze_out=True).shape == mat12_x.shape
        else:
            assert (
                domain.pull(fun_form, mat12_x, mat12_y, 0.5, kind=p_str, squeeze_out=True).shape == (3,) + mat12_x.shape
            )

        # eta1-eta3 matrix pullback:
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, mat13_x, 0.5, mat13_z, kind=p_str, squeeze_out=True).shape == mat13_x.shape
        else:
            assert (
                domain.pull(fun_form, mat13_x, 0.5, mat13_z, kind=p_str, squeeze_out=True).shape == (3,) + mat13_x.shape
            )

        # eta2-eta3 matrix pullback:
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, 0.5, mat23_y, mat23_z, kind=p_str, squeeze_out=True).shape == mat23_z.shape
        else:
            assert (
                domain.pull(fun_form, 0.5, mat23_y, mat23_z, kind=p_str, squeeze_out=True).shape == (3,) + mat23_z.shape
            )

        # matrix pullbacks for sparse meshgrid
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing="ij", sparse=True)
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == (
                mat_x.shape[0],
                mat_y.shape[1],
                mat_z.shape[2],
            )
        else:
            assert domain.pull(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == (
                3,
                mat_x.shape[0],
                mat_y.shape[1],
                mat_z.shape[2],
            )

        # matrix pullbacks
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing="ij")
        if p_str == "0" or p_str == "3":
            assert domain.pull(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == mat_x.shape
        else:
            assert domain.pull(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == (3,) + mat_x.shape


def test_pushforward():
    """Tests pushforward of p-forms."""

    import numpy as np

    from struphy.geometry import domains
    from struphy.geometry.base import Domain

    # arrays:
    arr1 = np.linspace(0.0, 1.0, 4)
    arr2 = np.linspace(0.0, 1.0, 5)
    arr3 = np.linspace(0.0, 1.0, 6)
    print()
    print('Testing "push"...')
    print("array shapes:", arr1.shape, arr2.shape, arr3.shape)

    markers = np.random.rand(13, 6)

    # logical function to push (used as components of forms too):
    def fun(e1, e2, e3):
        return np.exp(e1) * np.sin(e2) * np.cos(e3)

    domain_class = getattr(domains, "Colella")
    domain = domain_class()
    print()
    print("Domain object set.")

    assert isinstance(domain, Domain)
    print("domain's kind_map   :", domain.kind_map)
    print("domain's params :", domain.params)

    for p_str in domain.dict_transformations["push"]:
        print("component:", p_str)

        if p_str == "0" or p_str == "3":
            fun_form = fun
        else:
            fun_form = [fun, fun, fun]

        # point-wise push:
        if p_str == "0" or p_str == "3":
            assert isinstance(domain.push(fun_form, 0.5, 0.5, 0.5, kind=p_str, squeeze_out=True), float)
        else:
            assert domain.push(fun_form, 0.5, 0.5, 0.5, kind=p_str, squeeze_out=True).shape == (3,)

        # markers push:
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, markers, kind=p_str).shape == (markers.shape[0],)
        else:
            assert domain.push(fun_form, markers, kind=p_str).shape == (3, markers.shape[0])

        # eta1-array push:
        # print('eta1 array push, shape:', domain.push(fun_form, arr1, .5, .5, p_str).shape)
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, arr1, 0.5, 0.5, kind=p_str, squeeze_out=True).shape == arr1.shape
        else:
            assert domain.push(fun_form, arr1, 0.5, 0.5, kind=p_str, squeeze_out=True).shape == (3,) + arr1.shape

        # eta2-array push:
        # print('eta2 array push, shape:', domain.push(fun_form, .5, arr2, .5, p_str).shape)
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, 0.5, arr2, 0.5, kind=p_str, squeeze_out=True).shape == arr2.shape
        else:
            assert domain.push(fun_form, 0.5, arr2, 0.5, kind=p_str, squeeze_out=True).shape == (3,) + arr2.shape

        # eta3-array push:
        # print('eta3 array push, shape:', domain.push(fun_form, .5, .5, arr3, p_str).shape)
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, 0.5, 0.5, arr3, kind=p_str, squeeze_out=True).shape == arr3.shape
        else:
            assert domain.push(fun_form, 0.5, 0.5, arr3, kind=p_str, squeeze_out=True).shape == (3,) + arr3.shape

        # eta1-eta2-array push:
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, arr1, arr2, 0.5, kind=p_str, squeeze_out=True).shape == arr1.shape + arr2.shape
        else:
            assert (
                domain.push(fun_form, arr1, arr2, 0.5, kind=p_str, squeeze_out=True).shape
                == (3,) + arr1.shape + arr2.shape
            )

        # eta1-eta3-array push:
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, arr1, 0.5, arr3, kind=p_str, squeeze_out=True).shape == arr1.shape + arr3.shape
        else:
            assert (
                domain.push(fun_form, arr1, 0.5, arr3, kind=p_str, squeeze_out=True).shape
                == (3,) + arr1.shape + arr3.shape
            )

        # eta2-eta3-array push:
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, 0.5, arr2, arr3, kind=p_str, squeeze_out=True).shape == arr2.shape + arr3.shape
        else:
            assert (
                domain.push(fun_form, 0.5, arr2, arr3, kind=p_str, squeeze_out=True).shape
                == (3,) + arr2.shape + arr3.shape
            )

        # eta1-eta2-eta3 array push:
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, arr1, arr2, arr3, kind=p_str).shape == arr1.shape + arr2.shape + arr3.shape
        else:
            assert (
                domain.push(fun_form, arr1, arr2, arr3, kind=p_str).shape == (3,) + arr1.shape + arr2.shape + arr3.shape
            )

        # matrix pushs at one point in third direction
        mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing="ij")
        mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing="ij")
        mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing="ij")

        # eta1-eta2 matrix push:
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, mat12_x, mat12_y, 0.5, kind=p_str, squeeze_out=True).shape == mat12_x.shape
        else:
            assert (
                domain.push(fun_form, mat12_x, mat12_y, 0.5, kind=p_str, squeeze_out=True).shape == (3,) + mat12_x.shape
            )

        # eta1-eta3 matrix push:
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, mat13_x, 0.5, mat13_z, kind=p_str, squeeze_out=True).shape == mat13_x.shape
        else:
            assert (
                domain.push(fun_form, mat13_x, 0.5, mat13_z, kind=p_str, squeeze_out=True).shape == (3,) + mat13_x.shape
            )

        # eta2-eta3 matrix push:
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, 0.5, mat23_y, mat23_z, kind=p_str, squeeze_out=True).shape == mat23_z.shape
        else:
            assert (
                domain.push(fun_form, 0.5, mat23_y, mat23_z, kind=p_str, squeeze_out=True).shape == (3,) + mat23_z.shape
            )

        # matrix pushs for sparse meshgrid
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing="ij", sparse=True)
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == (
                mat_x.shape[0],
                mat_y.shape[1],
                mat_z.shape[2],
            )
        else:
            assert domain.push(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == (
                3,
                mat_x.shape[0],
                mat_y.shape[1],
                mat_z.shape[2],
            )

        # matrix pushs
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing="ij")
        if p_str == "0" or p_str == "3":
            assert domain.push(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == mat_x.shape
        else:
            assert domain.push(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == (3,) + mat_x.shape


def test_transform():
    """Tests transformation of p-forms."""

    import numpy as np

    from struphy.geometry import domains
    from struphy.geometry.base import Domain

    # arrays:
    arr1 = np.linspace(0.0, 1.0, 4)
    arr2 = np.linspace(0.0, 1.0, 5)
    arr3 = np.linspace(0.0, 1.0, 6)
    print()
    print('Testing "transform"...')
    print("array shapes:", arr1.shape, arr2.shape, arr3.shape)

    markers = np.random.rand(13, 6)

    # logical function to push (used as components of forms too):
    def fun(e1, e2, e3):
        return np.exp(e1) * np.sin(e2) * np.cos(e3)

    domain_class = getattr(domains, "Colella")
    domain = domain_class()
    print()
    print("Domain object set.")

    assert isinstance(domain, Domain)
    print("domain's kind_map   :", domain.kind_map)
    print("domain's params :", domain.params)

    for p_str in domain.dict_transformations["tran"]:
        print("component:", p_str)

        if p_str == "0_to_3" or p_str == "3_to_0":
            fun_form = fun
        else:
            fun_form = [fun, fun, fun]

        # point-wise transform:
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert isinstance(domain.transform(fun_form, 0.5, 0.5, 0.5, kind=p_str, squeeze_out=True), float)
        else:
            assert domain.transform(fun_form, 0.5, 0.5, 0.5, kind=p_str, squeeze_out=True).shape == (3,)

        # markers transform:
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert domain.transform(fun_form, markers, kind=p_str).shape == (markers.shape[0],)
        else:
            assert domain.transform(fun_form, markers, kind=p_str).shape == (3, markers.shape[0])

        # eta1-array transform:
        # print('eta1 array transform, shape:', domain.transform(fun_form, arr1, .5, .5, p_str).shape)
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert domain.transform(fun_form, arr1, 0.5, 0.5, kind=p_str, squeeze_out=True).shape == arr1.shape
        else:
            assert domain.transform(fun_form, arr1, 0.5, 0.5, kind=p_str, squeeze_out=True).shape == (3,) + arr1.shape

        # eta2-array transform:
        # print('eta2 array transform, shape:', domain.transform(fun_form, .5, arr2, .5, p_str).shape)
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert domain.transform(fun_form, 0.5, arr2, 0.5, kind=p_str, squeeze_out=True).shape == arr2.shape
        else:
            assert domain.transform(fun_form, 0.5, arr2, 0.5, kind=p_str, squeeze_out=True).shape == (3,) + arr2.shape

        # eta3-array transform:
        # print('eta3 array transform, shape:', domain.transform(fun_form, .5, .5, arr3, p_str).shape)
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert domain.transform(fun_form, 0.5, 0.5, arr3, kind=p_str, squeeze_out=True).shape == arr3.shape
        else:
            assert domain.transform(fun_form, 0.5, 0.5, arr3, kind=p_str, squeeze_out=True).shape == (3,) + arr3.shape

        # eta1-eta2-array transform:
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert (
                domain.transform(fun_form, arr1, arr2, 0.5, kind=p_str, squeeze_out=True).shape
                == arr1.shape + arr2.shape
            )
        else:
            assert (
                domain.transform(fun_form, arr1, arr2, 0.5, kind=p_str, squeeze_out=True).shape
                == (3,) + arr1.shape + arr2.shape
            )

        # eta1-eta3-array transform:
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert (
                domain.transform(fun_form, arr1, 0.5, arr3, kind=p_str, squeeze_out=True).shape
                == arr1.shape + arr3.shape
            )
        else:
            assert (
                domain.transform(fun_form, arr1, 0.5, arr3, kind=p_str, squeeze_out=True).shape
                == (3,) + arr1.shape + arr3.shape
            )

        # eta2-eta3-array transform:
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert (
                domain.transform(fun_form, 0.5, arr2, arr3, kind=p_str, squeeze_out=True).shape
                == arr2.shape + arr3.shape
            )
        else:
            assert (
                domain.transform(fun_form, 0.5, arr2, arr3, kind=p_str, squeeze_out=True).shape
                == (3,) + arr2.shape + arr3.shape
            )

        # eta1-eta2-eta3 array transform:
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert (
                domain.transform(fun_form, arr1, arr2, arr3, kind=p_str).shape == arr1.shape + arr2.shape + arr3.shape
            )
        else:
            assert (
                domain.transform(fun_form, arr1, arr2, arr3, kind=p_str).shape
                == (3,) + arr1.shape + arr2.shape + arr3.shape
            )

        # matrix transforms at one point in third direction
        mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing="ij")
        mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing="ij")
        mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing="ij")

        # eta1-eta2 matrix transform:
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert (
                domain.transform(fun_form, mat12_x, mat12_y, 0.5, kind=p_str, squeeze_out=True).shape == mat12_x.shape
            )
        else:
            assert (
                domain.transform(fun_form, mat12_x, mat12_y, 0.5, kind=p_str, squeeze_out=True).shape
                == (3,) + mat12_x.shape
            )

        # eta1-eta3 matrix transform:
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert (
                domain.transform(fun_form, mat13_x, 0.5, mat13_z, kind=p_str, squeeze_out=True).shape == mat13_x.shape
            )
        else:
            assert (
                domain.transform(fun_form, mat13_x, 0.5, mat13_z, kind=p_str, squeeze_out=True).shape
                == (3,) + mat13_x.shape
            )

        # eta2-eta3 matrix transform:
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert (
                domain.transform(fun_form, 0.5, mat23_y, mat23_z, kind=p_str, squeeze_out=True).shape == mat23_z.shape
            )
        else:
            assert (
                domain.transform(fun_form, 0.5, mat23_y, mat23_z, kind=p_str, squeeze_out=True).shape
                == (3,) + mat23_z.shape
            )

        # matrix transforms for sparse meshgrid
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing="ij", sparse=True)
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert domain.transform(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == (
                mat_x.shape[0],
                mat_y.shape[1],
                mat_z.shape[2],
            )
        else:
            assert domain.transform(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == (
                3,
                mat_x.shape[0],
                mat_y.shape[1],
                mat_z.shape[2],
            )

        # matrix transforms
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing="ij")
        if p_str == "0_to_3" or p_str == "3_to_0":
            assert domain.transform(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == mat_x.shape
        else:
            assert domain.transform(fun_form, mat_x, mat_y, mat_z, kind=p_str).shape == (3,) + mat_x.shape


# def test_transform():
#    """ Tests transformation of p-forms.
#    """
#
#    from struphy.geometry import domains
#    import numpy as np
#
#    # arrays:
#    arr1 = np.linspace(0., 1., 4)
#    arr2 = np.linspace(0., 1., 5)
#    arr3 = np.linspace(0., 1., 6)
#    print()
#    print('Testing "transform"...')
#    print('array shapes:', arr1.shape, arr2.shape, arr3.shape)
#
#    # logical function to tranform (used as components of forms too):
#    fun = lambda eta1, eta2, eta3: np.exp(eta1)*np.sin(eta2)*np.cos(eta3)
#
#    domain_class = getattr(domains, 'Colella')
#    domain = domain_class()
#    print()
#    print('Domain object set.')
#
#    print('domain\'s kind_map   :', domain.kind_map)
#    print('domain\'s params :', domain.params)
#
#    for p_str in domain.keys_transform:
#
#        print('component:', p_str)
#
#        if p_str == '0_to_3' or p_str == '3_to_0':
#            fun_form = fun
#        else:
#            fun_form = [fun, fun, fun]
#
#        # point-wise transformation:
#        assert isinstance(domain.transform(fun_form, .5, .5, .5, p_str), float)
#        #print('pointwise transformation, size:', domain.transform(fun_form, .5, .5, .5, p_str).size)
#
#        # flat transformation:
#        #assert domain.transform(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
#        #assert domain.transform(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
#        #assert domain.transform(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
#
#        # eta1-array transformation:
#        #print('eta1 array transformation, shape:', domain.transform(fun_form, arr1, .5, .5, p_str).shape)
#        assert domain.transform(fun_form, arr1, .5, .5, p_str).shape == arr1.shape
#        # eta2-array transformation:
#        #print('eta2 array transformation, shape:', domain.transform(fun_form, .5, arr2, .5, p_str).shape)
#        assert domain.transform(fun_form, .5, arr2, .5, p_str).shape == arr2.shape
#        # eta3-array transformation:
#        #print('eta3 array transformation, shape:', domain.transform(fun_form, .5, .5, arr3, p_str).shape)
#        assert domain.transform(fun_form, .5, .5, arr3, p_str).shape == arr3.shape
#
#        # eta1-eta2-array transformation:
#        a = domain.transform(fun_form, arr1, arr2, .5, p_str)
#        #print('eta1-eta2 array transformation, shape:', a.shape)
#        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
#        # eta1-eta3-array transformation:
#        a = domain.transform(fun_form, arr1, .5, arr3, p_str)
#        #print('eta1-eta3 array transformation, shape:', a.shape)
#        assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
#        # eta2-eta3-array transformation:
#        a = domain.transform(fun_form, .5, arr2, arr3, p_str)
#        #print('eta2-eta3 array transformation, shape:', a.shape)
#        assert a.shape[0] == arr2.size and a.shape[1] == arr3.size
#
#        # eta1-eta2-eta3 array transformation:
#        a = domain.transform(fun_form, arr1, arr2, arr3, p_str)
#        #print('eta1-eta2-eta3-array transformation, shape:', a.shape)
#        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size
#
#        # matrix transformation at one point in third direction
#        mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing='ij')
#        mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing='ij')
#        mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing='ij')
#
#        # eta1-eta2 matrix transformation:
#        a = domain.transform(fun_form, mat12_x, mat12_y, .5, p_str)
#        #print('eta1-eta2 matrix transformation, shape:', a.shape)
#        assert a.shape == mat12_x.shape
#        # eta1-eta3 matrix transformation:
#        a = domain.transform(fun_form, mat13_x, .5, mat13_z, p_str)
#        #print('eta1-eta3 matrix transformation, shape:', a.shape)
#        assert a.shape == mat13_x.shape
#        # eta2-eta3 matrix transformation:
#        a = domain.transform(fun_form, .5, mat23_y, mat23_z, p_str)
#        #print('eta2-eta3 matrix transformation, shape:', a.shape)
#        assert a.shape == mat23_y.shape
#
#        # matrix transformation for sparse meshgrid
#        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij', sparse=True)
#        a = domain.transform(fun_form, mat_x, mat_y, mat_z, p_str)
#        #print('sparse meshgrid matrix transformation, shape:', a.shape)
#        assert a.shape[0] == mat_x.shape[0] and a.shape[1] == mat_y.shape[1] and a.shape[2] == mat_z.shape[2]
#
#        # matrix transformation
#        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij')
#        a = domain.transform(fun_form, mat_x, mat_y, mat_z, p_str)
#        #print('matrix transformation, shape:', a.shape)
#        assert a.shape == mat_x.shape


if __name__ == "__main__":
    test_prepare_arg()
    test_evaluation_mappings("GVECunit")
    test_pullback()
    test_pushforward()
    test_transform()
