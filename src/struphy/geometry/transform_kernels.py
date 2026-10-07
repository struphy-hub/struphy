# coding: utf-8

"""
1. Basic pull-back (physical --> logical) operations between scalar fields, vector fields and differential p-forms:

- 0-form :  a^0                  = a
- 1-form : (a^1_1, a^1_2, a^1_3) =           DF^T    (ax, ay, az)
- 2-form : (a^2_1, a^2_2, a^2_3) = |det(DF)| DF^(-1) (ax, ay, az)
- 3-form :  a^3                  = |det(DF)| a
- vector : (a_1  , a_2  , a_3  ) =           DF^(-1) (ax, ay, az)

2. Basic push-forward (logical --> physical) operations between scalar fields, vector fields and differential p-forms:

- 0-form :  a           = a^0
- 1-form : (ax, ay, az) =             DF^(-T) (a^1_1, a^1_2, a^1_3)
- 2-form : (ax, ay, az) = 1/|det(DF)| DF      (a^2_1, a^2_2, a^2_3)
- 3-form :  a           = 1/|det(DF)| a^3
- vector : (ax, ay, az) =             DF      (a_1  , a_2  , a_3  )

3. Basic transformations between scalar fields, vector fields and differential p-forms:

- 0-form --> 3-form : a^3 = a^0 * |det(DF)|
- 3-form --> 0-form : a^0 = a^3 / |det(DF)|
- 1-form --> 2-form : (a^2_1, a^2_2, a^2_3) = G^(-1) * (a^1_1, a^1_2, a^1_3) * |det(DF)|
- 2-form --> 1-form : (a^1_1, a^1_2, a^1_3) = G      * (a^2_1, a^2_2, a^2_3) / |det(DF)|

- norm vector --> vector : (a_1, a_2, a_3) = (a^*_1/sqrt(sum(DF[:,0]^2)),
                                              a^*_2/sqrt(sum(DF[:,1]^2)),
                                              a^*_3/sqrt(sum(DF[:,2]^2)))

- norm vector --> 1-form : (a^1_1, a^1_2, a^1_3) = G * (a^*_1/sqrt(sum(DF[:,0]^2)),
                                                        a^*_2/sqrt(sum(DF[:,1]^2)),
                                                        a^*_3/sqrt(sum(DF[:,2]^2)))

- norm vector --> 2-form : (a^2_1, a^2_2, a^2_3) =     (a^*_1/sqrt(sum(DF[:,0]^2)),
                                                        a^*_2/sqrt(sum(DF[:,1]^2)),
                                                        a^*_3/sqrt(sum(DF[:,2]^2))) * |det(DF)|

- vector --> 1-form : (a^1_1, a^1_2, a^1_3) = G * (a_1, a_2, a_3)
- vector --> 2-form : (a^2_1, a^2_2, a^2_3) =     (a_1, a_2, a_3) * |det(DF)|

- 1-form --> vector : (a_1, a_2, a_3) = G^(-1) * (a^1_1, a^1_2, a^1_3)
- 2-form --> vector : (a_1, a_2, a_3) =          (a^2_1, a^2_2, a^2_3) / |det(DF)|

Let h_i = ||DF[:, i]||. The normalized-vector representation stores a^*_i = h_i a_i.
Consequently, the inverse transformations added here are 1-form --> norm, 2-form --> norm,
and vector --> norm: each first obtains the Cartesian vector and then multiplies component i
by h_i. These are used when :meth:`Domain.transform` converts a saved FEEC field to the
``"norm"`` representation.
"""

from numpy import empty, shape, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments


@stack_array("dfmat1", "dfmat2")
def pull(
    a: "float[:]",
    eta1: float,
    eta2: float,
    eta3: float,
    kind_fun: int,
    args_domain: "DomainArguments",
    out: "float[:]",
):
    """
    Pull-back of a Cartesian scalar/vector field to a differential p-form.

    Parameters
    ----------
    a : float[:]
        Value of scalar field a[0] or values of Cartesian components of vector field (a[0], a[1], a[2]).

    eta1, eta2, eta3 : float
        Logical evaluation points.

    kind_fun : int
        Which pull-back to be performed.

    args_domain : DomainArguments
        Domain info.

    out : float[:]
        Output values.
    """

    dfmat1 = empty((3, 3), dtype=float)
    dfmat2 = empty((3, 3), dtype=float)

    # evaluate Jacobian matrix and its determinant
    if kind_fun > 0:
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfmat1)
        detdf = linalg_kernels.det(dfmat1)

    # 0-form
    if kind_fun == 0:
        out[0] = a[0]

    # 3-form
    elif kind_fun == 1:
        out[0] = a[0] * abs(detdf)

    # 1-form
    elif kind_fun == 10:
        linalg_kernels.transpose(dfmat1, dfmat2)
        linalg_kernels.matrix_vector(dfmat2, a, out)

    # 2-form
    elif kind_fun == 11:
        linalg_kernels.matrix_inv_with_det(dfmat1, 1.0, dfmat2)
        linalg_kernels.matrix_vector(dfmat2, a, out)

        if detdf < 0.0:
            out[:] = -out

    # vector
    elif kind_fun == 12:
        linalg_kernels.matrix_inv(dfmat1, dfmat2)
        linalg_kernels.matrix_vector(dfmat2, a, out)


@stack_array("dfmat1", "dfmat2", "dfmat3")
def push(
    a: "float[:]",
    eta1: float,
    eta2: float,
    eta3: float,
    kind_fun: int,
    args_domain: "DomainArguments",
    out: "float[:]",
):
    """
    Pushforward of a differential p-forms to a Cartesian scalar/vector field.

    Parameters
    ----------
    a : float[:]
        Value of scalar p-form a[0] or values of components of vector valued p-form (a[0], a[1], a[2]).

    eta1, eta2, eta3 : float
        Logical evaluation points.

    kind_fun : int
        Which pushforward to be performed.

    args_domain : DomainArguments
        Domain info.

    out : float[:]
        Output values.
    """

    dfmat1 = empty((3, 3), dtype=float)
    dfmat2 = empty((3, 3), dtype=float)
    dfmat3 = empty((3, 3), dtype=float)

    # evaluate Jacobian matrix and its determinant
    if kind_fun > 0:
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfmat1)
        detdf = linalg_kernels.det(dfmat1)

    # 0-form
    if kind_fun == 0:
        out[0] = a[0]

    # 3-form
    elif kind_fun == 1:
        out[0] = a[0] / abs(detdf)

    # 1-form
    elif kind_fun == 10:
        linalg_kernels.matrix_inv_with_det(dfmat1, detdf, dfmat2)
        linalg_kernels.transpose(dfmat2, dfmat3)
        linalg_kernels.matrix_vector(dfmat3, a, out)

    # 2-form
    elif kind_fun == 11:
        linalg_kernels.matrix_vector(dfmat1, a, out)
        out[:] = out / abs(detdf)

    # vector
    elif kind_fun == 12:
        linalg_kernels.matrix_vector(dfmat1, a, out)


@stack_array("dfmat1", "dfmat2", "dfmat3", "vec1", "vec2")
def tran(
    a: "float[:]",
    eta1: float,
    eta2: float,
    eta3: float,
    kind_fun: int,
    args_domain: "DomainArguments",
    out: "float[:]",
):
    """
    Transformations between differential p-forms and/or vector fields.

    Parameters
    ----------
    a : float[:]
        Value of scalar function a[0] or values of components of vector valued functions (a[0], a[1], a[2]).

    eta1, eta2, eta3 : float
        Logical evaluation points.

    kind_fun : int
        Which transformation to perform. The values are assigned by
        ``Domain.dict_transformations["tran"]``:

        - 0, 1: ``0_to_3``, ``3_to_0``
        - 10, 11: ``1_to_2``, ``2_to_1``
        - 12, 13, 14: ``norm_to_v``, ``norm_to_1``, ``norm_to_2``
        - 15, 16: ``v_to_1``, ``v_to_2``
        - 17, 18: ``1_to_v``, ``2_to_v``
        - 19, 20, 21: ``1_to_norm``, ``2_to_norm``, ``v_to_norm``

    args_domain : DomainArguments
        Domain info.

    out : float[:]
        Output values.
    """

    dfmat1 = empty((3, 3), dtype=float)
    dfmat2 = empty((3, 3), dtype=float)
    dfmat3 = empty((3, 3), dtype=float)

    vec1 = empty(3, dtype=float)
    vec2 = empty(3, dtype=float)

    # evaluate Jacobian matrix and its determinant
    evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfmat1)
    detdf = linalg_kernels.det(dfmat1)

    # 0-form to 3-form
    if kind_fun == 0:
        out[0] = a[0] * abs(detdf)

    # 3-form to 0-form
    elif kind_fun == 1:
        out[0] = a[0] / abs(detdf)

    # 1-form to 2-form (a^2 = G^(-1) * a^1 * |det(DF)|)
    elif kind_fun == 10:
        linalg_kernels.matrix_inv_with_det(dfmat1, detdf, dfmat2)
        linalg_kernels.transpose(dfmat2, dfmat3)
        linalg_kernels.matrix_vector(dfmat3, a, vec1)
        linalg_kernels.matrix_vector(dfmat2, vec1, out)
        out[:] = out * abs(detdf)

    # 2-form to 1-form (a^1 = G * a^2 / |det(DF)|)
    elif kind_fun == 11:
        linalg_kernels.transpose(dfmat1, dfmat2)
        linalg_kernels.matrix_vector(dfmat1, a, vec1)
        linalg_kernels.matrix_vector(dfmat2, vec1, out)
        out[:] = out / abs(detdf)

    # norm vector to vector
    elif kind_fun == 12:
        out[0] = a[0] / sqrt(dfmat1[0, 0] ** 2 + dfmat1[1, 0] ** 2 + dfmat1[2, 0] ** 2)
        out[1] = a[1] / sqrt(dfmat1[0, 1] ** 2 + dfmat1[1, 1] ** 2 + dfmat1[2, 1] ** 2)
        out[2] = a[2] / sqrt(dfmat1[0, 2] ** 2 + dfmat1[1, 2] ** 2 + dfmat1[2, 2] ** 2)

    # norm vector to 1-form (a^1 = G * a)
    elif kind_fun == 13:
        vec1[0] = a[0] / sqrt(dfmat1[0, 0] ** 2 + dfmat1[1, 0] ** 2 + dfmat1[2, 0] ** 2)
        vec1[1] = a[1] / sqrt(dfmat1[0, 1] ** 2 + dfmat1[1, 1] ** 2 + dfmat1[2, 1] ** 2)
        vec1[2] = a[2] / sqrt(dfmat1[0, 2] ** 2 + dfmat1[1, 2] ** 2 + dfmat1[2, 2] ** 2)
        linalg_kernels.transpose(dfmat1, dfmat2)
        linalg_kernels.matrix_vector(dfmat1, vec1, vec2)
        linalg_kernels.matrix_vector(dfmat2, vec2, out)

    # norm vector to 2-form (a^2 = |det(DF)| * a)
    elif kind_fun == 14:
        out[0] = a[0] / sqrt(dfmat1[0, 0] ** 2 + dfmat1[1, 0] ** 2 + dfmat1[2, 0] ** 2)
        out[1] = a[1] / sqrt(dfmat1[0, 1] ** 2 + dfmat1[1, 1] ** 2 + dfmat1[2, 1] ** 2)
        out[2] = a[2] / sqrt(dfmat1[0, 2] ** 2 + dfmat1[1, 2] ** 2 + dfmat1[2, 2] ** 2)
        out[:] = out * abs(detdf)

    # vector to 1-form (a^1 = G * a)
    elif kind_fun == 15:
        linalg_kernels.transpose(dfmat1, dfmat2)
        linalg_kernels.matrix_vector(dfmat1, a, vec1)
        linalg_kernels.matrix_vector(dfmat2, vec1, out)

    # vector to 2-form (a^2 = |det(DF)| * a)
    elif kind_fun == 16:
        out[:] = a * abs(detdf)

    # 1-form to vector (a = G^(-1) * a^1)
    elif kind_fun == 17:
        linalg_kernels.matrix_inv_with_det(dfmat1, detdf, dfmat2)
        linalg_kernels.transpose(dfmat2, dfmat3)
        linalg_kernels.matrix_vector(dfmat3, a, vec1)
        linalg_kernels.matrix_vector(dfmat2, vec1, out)

    # 2-form to vector (a = a^2 / |det(DF)|)
    elif kind_fun == 18:
        out[:] = a / abs(detdf)

    # 1-form to normalized vector
    elif kind_fun == 19:
        linalg_kernels.matrix_inv_with_det(dfmat1, detdf, dfmat2)
        linalg_kernels.transpose(dfmat2, dfmat3)
        linalg_kernels.matrix_vector(dfmat3, a, vec1)
        linalg_kernels.matrix_vector(dfmat2, vec1, out)
        out[0] = out[0] * sqrt(dfmat1[0, 0] ** 2 + dfmat1[1, 0] ** 2 + dfmat1[2, 0] ** 2)
        out[1] = out[1] * sqrt(dfmat1[0, 1] ** 2 + dfmat1[1, 1] ** 2 + dfmat1[2, 1] ** 2)
        out[2] = out[2] * sqrt(dfmat1[0, 2] ** 2 + dfmat1[1, 2] ** 2 + dfmat1[2, 2] ** 2)

    # 2-form to normalized vector
    elif kind_fun == 20:
        out[:] = a / abs(detdf)
        out[0] = out[0] * sqrt(dfmat1[0, 0] ** 2 + dfmat1[1, 0] ** 2 + dfmat1[2, 0] ** 2)
        out[1] = out[1] * sqrt(dfmat1[0, 1] ** 2 + dfmat1[1, 1] ** 2 + dfmat1[2, 1] ** 2)
        out[2] = out[2] * sqrt(dfmat1[0, 2] ** 2 + dfmat1[1, 2] ** 2 + dfmat1[2, 2] ** 2)

    # Cartesian vector to normalized vector
    elif kind_fun == 21:
        out[0] = a[0] * sqrt(dfmat1[0, 0] ** 2 + dfmat1[1, 0] ** 2 + dfmat1[2, 0] ** 2)
        out[1] = a[1] * sqrt(dfmat1[0, 1] ** 2 + dfmat1[1, 1] ** 2 + dfmat1[2, 1] ** 2)
        out[2] = a[2] * sqrt(dfmat1[0, 2] ** 2 + dfmat1[1, 2] ** 2 + dfmat1[2, 2] ** 2)
