"""Pyccel kernels for spline mappings and their Jacobian matrices (see :class:`struphy.geometry.base.Spline`,
:class:`struphy.geometry.base.PoloidalSplineStraight` and :class:`struphy.geometry.base.PoloidalSplineTorus`)."""

from numpy import cos, pi, sin, zeros
from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_2d as evaluation_kernels_2d
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments


@stack_array("b1", "b2", "b3", "tmp1", "tmp2", "tmp3")
def spline_3d(
    eta1: float,
    eta2: float,
    eta3: float,
    degree: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    ind3: "int[:, :]",
    args: "DomainArguments",
    f_out: "float[:]",
):
    r"""Point-wise evaluation of a 3d spline map :math:`F = (F_n)_{(n=x,y,z)}` with

    .. math::

        F_n = \sum_{ijk} c^n_{ijk} N_i(\eta_1) N_j(\eta_2) N_k(\eta_3)\,,

    where :math:`c^n_{ijk}` are the control points of component :math:`n`.
    """

    # mapping spans
    span1 = bsplines_kernels.find_span(args.t1, int(degree[0]), eta1)
    span2 = bsplines_kernels.find_span(args.t2, int(degree[1]), eta2)
    span3 = bsplines_kernels.find_span(args.t3, int(degree[2]), eta3)

    # degree + 1 non-zero mapping splines
    b1 = zeros(int(degree[0]) + 1, dtype=float)
    b2 = zeros(int(degree[1]) + 1, dtype=float)
    b3 = zeros(int(degree[2]) + 1, dtype=float)

    bsplines_kernels.b_splines_slim(args.t1, int(degree[0]), eta1, span1, b1)
    bsplines_kernels.b_splines_slim(args.t2, int(degree[1]), eta2, span2, b2)
    bsplines_kernels.b_splines_slim(args.t3, int(degree[2]), eta3, span3, b3)

    # Evaluate spline mapping
    tmp1 = ind1[span1 - int(degree[0]), :]
    tmp2 = ind2[span2 - int(degree[1]), :]
    tmp3 = ind3[span3 - int(degree[2]), :]

    f_out[0] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        b1,
        b2,
        b3,
        tmp1,
        tmp2,
        tmp3,
        args.cx,
    )
    f_out[1] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        b1,
        b2,
        b3,
        tmp1,
        tmp2,
        tmp3,
        args.cy,
    )
    f_out[2] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        b1,
        b2,
        b3,
        tmp1,
        tmp2,
        tmp3,
        args.cz,
    )


@stack_array("b1", "b2", "b3", "der1", "der2", "der3", "tmp1", "tmp2", "tmp3")
def spline_3d_df(
    eta1: float,
    eta2: float,
    eta3: float,
    degree: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    ind3: "int[:, :]",
    args: "DomainArguments",
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.spline_mappings_kernels.spline_3d`."""

    # mapping spans
    span1 = bsplines_kernels.find_span(args.t1, int(degree[0]), eta1)
    span2 = bsplines_kernels.find_span(args.t2, int(degree[1]), eta2)
    span3 = bsplines_kernels.find_span(args.t3, int(degree[2]), eta3)

    # non-zero splines of mapping, and derivatives
    b1 = zeros(int(degree[0]) + 1, dtype=float)
    b2 = zeros(int(degree[1]) + 1, dtype=float)
    b3 = zeros(int(degree[2]) + 1, dtype=float)

    der1 = zeros(int(degree[0]) + 1, dtype=float)
    der2 = zeros(int(degree[1]) + 1, dtype=float)
    der3 = zeros(int(degree[2]) + 1, dtype=float)

    bsplines_kernels.b_der_splines_slim(args.t1, int(degree[0]), eta1, span1, b1, der1)
    bsplines_kernels.b_der_splines_slim(args.t2, int(degree[1]), eta2, span2, b2, der2)
    bsplines_kernels.b_der_splines_slim(args.t3, int(degree[2]), eta3, span3, b3, der3)

    # Evaluation of Jacobian
    tmp1 = ind1[span1 - int(degree[0]), :]
    tmp2 = ind2[span2 - int(degree[1]), :]
    tmp3 = ind3[span3 - int(degree[2]), :]

    df_out[0, 0] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        der1,
        b2,
        b3,
        tmp1,
        tmp2,
        tmp3,
        args.cx,
    )
    df_out[0, 1] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        b1,
        der2,
        b3,
        tmp1,
        tmp2,
        tmp3,
        args.cx,
    )
    df_out[0, 2] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        b1,
        b2,
        der3,
        tmp1,
        tmp2,
        tmp3,
        args.cx,
    )
    df_out[1, 0] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        der1,
        b2,
        b3,
        tmp1,
        tmp2,
        tmp3,
        args.cy,
    )
    df_out[1, 1] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        b1,
        der2,
        b3,
        tmp1,
        tmp2,
        tmp3,
        args.cy,
    )
    df_out[1, 2] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        b1,
        b2,
        der3,
        tmp1,
        tmp2,
        tmp3,
        args.cy,
    )
    df_out[2, 0] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        der1,
        b2,
        b3,
        tmp1,
        tmp2,
        tmp3,
        args.cz,
    )
    df_out[2, 1] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        b1,
        der2,
        b3,
        tmp1,
        tmp2,
        tmp3,
        args.cz,
    )
    df_out[2, 2] = evaluation_kernels_3d.evaluation_kernel_3d(
        int(degree[0]),
        int(degree[1]),
        int(degree[2]),
        b1,
        b2,
        der3,
        tmp1,
        tmp2,
        tmp3,
        args.cz,
    )


@stack_array("b1", "b2", "tmp1", "tmp2")
def spline_2d_straight(
    eta1: float,
    eta2: float,
    eta3: float,
    degree: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    args: "DomainArguments",
    lz: float,
    f_out: "float[:]",
):
    r"""Point-wise evaluation of a 2d spline map :math:`F = (F_n)_{(n=x,y,z)}` with

    .. math::

        F_{x(y)} &= \sum_{ij} c^{x(y)}_{ij} N_i(\eta_1) N_j(\eta_2) \,,

        F_z &= L_z*\eta_3\,.

    where :math:`c^{x(y)}_{ij}` are the control points in the :math:`\eta_1-\eta_2`-plane, independent of :math:`\eta_3`.
    """

    cx = args.cx[:, :, 0]
    cy = args.cy[:, :, 0]

    # mapping spans
    span1 = bsplines_kernels.find_span(args.t1, int(degree[0]), eta1)
    span2 = bsplines_kernels.find_span(args.t2, int(degree[1]), eta2)

    # degree + 1 non-zero mapping splines
    b1 = zeros(int(degree[0]) + 1, dtype=float)
    b2 = zeros(int(degree[1]) + 1, dtype=float)

    bsplines_kernels.b_splines_slim(args.t1, int(degree[0]), eta1, span1, b1)
    bsplines_kernels.b_splines_slim(args.t2, int(degree[1]), eta2, span2, b2)

    # Evaluate mapping
    tmp1 = ind1[span1 - int(degree[0]), :]
    tmp2 = ind2[span2 - int(degree[1]), :]

    f_out[0] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, b2, tmp1, tmp2, cx)
    f_out[1] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, b2, tmp1, tmp2, cy)
    f_out[2] = lz * eta3

    # TODO: explanation
    if eta1 == 0.0 and cx[0, 0] == cx[0, 1]:
        f_out[0] = cx[0, 0]

    if eta1 == 0.0 and cy[0, 0] == cy[0, 1]:
        f_out[1] = cy[0, 0]


@stack_array("b1", "b2", "der1", "der2", "tmp1", "tmp2")
def spline_2d_straight_df(
    eta1: float,
    eta2: float,
    degree: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    args: "DomainArguments",
    lz: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.spline_mappings_kernels.spline_2d_straight`."""

    cx = args.cx[:, :, 0]
    cy = args.cy[:, :, 0]

    # mapping spans
    span1 = bsplines_kernels.find_span(args.t1, int(degree[0]), eta1)
    span2 = bsplines_kernels.find_span(args.t2, int(degree[1]), eta2)

    # non-zero splines of mapping, and derivatives
    b1 = zeros(int(degree[0]) + 1, dtype=float)
    b2 = zeros(int(degree[1]) + 1, dtype=float)

    der1 = zeros(int(degree[0]) + 1, dtype=float)
    der2 = zeros(int(degree[1]) + 1, dtype=float)

    bsplines_kernels.b_der_splines_slim(args.t1, int(degree[0]), eta1, span1, b1, der1)
    bsplines_kernels.b_der_splines_slim(args.t2, int(degree[1]), eta2, span2, b2, der2)

    # Evaluation of Jacobian
    tmp1 = ind1[span1 - int(degree[0]), :]
    tmp2 = ind2[span2 - int(degree[1]), :]

    df_out[0, 0] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), der1, b2, tmp1, tmp2, cx)
    df_out[0, 1] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, der2, tmp1, tmp2, cx)
    df_out[0, 2] = 0.0
    df_out[1, 0] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), der1, b2, tmp1, tmp2, cy)
    df_out[1, 1] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, der2, tmp1, tmp2, cy)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz

    # TODO: explanation
    if eta1 == 0.0 and cx[0, 0] == cx[0, 1]:
        df_out[0, 1] = 0.0

    if eta1 == 0.0 and cy[0, 0] == cy[0, 1]:
        df_out[1, 1] = 0.0


@stack_array("b1", "b2", "tmp1", "tmp2")
def spline_2d_torus(
    eta1: float,
    eta2: float,
    eta3: float,
    degree: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    args: "DomainArguments",
    tor_period: float,
    f_out: "float[:]",
):
    r"""Point-wise evaluation of a 2d spline map :math:`F = (F_n)_{(n=x,y,z)}` with

    .. math::

        S_{R(z)}(\eta_1, \eta_2) &= \sum_{ij} c^{R(z)}_{ij} N_i(\eta_1) N_j(\eta_2) \,,

        F_x &= S_R(\eta_1, \eta_2) * \cos(2\pi\eta_3)

        F_y &= - S_R(\eta_1, \eta_2) * \sin(2\pi\eta_3)

        F_z &= S_z(\eta_1, \eta_2)\,.

    where :math:`c^{R(z)}_{ij}` are the control points in the :math:`\eta_1-\eta_2`-plane, independent of :math:`\eta_3`.
    """

    cx = args.cx[:, :, 0]
    cy = args.cy[:, :, 0]

    # mapping spans
    span1 = bsplines_kernels.find_span(args.t1, int(degree[0]), eta1)
    span2 = bsplines_kernels.find_span(args.t2, int(degree[1]), eta2)

    # degree + 1 non-zero mapping splines
    b1 = zeros(int(degree[0]) + 1, dtype=float)
    b2 = zeros(int(degree[1]) + 1, dtype=float)

    bsplines_kernels.b_splines_slim(args.t1, int(degree[0]), eta1, span1, b1)
    bsplines_kernels.b_splines_slim(args.t2, int(degree[1]), eta2, span2, b2)

    # Evaluate mapping
    tmp1 = ind1[span1 - int(degree[0]), :]
    tmp2 = ind2[span2 - int(degree[1]), :]

    f_out[0] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, b2, tmp1, tmp2, cx) * cos(
        2 * pi * eta3 / tor_period,
    )
    f_out[1] = (
        evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, b2, tmp1, tmp2, cx)
        * (-1)
        * sin(2 * pi * eta3 / tor_period)
    )
    f_out[2] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, b2, tmp1, tmp2, cy)

    # TODO: explanation
    if eta1 == 0.0 and cx[0, 0] == cx[0, 1]:
        f_out[0] = cx[0, 0] * cos(2 * pi * eta3 / tor_period)
        f_out[1] = cx[0, 0] * (-1) * sin(2 * pi * eta3 / tor_period)

    if eta1 == 0.0 and cy[0, 0] == cy[0, 1]:
        f_out[2] = cy[0, 0]


@stack_array("b1", "b2", "der1", "der2", "tmp1", "tmp2")
def spline_2d_torus_df(
    eta1: float,
    eta2: float,
    eta3: float,
    degree: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    args: "DomainArguments",
    tor_period: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.spline_mappings_kernels.spline_2d_torus`."""

    cx = args.cx[:, :, 0]
    cy = args.cy[:, :, 0]

    # mapping spans
    span1 = bsplines_kernels.find_span(args.t1, int(degree[0]), eta1)
    span2 = bsplines_kernels.find_span(args.t2, int(degree[1]), eta2)

    # non-zero splines of mapping, and derivatives
    b1 = zeros(int(degree[0]) + 1, dtype=float)
    b2 = zeros(int(degree[1]) + 1, dtype=float)

    der1 = zeros(int(degree[0]) + 1, dtype=float)
    der2 = zeros(int(degree[1]) + 1, dtype=float)

    bsplines_kernels.b_der_splines_slim(args.t1, int(degree[0]), eta1, span1, b1, der1)
    bsplines_kernels.b_der_splines_slim(args.t2, int(degree[1]), eta2, span2, b2, der2)

    tmp1 = ind1[span1 - int(degree[0]), :]
    tmp2 = ind2[span2 - int(degree[1]), :]

    df_out[0, 0] = evaluation_kernels_2d.evaluation_kernel_2d(
        int(degree[0]), int(degree[1]), der1, b2, tmp1, tmp2, cx
    ) * cos(
        2 * pi * eta3 / tor_period,
    )
    df_out[0, 1] = evaluation_kernels_2d.evaluation_kernel_2d(
        int(degree[0]), int(degree[1]), b1, der2, tmp1, tmp2, cx
    ) * cos(
        2 * pi * eta3 / tor_period,
    )
    df_out[0, 2] = (
        evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, b2, tmp1, tmp2, cx)
        * sin(2 * pi * eta3 / tor_period)
        * (-2 * pi / tor_period)
    )
    df_out[1, 0] = (
        evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), der1, b2, tmp1, tmp2, cx)
        * (-1)
        * sin(2 * pi * eta3 / tor_period)
    )
    df_out[1, 1] = (
        evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, der2, tmp1, tmp2, cx)
        * (-1)
        * sin(2 * pi * eta3 / tor_period)
    )
    df_out[1, 2] = (
        evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, b2, tmp1, tmp2, cx)
        * (-1)
        * cos(2 * pi * eta3 / tor_period)
        * 2
        * pi
        / tor_period
    )
    df_out[2, 0] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), der1, b2, tmp1, tmp2, cy)
    df_out[2, 1] = evaluation_kernels_2d.evaluation_kernel_2d(int(degree[0]), int(degree[1]), b1, der2, tmp1, tmp2, cy)
    df_out[2, 2] = 0.0

    # TODO: explanation
    if eta1 == 0.0 and cx[0, 0] == cx[0, 1]:
        df_out[0, 1] = 0.0
        df_out[1, 1] = 0.0

    if eta1 == 0.0 and cy[0, 0] == cy[0, 1]:
        df_out[2, 1] = 0.0
