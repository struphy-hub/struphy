from numpy import arcsin, arctan, cos, pi, sin, sqrt, tan, zeros
from pyccel.decorators import pure, stack_array

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
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.spline_3d`."""

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


@stack_array(
    "b1",
    "b2",
    "b3",
    "d1",
    "d2",
    "d3",
    "dd1",
    "dd2",
    "dd3",
    "tmp1",
    "tmp2",
    "tmp3",
)
def spline_3d_d2f(
    eta1: float,
    eta2: float,
    eta3: float,
    degree: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    ind3: "int[:, :]",
    args: "DomainArguments",
    d2f_out: "float[:,:,:]",
):
    """
    Exact Hessian of the 3D tensor-product spline mapping.

    ``d2f_out[i,j,k]`` contains

        d²F_i / (d eta_j d eta_k).
    """
    # Stack-array shapes must be expressed directly in terms of formal
    # arguments. Pyccel does not infer them through local aliases p1/p2/p3.
    b1 = zeros(int(degree[0]) + 1, dtype=float)
    b2 = zeros(int(degree[1]) + 1, dtype=float)
    b3 = zeros(int(degree[2]) + 1, dtype=float)

    d1 = zeros(int(degree[0]) + 1, dtype=float)
    d2 = zeros(int(degree[1]) + 1, dtype=float)
    d3 = zeros(int(degree[2]) + 1, dtype=float)

    dd1 = zeros(int(degree[0]) + 1, dtype=float)
    dd2 = zeros(int(degree[1]) + 1, dtype=float)
    dd3 = zeros(int(degree[2]) + 1, dtype=float)

    # Local aliases may be introduced after the stack-array allocations.
    p1 = int(degree[0])
    p2 = int(degree[1])
    p3 = int(degree[2])

    span1 = bsplines_kernels.find_span(args.t1, p1, eta1)
    span2 = bsplines_kernels.find_span(args.t2, p2, eta2)
    span3 = bsplines_kernels.find_span(args.t3, p3, eta3)

    bsplines_kernels.b_2der_splines_slim(args.t1, p1, eta1, span1, b1, d1, dd1)
    bsplines_kernels.b_2der_splines_slim(args.t2, p2, eta2, span2, b2, d2, dd2)
    bsplines_kernels.b_2der_splines_slim(args.t3, p3, eta3, span3, b3, d3, dd3)

    tmp1 = ind1[span1 - p1, :]
    tmp2 = ind2[span2 - p2, :]
    tmp3 = ind3[span3 - p3, :]

    d2f_out[:, :, :] = 0.0

    # Evaluate one physical component at a time. Pyccel does not make
    # selecting args.cx/args.cy/args.cz dynamically convenient, so the
    # calls are written explicitly.

    # F_x
    d2f_out[0, 0, 0] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, dd1, b2, b3, tmp1, tmp2, tmp3, args.cx)
    d2f_out[0, 1, 1] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, b1, dd2, b3, tmp1, tmp2, tmp3, args.cx)
    d2f_out[0, 2, 2] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, b1, b2, dd3, tmp1, tmp2, tmp3, args.cx)
    d2f_out[0, 0, 1] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, d1, d2, b3, tmp1, tmp2, tmp3, args.cx)
    d2f_out[0, 1, 0] = d2f_out[0, 0, 1]
    d2f_out[0, 0, 2] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, d1, b2, d3, tmp1, tmp2, tmp3, args.cx)
    d2f_out[0, 2, 0] = d2f_out[0, 0, 2]
    d2f_out[0, 1, 2] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, b1, d2, d3, tmp1, tmp2, tmp3, args.cx)
    d2f_out[0, 2, 1] = d2f_out[0, 1, 2]

    # F_y
    d2f_out[1, 0, 0] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, dd1, b2, b3, tmp1, tmp2, tmp3, args.cy)
    d2f_out[1, 1, 1] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, b1, dd2, b3, tmp1, tmp2, tmp3, args.cy)
    d2f_out[1, 2, 2] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, b1, b2, dd3, tmp1, tmp2, tmp3, args.cy)
    d2f_out[1, 0, 1] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, d1, d2, b3, tmp1, tmp2, tmp3, args.cy)
    d2f_out[1, 1, 0] = d2f_out[1, 0, 1]
    d2f_out[1, 0, 2] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, d1, b2, d3, tmp1, tmp2, tmp3, args.cy)
    d2f_out[1, 2, 0] = d2f_out[1, 0, 2]
    d2f_out[1, 1, 2] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, b1, d2, d3, tmp1, tmp2, tmp3, args.cy)
    d2f_out[1, 2, 1] = d2f_out[1, 1, 2]

    # F_z
    d2f_out[2, 0, 0] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, dd1, b2, b3, tmp1, tmp2, tmp3, args.cz)
    d2f_out[2, 1, 1] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, b1, dd2, b3, tmp1, tmp2, tmp3, args.cz)
    d2f_out[2, 2, 2] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, b1, b2, dd3, tmp1, tmp2, tmp3, args.cz)
    d2f_out[2, 0, 1] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, d1, d2, b3, tmp1, tmp2, tmp3, args.cz)
    d2f_out[2, 1, 0] = d2f_out[2, 0, 1]
    d2f_out[2, 0, 2] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, d1, b2, d3, tmp1, tmp2, tmp3, args.cz)
    d2f_out[2, 2, 0] = d2f_out[2, 0, 2]
    d2f_out[2, 1, 2] = evaluation_kernels_3d.evaluation_kernel_3d(p1, p2, p3, b1, d2, d3, tmp1, tmp2, tmp3, args.cz)
    d2f_out[2, 2, 1] = d2f_out[2, 1, 2]


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
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.spline_2d_straight`."""

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


@stack_array(
    "b1",
    "b2",
    "d1",
    "d2",
    "dd1",
    "dd2",
    "tmp1",
    "tmp2",
)
def spline_2d_straight_d2f(
    eta1: float,
    eta2: float,
    degree: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    args: "DomainArguments",
    d2f_out: "float[:,:,:]",
):
    r"""
    Exact mapping Hessian for the straight 2D spline mapping

    .. math::

        F(\eta_1,\eta_2,\eta_3)
        =
        \left(
            X(\eta_1,\eta_2),
            Y(\eta_1,\eta_2),
            L_z\eta_3
        \right).

    The output convention is

    .. math::

        \mathtt{d2f\_out[i,j,k]}
        =
        \frac{\partial^2 F_i}
        {\partial\eta_j\partial\eta_k}.
    """
    cx = args.cx[:, :, 0]
    cy = args.cy[:, :, 0]

    b1 = zeros(int(degree[0]) + 1, dtype=float)
    b2 = zeros(int(degree[1]) + 1, dtype=float)

    d1 = zeros(int(degree[0]) + 1, dtype=float)
    d2 = zeros(int(degree[1]) + 1, dtype=float)

    dd1 = zeros(int(degree[0]) + 1, dtype=float)
    dd2 = zeros(int(degree[1]) + 1, dtype=float)

    p1 = int(degree[0])
    p2 = int(degree[1])

    span1 = bsplines_kernels.find_span(args.t1, p1, eta1)
    span2 = bsplines_kernels.find_span(args.t2, p2, eta2)

    bsplines_kernels.b_2der_splines_slim(
        args.t1,
        p1,
        eta1,
        span1,
        b1,
        d1,
        dd1,
    )
    bsplines_kernels.b_2der_splines_slim(
        args.t2,
        p2,
        eta2,
        span2,
        b2,
        d2,
        dd2,
    )

    tmp1 = ind1[span1 - p1, :]
    tmp2 = ind2[span2 - p2, :]

    d2f_out[:, :, :] = 0.0

    # ----------------------------------------------------------
    # F_0 = X(eta1, eta2)
    # ----------------------------------------------------------
    d2f_out[0, 0, 0] = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        dd1,
        b2,
        tmp1,
        tmp2,
        cx,
    )

    d2f_out[0, 0, 1] = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        d1,
        d2,
        tmp1,
        tmp2,
        cx,
    )
    d2f_out[0, 1, 0] = d2f_out[0, 0, 1]

    d2f_out[0, 1, 1] = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        b1,
        dd2,
        tmp1,
        tmp2,
        cx,
    )

    # ----------------------------------------------------------
    # F_1 = Y(eta1, eta2)
    # ----------------------------------------------------------
    d2f_out[1, 0, 0] = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        dd1,
        b2,
        tmp1,
        tmp2,
        cy,
    )

    d2f_out[1, 0, 1] = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        d1,
        d2,
        tmp1,
        tmp2,
        cy,
    )
    d2f_out[1, 1, 0] = d2f_out[1, 0, 1]

    d2f_out[1, 1, 1] = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        b1,
        dd2,
        tmp1,
        tmp2,
        cy,
    )

    # F_2 = Lz * eta3 is affine, so all its second derivatives
    # are zero. All derivatives involving eta3 are also zero.

    # Match the exact polar-axis convention already used by
    # spline_2d_straight and spline_2d_straight_df.
    if eta1 == 0.0 and cx[0, 0] == cx[0, 1]:
        d2f_out[0, 1, 1] = 0.0

    if eta1 == 0.0 and cy[0, 0] == cy[0, 1]:
        d2f_out[1, 1, 1] = 0.0


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
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.spline_2d_torus`."""

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


@stack_array(
    "b1",
    "b2",
    "d1",
    "d2",
    "dd1",
    "dd2",
    "tmp1",
    "tmp2",
)
def spline_2d_torus_d2f(
    eta1: float,
    eta2: float,
    eta3: float,
    degree: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    args: "DomainArguments",
    tor_period: float,
    d2f_out: "float[:,:,:]",
):
    r"""
    Exact Hessian of

        F_0 = R(eta1, eta2) cos(phi),
        F_1 = -R(eta1, eta2) sin(phi),
        F_2 = Z(eta1, eta2),

    where

        phi = 2*pi*eta3/tor_period.

    Output convention:

        d2f_out[i,j,k] = d²F_i/(d eta_j d eta_k).
    """
    cx = args.cx[:, :, 0]
    cy = args.cy[:, :, 0]

    # Pyccel may require array sizes to use the degree expressions
    # directly rather than local aliases p1 and p2.
    b1 = zeros(int(degree[0]) + 1, dtype=float)
    b2 = zeros(int(degree[1]) + 1, dtype=float)

    d1 = zeros(int(degree[0]) + 1, dtype=float)
    d2 = zeros(int(degree[1]) + 1, dtype=float)

    dd1 = zeros(int(degree[0]) + 1, dtype=float)
    dd2 = zeros(int(degree[1]) + 1, dtype=float)

    p1 = int(degree[0])
    p2 = int(degree[1])

    span1 = bsplines_kernels.find_span(
        args.t1,
        p1,
        eta1,
    )
    span2 = bsplines_kernels.find_span(
        args.t2,
        p2,
        eta2,
    )

    # Compute the B-spline values and their exact first and second
    # derivatives. Without these calls all arrays remain zero.
    bsplines_kernels.b_2der_splines_slim(
        args.t1,
        p1,
        eta1,
        span1,
        b1,
        d1,
        dd1,
    )

    bsplines_kernels.b_2der_splines_slim(
        args.t2,
        p2,
        eta2,
        span2,
        b2,
        d2,
        dd2,
    )

    tmp1 = ind1[span1 - p1, :]
    tmp2 = ind2[span2 - p2, :]

    # R and derivatives.
    radius = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        b1,
        b2,
        tmp1,
        tmp2,
        cx,
    )

    radius_1 = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        d1,
        b2,
        tmp1,
        tmp2,
        cx,
    )

    radius_2 = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        b1,
        d2,
        tmp1,
        tmp2,
        cx,
    )

    radius_11 = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        dd1,
        b2,
        tmp1,
        tmp2,
        cx,
    )

    radius_12 = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        d1,
        d2,
        tmp1,
        tmp2,
        cx,
    )

    radius_22 = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        b1,
        dd2,
        tmp1,
        tmp2,
        cx,
    )

    # Z derivatives.
    height_11 = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        dd1,
        b2,
        tmp1,
        tmp2,
        cy,
    )

    height_12 = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        d1,
        d2,
        tmp1,
        tmp2,
        cy,
    )

    height_22 = evaluation_kernels_2d.evaluation_kernel_2d(
        p1,
        p2,
        b1,
        dd2,
        tmp1,
        tmp2,
        cy,
    )

    # Use exactly the same toroidal-angle convention as
    # spline_2d_torus() and spline_2d_torus_df().
    wave_number = 2.0 * pi / tor_period
    angle = wave_number * eta3

    cosine = cos(angle)
    sine = sin(angle)

    d2f_out[:, :, :] = 0.0

    # ----------------------------------------------------------
    # F_0 = R cos(phi)
    # ----------------------------------------------------------
    d2f_out[0, 0, 0] = radius_11 * cosine

    d2f_out[0, 0, 1] = radius_12 * cosine
    d2f_out[0, 1, 0] = radius_12 * cosine

    d2f_out[0, 1, 1] = radius_22 * cosine

    d2f_out[0, 0, 2] = -wave_number * radius_1 * sine
    d2f_out[0, 2, 0] = -wave_number * radius_1 * sine

    d2f_out[0, 1, 2] = -wave_number * radius_2 * sine
    d2f_out[0, 2, 1] = -wave_number * radius_2 * sine

    d2f_out[0, 2, 2] = -wave_number * wave_number * radius * cosine

    # ----------------------------------------------------------
    # F_1 = -R sin(phi)
    # ----------------------------------------------------------
    d2f_out[1, 0, 0] = -radius_11 * sine

    d2f_out[1, 0, 1] = -radius_12 * sine
    d2f_out[1, 1, 0] = -radius_12 * sine

    d2f_out[1, 1, 1] = -radius_22 * sine

    d2f_out[1, 0, 2] = -wave_number * radius_1 * cosine
    d2f_out[1, 2, 0] = -wave_number * radius_1 * cosine

    d2f_out[1, 1, 2] = -wave_number * radius_2 * cosine
    d2f_out[1, 2, 1] = -wave_number * radius_2 * cosine

    d2f_out[1, 2, 2] = wave_number * wave_number * radius * sine

    # ----------------------------------------------------------
    # F_2 = Z
    # ----------------------------------------------------------
    d2f_out[2, 0, 0] = height_11

    d2f_out[2, 0, 1] = height_12
    d2f_out[2, 1, 0] = height_12

    d2f_out[2, 1, 1] = height_22


@pure
def cuboid(
    eta1: float,
    eta2: float,
    eta3: float,
    l1: float,
    r1: float,
    l2: float,
    r2: float,
    l3: float,
    r3: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= l_1 + (r_1 - l_1)\,\eta_1\,,

        F_y &= l_2 + (r_2 - l_2)\,\eta_2\,,

        F_z &= l_3 + (r_3 - l_3)\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    l1, l2, l3 : float
        Left domain boundary.

    r1, r2, r3 : float
        Right domain boundary.

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    # value =  begin + (end - begin) * eta
    f_out[0] = l1 + (r1 - l1) * eta1
    f_out[1] = l2 + (r2 - l2) * eta2
    f_out[2] = l3 + (r3 - l3) * eta3


@pure
def cuboid_df(l1: float, r1: float, l2: float, r2: float, l3: float, r3: float, df_out: "float[:,:]"):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.cuboid`."""

    df_out[0, 0] = r1 - l1
    df_out[0, 1] = 0.0
    df_out[0, 2] = 0.0
    df_out[1, 0] = 0.0
    df_out[1, 1] = r2 - l2
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = r3 - l3


@pure
def cuboid_d2f(
    d2f_out: "float[:,:,:]",
):
    """Exact Hessian of the affine Cuboid mapping."""
    d2f_out[:, :, :] = 0.0


@pure
def orthogonal(eta1: float, eta2: float, eta3: float, lx: float, ly: float, alpha: float, lz: float, f_out: "float[:]"):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= L_x\,\left[\,\eta_1 + \\alpha\sin(2\pi\,\eta_1)\,\\right]\,,

        F_y &= L_y\,\left[\,\eta_2 + \\alpha\sin(2\pi\,\eta_2)\,\\right]\,,

        F_z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    lx : float
        Length in x-direction.

    ly : float
        Length in yy-direction.

    alpha : float
        Distortion factor.

    lz : float
        Length in third direction.

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    f_out[0] = lx * (eta1 + alpha * sin(2 * pi * eta1))
    f_out[1] = ly * (eta2 + alpha * sin(2 * pi * eta2))
    f_out[2] = lz * eta3


@pure
def orthogonal_df(eta1: float, eta2: float, lx: float, ly: float, alpha: float, lz: float, df_out: "float[:,:]"):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.orthogonal`."""

    df_out[0, 0] = lx * (1 + alpha * cos(2 * pi * eta1) * 2 * pi)
    df_out[0, 1] = 0.0
    df_out[0, 2] = 0.0
    df_out[1, 0] = 0.0
    df_out[1, 1] = ly * (1 + alpha * cos(2 * pi * eta2) * 2 * pi)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz


@pure
def colella(eta1: float, eta2: float, eta3: float, lx: float, ly: float, alpha: float, lz: float, f_out: "float[:]"):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= L_x\,\left[\,\eta_1 + \\alpha\sin(2\pi\,\eta_1)\sin(2\pi\,\eta_2)\,\\right]\,,

        F_y &= L_y\,\left[\,\eta_2 + \\alpha\sin(2\pi\,\eta_2)\sin(2\pi\,\eta_1)\,\\right]\,,

        F_z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    lx : float
        Length in x-direction.

    ly : float
        Length in y-direction.

    alpha : float
        Distortion factor.

    lz : float
        Length in z-direction.

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    f_out[0] = lx * (eta1 + alpha * sin(2 * pi * eta1) * sin(2 * pi * eta2))
    f_out[1] = ly * (eta2 + alpha * sin(2 * pi * eta1) * sin(2 * pi * eta2))
    f_out[2] = lz * eta3


@pure
def colella_df(eta1: float, eta2: float, lx: float, ly: float, alpha: float, lz: float, df_out: "float[:,:]"):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.colella`."""

    df_out[0, 0] = lx * (1 + alpha * cos(2 * pi * eta1) * sin(2 * pi * eta2) * 2 * pi)
    df_out[0, 1] = lx * alpha * sin(2 * pi * eta1) * cos(2 * pi * eta2) * 2 * pi
    df_out[0, 2] = 0.0
    df_out[1, 0] = ly * alpha * cos(2 * pi * eta1) * sin(2 * pi * eta2) * 2 * pi
    df_out[1, 1] = ly * (1 + alpha * sin(2 * pi * eta1) * cos(2 * pi * eta2) * 2 * pi)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz


@pure
def hollow_cyl(eta1: float, eta2: float, eta3: float, a1: float, a2: float, lz: float, poc: float, f_out: "float[:]"):
    r"""Point-wise evaluation of

    .. math::

        F_x &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\cos(2\pi\,\eta_2 / poc)\,,

        F_y &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\sin(2\pi\,\eta_2 / poc)\,,

        F_z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    a1 : float
        Inner radius.

    a2 : float
        Outer radius.

    lz : float
        Length in third direction.

    poc : int
        periodicity in second direction.

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    da = a2 - a1

    f_out[0] = (a1 + eta1 * da) * cos(2 * pi * eta2 / poc)
    f_out[1] = (a1 + eta1 * da) * sin(2 * pi * eta2 / poc)
    f_out[2] = lz * eta3


@pure
def hollow_cyl_df(eta1: float, eta2: float, a1: float, a2: float, lz: float, poc: float, df_out: "float[:,:]"):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.hollow_cyl`."""

    da = a2 - a1

    df_out[0, 0] = da * cos(2 * pi * eta2)
    df_out[0, 1] = -2 * pi / poc * (a1 + eta1 * da) * sin(2 * pi * eta2 / poc)
    df_out[0, 2] = 0.0
    df_out[1, 0] = da * sin(2 * pi * eta2)
    df_out[1, 1] = 2 * pi / poc * (a1 + eta1 * da) * cos(2 * pi * eta2 / poc)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz


@pure
def powered_ellipse(
    eta1: float,
    eta2: float,
    eta3: float,
    rx: float,
    ry: float,
    lz: float,
    s: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::
        F_x &= r_x\,\eta_1^s\cos(2\pi\,\eta_2)\,,

        F_y &= r_y\,\eta_1^s\sin(2\pi\,\eta_2)\,,

        F_z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    rx, ry : float
        Axes lengths.

    lz : float
        Length in third direction.

    s : float
        Power of eta1

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    f_out[0] = (eta1**s) * rx * cos(2 * pi * eta2)
    f_out[1] = (eta1**s) * ry * sin(2 * pi * eta2)
    f_out[2] = eta3 * lz


@pure
def powered_ellipse_df(
    eta1: float,
    eta2: float,
    eta3: float,
    rx: float,
    ry: float,
    lz: float,
    s: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.powered_ellipse`."""

    df_out[0, 0] = (eta1 ** (s - 1)) * rx * cos(2 * pi * eta2)
    df_out[0, 1] = -2 * pi * (eta1**s) * rx * sin(2 * pi * eta2)
    df_out[0, 2] = 0.0
    df_out[1, 0] = (eta1 ** (s - 1)) * ry * sin(2 * pi * eta2)
    df_out[1, 1] = 2 * pi * (eta1**s) * ry * cos(2 * pi * eta2)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz


@pure
def hollow_torus(
    eta1: float,
    eta2: float,
    eta3: float,
    a1: float,
    a2: float,
    r0: float,
    sfl: float,
    pol_period: float,
    tor_period: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\cos(\theta(\eta_1,\eta_2))+R_0\\rbrace\cos(2\pi\,\eta_3)\,,

        F_y &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\cos(\theta(\eta_1,\eta_2))+R_0\\rbrace\sin(2\pi\,\eta_3) \,,

        F_z &= \,\,\,\left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\sin(\theta(\eta_1,\eta_2)) \,,

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    a1 : float
        Inner radius.

    a2 : float
        Outer radius.

    r0 : float
        Major radius.

    sfl : float
        Whether to use straight field line angular parametrization (yes: 1., no: 0.).

    pol_period: float
        periodicity of theta used in the mapping: theta = 2*pi * eta2 / pol_period (if not sfl)

    tor_period : int
        Toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    # straight field lines coordinates
    if sfl == 1.0:
        da = a2 - a1

        r = a1 + eta1 * da
        theta = 2 * arctan(sqrt((1 + r / r0) / (1 - r / r0)) * tan(pi * eta2))

        f_out[0] = (r * cos(theta) + r0) * cos(2 * pi * eta3 / tor_period)
        f_out[1] = (r * cos(theta) + r0) * (-1) * sin(2 * pi * eta3 / tor_period)
        f_out[2] = r * sin(theta)

    # equal angle coordinates
    else:
        da = a2 - a1

        f_out[0] = ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0) * cos(2 * pi * eta3 / tor_period)
        f_out[1] = ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0) * (-1) * sin(2 * pi * eta3 / tor_period)
        f_out[2] = (a1 + eta1 * da) * sin(2 * pi * eta2 / pol_period)


@pure
def hollow_torus_df(
    eta1: float,
    eta2: float,
    eta3: float,
    a1: float,
    a2: float,
    r0: float,
    sfl: float,
    pol_period: float,
    tor_period: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.hollow_torus`."""

    # straight field lines coordinates
    if sfl == 1.0:
        da = a2 - a1

        r = a1 + da * eta1

        eps = r / r0
        eps_p = da / r0

        tpe = tan(pi * eta2)
        tpe_p = pi / cos(pi * eta2) ** 2

        g = sqrt((1 + eps) / (1 - eps))
        g_p = 1 / (2 * g) * (eps_p * (1 - eps) + (1 + eps) * eps_p) / (1 - eps) ** 2

        theta = 2 * arctan(g * tpe)

        dtheta_deta1 = 2 / (1 + (g * tpe) ** 2) * g_p * tpe
        dtheta_deta2 = 2 / (1 + (g * tpe) ** 2) * g * tpe_p

        df_out[0, 0] = (da * cos(theta) - r * sin(theta) * dtheta_deta1) * cos(2 * pi * eta3 / tor_period)
        df_out[0, 1] = -r * sin(theta) * dtheta_deta2 * cos(2 * pi * eta3 / tor_period)
        df_out[0, 2] = -2 * pi / tor_period * (r * cos(theta) + r0) * sin(2 * pi * eta3 / tor_period)

        df_out[1, 0] = (da * cos(theta) - r * sin(theta) * dtheta_deta1) * (-1) * sin(2 * pi * eta3 / tor_period)
        df_out[1, 1] = -r * sin(theta) * dtheta_deta2 * (-1) * sin(2 * pi * eta3 / tor_period)
        df_out[1, 2] = 2 * pi / tor_period * (r * cos(theta) + r0) * (-1) * cos(2 * pi * eta3 / tor_period)

        df_out[2, 0] = da * sin(theta) + r * cos(theta) * dtheta_deta1
        df_out[2, 1] = r * cos(theta) * dtheta_deta2
        df_out[2, 2] = 0.0

    # equal angle coordinates
    else:
        da = a2 - a1

        df_out[0, 0] = da * cos(2 * pi * eta2 / pol_period) * cos(2 * pi * eta3 / tor_period)
        df_out[0, 1] = (
            -2 * pi / pol_period * (a1 + eta1 * da) * sin(2 * pi * eta2 / pol_period) * cos(2 * pi * eta3 / tor_period)
        )
        df_out[0, 2] = (
            -2
            * pi
            / tor_period
            * ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0)
            * sin(2 * pi * eta3 / tor_period)
        )
        df_out[1, 0] = da * cos(2 * pi * eta2 / pol_period) * (-1) * sin(2 * pi * eta3 / tor_period)
        df_out[1, 1] = (
            -2
            * pi
            / pol_period
            * (a1 + eta1 * da)
            * sin(2 * pi * eta2 / pol_period)
            * (-1)
            * sin(2 * pi * eta3 / tor_period)
        )
        df_out[1, 2] = (
            ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0)
            * (-1)
            * cos(2 * pi * eta3 / tor_period)
            * 2
            * pi
            / tor_period
        )
        df_out[2, 0] = da * sin(2 * pi * eta2 / pol_period)
        df_out[2, 1] = (a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) * 2 * pi / pol_period
        df_out[2, 2] = 0.0


@pure
def shafranov_shift(
    eta1: float,
    eta2: float,
    eta3: float,
    rx: float,
    ry: float,
    lz: float,
    de: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\eta_1^2)r_x\Delta\,,

        F_y &= r_y\,\eta_1\sin(2\pi\,\eta_2)\,,

        F_z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    rx, ry : float
        Axes lengths.

    lz : float
        Length in third direction.

    de : float
        Shift factor, should be in [0, 0.1].

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    f_out[0] = (eta1 * rx) * cos(2 * pi * eta2) + (1 - eta1**2) * rx * de
    f_out[1] = (eta1 * ry) * sin(2 * pi * eta2)
    f_out[2] = eta3 * lz


@pure
def shafranov_shift_df(
    eta1: float,
    eta2: float,
    eta3: float,
    rx: float,
    ry: float,
    lz: float,
    de: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.shafranov_shift`."""

    df_out[0, 0] = rx * cos(2 * pi * eta2) - 2 * eta1 * rx * de
    df_out[0, 1] = -2 * pi * (eta1 * rx) * sin(2 * pi * eta2)
    df_out[0, 2] = 0.0
    df_out[1, 0] = ry * sin(2 * pi * eta2)
    df_out[1, 1] = 2 * pi * (eta1 * ry) * cos(2 * pi * eta2)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz


@pure
def shafranov_sqrt(
    eta1: float,
    eta2: float,
    eta3: float,
    rx: float,
    ry: float,
    lz: float,
    de: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\sqrt \eta_1)r_x\Delta\,,

        F_y &= r_y\,\eta_1\sin(2\pi\,\eta_2)\,,

        F_z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    rx, ry : float
        Axes lengths.

    lz : float
        Length in third direction.

    de : float
        Shift factor, should be in [0, 0.1].

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    f_out[0] = (eta1 * rx) * cos(2 * pi * eta2) + (1 - sqrt(eta1)) * rx * de
    f_out[1] = (eta1 * ry) * sin(2 * pi * eta2)
    f_out[2] = eta3 * lz


@pure
def shafranov_sqrt_df(
    eta1: float,
    eta2: float,
    eta3: float,
    rx: float,
    ry: float,
    lz: float,
    de: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.shafranov_sqrt`."""

    df_out[0, 0] = rx * cos(2 * pi * eta2) - 0.5 / sqrt(eta1) * rx * de
    df_out[0, 1] = -2 * pi * (eta1 * rx) * sin(2 * pi * eta2)
    df_out[0, 2] = 0.0
    df_out[1, 0] = ry * sin(2 * pi * eta2)
    df_out[1, 1] = 2 * pi * (eta1 * ry) * cos(2 * pi * eta2)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz


@pure
def shafranov_dshaped(
    eta1: float,
    eta2: float,
    eta3: float,
    r0: float,
    lz: float,
    dx: float,
    dy: float,
    dg: float,
    eg: float,
    kg: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::

        x &= R_0\left[1 + (1 - \eta_1^2)\Delta_x + \eta_1\epsilon\cos(2\pi\,\eta_2 + \\arcsin(\delta)\eta_1\sin(2\pi\,\eta_2)) \\right]\,,

        y &= R_0\left[    (1 - \eta_1^2)\Delta_y + \eta_1\epsilon\kappa\sin(2\pi\,\eta_2)\\right]\,,

        z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    r0 : float
        Base radius.

    lz : float
        Length in third direction.

    dx : float
        Shafranov shift in x-direction.

    dy : float
        Shafranov shift in y-direction.

    dg : float
        Delta = sin(alpha): Triangularity, shift of high point.

    eg : float
        Epsilon: Inverse aspect ratio a/r0.

    kg : float
        Kappa: Ellipticity (elongation).

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    f_out[0] = r0 * (1 + (1 - eta1**2) * dx + eg * eta1 * cos(2 * pi * eta2 + arcsin(dg) * eta1 * sin(2 * pi * eta2)))
    f_out[1] = r0 * ((1 - eta1**2) * dy + eg * kg * eta1 * sin(2 * pi * eta2))
    f_out[2] = eta3 * lz


@pure
def shafranov_dshaped_df(
    eta1: float,
    eta2: float,
    eta3: float,
    r0: float,
    lz: float,
    dx: float,
    dy: float,
    dg: float,
    eg: float,
    kg: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.shafranov_dshaped`."""

    df_out[0, 0] = r0 * (
        -2 * dx * eta1
        - eg * eta1 * sin(2 * pi * eta2) * arcsin(dg) * sin(eta1 * sin(2 * pi * eta2) * arcsin(dg) + 2 * pi * eta2)
        + eg * cos(eta1 * sin(2 * pi * eta2) * arcsin(dg) + 2 * pi * eta2)
    )
    df_out[0, 1] = (
        -r0
        * eg
        * eta1
        * (2 * pi * eta1 * cos(2 * pi * eta2) * arcsin(dg) + 2 * pi)
        * sin(eta1 * sin(2 * pi * eta2) * arcsin(dg) + 2 * pi * eta2)
    )
    df_out[0, 2] = 0.0
    df_out[1, 0] = r0 * (-2 * dy * eta1 + eg * kg * sin(2 * pi * eta2))
    df_out[1, 1] = 2 * pi * r0 * eg * eta1 * kg * cos(2 * pi * eta2)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz
