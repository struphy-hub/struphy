"""
Module containing accelerated (pyccelized) functions for evaluating metric coefficients
corresponding to mappings (x, y, z) = F(eta_1, eta_2, eta_3).
"""

from numpy import empty, shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.mappings_kernels as mappings_kernels

# do not remove; needed to identify dependencies
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments


def f(
    eta1: float,
    eta2: float,
    eta3: float,
    args: "DomainArguments",
    f_out: "float[:]",
):
    """Point-wise evaluation of (x, y, z) = F(eta1, eta2, eta3).

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Position on the unit cube.

    args: DomainArguments
        Arguments for the mapping.

    f_out : np.array
        Output array of shape (3,).
    """

    if args.kind_map == 0:
        mappings_kernels.spline_3d(
            eta1,
            eta2,
            eta3,
            args.p,
            args.ind1,
            args.ind2,
            args.ind3,
            args,
            f_out,
        )
    elif args.kind_map == 1:
        mappings_kernels.spline_2d_straight(
            eta1,
            eta2,
            eta3,
            args.p,
            args.ind1,
            args.ind2,
            args,
            args.params[0],
            f_out,
        )
    elif args.kind_map == 2:
        mappings_kernels.spline_2d_torus(
            eta1,
            eta2,
            eta3,
            args.p,
            args.ind1,
            args.ind2,
            args,
            args.params[0],
            f_out,
        )
    elif args.kind_map == 10:
        mappings_kernels.cuboid(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            args.params[4],
            args.params[5],
            f_out,
        )
    elif args.kind_map == 11:
        mappings_kernels.orthogonal(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            f_out,
        )
    elif args.kind_map == 12:
        mappings_kernels.colella(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            f_out,
        )
    elif args.kind_map == 20:
        mappings_kernels.hollow_cyl(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            f_out,
        )
    elif args.kind_map == 21:
        mappings_kernels.powered_ellipse(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            f_out,
        )
    elif args.kind_map == 22:
        mappings_kernels.hollow_torus(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            args.params[4],
            args.params[5],
            f_out,
        )
    elif args.kind_map == 30:
        mappings_kernels.shafranov_shift(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            f_out,
        )
    elif args.kind_map == 31:
        mappings_kernels.shafranov_sqrt(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            f_out,
        )
    elif args.kind_map == 32:
        mappings_kernels.shafranov_dshaped(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            args.params[4],
            args.params[5],
            args.params[6],
            f_out,
        )


def df(
    eta1: float,
    eta2: float,
    eta3: float,
    args: "DomainArguments",
    df_out: "float[:,:]",
):
    """Point-wise evaluation of the Jacobian matrix DF = (dF_i/deta_j)_(i,j=1,2,3).

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Position on the unit cube.

    args: DomainArguments
        Arguments for the mapping.

    df_out : np.array
        Output array of shape (3, 3).
    """

    if args.kind_map == 0:
        mappings_kernels.spline_3d_df(
            eta1,
            eta2,
            eta3,
            args.p,
            args.ind1,
            args.ind2,
            args.ind3,
            args,
            df_out,
        )
    elif args.kind_map == 1:
        mappings_kernels.spline_2d_straight_df(
            eta1,
            eta2,
            args.p,
            args.ind1,
            args.ind2,
            args,
            args.params[0],
            df_out,
        )
    elif args.kind_map == 2:
        mappings_kernels.spline_2d_torus_df(
            eta1,
            eta2,
            eta3,
            args.p,
            args.ind1,
            args.ind2,
            args,
            args.params[0],
            df_out,
        )
    elif args.kind_map == 10:
        mappings_kernels.cuboid_df(
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            args.params[4],
            args.params[5],
            df_out,
        )
    elif args.kind_map == 11:
        mappings_kernels.orthogonal_df(
            eta1,
            eta2,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            df_out,
        )
    elif args.kind_map == 12:
        mappings_kernels.colella_df(
            eta1,
            eta2,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            df_out,
        )
    elif args.kind_map == 20:
        mappings_kernels.hollow_cyl_df(
            eta1,
            eta2,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            df_out,
        )
    elif args.kind_map == 21:
        mappings_kernels.powered_ellipse_df(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            df_out,
        )
    elif args.kind_map == 22:
        mappings_kernels.hollow_torus_df(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            args.params[4],
            args.params[5],
            df_out,
        )
    elif args.kind_map == 30:
        mappings_kernels.shafranov_shift_df(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            df_out,
        )
    elif args.kind_map == 31:
        mappings_kernels.shafranov_sqrt_df(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            df_out,
        )
    elif args.kind_map == 32:
        mappings_kernels.shafranov_dshaped_df(
            eta1,
            eta2,
            eta3,
            args.params[0],
            args.params[1],
            args.params[2],
            args.params[3],
            args.params[4],
            args.params[5],
            args.params[6],
            df_out,
        )


def det_df(
    eta1: float,
    eta2: float,
    eta3: float,
    args: "DomainArguments",
    tmp1: "float[:,:]",
) -> float:
    """Point-wise evaluation of the Jacobian determinant det(dF) = dF/deta1.dot(dF/deta2 x dF/deta3).

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Position on the unit cube.

    args: DomainArguments
        Arguments for the mapping.

    tmp1 : np.array
        Temporary array of shape (3, 3).
    """

    df(
        eta1,
        eta2,
        eta3,
        args,
        tmp1,
    )
    detdf = linalg_kernels.det(tmp1)
    return detdf


def df_inv(
    eta1: float,
    eta2: float,
    eta3: float,
    args: "DomainArguments",
    tmp1: "float[:,:]",
    avoid_round_off: bool,
    dfinv_out: "float[:,:]",
):
    """Point-wise evaluation of the inverse Jacobian matrix DF^(-1).

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Position on the unit cube.

    args: DomainArguments
        Arguments for the mapping.

    tmp1: np.array
        Temporary array of shape (3, 3).

    avoid_round_off: bool
        Whether to manually set exact zeros in arrays.

    dfinv_out: np.array
        Output array of shape (3, 3).
    """

    df(
        eta1,
        eta2,
        eta3,
        args,
        tmp1,
    )
    linalg_kernels.matrix_inv(tmp1, dfinv_out)

    # set known (analytical) zero components manually to zero to avoid round-off error remainders!
    if avoid_round_off:
        if args.kind_map == 1:
            dfinv_out[0, 2] = 0.0
            dfinv_out[1, 2] = 0.0
            dfinv_out[2, 0] = 0.0
            dfinv_out[2, 1] = 0.0
        elif args.kind_map == 2:
            dfinv_out[2, 2] = 0
        elif args.kind_map == 10:
            dfinv_out[0, 1] = 0.0
            dfinv_out[0, 2] = 0.0
            dfinv_out[1, 0] = 0.0
            dfinv_out[1, 2] = 0.0
            dfinv_out[2, 0] = 0.0
            dfinv_out[2, 1] = 0.0
        elif args.kind_map == 11:
            dfinv_out[0, 1] = 0.0
            dfinv_out[0, 2] = 0.0
            dfinv_out[1, 0] = 0.0
            dfinv_out[1, 2] = 0.0
            dfinv_out[2, 0] = 0.0
            dfinv_out[2, 1] = 0.0
        elif args.kind_map == 12:
            dfinv_out[0, 2] = 0.0
            dfinv_out[1, 2] = 0.0
            dfinv_out[2, 0] = 0.0
            dfinv_out[2, 1] = 0.0
        elif args.kind_map == 20:
            dfinv_out[0, 2] = 0.0
            dfinv_out[1, 2] = 0.0
            dfinv_out[2, 0] = 0.0
            dfinv_out[2, 1] = 0.0
        elif args.kind_map == 21:
            dfinv_out[0, 2] = 0.0
            dfinv_out[1, 2] = 0.0
            dfinv_out[2, 0] = 0.0
            dfinv_out[2, 1] = 0.0
        elif args.kind_map == 22:
            dfinv_out[2, 2] = 0.0
        elif args.kind_map == 30:
            dfinv_out[0, 2] = 0.0
            dfinv_out[1, 2] = 0.0
            dfinv_out[2, 0] = 0.0
            dfinv_out[2, 1] = 0.0
        elif args.kind_map == 31:
            dfinv_out[0, 2] = 0.0
            dfinv_out[1, 2] = 0.0
            dfinv_out[2, 0] = 0.0
            dfinv_out[2, 1] = 0.0
        elif args.kind_map == 32:
            dfinv_out[0, 2] = 0.0
            dfinv_out[1, 2] = 0.0
            dfinv_out[2, 0] = 0.0
            dfinv_out[2, 1] = 0.0


def g(
    eta1: float,
    eta2: float,
    eta3: float,
    args: "DomainArguments",
    tmp1: "float[:,:]",
    tmp2: "float[:,:]",
    avoid_round_off: bool,
    g_out: "float[:,:]",
):
    """Point-wise evaluation of the metric tensor G = DF^T * DF.

    Parameters
    ----------
    eta1, eta2, eta3: float
        Position on the unit cube.

    args: DomainArguments
        Arguments for the mapping.

    tmp1, tmp2: np.array
        Temporary arrays of shape (3, 3).

    avoid_round_off: bool
        Whether to manually set exact zeros in arrays.

    g_out: np.array
        Output array of shape (3, 3).
    """
    df(
        eta1,
        eta2,
        eta3,
        args,
        tmp1,
    )
    linalg_kernels.transpose(tmp1, tmp2)
    linalg_kernels.matrix_matrix(tmp2, tmp1, g_out)

    # set known (analytical) zero components manually to zero to avoid round-off error remainders!
    if avoid_round_off:
        if args.kind_map == 1:
            g_out[0, 2] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0
        elif args.kind_map == 2:
            g_out[0, 2] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0
        elif args.kind_map == 10:
            g_out[0, 1] = 0.0
            g_out[0, 2] = 0.0
            g_out[1, 0] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0
        elif args.kind_map == 11:
            g_out[0, 1] = 0.0
            g_out[0, 2] = 0.0
            g_out[1, 0] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0
        elif args.kind_map == 12:
            g_out[0, 2] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0
        elif args.kind_map == 20:
            g_out[0, 1] = 0.0
            g_out[0, 2] = 0.0
            g_out[1, 0] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0
        elif args.kind_map == 21:
            g_out[0, 2] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0
        elif args.kind_map == 22:
            # straight field line coordinates
            if args.params[3] == 1.0:
                g_out[0, 2] = 0.0
                g_out[1, 2] = 0.0
                g_out[2, 0] = 0.0
                g_out[2, 1] = 0.0

            # equal angle coordinates
            else:
                g_out[0, 1] = 0.0
                g_out[0, 2] = 0.0
                g_out[1, 0] = 0.0
                g_out[1, 2] = 0.0
                g_out[2, 0] = 0.0
                g_out[2, 1] = 0.0

        elif args.kind_map == 30:
            g_out[0, 2] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0
        elif args.kind_map == 31:
            g_out[0, 2] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0
        elif args.kind_map == 32:
            g_out[0, 2] = 0.0
            g_out[1, 2] = 0.0
            g_out[2, 0] = 0.0
            g_out[2, 1] = 0.0


def g_inv(
    eta1: float,
    eta2: float,
    eta3: float,
    args: "DomainArguments",
    tmp1: "float[:,:]",
    tmp2: "float[:,:]",
    tmp3: "float[:,:]",
    avoid_round_off: bool,
    ginv_out: "float[:,:]",
):
    """Point-wise evaluation of the inverse metric tensor G^(-1) = DF^(-1) * DF^(-T).

    Parameters
    ----------
    eta1, eta2, eta3: float
        Position on the unit cube.

    args: DomainArguments
        Arguments for the mapping.

    tmp1, tmp2, tmp3: np.array
        Temporary arrays of shape (3, 3).

    avoid_round_off: bool
        Whether to manually set exact zeros in arrays.

    ginv_out: np.array
        Output array of shape (3, 3).
    """
    g(
        eta1,
        eta2,
        eta3,
        args,
        tmp1,
        tmp2,
        avoid_round_off,
        tmp3,
    )
    linalg_kernels.matrix_inv(tmp3, ginv_out)

    # set known (analytical) zero components manually to zero to avoid round-off error remainders!
    if avoid_round_off:
        if args.kind_map == 1:
            ginv_out[0, 2] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0
        elif args.kind_map == 2:
            ginv_out[0, 2] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0
        elif args.kind_map == 10:
            ginv_out[0, 1] = 0.0
            ginv_out[0, 2] = 0.0
            ginv_out[1, 0] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0
        elif args.kind_map == 11:
            ginv_out[0, 1] = 0.0
            ginv_out[0, 2] = 0.0
            ginv_out[1, 0] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0
        elif args.kind_map == 12:
            ginv_out[0, 2] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0
        elif args.kind_map == 20:
            ginv_out[0, 1] = 0.0
            ginv_out[0, 2] = 0.0
            ginv_out[1, 0] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0
        elif args.kind_map == 21:
            ginv_out[0, 2] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0
        elif args.kind_map == 22:
            # straight field line coordinates
            if args.params[3] == 1.0:
                ginv_out[0, 2] = 0.0
                ginv_out[1, 2] = 0.0
                ginv_out[2, 0] = 0.0
                ginv_out[2, 1] = 0.0

            # equal angle coordinates
            else:
                ginv_out[0, 1] = 0.0
                ginv_out[0, 2] = 0.0
                ginv_out[1, 0] = 0.0
                ginv_out[1, 2] = 0.0
                ginv_out[2, 0] = 0.0
                ginv_out[2, 1] = 0.0

        elif args.kind_map == 30:
            ginv_out[0, 2] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0
        elif args.kind_map == 31:
            ginv_out[0, 2] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0
        elif args.kind_map == 32:
            ginv_out[0, 2] = 0.0
            ginv_out[1, 2] = 0.0
            ginv_out[2, 0] = 0.0
            ginv_out[2, 1] = 0.0


def select_metric_coeff(
    eta1: float,
    eta2: float,
    eta3: float,
    kind_coeff: int,
    args: "DomainArguments",
    tmp0: "float[:]",
    tmp1: "float[:,:]",
    tmp2: "float[:,:]",
    tmp3: "float[:,:]",
    avoid_round_off: bool,
    out: "float[:,:]",
):
    """Point-wise evaluation of metric coefficients.

    Parameters
    ----------
    eta1, eta2, eta3: float
        Position on the unit cube.

    kind_coeff : int
        Which metric coefficient to evaluate.
            * -1 : identity
            *  0 : mapping
            *  1 : Jacobian matrix
            *  2 : Jacobian determinant
            *  3 : inverse Jacobian matrix
            *  4 : metric tensor
            *  5 : inverse metric tensor

    args: DomainArguments
        Arguments for the mapping.

    tmp0: np.array
        Temporary array of shape (3,).

    tmp1, tmp2, tmp3: np.array
        Temporary arrays of shape (3, 3).

    avoid_round_off: bool
        Whether to manually set exact zeros in arrays.

    out: np.array
        Output array of shape (3, 3).
    """
    # identity map
    if kind_coeff == -1:
        out[0, 0] = eta1
        out[1, 0] = eta2
        out[2, 0] = eta3

    # mapping F
    elif kind_coeff == 0:
        f(
            eta1,
            eta2,
            eta3,
            args,
            tmp0,
        )
        out[:, 0] = tmp0

    # Jacobian matrix DF
    elif kind_coeff == 1:
        df(
            eta1,
            eta2,
            eta3,
            args,
            out,
        )

    # Jacobian determinant det(dF)
    elif kind_coeff == 2:
        out[0, 0] = det_df(
            eta1,
            eta2,
            eta3,
            args,
            tmp1,
        )

    # inverse Jacobian matrix DF^(-1)
    elif kind_coeff == 3:
        df_inv(
            eta1,
            eta2,
            eta3,
            args,
            tmp1,
            avoid_round_off,
            out,
        )

    # metric tensor G = DF^T * DF
    elif kind_coeff == 4:
        g(
            eta1,
            eta2,
            eta3,
            args,
            tmp1,
            tmp2,
            avoid_round_off,
            out,
        )

    # inverse metric tensor G^(-1) = DF^(-1) * DF^(-T)
    elif kind_coeff == 5:
        g_inv(
            eta1,
            eta2,
            eta3,
            args,
            tmp1,
            tmp2,
            tmp3,
            avoid_round_off,
            out,
        )


@stack_array("tmp0", "tmp1", "tmp2", "tmp3", "out")
def kernel_evaluate(
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    kind_coeff: int,
    args: "DomainArguments",
    mat_f: "float[:,:,:,:,:]",
    is_sparse_meshgrid: bool,
    avoid_round_off: bool,
):
    """
    Evaluation of metric coefficients on a given 3d grid of evaluation points.

    Parameters
    ----------
    is_sparse_meshgrid : bool
        Whether the 3d evaluation points were obtained from a sparse meshgrid.
    """
    tmp0 = zeros(3, dtype=float)
    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros((3, 3), dtype=float)
    tmp3 = zeros((3, 3), dtype=float)
    out = zeros((3, 3), dtype=float)

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    if is_sparse_meshgrid:
        sparse_factor = 0
    else:
        sparse_factor = 1

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                e1 = eta1[i1, i2 * sparse_factor, i3 * sparse_factor]
                e2 = eta2[i1 * sparse_factor, i2, i3 * sparse_factor]
                e3 = eta3[i1 * sparse_factor, i2 * sparse_factor, i3]

                out[:] = mat_f[i1, i2, i3, :, :]

                select_metric_coeff(
                    e1,
                    e2,
                    e3,
                    kind_coeff,
                    args,
                    tmp0,
                    tmp1,
                    tmp2,
                    tmp3,
                    avoid_round_off,
                    out,
                )

                mat_f[i1, i2, i3, :, :] = out


@stack_array("tmp0", "tmp1", "tmp2", "tmp3", "out")
def kernel_evaluate_pic(
    markers: "float[:,:]",
    kind_coeff: int,
    args: "DomainArguments",
    mat_f: "float[:,:,:]",
    remove_outside: bool,
    avoid_round_off: bool,
) -> int:
    """
    Evaluation of metric coefficients for given markers.

    Parameters
    ----------
    remove_outside : bool
        Whether to remove values that originate from markers outside of [0, 1]^d.

    Returns
    -------
    counter : int
        How many markers have been treated (not been skipped).
    """
    tmp0 = zeros(3, dtype=float)
    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros((3, 3), dtype=float)
    tmp3 = zeros((3, 3), dtype=float)
    out = zeros((3, 3), dtype=float)

    np = shape(markers)[0]
    counter = 0

    for i in range(np):
        e1 = markers[i, 0]
        e2 = markers[i, 1]
        e3 = markers[i, 2]

        if e1 < 0.0 or e1 > 1.0 or e2 < 0.0 or e2 > 1.0 or e3 < 0.0 or e3 > 1.0:
            if remove_outside:
                continue
            else:
                if kind_coeff >= 0:
                    mat_f[counter, :, :] = -1.0
                else:
                    mat_f[counter, 0, 0] = e1
                    mat_f[counter, 1, 0] = e2
                    mat_f[counter, 2, 0] = e3
                counter += 1
        else:
            out[:] = mat_f[counter, :, :]

            select_metric_coeff(
                e1,
                e2,
                e3,
                kind_coeff,
                args,
                tmp0,
                tmp1,
                tmp2,
                tmp3,
                avoid_round_off,
                out,
            )

            mat_f[counter, :, :] = out

            counter += 1

    return counter
