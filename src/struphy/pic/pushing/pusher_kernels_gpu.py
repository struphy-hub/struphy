# "Pusher kernels for full orbit (6D) particles."

# from pyccel.stdlib.internal.openmp import omp_set_num_threads, omp_get_num_threads, omp_get_thread_num

from numpy import cos, empty, floor, log, shape, sin, sqrt, zeros, copy
from pyccel.decorators import pure, stack_array
from pyccel.decorators import inline

# import struphy.bsplines.bsplines_kernels as bsplines_kernels
# import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
# import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels

# do not remove; needed to identify dependencies
# import struphy.pic.pushing.pusher_args_kernels as pusher_args_kernels
# import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
# import struphy.pic.sph_eval_kernels as sph_eval_kernels
# from struphy.bsplines.evaluation_kernels_3d import (
#     # eval_0form_spline_mpi,
#     # eval_1form_spline_mpi,
#     # eval_2form_spline_mpi,
#     # eval_3form_spline_mpi,
#     # eval_vectorfield_spline_mpi,
#     get_spans,
# )

# import struphy.geometry.mappings_kernels as mappings_kernels
# import struphy.linear_algebra.linalg_kernels as linalg_kernels

# # do not remove; needed to identify dependencies
import struphy.pic.pushing.pusher_args_kernels as pusher_args_kernels
from struphy.pic.pushing.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments
# # from struphy.linear_algebra.linalg_kernels import matrix_inv, matrix_vector
# # from struphy.geometry.evaluation_kernels import df

# def _tmp_floor_division_pusher_kernels(x: int):
#     y = zeros(10)
#     z = copy(y)
#     return x // 2



def matmul_cpu(A: 'float[:,:]', B: 'float[:,:]', C: 'float[:,:]'):
    N: int = shape(A)[0]
    s: float = 0.0
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += A[i, k] * B[k, j]
            C[i, j] = s

def matmul_gpu(A: 'float[:,:]', B: 'float[:,:]', C: 'float[:,:]'):
    N: int = shape(A)[0]
    s: float = 0.0
    #$ omp target teams distribute parallel for collapse(2)
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += A[i, k] * B[k, j]
            C[i, j] = s




# @stack_array("dfm", "dfinv", "v", "k")
def push_eta_stage_gpu(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)

    # marker position e and velocity v
    v = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0
    
    args_domain_params = args_domain.params

    #$ omp target teams distribute parallel for
    for ip in range(n_markers):
        if markers[ip, first_init_idx] == -1.0 or markers[ip, -1] == -2.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]
        # # evaluate Jacobian, result in dfm
        # evaluation_kernels.df(

        # ------------ START DF ------------ #
        df_pusher_inline_nodomainargs(
            e1,
            e2,
            e3,
            args_domain_params,
            dfm,
        )
        # ------------ END DF ------------ #

        # evaluate inverse Jacobian matrix
        matrix_inv_inline(dfm, dfinv)
        # linalg_kernels.matrix_inv_inline(dfm, dfinv)

        # pull-back of velocity
        matrix_vector_inline(dfinv, v, k)
        # linalg_kernels.matrix_vector_inline(dfinv, v, k)

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * a[stage] * k
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

# @stack_array("dfm", "b_form", "b_cart", "b_norm", "v", "vperp", "vxb_norm", "b_normxvperp")
def push_vxb_analytic_gpu(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
):
    r"""Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form components, Cartesian components and normalized Cartesian components)
    b_form = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_norm = empty(3, dtype=float)

    # particle velocity
    v = empty(3, dtype=float)

    # perpendicular velocity, v x b_norm and b_norm x vperp
    vperp = empty(3, dtype=float)
    vxb_norm = empty(3, dtype=float)
    b_normxvperp = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    args_domain_params = args_domain.params
    
    # -- removed omp: #$ omp parallel private (ip, e1, e2, e3, v, dfm, det_df, span1, span2, span3, b_form, b_cart, b_abs, b_norm, vpar, vxb_norm, vperp, b_normxvperp)
    # -- removed omp: #$ omp for

    #$ omp target teams distribute parallel for
    for ip in range(n_markers):
        # check if marker is a hole
        if markers[ip, first_init_idx] == -1.0 or markers[ip, -1] == -2.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        # evaluation_kernels.df(
        #     e1,
        #     e2,
        #     e3,
        #     args_domain,
        #     dfm,
        # )
        df_pusher_inline_nodomainargs(
            e1,
            e2,
            e3,
            args_domain_params,
            dfm,
        )

        # # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # # spline evaluation
        span1, span2, span3 = get_spans_inline(e1, e2, e3, args_derham)

        # magnetic field 2-form
        eval_2form_spline_mpi_inline(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        # matrix_vector_inline(dfm, b_form, b_cart)
        # Temporary start (replacement of linalg_kernels.matrix_vector)
        # b_cart[:] = 0.
        # for i in range(3):
        #     for j in range(3):
        #         b_cart[i] += dfm[i, j] * b_form[j]
        # Temporary end

        b_cart[:] = b_cart / det_df

        # magnetic field: magnitude
        b_abs = sqrt(b_cart[0] ** 2 + b_cart[1] ** 2 + b_cart[2] ** 2)

        # only push vxb if magnetic field is non-zero
        if b_abs != 0.0:
            # normalized magnetic field direction
            b_norm[:] = b_cart / b_abs

            # parallel velocity v.b_norm
            vpar = linalg_kernels.scalar_dot(v, b_norm)

            # first component of perpendicular velocity
            linalg_kernels.cross(v, b_norm, vxb_norm)
            linalg_kernels.cross(b_norm, vxb_norm, vperp)

            # second component of perpendicular velocity
            linalg_kernels.cross(b_norm, vperp, b_normxvperp)

            # analytic rotation
            markers[ip, 3:6] = vpar * b_norm + cos(b_abs * dt) * vperp - sin(b_abs * dt) * b_normxvperp

    # -- removed omp: #$ omp end parallel



@pure
@inline
def matrix_vector_inline(a: 'float[:,:]', b: 'float[:]', c: 'float[:]'):
    """
    Performs the matrix-vector product of a 3x3 matrix with a vector.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3).

        b : array[float]
            The input array (vector) of shape (3,).

        c : array[float]
            The output array (vector) of shape (3,) which is the result of the matrix-vector product a.dot(b).
    """

    c[:] = 0.

    for i in range(3):
        for j in range(3):
            c[i] += a[i, j] * b[j]


# # @stack_array('det_a')
@inline
def matrix_inv_inline(a: 'float[:,:]', b: 'float[:,:]'):
    """
    Computes the inverse of a 3x3 matrix.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3).

        b : array[float]
            The output array (matrix) of shape (3,3).
    """

    #det_a = det(a)
    plus = a[0, 0]*a[1, 1]*a[2, 2] + a[0, 1] * \
        a[1, 2]*a[2, 0] + a[0, 2]*a[1, 0]*a[2, 1]
    minus = a[2, 0]*a[1, 1]*a[0, 2] + a[2, 1] * \
        a[1, 2]*a[0, 0] + a[2, 2]*a[1, 0]*a[0, 1]

    det_a = plus - minus

    

    b[0, 0] = (a[1, 1]*a[2, 2] - a[2, 1]*a[1, 2]) / det_a
    b[0, 1] = (a[2, 1]*a[0, 2] - a[0, 1]*a[2, 2]) / det_a
    b[0, 2] = (a[0, 1]*a[1, 2] - a[1, 1]*a[0, 2]) / det_a

    b[1, 0] = (a[1, 2]*a[2, 0] - a[2, 2]*a[1, 0]) / det_a
    b[1, 1] = (a[2, 2]*a[0, 0] - a[0, 2]*a[2, 0]) / det_a
    b[1, 2] = (a[0, 2]*a[1, 0] - a[1, 2]*a[0, 0]) / det_a

    b[2, 0] = (a[1, 0]*a[2, 1] - a[2, 0]*a[1, 1]) / det_a
    b[2, 1] = (a[2, 0]*a[0, 1] - a[0, 0]*a[2, 1]) / det_a
    b[2, 2] = (a[0, 0]*a[1, 1] - a[1, 0]*a[0, 1]) / det_a

@inline
def cuboid_df_inline(l1: float, r1: float, l2: float, r2: float, l3: float, r3: float, df_out: "float[:,:]"):
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


@inline
def df_pusher_inline_nodomainargs(
    eta1: float,
    eta2: float,
    eta3: float,
    args_params: "float[:]",
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
    
    cuboid_df_inline(
        args_params[0],
        args_params[1],
        args_params[2],
        args_params[3],
        args_params[4],
        args_params[5],
        df_out,
    )

@inline
def get_spans_inline(eta1: float, eta2: float, eta3: float, args_derham: "DerhamArguments"):
    """Compute the knot span index,
    the N-spline values (in bn) and the D-spline values (in bd)
    at (eta1, eta2, eta3)."""

    # find spans
    span1 = find_span_inline(args_derham.tn1, args_derham.pn[0], eta1)
    span2 = find_span_inline(args_derham.tn2, args_derham.pn[1], eta2)
    span3 = find_span_inline(args_derham.tn3, args_derham.pn[2], eta3)

    # get spline values at eta
    b_d_splines_slim_inline(
        args_derham.tn1, args_derham.pn[0], eta1, int(span1), args_derham.bn1, args_derham.bd1
    )
    b_d_splines_slim_inline(
        args_derham.tn2, args_derham.pn[1], eta2, int(span2), args_derham.bn2, args_derham.bd2
    )
    b_d_splines_slim_inline(
        args_derham.tn3, args_derham.pn[2], eta3, int(span3), args_derham.bn3, args_derham.bd3
    )

    return span1, span2, span3



@pure
@inline
def find_span_inline(t: 'float[:]', p: 'int', eta: 'float') -> 'int':
    """
    Computes the knot span index i for which the B-splines i-p until i are non-vanishing at point eta.

    Parameters:
    -----------
        t : array
            knot sequence

        p : integer
            degree of the basis splines

        eta : float
            Evaluation point

    Returns:
    --------
        span-index 
    """

    # Knot index at left/right boundary
    low = p
    high = len(t) - 1 - p

    # Check if point is exactly on left/right boundary, or outside domain
    if eta <= t[low]:
        returnVal = low
    elif eta >= t[high]:
        returnVal = high - 1
    else:
        # Perform binary search
        span = (low + high)//2

        while eta < t[span] or eta >= t[span + 1]:

            if eta < t[span]:
                high = span
            else:
                low = span
            span = (low + high)//2

        returnVal = span

    return returnVal

# @inline
@pure
def b_d_splines_slim_inline(tn: 'float[:]', pn: 'int', eta: 'float', span: 'int', bn: 'float[:]', bd: 'float[:]'):
    """
    One function to compute the values of non-vanishing B-splines and D-splines.

    Parameters
    ---------- 
        tn : array
            Knot sequence of B-splines.

        pn : int
            Polynomial degree of B-splines.

        span : integer
            Knot span index i -> [i-p,i] basis functions are non-vanishing.

        eta : float
            Evaluation point.

        bn : array[float]
            Output: pn + 1 non-vanishing B-splines at eta

        bd : array[float]
            Output: pn non-vanishing D-splines at eta
    """

    # compute D-spline degree
    pd = pn - 1

    # make sure the arrays we are writing to are empty
    bn[:] = 0.
    bd[:] = 0.

    # Initialize variables left and right used for computing the value
    left = zeros(pn, dtype=float)
    right = zeros(pn, dtype=float)

    bn[0] = 1.

    for j in range(pn):
        left[j] = eta - tn[span - j]
        right[j] = tn[span + 1 + j] - eta
        saved = 0.

        if j == pn-1:
            # compute D-splines values by scaling B-splines of degree pn-1
            for il in range(pd + 1):
                bd[pd - il] = pn/(
                    tn[span - il + pn] -
                    tn[span - il]
                ) * bn[pd - il]

        for r in range(j + 1):
            temp = bn[r]/(right[r] + left[j - r])
            bn[r] = saved + right[r] * temp
            saved = left[j - r] * temp

        bn[j + 1] = saved


def eval_2form_spline_mpi_inline(
    span1: int,
    span2: int,
    span3: int,
    args_derham: "DerhamArguments",
    form_coeffs_1: "float[:,:,:]",
    form_coeffs_2: "float[:,:,:]",
    form_coeffs_3: "float[:,:,:]",
    out: "float[:]",
):
    """Single-point evaluation of Derham 2-form spline defined by form_coeffs,
    given N-spline values (in bn), D-spline values (in bd)
    and knot span indices span."""

    out[0] = eval_spline_mpi_kernel_inline(
        args_derham.pn[0],
        args_derham.pn[1] - 1,
        args_derham.pn[2] - 1,
        args_derham.bn1,
        args_derham.bd2,
        args_derham.bd3,
        span1,
        span2,
        span3,
        form_coeffs_1,
        args_derham.starts,
    )

    out[1] = eval_spline_mpi_kernel_inline(
        args_derham.pn[0] - 1,
        args_derham.pn[1],
        args_derham.pn[2] - 1,
        args_derham.bd1,
        args_derham.bn2,
        args_derham.bd3,
        span1,
        span2,
        span3,
        form_coeffs_2,
        args_derham.starts,
    )

    out[2] = eval_spline_mpi_kernel_inline(
        args_derham.pn[0] - 1,
        args_derham.pn[1] - 1,
        args_derham.pn[2],
        args_derham.bd1,
        args_derham.bd2,
        args_derham.bn3,
        span1,
        span2,
        span3,
        form_coeffs_3,
        args_derham.starts,
    )


def eval_spline_mpi_kernel_inline(
    p1: "int",
    p2: "int",
    p3: "int",
    basis1: "float[:]",
    basis2: "float[:]",
    basis3: "float[:]",
    span1: "int",
    span2: "int",
    span3: "int",
    _data: "float[:,:,:]",
    starts: "int[:]",
) -> float:
    """
    Summing non-zero contributions of a spline function with distributed memory (domain decomposition).

    Parameters
    ----------
        p1, p2, p3 : int
            Degrees of the univariate splines in each direction.

        basis1, basis2, basis3 : array[float]
            The p + 1 values of non-zero basis splines at one point (eta1, eta2, eta3) in each direction.

        span1, span2, span3: int
            Knot span index in each direction.

        _data : array[float]
            The spline coefficients c_ijk of the current process, ie. the _data attribute of a StencilVector.

        starts : array[int]
            Starting indices of current process.

    Returns
    -------
        spline_value : float
            Value of tensor-product spline at point (eta1, eta2, eta3).
    """

    spline_value = 0.0

    for il1 in range(p1 + 1):
        i1 = span1 + il1 - starts[0]
        for il2 in range(p2 + 1):
            i2 = span2 + il2 - starts[1]
            for il3 in range(p3 + 1):
                i3 = span3 + il3 - starts[2]

                spline_value += _data[i1, i2, i3] * basis1[il1] * basis2[il2] * basis3[il3]

    return spline_value
