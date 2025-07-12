# "Pusher kernels for full orbit (6D) particles."
# from pyccel.stdlib.internal.openmp import omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from numpy import arcsin, arctan, cos, pi, sin, sqrt, tan
from numpy import abs, exp
from numpy import copy, empty, floor, log, shape, zeros
from pyccel.decorators import inline, pure, stack_array

class MarkerArguments:
    """Holds arguments pertaining to :class:`~struphy.pic.base.Particles`
    passed to particle kernels.

    Paramaters
    ----------
    markers : array[float]
        Markers array.

    Np : int
        Total number of particles.

    vdim : int
        Dimension of velocity space.

    weight_idx : int
        Column index of particle weight.

    first_diagnostics_idx : int
        Starting index for diagnostics columns:
        after 3 positions, vdim velocities, weight, s0 and w0.

    first_pusher_idx : int
        Starting buffer marker index number for pusher.

    first_shift_idx : int
        First index for storing shifts due to boundary conditions in eta-space.

    residual_idx: int
        Column for storing the residual in iterative pushers.

    first_free_idx : int
        First index for storing auxiliary quantities for each particle.
    """

    def __init__(
        self,
        markers: "float[:, :]",
        valid_mks: "bool[:]",
        Np: int,
        vdim: int,
        weight_idx: int,
        first_diagnostics_idx: int,
        first_pusher_idx: int,
        first_shift_idx: int,
        residual_idx: int,
        first_free_idx: int,
    ):
        self.markers = markers
        self.valid_mks = valid_mks
        self.Np = Np
        self.vdim = vdim
        self.weight_idx = weight_idx
        self.n_markers = markers.shape[0]

        # useful indices
        self.first_diagnostics_idx = first_diagnostics_idx
        self.first_init_idx = first_pusher_idx
        self.first_shift_idx = first_shift_idx  # starting idx for eta-shifts due to boundary conditions
        self.residual_idx = residual_idx  # residual in iterative solvers
        self.first_free_idx = first_free_idx  # index after which auxiliary saving is possible

        # only used for Particles5D
        self.energy_idx = 8  # particle energy
        self.mu_idx = 9  # particle magnetic moment
        self.toroidalmom_idx = 10  # particle toroidal momentum


class DerhamArguments:
    """Holds the mandatory arguments pertaining to :class:`~struphy.feec.psydac_derham.Derham` passed to particle pusher kernels.

    Paramaters
    ----------
    pn : array[int]
        Spline degrees of :class:`~struphy.feec.psydac_derham.Derham`.

    tn1, tn2, tn3 : array[float]
        Knot sequences of :class:`~struphy.feec.psydac_derham.Derham`.

    starts : array[int]
        Start indices (current MPI process) of :class:`~struphy.feec.psydac_derham.Derham`.
    """

    def __init__(
        self,
        pn: "int[:]",
        tn1: "float[:]",
        tn2: "float[:]",
        tn3: "float[:]",
        starts: "int[:]",
    ):
        self.pn = pn
        self.tn1 = tn1
        self.tn2 = tn2
        self.tn3 = tn3
        self.starts = starts

        self.bn1 = empty(pn[0] + 1, dtype=float)
        self.bn2 = empty(pn[1] + 1, dtype=float)
        self.bn3 = empty(pn[2] + 1, dtype=float)
        self.bd1 = empty(pn[0], dtype=float)
        self.bd2 = empty(pn[1], dtype=float)
        self.bd3 = empty(pn[2], dtype=float)


class DomainArguments:
    """Holds the mandatory arguments pertaining to :class:`~struphy.geometry.base.Domain` passed to particle pusher kernels.

    Paramaters
    ----------
    kind : int
        Mapping identifier of :class:`~struphy.geometry.base.Domain`.

    params : array[float]
        Mapping parameters of :class:`~struphy.geometry.base.Domain`.

    p : array[int]
        Spline degrees of :class:`~struphy.geometry.base.Domain`.

    t1, t2, t3 : array[float]
        Knot sequences of :class:`~struphy.geometry.base.Domain`.

    ind1, ind2, ind3 : array[float]
        Indices of non-vanishing splines in format (number of mapping grid cells, p + 1) of :class:`~struphy.geometry.base.Domain`.

    cx, cy, cz : array[float]
        Spline coefficients (control points) of :class:`~struphy.geometry.base.Domain`.
    """

    def __init__(
        self,
        kind_map: int,
        params: "float[:]",
        p: "int[:]",
        t1: "float[:]",
        t2: "float[:]",
        t3: "float[:]",
        ind1: "int[:,:]",
        ind2: "int[:,:]",
        ind3: "int[:,:]",
        cx: "float[:,:,:]",
        cy: "float[:,:,:]",
        cz: "float[:,:,:]",
    ):
        self.kind_map = kind_map
        self.params = copy(params)
        self.p = copy(p)
        self.t1 = copy(t1)
        self.t2 = copy(t2)
        self.t3 = copy(t3)
        self.ind1 = copy(ind1)
        self.ind2 = copy(ind2)
        self.ind3 = copy(ind3)
        self.cx = copy(cx)
        self.cy = copy(cy)
        self.cz = copy(cz)

# import struphy.bsplines.bsplines_kernels as bsplines_kernels
# import struphy.geometry.evaluation_kernels as evaluation_kernels
# import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
# import struphy.geometry.evaluation_kernels as evaluation_kernels
# import struphy.linear_algebra.linalg_kernels as linalg_kernels

# do not remove; needed to identify dependencies
# # import struphy.pic.pushing.pusher_args_kernels as pusher_args_kernels
# import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
# from struphy.bsplines.evaluation_kernels_3d import (
#     # eval_0form_spline_mpi,
#     # eval_1form_spline_mpi,
#     # eval_2form_spline_mpi,
#     # eval_3form_spline_mpi,
#     # eval_vectorfield_spline_mpi,
#     get_spans,
# )
# import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
# from evaluation_kernels_3d import get_spans
# import struphy.geometry.mappings_kernels as mappings_kernels
# import struphy.linear_algebra.linalg_kernels as linalg_kernels
# from linalg_kernels import det, scalar_dot
# # do not remove; needed to identify dependencies
# # import struphy.pic.pushing.pusher_args_kernels as pusher_args_kernels
# import struphy.pic.sph_eval_kernels as sph_eval_kernels
# from struphy.pic.sph_eval_kernels import boxed_based_kernel
# from struphy.pic.pushing.pusher_kernels_gpu import DerhamArguments, DomainArguments, MarkerArguments

# # from struphy.linear_algebra.linalg_kernels import matrix_inv, matrix_vector
# # from struphy.geometry.evaluation_kernels import df

def matmul_cpu(A: "float[:,:]", B: "float[:,:]", C: "float[:,:]"):
    N: int = shape(A)[0]
    s: float = 0.0
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += A[i, k] * B[k, j]
            C[i, j] = s


def matmul_gpu(A: "float[:,:]", B: "float[:,:]", C: "float[:,:]"):
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

    span1: int = 0
    span2: int = 0
    span3: int = 0

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

    pn0 = args_derham.pn[0]
    pn1 = args_derham.pn[1]
    pn2 = args_derham.pn[2]

    tn1 = args_derham.tn1
    tn2 = args_derham.tn2
    tn3 = args_derham.tn3

    bn1 = args_derham.bn1
    bn2 = args_derham.bn2
    bn3 = args_derham.bn3

    bd1 = args_derham.bd1
    bd2 = args_derham.bd2
    bd3 = args_derham.bd3

    args_derham_pn = args_derham.pn
    args_derham_bn1 = args_derham.bn1
    args_derham_bn2 = args_derham.bn2
    args_derham_bn3 = args_derham.bn3
    args_derham_bd1 = args_derham.bd1
    args_derham_bd2 = args_derham.bd2
    args_derham_bd3 = args_derham.bd3
    args_derham_starts = args_derham.starts

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
        # _tmp123 = 2 // 3
        # # metric coeffs
        det_df = det_inline(dfm)

        # # spline evaluation
        # span1, span2, span3 = get_spans(e1, e2, e3, args_derham,)
        # span1, span2, span3 = get_spans_inline_expanded(e1, e2, e3,
        #                                     #    args_derham,
        #                                     args_derham.tn1,
        #                                     args_derham.tn2,
        #                                     args_derham.tn3,
        #                                     args_derham.pn,
        #                                     args_derham.bn1,
        #                                     args_derham.bn2,
        #                                     args_derham.bn3,
        #                                     args_derham.bd1,
        #                                     args_derham.bd2,
        #                                     args_derham.bd3,
        #                                        )

        span1 = find_span_inline(tn1, pn0, e1)
        span2 = find_span_inline(tn2, pn1, e2)
        span3 = find_span_inline(tn3, pn2, e3)

        # get spline values at eta
        # b_d_splines_slim_inline(
        #    tn1, pn0, e1, span1, bn1, bd1
        # )
        # b_d_splines_slim_inline(
        #    tn2, pn1, e2, int(span2), bn2, bd2
        # )
        # b_d_splines_slim_inline(
        #    tn3, pn2, e3, int(span3), bn3, bd3
        # )
        # span1 = find_span_inline(tn1, pn0, e1)
        # span2 = find_span_inline(args_derham.tn2, args_derham.pn[1], e2)
        # span3 = find_span_inline(args_derham.tn3, args_derham.pn[2], e3)
        # magnetic field 2-form
        eval_2form_spline_mpi_inline(
            span1,
            span2,
            span3,
            args_derham_pn,
            args_derham_bn1,
            args_derham_bn2,
            args_derham_bn3,
            args_derham_bd1,
            args_derham_bd2,
            args_derham_bd3,
            args_derham_starts,
            b2_1,
            b2_2,
            b2_3,
            b_form,
        )

        # magnetic field: Cartesian components
        # linalg_kernels.matrix_vector(dfm, b_form, b_cart)
        # matrix_vector_inline(dfm, b_form, b_cart)
        # Temporary start (replacement of linalg_kernels.matrix_vector)
        b_cart[:] = 0.0
        for i in range(3):
            for j in range(3):
                b_cart[i] += dfm[i, j] * b_form[j]
        # Temporary end

        b_cart[:] = b_cart / det_df

        # magnetic field: magnitude
        b_abs = sqrt(b_cart[0] ** 2 + b_cart[1] ** 2 + b_cart[2] ** 2)

        # # only push vxb if magnetic field is non-zero
        if b_abs != 0.0:
            # normalized magnetic field direction
            b_norm[:] = b_cart / b_abs

            # parallel velocity v.b_norm
            # vpar = linalg_kernels.scalar_dot(v, b_norm)

            # vpar = scalar_dot_inline(v, b_norm)
            vpar = v[0] * b_norm[0] + v[1] * b_norm[1] + v[2] * b_norm[2]
            # # first component of perpendicular velocity
            # linalg_kernels.cross(v, b_norm, vxb_norm)
            cross_inline(v, b_norm, vxb_norm)

            # linalg_kernels.cross(b_norm, vxb_norm, vperp)
            cross_inline(b_norm, vxb_norm, vperp)

            # # second component of perpendicular velocity
            # linalg_kernels.cross(b_norm, vperp, b_normxvperp)
            cross_inline(b_norm, vperp, b_normxvperp)

            # # analytic rotation
            markers[ip, 3:6] = vpar * b_norm + cos(b_abs * dt) * vperp - sin(b_abs * dt) * b_normxvperp


# @stack_array("grad_u", "grad_u_cart", "tmp1", "dfinv", "dfinvT")
def push_v_sph_pressure_gpu(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:,:]",
    neighbours: "int[:, :]",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
):
    r"""Updates particle velocities as

    .. math::

        \frac{\mathbf v^{n+1} - \mathbf v^n}{\Delta t} = \kappa_p \sum_{q} w_p\,w_q \left( \frac{1}{\rho^{N,h}(\boldsymbol \eta_p)} + \frac{1}{\rho^{N,h}(\boldsymbol \eta_q)} \right) G^{-1}\nabla W_h(\boldsymbol \eta_p - \boldsymbol \eta_q) \,,

    where :math:`G^{-1}` denotes the inverse metric tensor, and with the smoothed density

    .. math::

        \rho^{N,h}(\boldsymbol \eta_p) = \frac 1N \sum_q w_q \, W_h(\boldsymbol \eta_p - \boldsymbol \eta_q)\,,

    where :math:`W_h(\boldsymbol \eta)` is a smoothing kernel from :mod:`~struphy.pic.sph_smoothing_kernels`.

    Parameters
    ----------
    boxes : 2d array
        Box array of the sorting boxes structure.

    neighbours : 2d array
        Array containing the 27 neighbouring boxes of each box.

    holes : bool
        1D array of length markers.shape[0]. True if markers[i] is a hole.

    periodic1, periodic2, periodic3 : bool
        True if periodic in that dimension.

    kernel_type : int
        Number of the smoothing kernel.

    h1, h2, h3 : float
        Kernel width in respective dimension.
    """

    # Variables
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    out_bbk = 0.0
    c = 0
    box_to_search = 0
    p = 0

    # allocate arrays
    grad_u = zeros(3, dtype=float)
    grad_u_cart = zeros(3, dtype=float)
    tmp1 = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvT = zeros((3, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    Np = args_markers.Np
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks

    # Get domain args
    args_domain_kind_map    = args_domain.kind_map
    args_domain_params      = args_domain.params
    args_domain_p           = args_domain.p
    args_domain_t1          = args_domain.t1
    args_domain_t2          = args_domain.t2
    args_domain_t3          = args_domain.t3
    args_domain_ind1        = args_domain.ind1
    args_domain_ind2        = args_domain.ind2
    args_domain_ind3        = args_domain.ind3
    args_domain_cx          = args_domain.cx
    args_domain_cy          = args_domain.cy
    args_domain_cz          = args_domain.cz

    # -- removed omp: #$ omp parallel private(ip, eta1, eta2, eta3, dfinv)
    # -- removed omp: #$ omp for
    #$ omp target teams distribute parallel for
    for ip in range(n_markers):
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        weight = markers[ip, weight_idx]
        kappa = 1.0  # markers[ip, first_diagnostics_idx]
        n_at_eta = markers[ip, first_free_idx]
        loc_box = int(markers[ip, -2])

        # first component
        grad_u[0] = boxed_based_kernel_inline(
            eta1,
            eta2,
            eta3,
            loc_box,
            boxes,
            neighbours,
            markers,
            Np,
            holes,
            periodic1,
            periodic2,
            periodic3,
            weight_idx,
            kernel_type + 1,
            h1,
            h2,
            h3,
            s1,
            s2,
            s3,
            c,
            box_to_search,
            p,
            out_bbk,
        )
        grad_u[0] *= kappa / n_at_eta

        sum2 = boxed_based_kernel_inline(
            eta1,
            eta2,
            eta3,
            loc_box,
            boxes,
            neighbours,
            markers,
            Np,
            holes,
            periodic1,
            periodic2,
            periodic3,
            first_free_idx + 1,
            kernel_type + 1,
            h1,
            h2,
            h3,
            s1,
            s2,
            s3,
            c,
            box_to_search,
            p,
            out_bbk,
        )
        sum2 *= kappa
        grad_u[0] += sum2

        if kernel_type >= 340:
            # second component
            grad_u[1] = boxed_based_kernel_inline(
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                markers,
                Np,
                holes,
                periodic1,
                periodic2,
                periodic3,
                weight_idx,
                kernel_type + 2,
                h1,
                h2,
                h3,
                s1,
                s2,
                s3,
                c,
                box_to_search,
                p,
                out_bbk,
            )
            grad_u[1] *= kappa / n_at_eta

            sum4 = boxed_based_kernel_inline(
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                markers,
                Np,
                holes,
                periodic1,
                periodic2,
                periodic3,
                first_free_idx + 1,
                kernel_type + 2,
                h1,
                h2,
                h3,
                s1,
                s2,
                s3,
                c,
                box_to_search,
                p,
                out_bbk,
            )
            sum4 *= kappa
            grad_u[1] += sum4

        if kernel_type >= 670:
            # third component
            grad_u[2] = boxed_based_kernel_inline(
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                markers,
                Np,
                holes,
                periodic1,
                periodic2,
                periodic3,
                weight_idx,
                kernel_type + 3,
                h1,
                h2,
                h3,
                s1,
                s2,
                s3,
                c,
                box_to_search,
                p,
                out_bbk,
            )
            grad_u[2] *= kappa / n_at_eta

            sum6 = boxed_based_kernel_inline(
                eta1,
                eta2,
                eta3,
                loc_box,
                boxes,
                neighbours,
                markers,
                Np,
                holes,
                periodic1,
                periodic2,
                periodic3,
                first_free_idx + 1,
                kernel_type + 3,
                h1,
                h2,
                h3,
                s1,
                s2,
                s3,
                c,
                box_to_search,
                p,
                out_bbk,
            )
            sum6 *= kappa
            grad_u[2] += sum6

        # push to Cartesian coordinates
        # evaluation_kernels.df_inv(
        #     eta1,
        #     eta2,
        #     eta3,
        #     args_domain,
        #     tmp1,
        #     False,
        #     dfinv,
        # )
        df_inv_inline(
            eta1,
            eta2,
            eta3,
            # args_domain,
            # args domain start
            args_domain_kind_map,
            args_domain_params,
            args_domain_p, 
            args_domain_t1,
            args_domain_t2,
            args_domain_t3,
            args_domain_ind1,
            args_domain_ind2,
            args_domain_ind3,
            args_domain_cx,      
            args_domain_cy,
            args_domain_cz,
            # args domain end
            tmp1,
            False,
            dfinv,
        )

        df_inv_inline(
            eta1,
            eta2,
            eta3,
            # args_domain,
            # args domain start
            args_domain_kind_map,
            args_domain_params,
            args_domain_p, 
            args_domain_t1,
            args_domain_t2,
            args_domain_t3,
            args_domain_ind1,
            args_domain_ind2,
            args_domain_ind3,
            args_domain_cx,      
            args_domain_cy,
            args_domain_cz,
            # args domain end
            tmp1,
            False,
            dfinv,
        )
        # linalg_kernels.transpose(dfinv, dfinvT)
        # transpose_inline(dfinv, dfinvT)
        for _i in range(3):
            for _j in range(3):
                dfinvT[_i, _j] = dfinv[_j, _i]
        # linalg_kernels.matrix_vector(dfinvT, grad_u, grad_u_cart)
        # matrix_vector_inline(dfinvT, grad_u, grad_u_cart)

        grad_u_cart[:] = 0.0

        for _i in range(3):
            for _j in range(3):
                grad_u_cart[_i] += dfinvT[_i, _j] * grad_u[_j]

        # update velocities
        markers[ip, 3:6] -= dt * grad_u_cart

    # -- removed omp: #$ omp end parallel

# @stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_isotherm_pressure_coeffs_gpu(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:, :]",
    neighbours: "int[:, :]",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
):
    r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

    .. math::

        H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
        + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

    that is

    .. math::

        \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
        + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

    where the evaluation point is the weighted average
    :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
    for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
    in markers array for each particle.
    """
    # Variables
    out_bbk = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    c = 0
    box_to_search = 0
    p = 0

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[1]
    Np = args_markers.Np
    weight_idx = args_markers.weight_idx
    valid_mks = args_markers.valid_mks

    #$ omp target teams distribute parallel for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        loc_box = int(markers[ip, n_cols - 2])
        # n_at_eta = sph_eval_kernels.boxed_based_kernel(
        n_at_eta = boxed_based_kernel_inline(
            eta1,
            eta2,
            eta3,
            loc_box,
            boxes,
            neighbours,
            markers,
            Np,
            holes,
            periodic1,
            periodic2,
            periodic3,
            weight_idx,
            kernel_type,
            h1,
            h2,
            h3,
            s1,
            s2,
            s3,
            c,
            box_to_search,
            p,
            out_bbk,
        )
        weight = markers[ip, weight_idx]
        # save
        markers[ip, column_nr] = n_at_eta
        markers[ip, column_nr + 1] = weight / n_at_eta


def compute_sorting_etas(
        markers: "float[:,:]",
        bi: int,
        vdim : int,
        alpha: "float[:]",
        sorting_etas: "float[:,:]"
        ):
    n_markers = markers.shape[0]

    #$ omp target teams distribute parallel for
    for i in range(n_markers):
        for d in range(3):
            pos   = markers[i, d]
            shift = markers[i, bi + 3 + vdim + d]
            ppos  = markers[i, bi + d]

            val = alpha[d] * (pos + shift) + (1.0 - alpha[d]) * ppos
            sorting_etas[i, d] = val - floor(val)  # mod(val, 1.0)

@inline
def df_inv_inline(
    eta1: float,
    eta2: float,
    eta3: float,
    # args: "DomainArguments",
    args_domain_kind_map: int,
    args_domain_params: "float[:]",
    args_domain_p: "int[:]", 
    args_domain_t1: "float[:]",
    args_domain_t2: "float[:]",
    args_domain_t3: "float[:]",
    args_domain_ind1: "int[:,:]",
    args_domain_ind2: "int[:,:]",
    args_domain_ind3: "int[:,:]",
    args_domain_cx: "float[:,:,:]",
    args_domain_cy: "float[:,:,:]",
    args_domain_cz: "float[:,:,:]",
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
    # TODO: This should be called with args_xyz
    df_inline(
        eta1,
        eta2,
        eta3,
        # args,
        # args domain start
        args_domain_kind_map,
        args_domain_params,
        args_domain_p, 
        args_domain_t1,
        args_domain_t2,
        args_domain_t3,
        args_domain_ind1,
        args_domain_ind2,
        args_domain_ind3,
        args_domain_cx,      
        args_domain_cy,
        args_domain_cz,
        # args domain end
        tmp1,
    )
    matrix_inv_inline(tmp1, dfinv_out)
    
    # TODO: Use args_kind_map here
    # set known (analytical) zero components manually to zero to avoid round-off error remainders!
    # if avoid_round_off:
    #     if args_domain_kind_map == 1:
    #         dfinv_out[0, 2] = 0.0
    #         dfinv_out[1, 2] = 0.0
    #         dfinv_out[2, 0] = 0.0
    #         dfinv_out[2, 1] = 0.0
    #     elif args_domain_kind_map == 2:
    #         dfinv_out[2, 2] = 0
    #     elif args_domain_kind_map == 10:
    #         dfinv_out[0, 1] = 0.0
    #         dfinv_out[0, 2] = 0.0
    #         dfinv_out[1, 0] = 0.0
    #         dfinv_out[1, 2] = 0.0
    #         dfinv_out[2, 0] = 0.0
    #         dfinv_out[2, 1] = 0.0
    #     elif args_domain_kind_map == 11:
    #         dfinv_out[0, 1] = 0.0
    #         dfinv_out[0, 2] = 0.0
    #         dfinv_out[1, 0] = 0.0
    #         dfinv_out[1, 2] = 0.0
    #         dfinv_out[2, 0] = 0.0
    #         dfinv_out[2, 1] = 0.0
    #     elif args_domain_kind_map == 12:
    #         dfinv_out[0, 2] = 0.0
    #         dfinv_out[1, 2] = 0.0
    #         dfinv_out[2, 0] = 0.0
    #         dfinv_out[2, 1] = 0.0
    #     elif args_domain_kind_map == 20:
    #         dfinv_out[0, 2] = 0.0
    #         dfinv_out[1, 2] = 0.0
    #         dfinv_out[2, 0] = 0.0
    #         dfinv_out[2, 1] = 0.0
    #     elif args_domain_kind_map == 21:
    #         dfinv_out[0, 2] = 0.0
    #         dfinv_out[1, 2] = 0.0
    #         dfinv_out[2, 0] = 0.0
    #         dfinv_out[2, 1] = 0.0
    #     elif args_domain_kind_map == 22:
    #         dfinv_out[2, 2] = 0.0
    #     elif args_domain_kind_map == 30:
    #         dfinv_out[0, 2] = 0.0
    #         dfinv_out[1, 2] = 0.0
    #         dfinv_out[2, 0] = 0.0
    #         dfinv_out[2, 1] = 0.0
    #     elif args_domain_kind_map == 31:
    #         dfinv_out[0, 2] = 0.0
    #         dfinv_out[1, 2] = 0.0
    #         dfinv_out[2, 0] = 0.0
    #         dfinv_out[2, 1] = 0.0
    #     elif args_domain_kind_map == 32:
    #         dfinv_out[0, 2] = 0.0
    #         dfinv_out[1, 2] = 0.0
    #         dfinv_out[2, 0] = 0.0
    #         dfinv_out[2, 1] = 0.0

@inline
def df_inline(
    eta1: float,
    eta2: float,
    eta3: float,
    # args: "DomainArguments",
    args_domain_kind_map: int,
    args_domain_params: "float[:]",
    args_domain_p: "int[:]", 
    args_domain_t1: "float[:]",
    args_domain_t2: "float[:]",
    args_domain_t3: "float[:]",
    args_domain_ind1: "int[:,:]",
    args_domain_ind2: "int[:,:]",
    args_domain_ind3: "int[:,:]",
    args_domain_cx: "float[:,:,:]",
    args_domain_cy: "float[:,:,:]",
    args_domain_cz: "float[:,:,:]",
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
    # Let's assume it's 10
    
    if args_domain_kind_map == 10:
        cuboid_df(
            args_domain_params[0],
            args_domain_params[1],
            args_domain_params[2],
            args_domain_params[3],
            args_domain_params[4],
            args_domain_params[5],
            df_out,
        )
    


@inline
def boxed_based_kernel_inline(
    eta1: "float",
    eta2: "float",
    eta3: "float",
    loc_box: "int",
    boxes: "int[:,:]",
    neighbours: "int[:,:]",
    markers: "float[:,:]",
    Np: "int",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    index: "int",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
    c: "int",
    box_to_search: "int",
    p: "int",
    out: "float",
) -> float:
    """Box-based single-point sph evaluation.
    The sum is done over the particles that are in the 26 + 1 neighboring boxes
    of the ``loc_box`` the evaluation point is in.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Evaluation point in logical space.

    loc_box : int
        Box of the evaluation point.

    boxes : 2d array
        Box array of the sorting boxes structure.

    neighbours : 2d array
        Array containing the 27 neighbouring boxes of each box.

    markers : array[float]
        Markers array.

    Np : int
        Total number of particles.

    holes : bool
        1D array of length markers.shape[0]. True if markers[i] is a hole.

    periodic1, periodic2, periodic3 : bool
        True if periodic in that dimension.

    index : int
        Column index in markers array where the value multiplying the kernel in the evaluation is stored.

    kernel_type : int
        Number of the smoothing kernel.

    h1, h2, h3 : float
        Kernel width in respective dimension.
    """

    # s1 = 0.0
    # s2 = 0.0
    # s3 = 0.0
    # out = 0.0

    for neigh in range(27):
        box_to_search = neighbours[loc_box, neigh]
        c = 0
        # loop over all particles in a box
        while boxes[box_to_search, c] != -1:
            p = boxes[box_to_search, c]
            c = c + 1
            if not holes[p]:
                r1 = distance_inline(eta1, markers[p, 0], periodic1)
                r2 = distance_inline(eta2, markers[p, 1], periodic2)
                r3 = distance_inline(eta3, markers[p, 2], periodic3)
                _tmp = smoothing_kernel_inline(kernel_type,r1,r2,r3,h1, h2, h3, s1, s2, s3)
                out = out + markers[p, index] * _tmp
                # out += markers[p, index] * smoothing_kernel_inline(kernel_type, r1, r2, r3, h1, h2, h3)
                pass
    return out / Np

@inline
def distance_inline(
        x: "float", y: "float", periodic: "bool",
        ) -> float:
    """Return the one dimensional distance of x and y taking in account the periodicity on [0,1]."""
    d = x - y
    if periodic:
        if d > 0.5:
            while d > 0.5:
                d = d - 1.0
        elif d < -0.5:
            while d < -0.5:
                d = d + 1.0
    return d

@inline
def transpose_inline(a: "float[:,:]", b: "float[:,:]"):
    """
    Assembles the transposed of a 3x3 matrix.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3).

        b : array[float]
            The output array (matrix) of shape (3,3).
    """

    for i in range(3):
        for j in range(3):
            b[i, j] = a[j, i]


# @pure
@inline
def matrix_vector_inline(a: "float[:,:]", b: "float[:]", c: "float[:]"):
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

    c[:] = 0.0

    for i in range(3):
        for j in range(3):
            c[i] += a[i, j] * b[j]


@inline
def scalar_dot_inline(a: "float[:]", b: "float[:]") -> float:
    """
    Computes scalar (dot) product of two vectors of length 3.

    Parameters
    ----------
        a : array[float]
            The first input array (vector) of shape (3,).

        b : array[float]
            The second input array (vector) of shape (3,).

    Returns
    -------
        value : float
            The scalar poduct of the two input vectors a and b.
    """
    value: float = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    return value


@pure
@inline
def det_inline(a: "float[:,:]") -> float:
    """
    Computes the determinant of a 3x3 matrix.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3) of which the determinant shall be computed.

    Returns
    -------
        det_a : float
            The determinant of the 3x3 matrix a.
    """

    plus = a[0, 0] * a[1, 1] * a[2, 2] + a[0, 1] * a[1, 2] * a[2, 0] + a[0, 2] * a[1, 0] * a[2, 1]
    minus = a[2, 0] * a[1, 1] * a[0, 2] + a[2, 1] * a[1, 2] * a[0, 0] + a[2, 2] * a[1, 0] * a[0, 1]

    det_a = plus - minus

    return det_a


@pure
@inline
def cross_inline(a: "float[:]", b: "float[:]", c: "float[:]"):
    """
    Computes the vector (cross) product of two vectors of length 3.

    Parameters
    ----------
        a : array[float]
            The first input array (vector) of shape (3,).

        b : array[float]
            The second input array (vector) of shape (3,).

        c : array[float]
            The output array (vector) of shape (3,) which is the vector product a x b.
    """

    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]


# # @stack_array('det_a')
@inline
def matrix_inv_inline(a: "float[:,:]", b: "float[:,:]"):
    """
    Computes the inverse of a 3x3 matrix.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3).

        b : array[float]
            The output array (matrix) of shape (3,3).
    """

    # det_a = det_inline(a)
    plus = a[0, 0] * a[1, 1] * a[2, 2] + a[0, 1] * a[1, 2] * a[2, 0] + a[0, 2] * a[1, 0] * a[2, 1]
    minus = a[2, 0] * a[1, 1] * a[0, 2] + a[2, 1] * a[1, 2] * a[0, 0] + a[2, 2] * a[1, 0] * a[0, 1]

    det_a = plus - minus

    b[0, 0] = (a[1, 1] * a[2, 2] - a[2, 1] * a[1, 2]) / det_a
    b[0, 1] = (a[2, 1] * a[0, 2] - a[0, 1] * a[2, 2]) / det_a
    b[0, 2] = (a[0, 1] * a[1, 2] - a[1, 1] * a[0, 2]) / det_a

    b[1, 0] = (a[1, 2] * a[2, 0] - a[2, 2] * a[1, 0]) / det_a
    b[1, 1] = (a[2, 2] * a[0, 0] - a[0, 2] * a[2, 0]) / det_a
    b[1, 2] = (a[0, 2] * a[1, 0] - a[1, 2] * a[0, 0]) / det_a

    b[2, 0] = (a[1, 0] * a[2, 1] - a[2, 0] * a[1, 1]) / det_a
    b[2, 1] = (a[2, 0] * a[0, 1] - a[0, 0] * a[2, 1]) / det_a
    b[2, 2] = (a[0, 0] * a[1, 1] - a[1, 0] * a[0, 1]) / det_a


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
def find_span_inline(t: "float[:]", p: "int", eta: "float"): #, returnVal: "int"):
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
        # span = (low + high)//2
        span = int((low + high) / 2)

        while eta < t[span] or eta >= t[span + 1]:
            if eta < t[span]:
                high = span
            else:
                low = span
            # span = (low + high)//2
            span = int((low + high) / 2)

        returnVal = span

    return returnVal


@inline
def b_d_splines_slim_inline(tn: "float[:]", pn: "int", eta: "float", span: "int", bn: "float[:]", bd: "float[:]"):
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
    bn[:] = 0.0
    bd[:] = 0.0

    # Initialize variables left and right used for computing the value
    left = zeros(10, dtype=float) # This should be zeros(pn, dtype=float)
    right = zeros(10, dtype=float)

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


@inline
def eval_2form_spline_mpi_inline(
    span1: int,
    span2: int,
    span3: int,
    # args_derham: "DerhamArguments",
    args_derham_pn: "int[:]",
    args_derham_bn1: "float[:]",
    args_derham_bn2: "float[:]",
    args_derham_bn3: "float[:]",
    args_derham_bd1: "float[:]",
    args_derham_bd2: "float[:]",
    args_derham_bd3: "float[:]",
    args_derham_starts: "int[:]",
    form_coeffs_1: "float[:,:,:]",
    form_coeffs_2: "float[:,:,:]",
    form_coeffs_3: "float[:,:,:]",
    out: "float[:]",
):
    """Single-point evaluation of Derham 2-form spline defined by form_coeffs,
    given N-spline values (in bn), D-spline values (in bd)
    and knot span indices span."""

    # out[0] =
    out[0] = eval_spline_mpi_kernel_inline(
        args_derham_pn[0],
        args_derham_pn[1] - 1,
        args_derham_pn[2] - 1,
        args_derham_bn1,
        args_derham_bd2,
        args_derham_bd3,
        span1,
        span2,
        span3,
        form_coeffs_1,
        args_derham_starts,
    )

    out[1] = eval_spline_mpi_kernel_inline(
       args_derham_pn[0] - 1,
       args_derham_pn[1],
       args_derham_pn[2] - 1,
       args_derham_bd1,
       args_derham_bn2,
       args_derham_bd3,
       span1,
       span2,
       span3,
       form_coeffs_2,
       args_derham_starts,
    )

    out[2] = eval_spline_mpi_kernel_inline(
       args_derham_pn[0] - 1,
       args_derham_pn[1] - 1,
       args_derham_pn[2],
       args_derham_bd1,
       args_derham_bd2,
       args_derham_bn3,
       span1,
       span2,
       span3,
       form_coeffs_3,
       args_derham_starts,
    )


@inline
def eval_spline_mpi_kernel_inline(
    _p1: "int",
    _p2: "int",
    _p3: "int",
    _basis1: "float[:]",
    _basis2: "float[:]",
    _basis3: "float[:]",
    _span1: "int",
    _span2: "int",
    _span3: "int",
    _data: "float[:,:,:]",
    _starts: "int[:]",
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

    for il1 in range(_p1 + 1):
        # _data[:,:,:] = 0.0
        i1 = _span1 + il1 - _starts[0]
        for il2 in range(_p2 + 1):
            i2 = _span2 + il2 - _starts[1]
            for il3 in range(_p3 + 1):
                i3 = _span3 + il3 - _starts[2]
                # spline_value +=               _data[i1, i2, i3] * basis1[il1] * basis2[il2] * basis3[il3]
                spline_value = spline_value + _data[i1, i2, i3] * _basis1[il1] * _basis2[il2] * _basis3[il3]
    return spline_value




###########################################
# Uni-variate kernels for tensor products #
###########################################
@inline
def trigonometric_uni(
    x: "float",
    h: "float",
) -> float:
    """Uni-variate kernel S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    
    if abs(x / h) <= 1.0:
        out = 0.785398163397448 / h * cos(x / h * pi / 2.0)
    else:
        out = 0.0
    return out

@inline
def grad_trigonometric_uni(
    x: "float",
    h: "float",
) -> float:
    """Derivative of S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""

    if abs(x / h) <= 1.0:
        out = -(1.2337005501361697 / h**2) * sin(x / h * pi / 2.0)
    else:
        out = 0.0
    return out

@inline
def gaussian_uni(
    x: "float",
    h: "float",
) -> float:
    """Uni-variate S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        out =  1 / (sqrt(pi) * h / 3) * exp(-(x**2) / (h / 3) ** 2)
    else:
        out = 0.0
    return out

@inline
def gaussian_uni_noreturn(
    x: "float",
    h: "float",
    out: "float",
) -> float:
    """Uni-variate S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        out =  1 / (sqrt(pi) * h / 3) * exp(-(x**2) / (h / 3) ** 2)
    else:
        out = 0.0



@inline
def grad_gaussian_uni(
    x: "float",
    h: "float",
) -> float:
    """Derivative of S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        out = -54 * x / (h**3 * sqrt(pi)) * exp(-(x**2) / (h / 3) ** 2)
    else:
        out = 0.0
    return out

@inline
def linear_uni(
    x: "float",
    h: "float",
) -> float:
    """Uni-variate S(x, h) = (1 - x)/h if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        out = (1.0 - abs(x / h)) / h
    else:
        out = 0.0
    return out

@inline
def grad_linear_uni(
    x: "float",
    h: "float",
) -> float:
    """Derivative of S(x, h) = (1 - x)/h if |x|<1, 0 else."""
    
    if abs(x / h) <= 1.0:
        if x > 0.0:
            out = -(1 / h**2)
        else:
            out = 1 / h**2
    else:
        out = 0.0
    return out

##############
# 1d kernels #
##############
@inline
def trigonometric_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """1d kernel S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = trigonometric_uni(r1, h1)
    return s1

@inline
def grad_trigonometric_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """Derivative of S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    ds1 = grad_trigonometric_uni(r1, h1)
    return ds1

@inline
def gaussian_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """1d kernel S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    return s1

@inline
def grad_gaussian_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """Derivative of S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    ds1 = grad_gaussian_uni(r1, h1)
    return ds1

@inline
def linear_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """1d kernel S(x, h) = (1 - x)/h if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    return s1

@inline
def grad_linear_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """Derivative of S(x, h) = (1 - x)/h if |x|<1, 0 else."""
    ds1 = grad_linear_uni(r1, h1)
    return ds1


##############
# 2d kernels #
##############

@inline
def trigonometric_2d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    _temporary =  testfunc()
    # s1 = trigonometric_uni(r1, h1)
    # s2 = trigonometric_uni(r2, h2)
    return 1.0 * 2.0 #s1 * s2

def testfunc() -> "float":
    return 2.0

@inline
def grad_trigonometric_2d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    ds1 = grad_trigonometric_uni(r1, h1)
    s2 = trigonometric_uni(r2, h2)
    return ds1 * s2

@inline
def grad_trigonometric_2d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = trigonometric_uni(r1, h1)
    ds2 = grad_trigonometric_uni(r2, h2)
    return s1 * ds2

@inline
def gaussian_2d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    s2 = gaussian_uni(r2, h2)
    return s1 * s2

@inline
def grad_gaussian_2d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    ds1 = grad_gaussian_uni(r1, h1)
    s2 = gaussian_uni(r2, h2)
    return ds1 * s2

@inline
def grad_gaussian_2d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    ds2 = grad_gaussian_uni(r2, h2)
    return s1 * ds2

@inline
def linear_2d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    return s1 * s2

@inline
def grad_linear_2d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    ds1 = grad_linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    return ds1 * s2

@inline
def grad_linear_2d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    ds2 = grad_linear_uni(r2, h2)
    return s1 * ds2


##############
# 3d kernels #
##############
@inline
def trigonometric_3d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = trigonometric_uni(r1, h1)
    s2 = trigonometric_uni(r2, h2)
    s3 = trigonometric_uni(r3, h3)
    return s1 * s2 * s3

@inline
def gaussian_3d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
    # out: "float",
) -> float:
    """Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h3)
    s2 = gaussian_uni(r2, h2)
    s3 = gaussian_uni(r3, h3)
    return s1 * s2 * s3

@inline
def grad_gaussian_3d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    ds1 = grad_gaussian_uni(r1, h1)
    s2 = gaussian_uni(r2, h2)
    s3 = gaussian_uni(r3, h3)
    return ds1 * s2 * s3

@inline
def grad_gaussian_3d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    ds2 = grad_gaussian_uni(r2, h2)
    s3 = gaussian_uni(r3, h3)
    return s1 * ds2 * s3

@inline
def grad_gaussian_3d_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """3rd component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    s2 = gaussian_uni(r2, h2)
    ds3 = grad_gaussian_uni(r3, h3)
    return s1 * s2 * ds3

@inline
def linear_isotropic_3d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """
    Smoothing kernel S(r,h) = C(h)F(r/h) with F(x) = 1-x if x<1, 0 else,
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral.
    """
    r = sqrt(r1**2 + r2**2 + r3**2)
    h = h1
    if r / h > 1.0:
        return 0.0
    else:
        return (1.0 - r / h) / (1.0471975512 * h**3)

@inline
def grad_linear_isotropic_3d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """
    1st component of gradient of S(r,h) = C(h)F(r/h) with F(x) = 1-x if x<1, 0 else,
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral.
    """
    r = sqrt(r1**2 + r2**2 + r3**2)
    h = h1
    if r / h > 1.0:
        return 0.0
    elif r == 0.0:
        return -1 / h / (1.0471975512 * h**3)
    else:
        return -r1 / (r * h) / (1.0471975512 * h**3)

@inline
def grad_linear_isotropic_3d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """
    1st component of gradient of S(r,h) = C(h)F(r/h) with F(x) = 1-x if x<1, 0 else,
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral.
    """
    r = sqrt(r1**2 + r2**2 + r3**2)
    h = h1
    if r / h > 1.0:
        return 0.0
    elif r == 0.0:
        return -1 / h / (1.0471975512 * h**3)
    else:
        return -r2 / (r * h) / (1.0471975512 * h**3)

@inline
def grad_linear_isotropic_3d_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """
    1st component of gradient of S(r,h) = C(h)F(r/h) with F(x) = 1-x if x<1, 0 else,
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral.
    """
    r = sqrt(r1**2 + r2**2 + r3**2)
    h = h1
    if r / h > 1.0:
        return 0.0
    elif r == 0.0:
        return -1 / h / (1.0471975512 * h**3)
    else:
        return -r3 / (r * h) / (1.0471975512 * h**3)

@inline
def linear_3d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    s3 = linear_uni(r3, h3)
    return s1 * s2 * s3

@inline
def grad_linear_3d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    ds1 = grad_linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    s3 = linear_uni(r3, h3)
    return ds1 * s2 * s3

@inline
def grad_linear_3d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    ds2 = grad_linear_uni(r2, h2)
    s3 = linear_uni(r3, h3)
    return s1 * ds2 * s3

@inline
def grad_linear_3d_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
) -> float:
    """3rd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    ds3 = grad_linear_uni(r3, h3)
    return s1 * s2 * ds3


############
# selector #
############
@inline
def smoothing_kernel_inline(
    kernel_type: "int",
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
    s1: "float",
    s2: "float",
    s3: "float",
):
    """Each smoothing kernel is normalized to 1.

    The kernel type numbers must have 3 digits, where the last digit is reserved for the gradient;
    if a kernel has the type number n, the i-th components of its gradient has the number n + i.
    This means we have space for 99 kernels (and its gradient components) in principle.

    - 1d kernels <= 330
    - 2d kernels <= 660
    - 3d kernels >= 670

    If you add a kernel, make sure it is also added to :meth:`~struphy.pic.base.Particles.ker_dct`."""
    out = 1.0

    # 1d kernels
    if kernel_type == 100:
        out = trigonometric_1d(r1, r2, r3, h1, h2, h3, s1, s2, s3)
    elif kernel_type == 101:
        out = grad_trigonometric_1d(r1, r2, r3, h1, h2, h3, s1, s2, s3)

    elif kernel_type == 110:
        out = gaussian_1d(r1, r2, r3, h1, h2, h3, s1, s2, s3)
    elif kernel_type == 111:
        out = grad_gaussian_1d(r1, r2, r3, h1, h2, h3, s1, s2, s3)

    elif kernel_type == 120:
        out = linear_1d(r1, r2, r3, h1, h2, h3, s1, s2, s3)
    elif kernel_type == 121:
        out = grad_linear_1d(r1, r2, r3, h1, h2, h3, s1, s2, s3)

    # 2d kernels
    # elif kernel_type == 340:
    #     out = trigonometric_2d(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 341:
    #     out = grad_trigonometric_2d_1(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 342:
    #     out = grad_trigonometric_2d_2(r1, r2, r3, h1, h2, h3)

    # elif kernel_type == 350:
    #     out = gaussian_2d(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 351:
    #     out = grad_gaussian_2d_1(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 352:
    #     out = grad_gaussian_2d_2(r1, r2, r3, h1, h2, h3)

    # elif kernel_type == 360:
    #     out = linear_2d(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 361:
    #     out = grad_linear_2d_1(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 362:
    #     out = grad_linear_2d_2(r1, r2, r3, h1, h2, h3)

    # # 3d kernels
    # elif kernel_type == 670:
    #     out = trigonometric_3d(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 671:
    #     out = grad_trigonometric_3d_1(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 672:
    #     out = grad_trigonometric_3d_2(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 673:
    #     out = grad_trigonometric_3d_3(r1, r2, r3, h1, h2, h3)

    # TODO: Make this work
    elif kernel_type == 680:
        out = gaussian_3d(r1, r2, r3, h1, h2, h3, s1, s2, s3)
        # gaussian_uni_noreturn(r1, h1, _s1, _s2, _s3)
        # out2 = 0.0
        # gaussian_uni_noreturn(r2, h2,out2)
        # out = out * gaussian_uni(r2, h2)
        # out = out * gaussian_uni(r3, h3)
    # elif kernel_type == 681:
    #     out = grad_gaussian_3d_1(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 682:
    #     out = grad_gaussian_3d_2(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 683:
    #     out = grad_gaussian_3d_3(r1, r2, r3, h1, h2, h3)

    # elif kernel_type == 690:
    #     out = linear_isotropic_3d(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 691:
    #     out = grad_linear_isotropic_3d_1(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 692:
    #     out = grad_linear_isotropic_3d_2(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 693:
    #     out = grad_linear_isotropic_3d_3(r1, r2, r3, h1, h2, h3)

    # elif kernel_type == 700:
    #     out = linear_3d(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 701:
    #     out = grad_linear_3d_1(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 702:
    #     out = grad_linear_3d_2(r1, r2, r3, h1, h2, h3)
    # elif kernel_type == 703:
    #     out = grad_linear_3d_3(r1, r2, r3, h1, h2, h3)
    return out


# @stack_array("b1", "b2", "b3", "tmp1", "tmp2", "tmp3")
@inline
def spline_3d(
    eta1: float,
    eta2: float,
    eta3: float,
    p: "int[:]",
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
    span1 = find_span_inline(args.t1, int(p[0]), eta1)
    span2 = find_span_inline(args.t2, int(p[1]), eta2)
    span3 = find_span_inline(args.t3, int(p[2]), eta3)

    # p + 1 non-zero mapping splines
    b1 = zeros(int(p[0]) + 1, dtype=float)
    b2 = zeros(int(p[1]) + 1, dtype=float)
    b3 = zeros(int(p[2]) + 1, dtype=float)

    b_splines_slim_inline(args.t1, int(p[0]), eta1, span1, b1)
    b_splines_slim_inline(args.t2, int(p[1]), eta2, span2, b2)
    b_splines_slim_inline(args.t3, int(p[2]), eta3, span3, b3)

    # Evaluate spline mapping
    tmp1 = ind1[span1 - int(p[0]), :]
    tmp2 = ind2[span2 - int(p[1]), :]
    tmp3 = ind3[span3 - int(p[2]), :]

    f_out[0] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), b1, b2, b3, tmp1, tmp2, tmp3, args.cx
    )
    f_out[1] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), b1, b2, b3, tmp1, tmp2, tmp3, args.cy
    )
    f_out[2] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), b1, b2, b3, tmp1, tmp2, tmp3, args.cz
    )


# @stack_array("b1", "b2", "b3", "der1", "der2", "der3", "tmp1", "tmp2", "tmp3")
@inline
def spline_3d_df(
    eta1: float,
    eta2: float,
    eta3: float,
    p: "int[:]",
    ind1: "int[:, :]",
    ind2: "int[:, :]",
    ind3: "int[:, :]",
    args: "DomainArguments",
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.mappings_kernels.spline_3d`."""

    # mapping spans
    span1 = find_span_inline(args.t1, int(p[0]), eta1)
    span2 = find_span_inline(args.t2, int(p[1]), eta2)
    span3 = find_span_inline(args.t3, int(p[2]), eta3)

    # non-zero splines of mapping, and derivatives
    b1 = zeros(int(p[0]) + 1, dtype=float)
    b2 = zeros(int(p[1]) + 1, dtype=float)
    b3 = zeros(int(p[2]) + 1, dtype=float)

    der1 = zeros(int(p[0]) + 1, dtype=float)
    der2 = zeros(int(p[1]) + 1, dtype=float)
    der3 = zeros(int(p[2]) + 1, dtype=float)

    b_der_splines_slim_inline(args.t1, int(p[0]), eta1, span1, b1, der1)
    b_der_splines_slim_inline(args.t2, int(p[1]), eta2, span2, b2, der2)
    b_der_splines_slim_inline(args.t3, int(p[2]), eta3, span3, b3, der3)

    # Evaluation of Jacobian
    tmp1 = ind1[span1 - int(p[0]), :]
    tmp2 = ind2[span2 - int(p[1]), :]
    tmp3 = ind3[span3 - int(p[2]), :]

    df_out[0, 0] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), der1, b2, b3, tmp1, tmp2, tmp3, args.cx
    )
    df_out[0, 1] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), b1, der2, b3, tmp1, tmp2, tmp3, args.cx
    )
    df_out[0, 2] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), b1, b2, der3, tmp1, tmp2, tmp3, args.cx
    )
    df_out[1, 0] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), der1, b2, b3, tmp1, tmp2, tmp3, args.cy
    )
    df_out[1, 1] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), b1, der2, b3, tmp1, tmp2, tmp3, args.cy
    )
    df_out[1, 2] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), b1, b2, der3, tmp1, tmp2, tmp3, args.cy
    )
    df_out[2, 0] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), der1, b2, b3, tmp1, tmp2, tmp3, args.cz
    )
    df_out[2, 1] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), b1, der2, b3, tmp1, tmp2, tmp3, args.cz
    )
    df_out[2, 2] = evaluation_kernel_3d_inline(
        int(p[0]), int(p[1]), int(p[2]), b1, b2, der3, tmp1, tmp2, tmp3, args.cz
    )


# @stack_array("b1", "b2", "tmp1", "tmp2")
@inline
def spline_2d_straight(
    eta1: float,
    eta2: float,
    eta3: float,
    p: "int[:]",
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
    span1 = find_span_inline(args.t1, int(p[0]), eta1)
    span2 = find_span_inline(args.t2, int(p[1]), eta2)

    # p + 1 non-zero mapping splines
    b1 = zeros(int(p[0]) + 1, dtype=float)
    b2 = zeros(int(p[1]) + 1, dtype=float)

    b_splines_slim_inline(args.t1, int(p[0]), eta1, span1, b1)
    b_splines_slim_inline(args.t2, int(p[1]), eta2, span2, b2)

    # Evaluate mapping
    tmp1 = ind1[span1 - int(p[0]), :]
    tmp2 = ind2[span2 - int(p[1]), :]

    f_out[0] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, b2, tmp1, tmp2, cx)
    f_out[1] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, b2, tmp1, tmp2, cy)
    f_out[2] = lz * eta3

    # TODO: explanation
    if eta1 == 0.0 and cx[0, 0] == cx[0, 1]:
        f_out[0] = cx[0, 0]

    if eta1 == 0.0 and cy[0, 0] == cy[0, 1]:
        f_out[1] = cy[0, 0]


# @stack_array("b1", "b2", "der1", "der2", "tmp1", "tmp2")
@inline
def spline_2d_straight_df(
    eta1: float,
    eta2: float,
    p: "int[:]",
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
    span1 = find_span_inline(args.t1, int(p[0]), eta1)
    span2 = find_span_inline(args.t2, int(p[1]), eta2)

    # non-zero splines of mapping, and derivatives
    b1 = zeros(int(p[0]) + 1, dtype=float)
    b2 = zeros(int(p[1]) + 1, dtype=float)

    der1 = zeros(int(p[0]) + 1, dtype=float)
    der2 = zeros(int(p[1]) + 1, dtype=float)

    b_der_splines_slim_inline(args.t1, int(p[0]), eta1, span1, b1, der1)
    b_der_splines_slim_inline(args.t2, int(p[1]), eta2, span2, b2, der2)

    # Evaluation of Jacobian
    tmp1 = ind1[span1 - int(p[0]), :]
    tmp2 = ind2[span2 - int(p[1]), :]

    df_out[0, 0] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), der1, b2, tmp1, tmp2, cx)
    df_out[0, 1] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, der2, tmp1, tmp2, cx)
    df_out[0, 2] = 0.0
    df_out[1, 0] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), der1, b2, tmp1, tmp2, cy)
    df_out[1, 1] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, der2, tmp1, tmp2, cy)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz

    # TODO: explanation
    if eta1 == 0.0 and cx[0, 0] == cx[0, 1]:
        df_out[0, 1] = 0.0

    if eta1 == 0.0 and cy[0, 0] == cy[0, 1]:
        df_out[1, 1] = 0.0


# @stack_array("b1", "b2", "tmp1", "tmp2")
@inline
def spline_2d_torus(
    eta1: float,
    eta2: float,
    eta3: float,
    p: "int[:]",
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
    span1 = find_span_inline(args.t1, int(p[0]), eta1)
    span2 = find_span_inline(args.t2, int(p[1]), eta2)

    # p + 1 non-zero mapping splines
    b1 = zeros(int(p[0]) + 1, dtype=float)
    b2 = zeros(int(p[1]) + 1, dtype=float)

    b_splines_slim_inline(args.t1, int(p[0]), eta1, span1, b1)
    b_splines_slim_inline(args.t2, int(p[1]), eta2, span2, b2)

    # Evaluate mapping
    tmp1 = ind1[span1 - int(p[0]), :]
    tmp2 = ind2[span2 - int(p[1]), :]

    f_out[0] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, b2, tmp1, tmp2, cx) * cos(
        2 * pi * eta3 / tor_period
    )
    f_out[1] = (
        evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, b2, tmp1, tmp2, cx)
        * (-1)
        * sin(2 * pi * eta3 / tor_period)
    )
    f_out[2] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, b2, tmp1, tmp2, cy)

    # TODO: explanation
    if eta1 == 0.0 and cx[0, 0] == cx[0, 1]:
        f_out[0] = cx[0, 0] * cos(2 * pi * eta3 / tor_period)
        f_out[1] = cx[0, 0] * (-1) * sin(2 * pi * eta3 / tor_period)

    if eta1 == 0.0 and cy[0, 0] == cy[0, 1]:
        f_out[2] = cy[0, 0]


# @stack_array("b1", "b2", "der1", "der2", "tmp1", "tmp2")
@inline
def spline_2d_torus_df(
    eta1: float,
    eta2: float,
    eta3: float,
    p: "int[:]",
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
    span1 = find_span_inline(args.t1, int(p[0]), eta1)
    span2 = find_span_inline(args.t2, int(p[1]), eta2)

    # non-zero splines of mapping, and derivatives
    b1 = zeros(int(p[0]) + 1, dtype=float)
    b2 = zeros(int(p[1]) + 1, dtype=float)

    der1 = zeros(int(p[0]) + 1, dtype=float)
    der2 = zeros(int(p[1]) + 1, dtype=float)

    b_der_splines_slim_inline(args.t1, int(p[0]), eta1, span1, b1, der1)
    b_der_splines_slim_inline(args.t2, int(p[1]), eta2, span2, b2, der2)

    tmp1 = ind1[span1 - int(p[0]), :]
    tmp2 = ind2[span2 - int(p[1]), :]

    df_out[0, 0] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), der1, b2, tmp1, tmp2, cx) * cos(
        2 * pi * eta3 / tor_period
    )
    df_out[0, 1] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, der2, tmp1, tmp2, cx) * cos(
        2 * pi * eta3 / tor_period
    )
    df_out[0, 2] = (
        evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, b2, tmp1, tmp2, cx)
        * sin(2 * pi * eta3 / tor_period)
        * (-2 * pi / tor_period)
    )
    df_out[1, 0] = (
        evaluation_kernel_2d_inline(int(p[0]), int(p[1]), der1, b2, tmp1, tmp2, cx)
        * (-1)
        * sin(2 * pi * eta3 / tor_period)
    )
    df_out[1, 1] = (
        evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, der2, tmp1, tmp2, cx)
        * (-1)
        * sin(2 * pi * eta3 / tor_period)
    )
    df_out[1, 2] = (
        evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, b2, tmp1, tmp2, cx)
        * (-1)
        * cos(2 * pi * eta3 / tor_period)
        * 2
        * pi
        / tor_period
    )
    df_out[2, 0] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), der1, b2, tmp1, tmp2, cy)
    df_out[2, 1] = evaluation_kernel_2d_inline(int(p[0]), int(p[1]), b1, der2, tmp1, tmp2, cy)
    df_out[2, 2] = 0.0

    # TODO: explanation
    if eta1 == 0.0 and cx[0, 0] == cx[0, 1]:
        df_out[0, 1] = 0.0
        df_out[1, 1] = 0.0

    if eta1 == 0.0 and cy[0, 0] == cy[0, 1]:
        df_out[2, 1] = 0.0


@pure
@inline
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

    Note
    ----
    Example with paramters :math:`l_1=0\,,r_1=1\,,l_2=0\,,r_2=1\,,l_3=0` and :math:`r_3=1`:

        .. image:: ../pics/mappings/cuboid.png

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
@inline
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
@inline
def orthogonal(eta1: float, eta2: float, eta3: float, lx: float, ly: float, alpha: float, lz: float, f_out: "float[:]"):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= L_x\,\left[\,\eta_1 + \\alpha\sin(2\pi\,\eta_1)\,\\right]\,,

        F_y &= L_y\,\left[\,\eta_2 + \\alpha\sin(2\pi\,\eta_2)\,\\right]\,,

        F_z &= L_z\,\eta_3\,.

    Note
    ----
    Example with paramters :math:`L_x=1\,,L_y=1\,,\\alpha=0.1` and :math:`L_z=1`:

        .. image:: ../pics/mappings/orthogonal.png

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
@inline
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
@inline
def colella(eta1: float, eta2: float, eta3: float, lx: float, ly: float, alpha: float, lz: float, f_out: "float[:]"):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= L_x\,\left[\,\eta_1 + \\alpha\sin(2\pi\,\eta_1)\sin(2\pi\,\eta_2)\,\\right]\,,

        F_y &= L_y\,\left[\,\eta_2 + \\alpha\sin(2\pi\,\eta_2)\sin(2\pi\,\eta_1)\,\\right]\,,

        F_z &= L_z\,\eta_3\,.

    Note
    ----
    Example with paramters :math:`L_x=1\,,L_y=1\,,\\alpha=0.1` and :math:`L_z=1`:

        .. image:: ../pics/mappings/colella.png

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
@inline
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
@inline
def hollow_cyl(eta1: float, eta2: float, eta3: float, a1: float, a2: float, lz: float, poc: float, f_out: "float[:]"):
    r"""Point-wise evaluation of

    .. math::

        F_x &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\cos(2\pi\,\eta_2 / poc)\,,

        F_y &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\sin(2\pi\,\eta_2 / poc)\,,

        F_z &= L_z\,\eta_3\,.

    Note
    ----
        Example with paramters :math:`a_1=0.2\,,a_2=1` and :math:`L_z=3`:

        .. image:: ../pics/mappings/hollow_cylinder.png

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
@inline
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
@inline
def powered_ellipse(
    eta1: float, eta2: float, eta3: float, rx: float, ry: float, lz: float, s: float, f_out: "float[:]"
):
    r"""
    Point-wise evaluation of

    .. math::
        F_x &= r_x\,\eta_1^s\cos(2\pi\,\eta_2)\,,

        F_y &= r_y\,\eta_1^s\sin(2\pi\,\eta_2)\,,

        F_z &= L_z\,\eta_3\,.

    Note
    ----
        Example with paramters :math:`r_x=1\,,r_y=2,s=0.5` and :math:`L_z=1`:

        .. image:: ../pics/mappings/ellipse.png

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
@inline
def powered_ellipse_df(
    eta1: float, eta2: float, eta3: float, rx: float, ry: float, lz: float, s: float, df_out: "float[:,:]"
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
@inline
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

    Note
    ----
        Example with paramters :math:`a_1=0.2\,,a_2=1` and :math:`R_0=3`:

        .. image:: ../pics/mappings/hollow_torus.png

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
@inline
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
@inline
def shafranov_shift(
    eta1: float, eta2: float, eta3: float, rx: float, ry: float, lz: float, de: float, f_out: "float[:]"
):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\eta_1^2)r_x\Delta\,,

        F_y &= r_y\,\eta_1\sin(2\pi\,\eta_2)\,,

        F_z &= L_z\,\eta_3\,.

    Note
    ----
    Example with paramters :math:`r_x=1\,,r_y=1\,,L_z=1` and :math:`\Delta=0.2`:

        .. image:: ../pics/mappings/shafranov_shift.png

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
@inline
def shafranov_shift_df(
    eta1: float, eta2: float, eta3: float, rx: float, ry: float, lz: float, de: float, df_out: "float[:,:]"
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
@inline
def shafranov_sqrt(
    eta1: float, eta2: float, eta3: float, rx: float, ry: float, lz: float, de: float, f_out: "float[:]"
):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\sqrt \eta_1)r_x\Delta\,,

        F_y &= r_y\,\eta_1\sin(2\pi\,\eta_2)\,,

        F_z &= L_z\,\eta_3\,.

    Note
    ----
    No example plot yet.

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
@inline
def shafranov_sqrt_df(
    eta1: float, eta2: float, eta3: float, rx: float, ry: float, lz: float, de: float, df_out: "float[:,:]"
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
@inline
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

    Note
    ----
    Example with paramters :math:`R_0=3\,,L_z=1\,,\Delta_x=0.1\,,\Delta_y=0\,,\delta=0.2\,,\epsilon=1/3` and :math:`\kappa=1.5`:

        .. image:: ../pics/mappings/shafranov_dshaped.png

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
@inline
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


# @pure
# @stack_array("left", "right", "diff")
@inline
def b_der_splines_slim_inline(tn: "float[:]", pn: "int", eta: "float", span: "int", bn: "float[:]", der: "float[:]"):
    """
    Parameters
    ----------
    tn : array_like
        Knots sequence of B-splines.

    pn : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    bn : array[float]
        Out: pn + 1 values of B-splines at eta.

    der : array[float]
        Out: pn + 1 values of derivatives of B-splines at eta.
    """

    left = zeros(pn, dtype=float)
    right = zeros(pn, dtype=float)
    diff = zeros(pn, dtype=float)

    values = zeros((pn + 1, pn + 1), dtype=float)
    values[0, 0] = 1.0

    for j in range(pn):
        left[j] = eta - tn[span - j]
        right[j] = tn[span + 1 + j] - eta
        saved = 0.0
        for r in range(j + 1):
            diff[r] = 1.0 / (right[r] + left[j - r])
            temp = values[j, r] * diff[r]
            values[j + 1, r] = saved + right[r] * temp
            saved = left[j - r] * temp
        values[j + 1, j + 1] = saved

    diff[:] = diff * pn

    # compute derivatives
    # j = 0
    saved = values[pn - 1, 0] * diff[0]
    der[0] = -saved

    # j = 1, ... , pn
    for j in range(1, pn):
        temp = saved
        saved = values[pn - 1, j] * diff[j]
        der[j] = temp - saved

    # j = pn
    bn[:] = values[pn, :]
    der[pn] = saved


# @pure
# @stack_array("left", "right")
@inline
def b_splines_slim_inline(tn: "float[:]", pn: "int", eta: "float", span: "int", values: "float[:]"):
    """
    Computes the values of pn + 1 non-vanishing B-splines at position eta.

    Parameters
    ----------
    tn : array
        Knots sequence.

    pn : int
        Polynomial degree of B-splines.

    eta : float
        Evaluation point.

    span : int
        Knot span index.

    values : array[float]
        Outout: values of pn + 1 non-vanishing B-Splines at location eta.
    """

    # Initialize variables left and right used for computing the values
    left = empty(pn, dtype=float)
    right = empty(pn, dtype=float)
    left[:] = 0.0
    right[:] = 0.0

    values[0] = 1.0

    for j in range(pn):
        left[j] = eta - tn[span - j]
        right[j] = tn[span + 1 + j] - eta
        saved = 0.0
        for r in range(j + 1):
            temp = values[r] / (right[r] + left[j - r])
            values[r] = saved + right[r] * temp
            saved = left[j - r] * temp
        values[j + 1] = saved


def evaluation_kernel_2d_inline(
    p1: int, p2: int, basis1: "float[:]", basis2: "float[:]", ind1: "int[:]", ind2: "int[:]", coeff: "float[:,:]"
) -> float:
    """
    Summing non-zero contributions of a spline function.

    Parameters
    ----------
        p1, p2 : int
            Degrees of the univariate splines in each direction.

        basis1, basis2 : array[float]
            The p + 1 values of non-zero basis splines at one point (eta1, eta2) in each direction.

        ind1, ind2 : array[int]
            Global indices of non-vanishing splines in the element of the considered point.

        coeff : array[float]
            The spline coefficients c_ij.

    Returns
    -------
        spline_value : float
            Value of tensor-product spline at point (eta1, eta2).
    """

    spline_value = 0.0

    for il1 in range(p1 + 1):
        i1 = ind1[il1]
        for il2 in range(p2 + 1):
            i2 = ind2[il2]

            spline_value += coeff[i1, i2] * basis1[il1] * basis2[il2]

    return spline_value


@inline
def evaluation_kernel_3d_inline(
    p1: int,
    p2: int,
    p3: int,
    basis1: "float[:]",
    basis2: "float[:]",
    basis3: "float[:]",
    ind1: "int[:]",
    ind2: "int[:]",
    ind3: "int[:]",
    coeff: "float[:,:,:]",
) -> float:
    """
    Summing non-zero contributions of a spline function (serial, needs global arrays).

    Parameters
    ----------
        p1, p2, p3 : int
            Degrees of the univariate splines in each direction.

        basis1, basis2, basis3 : array[float]
            The p + 1 values of non-zero basis splines at one point (eta1, eta2, eta3) in each direction.

        ind1, ind2, ind3 : array[int]
            Global indices of non-vanishing splines in the element of the considered point.

        coeff : array[float]
            The spline coefficients c_ijk.

    Returns
    -------
        spline_value : float
            Value of tensor-product spline at point (eta1, eta2, eta3).
    """

    spline_value = 0.0

    for il1 in range(p1 + 1):
        i1 = ind1[il1]
        for il2 in range(p2 + 1):
            i2 = ind2[il2]
            for il3 in range(p3 + 1):
                i3 = ind3[il3]

                spline_value = spline_value + coeff[i1, i2, i3] * basis1[il1] * basis2[il2] * basis3[il3]

    return spline_value

