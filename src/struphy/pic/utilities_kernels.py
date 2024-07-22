from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.pic.pushing.pusher_args_kernels as pusher_args_kernels # do not remove; needed to identify dependencies

from struphy.pic.pushing.pusher_args_kernels import DerhamArguments, DomainArguments
from struphy.bsplines.evaluation_kernels_3d import get_spans, eval_0form_spline_mpi, eval_1form_spline_mpi, eval_2form_spline_mpi, eval_3form_spline_mpi, eval_vectorfield_spline_mpi

from numpy import empty, shape, zeros, sqrt, log, abs, sign


@stack_array('b_cart', 'norm_b_cart', 'v', 'temp', 'v_perp')
def eval_magnetic_moment_6d(markers: 'float[:,:]',
                            args_derham: 'DerhamArguments',
                            b_cart_1: 'float[:,:,:]',
                            b_cart_2: 'float[:,:,:]',
                            b_cart_3: 'float[:,:,:]'):
    """
    Evaluate parallel velocity and magnetic moment of each particles 
    and assign it into markers[ip, 3] and markers[ip, 4], respectively. 
    """

    b_cart = empty(3, dtype=float)
    norm_b_cart = empty(3, dtype=float)
    v = empty(3, dtype=float)
    temp = empty(3, dtype=float)
    v_perp = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v[:] = markers[ip, 3:6]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b_cart, TODO: why vector-field?
        eval_vectorfield_spline_mpi(span1, span2, span3,
                                    args_derham,
                                    b_cart_1,
                                    b_cart_2,
                                    b_cart_3,
                                    b_cart)

        # calculate absB
        absB = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)

        if absB != 0.:
            norm_b_cart[:] = b_cart/absB
        else:
            norm_b_cart[:] = b_cart

        # calculate parallel velocity
        v_parallel = linalg_kernels.scalar_dot(norm_b_cart, v)

        # extract perpendicular velocity
        linalg_kernels.cross(v, norm_b_cart, temp)
        linalg_kernels.cross(norm_b_cart, temp, v_perp)

        v_perp_square = (v_perp[0]**2 + v_perp[1]**2 + v_perp[2]**2)

        # parallel velocity
        markers[ip, 3] = v_parallel
        # magnetic moment
        markers[ip, 4] = 1/2 * v_perp_square / absB
        # empty leftovers
        markers[ip, 5] = 0.


def eval_magnetic_moment_5d(markers: 'float[:,:]',
                            args_derham: 'DerhamArguments',
                            absB: 'float[:,:,:]'):
    """
    Evaluate parallel velocity and magnetic moment of each particles 
    and assign it into markers[ip,3] and markers[ip,4] respectively.
    """

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v_perp = markers[ip, 4]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        abs_B = eval_0form_spline_mpi(span1, span2, span3,
                                      args_derham,
                                      absB)

        # magnetic moment
        markers[ip, 9] = 1/2 * v_perp**2 / abs_B


def eval_energy_5d(markers: 'float[:,:]',
                   args_derham: 'DerhamArguments',
                   absB: 'float[:,:,:]'):
    """
    Evaluate total energy of each particles and assign it into markers[ip,10].
    Only equilibrium magnetic field is considered.
    """

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v_parallel = markers[ip, 3]
        mu = markers[ip, 9]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        abs_B = eval_0form_spline_mpi(span1, span2, span3,
                                      args_derham,
                                      absB)

        # total energy
        markers[ip, 8] = 1/2*v_parallel**2 + mu*abs_B


def eval_canonical_toroidal_moment_5d(markers: 'float[:,:]',
                                      args_derham: 'DerhamArguments',
                                      epsilon: float, 
                                      B0: float, 
                                      R0: float,
                                      absB: 'float[:,:,:]'):
    """
    Evaluate canonical toroidal momentum of each particles and assign it into markers[ip,11].
    Only equilibrium magnetic field is considered.   
    """

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v_para = markers[ip, 3]
        mu = markers[ip, 9]
        energy = markers[ip, 8]
        psi = markers[ip, 10]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        abs_B = eval_0form_spline_mpi(span1, span2, span3,
                                      args_derham,
                                      absB)

        # shifted canonical toroidal momentum
        markers[ip, 10] = psi - epsilon*B0*R0/abs_B*v_para

        if energy-mu*B0 >0:
            markers[ip, 10] += epsilon*sign(v_para)*sqrt(2*(energy-mu*B0))*R0


@stack_array('dfm', 'norm_b1', 'b')
def eval_magnetic_background_energy(markers: 'float[:,:]',
                                    args_derham: 'DerhamArguments',
                                    args_domain: 'DomainArguments',
                                    abs_B0: 'float[:,:,:]'):
    r"""
    Evaluate :math:`mu_p |B_0(\boldsymbol \eta_p)|` for each marker.
    The result is stored at markers[:, 8].
    """

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, mu, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, abs_B)
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        mu = markers[ip, 4]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # abs_B0; 0form
        abs_B = eval_0form_spline_mpi(span1, span2, span3,
                                      args_derham,
                                      abs_B0)

        markers[ip, 8] = mu*abs_B

    #$ omp end parallel


@stack_array('dfm', 'norm_b1', 'b')
def eval_magnetic_energy(markers: 'float[:,:]',
                         args_derham: 'DerhamArguments',
                         args_domain: 'DomainArguments',
                         abs_B0: 'float[:,:,:]',
                         norm_b11: 'float[:,:,:]',
                         norm_b12: 'float[:,:,:]',
                         norm_b13: 'float[:,:,:]',
                         b1: 'float[:,:,:]',
                         b2: 'float[:,:,:]',
                         b3: 'float[:,:,:]'):
    """
    Evaluate magnetic field energy of each particles

    TODO: explain what is calculated here!!!
    """
    norm_b1 = empty(3, dtype=float)
    b = empty(3, dtype=float)

    dfm = empty((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, mu, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_para, abs_B, norm_b1, dfm, det_df)
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        mu = markers[ip, 9]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        det_df = linalg_kernels.det(dfm)

        # abs_B0; 0form
        abs_B = eval_0form_spline_mpi(span1, span2, span3,
                                      args_derham,
                                      abs_B0)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              b1,
                              b2,
                              b3,
                              b)
        
        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b11,
                              norm_b12,
                              norm_b13,
                              norm_b1)

        b_para = linalg_kernels.scalar_dot(norm_b1, b)
        b_para /= det_df

        markers[ip, 8] = mu*(abs_B + b_para)

    #$ omp end parallel


@stack_array('grad_PB', 'tmp')
def accum_gradI_const(markers: 'float[:,:]', n_markers_tot: 'int',
                      args_derham: 'DerhamArguments',
                      grad_PB1: 'float[:,:,:]', grad_PB2: 'float[:,:,:]', grad_PB3: 'float[:,:,:]',
                      scale: 'float'):
    r"""TODO
    """
    # allocate for magnetic field evaluation
    grad_PB = empty(3, dtype=float)
    tmp = empty(3, dtype=float)

    # allocate for filling
    res = zeros(1, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]  # mid
        eta2 = markers[ip, 1]  # mid
        eta3 = markers[ip, 2]  # mid

        # marker weight and velocity
        weight = markers[ip, 5]
        mu = markers[ip, 9]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # grad_PB; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_PB1,
                              grad_PB2,
                              grad_PB3,
                              grad_PB)

        tmp[:] = markers[ip, 15:18]
        res += linalg_kernels.scalar_dot(tmp, grad_PB) * weight * mu * scale

    return res/n_markers_tot


def accum_en_fB(markers: 'float[:,:]',
                n_markers_tot: 'int',
                args_derham: 'DerhamArguments',
                PB: 'float[:,:,:]'):
    r"""TODO
    """

    # allocate for filling
    res = zeros(1, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        mu = markers[ip, 9]
        weight = markers[ip, 5]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        B0 = eval_0form_spline_mpi(span1, span2, span3,
                                      args_derham,
                                      PB)

        res += abs(B0)*mu*weight

    return res/n_markers_tot


@stack_array('e', 'e_diff')
def check_eta_diff(markers: 'float[:,:]'):
    r'''TODO
    '''
    # marker position e
    e = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 9:12]

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        markers[ip, 15:18] = e_diff[:]


@stack_array('e', 'e_diff')
def check_eta_diff2(markers: 'float[:,:]'):
    r'''TODO
    '''
    # marker position e
    e = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 12:15]

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        markers[ip, 15:18] = e_diff[:]


@stack_array('e', 'e_diff', 'e_mid')
def check_eta_mid(markers: 'float[:,:]'):
    r'''TODO
    '''
    # marker position e
    e = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)
    e_mid = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        markers[ip, 12:15] = e[:]

        e_diff[:] = e[:] - markers[ip, 9:12]
        e_mid[:] = (e[:] + markers[ip, 9:12])/2.

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_mid[axis] += 0.5
            elif e_diff[axis] < -0.5:
                e_mid[axis] += 0.5

        markers[ip, 0:3] = e_mid[:]


@stack_array('dfm', 'dfinv', 'dfinv_t', 'v', 'a_form', 'dfta_form')
def canonical_kinetic_particles(res: 'float[:]', markers: 'float[:,:]',
                                args_derham: 'DerhamArguments',
                                args_domain: 'DomainArguments',
                                a1_1: 'float[:,:,:]', a1_2: 'float[:,:,:]', a1_3: 'float[:,:,:]'):
    r'''
    Calculate kinetic energy of each particle and sum up the result.

    .. math:: 
    
        \frac{1}{2} \sum_p w_p |{\mathbf p} -  \hat{\mathbf A}^1({\boldsymbol \eta}_p)|^2.
    '''

    res[:] = 0.0
    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations (1-form components)
    a_form = empty(3, dtype=float)
    dfta_form = empty(3, dtype=float)
    # particle position and velocity
    v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, e, v, w, dfm, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, a_form, dfta_form)
    #$ omp for reduction( + : res)
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]  
        eta2 = markers[ip, 1]  
        eta3 = markers[ip, 2]  
        v[:] = markers[ip, 3:6]
        w = markers[ip,   6]
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # vector potential: 1-form components
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              a1_1,
                              a1_2,
                              a1_3,
                              a_form)

        dfta_form[0] = dfinv_t[0, 0] * a_form[0] + \
            dfinv_t[0, 1] * a_form[1] + dfinv_t[0, 2] * a_form[2]
        dfta_form[1] = dfinv_t[1, 0] * a_form[0] + \
            dfinv_t[1, 1] * a_form[1] + dfinv_t[1, 2] * a_form[2]
        dfta_form[2] = dfinv_t[2, 0] * a_form[0] + \
            dfinv_t[2, 1] * a_form[1] + dfinv_t[2, 2] * a_form[2]

        res[0] += 0.5 * w * ((v[0] - dfta_form[0]) ** 2.0 +
                             (v[1] - dfta_form[1]) ** 2.0 + (v[2] - dfta_form[2]) ** 2.0)

    #$ omp end parallel


@stack_array('det_df', 'dfm')
def thermal_energy(res: 'float[:]', density: 'float[:,:,:,:,:,:]',
                   pads1: int, pads2: int, pads3: int,
                   nel1: 'int', nel2: 'int', nel3: 'int',
                   nq1: int, nq2: int, nq3: int,
                   w1: 'float[:,:]', w2: 'float[:,:]', w3: 'float[:,:]',
                   pts1: 'float[:,:]', pts2: 'float[:,:]', pts3: 'float[:,:]',
                   args_domain: 'DomainArguments',):
    r'''
    Calculate thermal energy of electron.

    Parameters
    ----------
        res : array[float]
            array to store the thermal energy of electrons

        density : array[float]
            array to store values of density at quadrature points in each cell

        pads1 - pads3 : int
            size of ghost region in each direction

        nel1 - nel3 : array[int]
            number of cells in each direction

        nq1 - nq3 : array[int]
            number of quadrature points in each direction of each cell

        w1 - w3: array[float]
            quadrature weights in each cell 

        pts1 - pts3: array[float]
            quadrature points in each cell 

        starts1 : array[int]
            starts of the stencil objects

        kind_map ->  cz:
            domain information

    .. math:: 
        \begin{align*}
            \int \hat{n}^0 \ln \hat{n}^0 \sqrt{g} \mathrm{d}{\boldsymbol \eta}.
        \end{align*}
    '''

    res[:] = 0.0

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    #$ omp parallel private (iel1, iel2, iel3, q1, q2, q3, eta1, eta2, eta3, wvol, vv, dfm, det_df)
    #$ omp for reduction( + : res)
    for iel1 in range(nel1):
        for iel2 in range(nel2):
            for iel3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            eta1 = pts1[iel1, q1]
                            eta2 = pts2[iel2, q2]
                            eta3 = pts3[iel3, q3]

                            wvol = w1[iel1, q1] * w2[iel2, q2] * w3[iel3, q3]

                            vv = density[pads1 + iel1, pads2 +
                                         iel2, pads3 + iel3, q1, q2, q3]

                            if abs(vv) < 0.00001:
                                vv = 1.0

                            # evaluate Jacobian, result in dfm
                            evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

                            det_df = linalg_kernels.det(dfm)

                            res[0] += vv * det_df * log(vv) * wvol

    #$ omp end parallel
