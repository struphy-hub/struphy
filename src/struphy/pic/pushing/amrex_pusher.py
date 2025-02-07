"Pusher kernels for full orbit (6D) particles and AMReX data structures."

from numpy import cos, empty, floor, log, shape, sin, sqrt, zeros
from pyccel.decorators import stack_array

import amrex.space3d as amr

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels

# do not remove; needed to identify dependencies
import struphy.pic.pushing.pusher_args_kernels as pusher_args_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    eval_3form_spline_mpi,
    eval_vectorfield_spline_mpi,
    get_spans,
)
from struphy.pic.pushing.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments
from struphy.pic.base import Particles


def push_eta_stage(
    dt: float,
    stage: int,
    particles: "Particles",
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

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    markers_array = particles.markers.get_particles(
        0)[(0, 0)].get_struct_of_arrays().to_numpy().real

    for pti in particles.markers.iterator(particles.markers, 0):
        soa = pti.soa()
        soa_arr = soa.to_numpy()[0] # to_xp()?
        for ip in range(soa_arr["x"].size):
            e1 = soa_arr["x"][:] # use this for array operations
            e2 = soa_arr["y"][ip]
            e3 = soa_arr["z"][ip]
            v[0] = soa_arr["a"][ip]
            v[1] = soa_arr["b"][ip]
            v[2] = soa_arr["c"][ip]

            # evaluate Jacobian, result in dfm
            evaluation_kernels.df( # TODO: use the vector version
                e1,
                e2,
                e3,
                args_domain,
                dfm,
            )

            # evaluate inverse Jacobian matrix
            linalg_kernels.matrix_inv(dfm, dfinv) # TODO: use the vector version

            # pull-back of velocity
            linalg_kernels.matrix_vector(dfinv, v, k) # TODO: use matrix vector

            # accumulation for last stage
            temp = dt * b[stage] * k
            markers_array["g"][ip] = markers_array["g"][ip] + temp[0]
            markers_array["h"][ip] = markers_array["h"][ip] + temp[1]
            markers_array["i"][ip] = markers_array["i"][ip] + temp[2]

            # update positions for intermediate stages or last stage
            markers_array["x"][ip] = markers_array["g"][ip] + temp[0]
            markers_array["y"][ip] = markers_array["h"][ip] + temp[1]
            markers_array["z"][ip] = markers_array["i"][ip] + temp[2]
            # markers[ip, 0:3] = (
            #     markers[ip, first_init_idx : first_init_idx + 3]
            #     + dt * a[stage] * k
            #     + last * markers[ip, first_free_idx : first_free_idx + 3]
            # )

    # #$ omp parallel private(ip, e, v, dfm, dfinv, k)
    # #$ omp for
    # for ip in range(n_markers):
    #     # check if marker is a hole
    #     if markers[ip, first_init_idx] == -1.0:
    #         continue

    #     e1 = markers[ip, 0]
    #     e2 = markers[ip, 1]
    #     e3 = markers[ip, 2]
    #     v[:] = markers[ip, 3:6]

    #     # evaluate Jacobian, result in dfm
    #     evaluation_kernels.df(
    #         e1,
    #         e2,
    #         e3,
    #         args_domain,
    #         dfm,
    #     )

    #     # evaluate inverse Jacobian matrix
    #     linalg_kernels.matrix_inv(dfm, dfinv)

    #     # pull-back of velocity
    #     linalg_kernels.matrix_vector(dfinv, v, k)

    #     # accumulation for last stage
    #     markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

    #     # update positions for intermediate stages or last stage
    #     markers[ip, 0:3] = (
    #         markers[ip, first_init_idx : first_init_idx + 3]
    #         + dt * a[stage] * k
    #         + last * markers[ip, first_free_idx : first_free_idx + 3]
    #     )

    # #$ omp end parallel
