"Pusher kernels for full orbit (6D) particles and AMReX data structures."

from numpy import shape, array, newaxis, matmul
from struphy.pic.base import Particles


def push_eta_stage(
    dt: float,
    stage: int,
    particles: "Particles",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    """

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    markers_array = particles.markers.get_particles(0)[(0, 0)].get_struct_of_arrays().to_numpy().real

    e1 = markers_array["x"][:]
    e2 = markers_array["y"][:]
    e3 = markers_array["z"][:]
    v1 = markers_array["v1"][:]
    v2 = markers_array["v2"][:]
    v3 = markers_array["v3"][:]

    # evaluate inverse Jacobian matrices for each point
    etas = array([e1, e2, e3]).T.copy()  # needed for c kernels
    jacobian_inv = particles.domain.jacobian_inv(etas, change_out_order=True)  # Npx3x3

    # pull-back of velocity
    v = array([v1, v2, v3]).T
    v = v[..., newaxis]  # Npx3x1
    # If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly. Squeeze to take away the unnecessary 1 dim
    k = matmul(jacobian_inv, v).squeeze()

    # accumulation for last stage
    temp = dt * b[stage] * k
    markers_array["real_comp0"][:] = markers_array["real_comp0"][:] + temp[:, 0]
    markers_array["real_comp1"][:] = markers_array["real_comp1"][:] + temp[:, 1]
    markers_array["real_comp2"][:] = markers_array["real_comp2"][:] + temp[:, 2]

    # update positions for intermediate stages or last stage
    temp = dt * a[stage] * k
    markers_array["x"][:] = (
        markers_array["real_comp0"][:]
        + temp[:, 0]
        + last*markers_array["real_comp0"][:]
    )
    markers_array["y"][:] = (
        markers_array["real_comp1"][:]
        + temp[:, 1]
        + last*markers_array["real_comp1"][:]
    )
    markers_array["z"][:] = (
        markers_array["real_comp2"][:]
        + temp[:, 2]
        + last*markers_array["real_comp2"][:]
    )
