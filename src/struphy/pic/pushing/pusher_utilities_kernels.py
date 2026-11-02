from numpy import arctan2, cos, empty, floor, mod, pi, shape, sin, sqrt, zeros
from pyccel.decorators import pure, stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.geometry.evaluation_kernels as evaluation_kernels

# do not remove; needed to identify dependencies
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "dfinv", "v", "v_logical", "reflected")
def apply_kinetic_bc_marker(
    ip: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    newton: bool,
):
    r"""Apply the kinetic boundary conditions to the single marker ``markers[:, ip]``,
    to be called at the end of a pusher kernel's particle loop, after the position update.
    Same result as :meth:`~struphy.pic.base.Particles.apply_kinetic_bc` for this marker.

    If the domain has a pole at :math:`\eta_1=0` (``args_markers.pole``), a marker with :math:`\eta_1 < 0`
    has crossed the axis and is mapped to :math:`(-\eta_1, \eta_2 + 1/2)`, with unchanged velocities;
    :math:`\eta_1` is kept above ``1e-10``, where the mapping is regular.
    The boundary condition of axis 0 then applies at :math:`\eta_1 = 1` only.

    The boundary condition of each axis is read from ``args_markers.bc_type``:

    * 0 (periodic): wrap :math:`\eta_i` into :math:`[0, 1)` and store the shift :math:`\pm 1`
      at ``first_shift_idx + axis`` (set for ``newton=False``, accumulated for ``newton=True``).
    * 1 (reflect): mirror :math:`\eta_i` at the boundary, flip the logical velocity component
      (only for Cartesian velocities, ``vdim=3``) and mark the marker as done for
      multi-stage/iterative pushers (``first_pusher_idx = -1``).
    * 2 (remove): turn the marker into a hole (all columns except the ID set to -1).
    * 3 (refill): skipped here, handled by :meth:`~struphy.pic.base.Particles.apply_kinetic_bc`.

    Parameters
    ----------
    ip : int
        Row index of the marker (must be neither a hole nor a ghost particle).

    args_markers : MarkerArguments
        Marker arguments, including ``bc_type``.

    args_domain : DomainArguments
        Mapping arguments (needed for reflection of the velocity).

    newton : bool
        Whether the shift is accumulated (Newton step) or overwritten (explicit or Picard step).
    """

    markers = args_markers.markers
    bc_type = args_markers.bc_type
    first_pusher_idx = args_markers.first_pusher_idx
    first_shift_idx = args_markers.first_shift_idx

    # pole at eta1=0
    if args_markers.pole:
        if markers[0, ip] < 0.0:
            markers[0, ip] = -markers[0, ip]
            markers[1, ip] = mod(markers[1, ip] + 0.5, 1.0)
        if markers[0, ip] < 1e-10:
            markers[0, ip] = 1e-10

    # remove
    for axis in range(3):
        if bc_type[axis] == 2:
            if markers[axis, ip] > 1.0 or markers[axis, ip] < 0.0:
                n_cols = shape(markers)[0]
                markers[0 : n_cols - 1, ip] = -1.0
                return

    # periodic
    for axis in range(3):
        if bc_type[axis] == 0:
            if markers[axis, ip] > 1.0:
                markers[axis, ip] = mod(markers[axis, ip], 1.0)
                if newton:
                    markers[first_shift_idx + axis, ip] += 1.0
                else:
                    markers[first_shift_idx + axis, ip] = 1.0
            elif markers[axis, ip] < 0.0:
                markers[axis, ip] = mod(markers[axis, ip], 1.0)
                if newton:
                    markers[first_shift_idx + axis, ip] += -1.0
                else:
                    markers[first_shift_idx + axis, ip] = -1.0
            elif not newton:
                markers[first_shift_idx + axis, ip] = 0.0

    # reflect positions
    reflected = zeros(3, dtype=bool)
    n_reflected = 0
    for axis in range(3):
        if bc_type[axis] == 1:
            if markers[axis, ip] > 1.0:
                markers[axis, ip] = 2.0 - markers[axis, ip]
                reflected[axis] = True
                n_reflected += 1
            elif markers[axis, ip] < 0.0:
                markers[axis, ip] = -markers[axis, ip]
                reflected[axis] = True
                n_reflected += 1

    if n_reflected == 0:
        return

    markers[first_pusher_idx, ip] = -1.0

    # only Cartesian velocities are reflected (columns 3:6 hold other data for vdim != 3)
    if args_markers.vdim != 3:
        return

    # reflect velocities (Jacobian at the reflected position)
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    v = empty(3, dtype=float)
    v_logical = empty(3, dtype=float)

    for axis in range(3):
        if not reflected[axis]:
            continue

        v[:] = markers[3:6, ip]

        evaluation_kernels.df(
            markers[0, ip],
            markers[1, ip],
            markers[2, ip],
            args_domain,
            dfm,
        )

        linalg_kernels.matrix_inv(dfm, dfinv)

        # pull-back, reverse, push-forward
        linalg_kernels.matrix_vector(dfinv, v, v_logical)
        v_logical[axis] *= -1.0
        linalg_kernels.matrix_vector(dfm, v_logical, v)

        markers[3:6, ip] = v[:]


def update_position_stage(
    ip: int,
    k: "float[:]",
    dt: float,
    a_stage: float,
    b_stage: float,
    last: float,
    args_markers: "MarkerArguments",
):
    r"""Position update of one stage of an explicit Runge-Kutta pusher for the marker ``markers[:, ip]``,
    to be called after the logical velocity :math:`\mathbf k = \dot{\boldsymbol \eta}` has been evaluated at
    the current position ``markers[0:3, ip]``.

    The stage contribution :math:`\Delta t\, b_\textrm{stage}\, \mathbf k` is accumulated at
    ``first_free_idx``, and the new position is

    .. math::

        \boldsymbol \eta = \boldsymbol \eta^n + \Delta t\, a_\textrm{stage}\, \mathbf k + \textrm{last} \sum_\textrm{stages} \Delta t\, b\, \mathbf k\,,

    with :math:`\boldsymbol \eta^n` at ``first_pusher_idx``.

    If the domain has a pole at :math:`\eta_1=0` (``args_markers.pole``), the update is done in the coordinates

    .. math::

        (\xi_1, \xi_2, \eta_3) = \big(\eta_1 \cos(2\pi\eta_2),\, \eta_1 \sin(2\pi\eta_2),\, \eta_3\big)\,,

    in which the mapping (and thus the velocity :math:`\dot{\boldsymbol \xi} = \partial \boldsymbol \xi / \partial \boldsymbol \eta\, \mathbf k`)
    is regular at the pole, and markers move through the axis. The accumulated stages are then :math:`\dot{\boldsymbol \xi}`,
    and the position is transformed back with :math:`\eta_1 = |\boldsymbol \xi|` (kept above ``1e-10``)
    and :math:`\eta_2 = \arctan(\xi_2 / \xi_1) / 2\pi \in [0, 1)`.

    Parameters
    ----------
    ip : int
        Row index of the marker.

    k : array[float]
        Logical velocity at the current position.

    dt : float
        Time step.

    a_stage : float
        Butcher coefficient of the position update of this stage.

    b_stage : float
        Butcher weight of this stage.

    last : float
        1.0 for the last stage (adds the accumulated stages), 0.0 otherwise.

    args_markers : MarkerArguments
        Marker arguments.
    """

    markers = args_markers.markers
    first_pusher_idx = args_markers.first_pusher_idx
    first_free_idx = args_markers.first_free_idx

    if not args_markers.pole:
        # accumulation for last stage
        markers[first_free_idx : first_free_idx + 3, ip] += dt * b_stage * k

        # update positions for intermediate stages or last stage
        markers[0:3, ip] = (
            markers[first_pusher_idx : first_pusher_idx + 3, ip]
            + dt * a_stage * k
            + last * markers[first_free_idx : first_free_idx + 3, ip]
        )
        return

    # velocity in the regular coordinates (xi1, xi2), at the current position
    theta = 2.0 * pi * markers[1, ip]
    k_xi1 = cos(theta) * k[0] - 2.0 * pi * markers[0, ip] * sin(theta) * k[1]
    k_xi2 = sin(theta) * k[0] + 2.0 * pi * markers[0, ip] * cos(theta) * k[1]

    # accumulation for last stage
    markers[first_free_idx, ip] += dt * b_stage * k_xi1
    markers[first_free_idx + 1, ip] += dt * b_stage * k_xi2
    markers[first_free_idx + 2, ip] += dt * b_stage * k[2]

    # update positions for intermediate stages or last stage
    theta = 2.0 * pi * markers[first_pusher_idx + 1, ip]
    xi1 = markers[first_pusher_idx, ip] * cos(theta) + dt * a_stage * k_xi1 + last * markers[first_free_idx, ip]
    xi2 = markers[first_pusher_idx, ip] * sin(theta) + dt * a_stage * k_xi2 + last * markers[first_free_idx + 1, ip]

    markers[0, ip] = sqrt(xi1**2 + xi2**2)
    if markers[0, ip] < 1e-10:
        markers[0, ip] = 1e-10
    markers[1, ip] = mod(arctan2(xi2, xi1) / (2.0 * pi), 1.0)
    markers[2, ip] = markers[first_pusher_idx + 2, ip] + dt * a_stage * k[2] + last * markers[first_free_idx + 2, ip]
