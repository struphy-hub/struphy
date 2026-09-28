from numpy import empty, floor, mod, shape, sqrt, zeros
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
    r"""Apply the kinetic boundary conditions to the single marker ``markers[ip]``,
    to be called at the end of a pusher kernel's particle loop, after the position update.
    Same result as :meth:`~struphy.pic.base.Particles.apply_kinetic_bc` for this marker.

    The boundary condition of each axis is read from ``args_markers.bc_type``:

    * 0 (periodic): wrap :math:`\eta_i` into :math:`[0, 1)` and store the shift :math:`\pm 1`
      at ``first_shift_idx + axis`` (set for ``newton=False``, accumulated for ``newton=True``).
    * 1 (reflect): mirror :math:`\eta_i` at the boundary, flip the logical velocity component
      and mark the marker as done for multi-stage/iterative pushers (``first_init_idx = -1``).
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
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx

    # remove
    for axis in range(3):
        if bc_type[axis] == 2:
            if markers[ip, axis] > 1.0 or markers[ip, axis] < 0.0:
                n_cols = shape(markers)[1]
                markers[ip, 0 : n_cols - 1] = -1.0
                return

    # periodic
    for axis in range(3):
        if bc_type[axis] == 0:
            if markers[ip, axis] > 1.0:
                markers[ip, axis] = mod(markers[ip, axis], 1.0)
                if newton:
                    markers[ip, first_shift_idx + axis] += 1.0
                else:
                    markers[ip, first_shift_idx + axis] = 1.0
            elif markers[ip, axis] < 0.0:
                markers[ip, axis] = mod(markers[ip, axis], 1.0)
                if newton:
                    markers[ip, first_shift_idx + axis] += -1.0
                else:
                    markers[ip, first_shift_idx + axis] = -1.0
            elif not newton:
                markers[ip, first_shift_idx + axis] = 0.0

    # reflect positions
    reflected = zeros(3, dtype=bool)
    n_reflected = 0
    for axis in range(3):
        if bc_type[axis] == 1:
            if markers[ip, axis] > 1.0:
                markers[ip, axis] = 2.0 - markers[ip, axis]
                reflected[axis] = True
                n_reflected += 1
            elif markers[ip, axis] < 0.0:
                markers[ip, axis] = -markers[ip, axis]
                reflected[axis] = True
                n_reflected += 1

    if n_reflected == 0:
        return

    markers[ip, first_init_idx] = -1.0

    # reflect velocities (Jacobian at the reflected position)
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    v = empty(3, dtype=float)
    v_logical = empty(3, dtype=float)

    for axis in range(3):
        if not reflected[axis]:
            continue

        v[:] = markers[ip, 3:6]

        evaluation_kernels.df(
            markers[ip, 0],
            markers[ip, 1],
            markers[ip, 2],
            args_domain,
            dfm,
        )

        linalg_kernels.matrix_inv(dfm, dfinv)

        # pull-back, reverse, push-forward
        linalg_kernels.matrix_vector(dfinv, v, v_logical)
        v_logical[axis] *= -1.0
        linalg_kernels.matrix_vector(dfm, v_logical, v)

        markers[ip, 3:6] = v[:]


@stack_array("dfm", "dfinv", "eta", "v", "v_logical")
def reflect(
    markers: "float[:,:]",
    args_domain: "DomainArguments",
    outside_inds: "int[:]",
    axis: "int",
):
    r"""
    Reflect the particles which are pushed outside of the logical cube.

    .. math::

        \hat{v} = DF^{-1} v \,, \\
        \hat{v}_\text{reflected}[\text{axis}] = -1 * \hat{v} \,, \\
        v_\text{reflected} = DF \hat{v}_\text{reflected} \,.

    Parameters
    ----------
        markers : array[float]
            Local markers array

        args_domain : DomainArguments
            kind_map, params_map, ..., cx, cy, cz

        outside_inds : array[int]
            inds indicate the particles which are pushed outside of the local cube

        axis : int
            0, 1 or 2
    """

    # allocate metric coeffs
    dfm = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)
    v_logical = empty(3, dtype=float)

    for ip in outside_inds:
        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta[0],
            eta[1],
            eta[2],
            args_domain,
            dfm,
        )

        linalg_kernels.matrix_inv(dfm, dfinv)

        # pull back of the velocity
        linalg_kernels.matrix_vector(dfinv, v, v_logical)

        # reverse the velocity
        v_logical[axis] *= -1

        # push forwward of the velocity
        linalg_kernels.matrix_vector(dfm, v_logical, v)

        # update the particle velocities
        markers[ip, 3:6] = v[:]
