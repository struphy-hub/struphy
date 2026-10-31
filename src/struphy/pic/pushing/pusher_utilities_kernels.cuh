#pragma once
// NVRTC provides the device math function floor without a host math.h header.
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
namespace struphy_cuda::pusher_utilities_kernels {
/**
 * Reflect one marker's velocity using the Pyccel boundary helper's calculation.
 *
 * @param ip Marker index.
 * @param markers Marker array view.
 * @param args_domain Mapping arguments for the Jacobian at the reflected position.
 * @param axis Logical velocity component to reverse (0, 1 or 2).
 *
 * This CUDA-only helper extracts the velocity reflection from
 * pusher_utilities_kernels.apply_kinetic_bc_marker. dfm, dfinv, v and
 * v_logical retain the Pyccel names. Pull the velocity back to logical
 * coordinates, reverse its axis component, then push it forward again.
 */
__device__ inline void reflect_velocity(long long ip, Array2D<double> markers,
                                       const DomainArgs& args_domain, int axis) {
    double dfm[9], dfinv[9], v[3], v_logical[3];
    evaluation_kernels::df(markers(0, ip), markers(1, ip), markers(2, ip), args_domain, dfm);
    linalg_kernels::matrix_inv(dfm, dfinv);
    for (int j = 0; j < 3; ++j) v[j] = markers(3 + j, ip);
    linalg_kernels::matrix_vector(dfinv, v, v_logical);
    v_logical[axis] *= -1.;
    linalg_kernels::matrix_vector(dfm, v_logical, v);
    for (int j = 0; j < 3; ++j) markers(3 + j, ip) = v[j];
}

/**
 * Prepare marker ip for a push, as the kernel prepare_push (prepare_push_cuda.cu) does for every marker: copy the
 * phase space coordinates (rows 0 to 3 + vdim of the buffer) to the rows from first_pusher_idx and zero the rows after
 * them up to n_cols - 2. Called at stage 0 by the push kernels that take over the preparation from prepare_push
 * (args_markers.prepare_rows, see Pusher._push), for every marker including holes, so that it is exactly the same.
 */
__device__ inline void prepare_marker_row(int ip, const MarkerArgs& args_markers) {
    const int first = args_markers.first_pusher_idx;
    const int n_init = 3 + args_markers.vdim;
    const int end = (int)args_markers.markers.shape[0] - 2;
    for (int j = first; j < end; ++j)
        args_markers.markers(j, ip) = j < first + n_init ? args_markers.markers(j - first, ip) : 0.0;
}

/**
 * Apply kinetic boundary conditions, as in pusher_utilities_kernels.apply_kinetic_bc_marker.
 *
 * @param ip Row index of a marker that is neither a hole nor a ghost.
 * @param args_markers Marker buffer, boundary types and bookkeeping columns.
 * @param args_domain Mapping arguments needed for velocity reflection.
 * @param newton Accumulate periodic shifts if true; overwrite them otherwise.
 *
 * Call after updating the position. With a pole at eta1=0, a marker with
 * eta1 < 0 is mapped to (-eta1, eta2 + 1/2) first. Boundary type 0 wraps periodically,
 * type 1 mirrors position and logical velocity (for vdim=3) and sets first_pusher_idx to -1,
 * type 2 clears all columns except the ID to -1, and type 3 is handled on
 * the host. The marker view carries shape and strides; j is a CUDA-only
 * column index replacing Pyccel array slices.
 */
__device__ inline void apply_kinetic_bc_marker(int ip, const MarkerArgs& args_markers,
                                             const DomainArgs& args_domain, bool newton) {
    const long long* bc_type = args_markers.bc_type;
    int first_pusher_idx = args_markers.first_pusher_idx;
    int first_shift_idx = args_markers.first_shift_idx;

    // Pole at eta1=0: a marker with eta1 < 0 has crossed the axis.
    if (args_markers.pole) {
        if (args_markers.markers(0, ip) < 0.) {
            args_markers.markers(0, ip) = -args_markers.markers(0, ip);
            double e2 = args_markers.markers(1, ip) + 0.5;
            args_markers.markers(1, ip) = e2 - floor(e2);
        }
        if (args_markers.markers(0, ip) < 1e-10) args_markers.markers(0, ip) = 1e-10;
    }

    // Remove markers before applying periodic or reflecting boundaries.
    for (int axis = 0; axis < 3; ++axis) {
        if (bc_type[axis] == 2 &&
            (args_markers.markers(axis, ip) > 1. || args_markers.markers(axis, ip) < 0.)) {
            int n_cols = args_markers.markers.shape[0];
            for (int j = 0; j < n_cols - 1; ++j) args_markers.markers(j, ip) = -1.;
            return;
        }
    }

    // Wrap positions and update the shift exactly as in the Pyccel helper.
    for (int axis = 0; axis < 3; ++axis) {
        if (bc_type[axis] == 0) {
            if (args_markers.markers(axis, ip) > 1.) {
                args_markers.markers(axis, ip) -= floor(args_markers.markers(axis, ip));
                if (newton) args_markers.markers(first_shift_idx + axis, ip) += 1.;
                else args_markers.markers(first_shift_idx + axis, ip) = 1.;
            } else if (args_markers.markers(axis, ip) < 0.) {
                args_markers.markers(axis, ip) -= floor(args_markers.markers(axis, ip));
                if (newton) args_markers.markers(first_shift_idx + axis, ip) += -1.;
                else args_markers.markers(first_shift_idx + axis, ip) = -1.;
            } else if (!newton) {
                args_markers.markers(first_shift_idx + axis, ip) = 0.;
            }
        }
    }

    bool reflected[3] = {false, false, false};
    int n_reflected = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (bc_type[axis] == 1) {
            if (args_markers.markers(axis, ip) > 1.) {
                args_markers.markers(axis, ip) = 2. - args_markers.markers(axis, ip);
                reflected[axis] = true;
                ++n_reflected;
            } else if (args_markers.markers(axis, ip) < 0.) {
                args_markers.markers(axis, ip) = -args_markers.markers(axis, ip);
                reflected[axis] = true;
                ++n_reflected;
            }
        }
    }
    if (n_reflected == 0) return;
    args_markers.markers(first_pusher_idx, ip) = -1.;
    // Only Cartesian velocities are reflected (columns 3:6 hold other data for vdim != 3).
    if (args_markers.vdim != 3) return;
    for (int axis = 0; axis < 3; ++axis) {
        if (reflected[axis]) reflect_velocity(ip, args_markers.markers, args_domain, axis);
    }
}

/**
 * Position update of one stage of an explicit Runge-Kutta pusher, as in pusher_utilities_kernels.update_position_stage.
 *
 * @param ip Row index of the marker; call after k has been evaluated at markers(0:3, ip).
 * @param k Logical velocity d eta / dt at the current position (three entries).
 * @param dt Time step.
 * @param a_stage Butcher coefficient of the position update of this stage.
 * @param b_stage Butcher weight of this stage.
 * @param last 1.0 for the last stage (adds the accumulated stages), 0.0 otherwise.
 * @param args_markers Marker buffer; the stages are accumulated at first_free_idx, eta^n is at first_pusher_idx.
 *
 * Without a pole, eta = eta^n + dt a_stage k + last * (sum of dt b k), with dt b_stage k accumulated at first_free_idx.
 * With a pole at eta1 = 0 (args_markers.pole) the update is done in the regular coordinates
 * (xi1, xi2) = eta1 (cos 2 pi eta2, sin 2 pi eta2) and transformed back with eta1 = |xi| (at least 1e-10) and
 * eta2 = mod(atan2(xi2, xi1) / 2 pi, 1).
 */
__device__ inline void update_position_stage(int ip, const double* k, double dt, double a_stage, double b_stage,
                                             double last, const MarkerArgs& args_markers) {
    int first_pusher_idx = args_markers.first_pusher_idx;
    int first_free_idx = args_markers.first_free_idx;

    if (!args_markers.pole) {
        for (int j = 0; j < 3; ++j) {
            // accumulation for last stage
            args_markers.markers(first_free_idx + j, ip) += dt * b_stage * k[j];
            // update positions for intermediate stages or last stage
            args_markers.markers(j, ip) = args_markers.markers(first_pusher_idx + j, ip) + dt * a_stage * k[j] +
                                          last * args_markers.markers(first_free_idx + j, ip);
        }
        return;
    }

    // velocity in the regular coordinates (xi1, xi2), at the current position
    double theta = 2.0 * pi * args_markers.markers(1, ip);
    double k_xi1 = cos(theta) * k[0] - 2.0 * pi * args_markers.markers(0, ip) * sin(theta) * k[1];
    double k_xi2 = sin(theta) * k[0] + 2.0 * pi * args_markers.markers(0, ip) * cos(theta) * k[1];

    // accumulation for last stage
    args_markers.markers(first_free_idx, ip) += dt * b_stage * k_xi1;
    args_markers.markers(first_free_idx + 1, ip) += dt * b_stage * k_xi2;
    args_markers.markers(first_free_idx + 2, ip) += dt * b_stage * k[2];

    // update positions for intermediate stages or last stage
    theta = 2.0 * pi * args_markers.markers(first_pusher_idx + 1, ip);
    double xi1 = args_markers.markers(first_pusher_idx, ip) * cos(theta) + dt * a_stage * k_xi1 +
                 last * args_markers.markers(first_free_idx, ip);
    double xi2 = args_markers.markers(first_pusher_idx, ip) * sin(theta) + dt * a_stage * k_xi2 +
                 last * args_markers.markers(first_free_idx + 1, ip);

    args_markers.markers(0, ip) = sqrt(xi1 * xi1 + xi2 * xi2);
    if (args_markers.markers(0, ip) < 1e-10) args_markers.markers(0, ip) = 1e-10;
    // numpy.mod(x, 1.0) (Fortran MODULO): the result has the sign of the divisor
    double e2 = atan2(xi2, xi1) / (2.0 * pi);
    args_markers.markers(1, ip) = e2 - floor(e2);
    args_markers.markers(2, ip) = args_markers.markers(first_pusher_idx + 2, ip) + dt * a_stage * k[2] +
                                  last * args_markers.markers(first_free_idx + 2, ip);
}
}  // namespace struphy_cuda::pusher_utilities_kernels
