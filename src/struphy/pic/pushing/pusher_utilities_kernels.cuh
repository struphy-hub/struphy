#pragma once
// NVRTC provides the device math function floor without a host math.h header.
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
namespace struphy_cuda {
/**
 * Reflect one marker's velocity using the Pyccel boundary helper's calculation.
 *
 * @param ip Marker row index.
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
    df(markers(ip, 0), markers(ip, 1), markers(ip, 2),
       args_domain, dfm);
    matrix_inv(dfm, dfinv);
    for (int j = 0; j < 3; ++j) v[j] = markers(ip, 3 + j);
    matrix_vector(dfinv, v, v_logical);
    v_logical[axis] *= -1.;
    matrix_vector(dfm, v_logical, v);
    for (int j = 0; j < 3; ++j) markers(ip, 3 + j) = v[j];
}

/**
 * Apply kinetic boundary conditions, as in pusher_utilities_kernels.apply_kinetic_bc_marker.
 *
 * @param ip Row index of a marker that is neither a hole nor a ghost.
 * @param args_markers Marker buffer, boundary types and bookkeeping columns.
 * @param args_domain Mapping arguments needed for velocity reflection.
 * @param newton Accumulate periodic shifts if true; overwrite them otherwise.
 *
 * Call after updating the position. Boundary type 0 wraps periodically,
 * type 1 mirrors position and logical velocity and sets first_init_idx to -1,
 * type 2 clears all columns except the ID to -1, and type 3 is handled on
 * the host. MARKER accesses the flat row-major buffer; j is a CUDA-only
 * column index replacing Pyccel array slices.
 */
__device__ inline void apply_kinetic_bc_marker(int ip, const MarkerArgs& args_markers,
                                             const DomainArgs& args_domain, bool newton) {
    const long long* bc_type = args_markers.bc_type;
    int first_init_idx = args_markers.first_init_idx;
    int first_shift_idx = args_markers.first_shift_idx;

    // Remove markers before applying periodic or reflecting boundaries.
    for (int axis = 0; axis < 3; ++axis) {
        if (bc_type[axis] == 2 &&
            (args_markers.markers(ip, axis) > 1. || args_markers.markers(ip, axis) < 0.)) {
            int n_cols = args_markers.markers.shape[1];
            for (int j = 0; j < n_cols - 1; ++j) args_markers.markers(ip, j) = -1.;
            return;
        }
    }

    // Wrap positions and update the shift exactly as in the Pyccel helper.
    for (int axis = 0; axis < 3; ++axis) {
        if (bc_type[axis] == 0) {
            if (args_markers.markers(ip, axis) > 1.) {
                args_markers.markers(ip, axis) -= floor(args_markers.markers(ip, axis));
                if (newton) args_markers.markers(ip, first_shift_idx + axis) += 1.;
                else args_markers.markers(ip, first_shift_idx + axis) = 1.;
            } else if (args_markers.markers(ip, axis) < 0.) {
                args_markers.markers(ip, axis) -= floor(args_markers.markers(ip, axis));
                if (newton) args_markers.markers(ip, first_shift_idx + axis) += -1.;
                else args_markers.markers(ip, first_shift_idx + axis) = -1.;
            } else if (!newton) {
                args_markers.markers(ip, first_shift_idx + axis) = 0.;
            }
        }
    }

    bool reflected[3] = {false, false, false};
    int n_reflected = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (bc_type[axis] == 1) {
            if (args_markers.markers(ip, axis) > 1.) {
                args_markers.markers(ip, axis) = 2. - args_markers.markers(ip, axis);
                reflected[axis] = true;
                ++n_reflected;
            } else if (args_markers.markers(ip, axis) < 0.) {
                args_markers.markers(ip, axis) = -args_markers.markers(ip, axis);
                reflected[axis] = true;
                ++n_reflected;
            }
        }
    }
    if (n_reflected == 0) return;
    args_markers.markers(ip, first_init_idx) = -1.;
    for (int axis = 0; axis < 3; ++axis) {
        if (reflected[axis]) reflect_velocity(ip, args_markers.markers, args_domain, axis);
    }
}
}
