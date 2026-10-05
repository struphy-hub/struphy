// CUDA versions of the argument classes in pusher_args_kernels.py, one struct per class.
//
// CUDA kernels take these structs by value, in the place of the pyccel argument classes, e.g.
//
//     #include "struphy/kernel_arguments/pusher_args.cuh"
//
//     extern "C" __global__
//     void push_eta_stage(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain, ...)
//
// The structs are filled on the host by the classes in struphy/utils/cuda_arguments.py, whose `fields` list the
// members below in the same order and with the same types; test_cuda_argument_structs checks that they agree.
// The member names are the attribute names of the pyccel classes. Pointers are device pointers.
#pragma once
#include "struphy/kernel_arguments/array_view.cuh"

// CUDA version of MarkerArguments (struphy.utils.cuda_arguments.CudaMarkerArguments).
struct MarkerArgs {
    Array2D<double> markers;  // device pointer, shape and element strides
    bool* valid_mks;  // (n_markers,), true for markers that are neither holes nor ghosts
    int n_markers;
    int Np;
    int vdim;
    int weight_idx;
    int first_diagnostics_idx;
    int first_init_idx;
    int first_shift_idx;
    int residual_idx;
    int first_free_idx;
    int mu_idx;
    long long* bc_type;  // (3,)
};

// CUDA version of DerhamArguments (struphy.utils.cuda_arguments.CudaDerhamArguments).
// The scratch arrays of the pyccel class (bn1, ..., bd3) are local arrays in the kernels.
struct DerhamArgs {
    long long* pn;  // (3,)
    double* tn1;
    double* tn2;
    double* tn3;
    long long* starts;  // (3,)
    int nt1;
    int nt2;
    int nt3;
};

// CUDA version of DomainArguments (struphy.utils.cuda_arguments.CudaDomainArguments).
struct DomainArgs {
    int kind_map;
    double* params;
    long long* degree;  // (3,)
    double* t1;
    double* t2;
    double* t3;
    long long* ind1;  // (number of mapping grid cells, degree + 1)
    long long* ind2;
    long long* ind3;
    double* cx;  // control points
    double* cy;
    double* cz;
};
