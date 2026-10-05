// Device versions of the fillers in filler_kernels.py: add the contribution of one marker to a stencil vector.
// Many markers write to the same entries, so every addition is atomic.
#pragma once
#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
namespace struphy_cuda {
__device__ inline void fill_vec(int pi1, int pi2, int pi3, const double* bi1, const double* bi2, const double* bi3,
                                int span1, int span2, int span3, const long long* starts, Array3D<double> vec, double filling) {
    for(int il1=0;il1<=pi1;++il1) {
        long long i1=span1+il1-starts[0];
        double b1=bi1[il1]*filling;
        for(int il2=0;il2<=pi2;++il2) {
            long long i2=span2+il2-starts[1];
            double b2=b1*bi2[il2];
            for(int il3=0;il3<=pi3;++il3) {
                long long i3=span3+il3-starts[2];
                cunumpy_atomic_add(&vec(i1,i2,i3),b2*bi3[il3]);
            }
        }
    }
}
}
