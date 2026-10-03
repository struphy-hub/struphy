#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"
extern "C" __global__ void reflect(MarkerArgs m, DomainArgs d, const long long* outside, int axis, int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<n) struphy_cuda::reflect_velocity(outside[i],m,d,axis);
}
