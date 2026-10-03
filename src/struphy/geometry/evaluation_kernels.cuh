#pragma once
#include "struphy/kernel_arguments/pusher_args.cuh"
#include "struphy/geometry/domains/cuboid/cuboid_cuda.cuh"
namespace struphy_cuda {
// Unsupported mappings are rejected at pusher setup; never silently use Cuboid.
__device__ inline void df(double x, double y, double z, const DomainArgs& args, double* out) {
    switch(args.kind_map) {
        case 10: cuboid_df(args.params, out); return;
        default: asm("trap;");
    }
}
}
