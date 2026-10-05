#pragma once
#include "struphy/kernel_arguments/pusher_args.cuh"
#include "struphy/geometry/domains/cuboid/cuboid_cuda.cuh"
namespace struphy_cuda {
/**
 * Evaluate the mapping Jacobian DF, as in geometry.evaluation_kernels.df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param args Mapping identifier and parameters.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 *
 * Currently only Cuboid (kind_map == 10) is supported. Its Jacobian is
 * independent of eta1/eta2/eta3. Setup rejects other mappings; the device
 * trap guards against unsupported identifiers reaching this helper.
 */
__device__ inline void df(double eta1, double eta2, double eta3, const DomainArgs& args, double* df_out) {
    switch (args.kind_map) {
        case 10:
            cuboid_df(args.params[0], args.params[1], args.params[2],
                      args.params[3], args.params[4], args.params[5], df_out);
            return;
        default: asm("trap;");
    }
}
}
