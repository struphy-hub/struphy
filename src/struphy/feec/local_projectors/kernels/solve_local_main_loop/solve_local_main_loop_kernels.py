"""Kernel for obtaining the FEEC coefficients with local projectors."""

import struphy.kernel_arguments.local_projectors_args_kernels as local_projectors_args_kernels  # do not remove; needed to identify dependencies
from struphy.feec.local_projectors_kernels import get_local_problem_size, select_quasi_points
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments


def solve_local_main_loop(args_solve: LocalProjectorsArguments, rhs: "float[:,:,:]", out: "float[:,:,:]"):
    """Kernel for obtaining the FEEC coefficients with local projectors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        rhs : 3d float array
            Array with the evaluated degrees of freedom.

        out : 3d float array
            Array of FEEC coefficients.
    """
    lenj1, lenj2, lenj3 = get_local_problem_size(args_solve.periodic, args_solve.degree, args_solve.IoH)

    # We iterate over all the entries that belong to the current rank
    counteri0 = 0
    for i0 in range(args_solve.starts[0], args_solve.ends[0] + 1):
        counteri1 = 0
        for i1 in range(args_solve.starts[1], args_solve.ends[1] + 1):
            counteri2 = 0
            for i2 in range(args_solve.starts[2], args_solve.ends[2] + 1):
                L123 = 0.0
                startj1, endj1 = select_quasi_points(
                    i0,
                    args_solve.degree[0],
                    args_solve.B_nbasis[0],
                    args_solve.periodic[0],
                )
                startj2, endj2 = select_quasi_points(
                    i1,
                    args_solve.degree[1],
                    args_solve.B_nbasis[1],
                    args_solve.periodic[1],
                )
                startj3, endj3 = select_quasi_points(
                    i2,
                    args_solve.degree[2],
                    args_solve.B_nbasis[2],
                    args_solve.periodic[2],
                )
                for j1 in range(lenj1):
                    # We only bother to compute this contribution if the weight wij is not zero. For if it is zero the contribution will be zero as well.
                    if args_solve.wij0[i0][j1] != 0.0:
                        # position 1 to evaluate rhs. The module is only necessary for periodic boundary conditions. But it does not hurt the clamped boundary conditions so we just leave it as is to avoid an extra if.
                        if startj1 + j1 < args_solve.original_size[0]:
                            pos1 = args_solve.index_translation1[startj1 + j1]
                        else:
                            pos1 = args_solve.index_translation1[int(startj1 + j1 + args_solve.shift[0])]
                        auxL2 = 0.0
                        for j2 in range(lenj2):
                            # We only bother to compute this contribution if the weight wij is not zero. For if it is zero the contribution will be zero as well.
                            if args_solve.wij1[i1][j2] != 0.0:
                                # position 2 to evaluate rhs
                                if startj2 + j2 < args_solve.original_size[1]:
                                    pos2 = args_solve.index_translation2[startj2 + j2]
                                else:
                                    pos2 = args_solve.index_translation2[
                                        int(
                                            startj2 + j2 + args_solve.shift[1],
                                        )
                                    ]
                                auxL3 = 0.0
                                for j3 in range(lenj3):
                                    # We only bother to compute this contribution if the weight wij is not zero. For if it is zero the contribution will be zero as well.
                                    if args_solve.wij2[i2][j3] != 0.0:
                                        # position 3 to evaluate rhs
                                        if startj3 + j3 < args_solve.original_size[2]:
                                            pos3 = args_solve.index_translation3[startj3 + j3]
                                        else:
                                            pos3 = args_solve.index_translation3[
                                                int(
                                                    startj3 + j3 + args_solve.shift[2],
                                                )
                                            ]
                                        auxL3 += args_solve.wij2[i2][j3] * rhs[pos1, pos2, pos3]
                                auxL2 += args_solve.wij1[i1][j2] * auxL3
                        L123 += args_solve.wij0[i0][j1] * auxL2
                out[args_solve.pds[0] + counteri0, args_solve.pds[1] + counteri1, args_solve.pds[2] + counteri2] = L123
                counteri2 += 1
            counteri1 += 1
        counteri0 += 1
