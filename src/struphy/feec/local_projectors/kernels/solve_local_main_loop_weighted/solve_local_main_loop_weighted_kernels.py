"""Kernel for obtaining the FEEC coefficients of three forms with local projectors."""

import struphy.kernel_arguments.local_projectors_args_kernels as local_projectors_args_kernels  # do not remove; needed to identify dependencies
from struphy.feec.local_projectors_kernels import get_local_problem_size, select_quasi_points
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments


def solve_local_main_loop_weighted(
    args_solve: LocalProjectorsArguments,
    rhs: "float[:,:,:]",
    rows0: "int[:]",
    rows1: "int[:]",
    rows2: "int[:]",
    rowe0: "int[:]",
    rowe1: "int[:]",
    rowe2: "int[:]",
    out: "float[:,:,:]",
    basis0: "float[:]",
    basis1: "float[:]",
    basis2: "float[:]",
):
    """Kernel for obtaining the FEEC coefficients of three forms with local projectors.

    Parameters
    ----
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        rhs : 3d float array
            Array with the evaluated degrees of freedom.

        rows0: 1d int array
            Array that tell us for which rows the basis function in the e1 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the start indices of said regions.

        rows1: 1d int array
            Array that tell us for which rows the basis function in the e2 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the start indices of said regions.

        rows2: 1d int array
            Array that tell us for which rows the basis function in the e3 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the start indices of said regions.

        rowe0: 1d int array
            Array that tell us for which rows the basis function in the e1 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the end indices of said regions.

        rowe1: 1d int array
            Array that tell us for which rows the basis function in the e2 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the end indices of said regions.

        rowe2: 1d int array
            Array that tell us for which rows the basis function in the e3 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the end indices of said regions.

        out : 3d float array
            Array of FEEC coefficients.

        basis0 : 1d float array
            Array with the evaluated basis functions for the e1 direction. Only relevant for the 0 and 0V spaces since they are the only ones who did not multiply the rhs
            by the basis function duting the get_dofs_weigthed function.

        basis1 : 1d float array
            Array with the evaluated basis functions for the e2 direction. Only relevant for the 0 and 0V spaces since they are the only ones who did not multiply the rhs
            by the basis function duting the get_dofs_weigthed function.

        basis2 : 1d float array
            Array with the evaluated basis functions for the e3 direction. Only relevant for the H1 and H1vec spaces since they are the only ones who did not multiply the rhs
            by the basis function during the get_dofs_weigthed function.
    """
    # First we determine if we must multiply the rhs by the basis functions. This is only required for the H1 and H1vec spaces.
    if args_solve.space_key == 0 or args_solve.space_key == 4:
        Need_basis = True
    else:
        Need_basis = False

    lenj1, lenj2, lenj3 = get_local_problem_size(args_solve.periodic, args_solve.degree, args_solve.IoH)

    # We iterate over all the entries that belong to the current rank
    counteri0 = 0
    for i0 in range(args_solve.starts[0], args_solve.ends[0] + 1):
        # This bool variable tell us if this row has a non-zero FE coefficient, based on the current basis function we are using on our projection
        compute0 = False
        # We iterate over the arrays with the start and end indices of non-zero row regions to check if our current row falls in one of them.
        for i00 in range(len(rows0)):
            if counteri0 >= rows0[i00] and counteri0 <= rowe0[i00]:
                compute0 = True
                break
        if compute0:
            counteri1 = 0
            for i1 in range(args_solve.starts[1], args_solve.ends[1] + 1):
                # This bool variable tell us if this row has a non-zero FE coefficient, based on the current basis function we are using on our projection
                compute1 = False
                # We iterate over the arrays with the start and end indices of non-zero row regions to check if our current row falls in one of them.
                for i11 in range(len(rows1)):
                    if counteri1 >= rows1[i11] and counteri1 <= rowe1[i11]:
                        compute1 = True
                        break
                if compute1:
                    counteri2 = 0
                    for i2 in range(args_solve.starts[2], args_solve.ends[2] + 1):
                        # This bool variable tell us if this row has a non-zero FE coefficient, based on the current basis function we are using on our projection
                        compute2 = False
                        # We iterate over the arrays with the start and end indices of non-zero row regions to check if our current row falls in one of them.
                        for i22 in range(len(rows2)):
                            if counteri2 >= rows2[i22] and counteri2 <= rowe2[i22]:
                                compute2 = True
                                break
                        if compute2:
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
                                if args_solve.wij0[i0][j1] != 0.0:
                                    # position 1 to evaluate rhs
                                    if startj1 + j1 < args_solve.original_size[0]:
                                        pos1 = args_solve.index_translation1[startj1 + j1]
                                    else:
                                        pos1 = args_solve.index_translation1[int(startj1 + j1 + args_solve.shift[0])]
                                    auxL2 = 0.0
                                    for j2 in range(lenj2):
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
                                                    if Need_basis:
                                                        auxL3 += (
                                                            args_solve.wij2[i2][j3]
                                                            * rhs[pos1, pos2, pos3]
                                                            * basis2[pos3]
                                                        )
                                                    else:
                                                        auxL3 += args_solve.wij2[i2][j3] * rhs[pos1, pos2, pos3]
                                            if Need_basis:
                                                auxL2 += args_solve.wij1[i1][j2] * auxL3 * basis1[pos2]
                                            else:
                                                auxL2 += args_solve.wij1[i1][j2] * auxL3
                                    if Need_basis:
                                        L123 += args_solve.wij0[i0][j1] * auxL2 * basis0[pos1]
                                    else:
                                        L123 += args_solve.wij0[i0][j1] * auxL2
                            out[counteri0, counteri1, counteri2] = L123
                        counteri2 += 1
                counteri1 += 1
        counteri0 += 1
