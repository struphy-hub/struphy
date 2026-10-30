"""Kernel for evaluating the degrees of freedom for the c-th component of 2-forms.  This function is for local commuting projetors."""

from numpy import shape, zeros

import struphy.kernel_arguments.local_projectors_args_kernels as local_projectors_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments


def get_dofs_local_2_form_ec_component_weighted(
    args_solve: LocalProjectorsArguments,
    fc: "float[:,:,:]",
    basis0: "float[:]",
    basis1: "float[:]",
    basis2: "float[:]",
    arezero_a: "int[:]",
    arezero_b: "int[:]",
    f_eval_aux: "float[:,:,:]",
    c: int,
):
    """Kernel for evaluating the degrees of freedom for the c-th component of 2-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        fc : 3d float array
            Evaluation for the c-th component of the 2-form function over all the interpolation points in e_c, as well as all the Gauss-Legendre quadrature point in e_a and e_b.
            Of the two spatial directions different from e_c, e_a is the one with the smaller index, and e_b is the one with the larger index.

        basis0 : 1d float array
            Array with the evaluated basis functions for the e1 direction.

        basis1 : 1d float array
            Array with the evaluated basis functions for the e2 direction.

        basis2 : 1d float array
            Array with the evaluated basis functions for the e3 direction.

        arezero_a : 1d int array
            Array zeros or ones. A one means that for this particular set of quadrature points, in the e_a direction, the basis function is not zero for at least one of them.

        arezero_b : 1d int array
            Array zeros or ones. A one means that for this particular set of quadrature points, in the e_b direction, the basis function is not zero for at least one of them.

        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.

        c : int
            This int tell us whichone of the three components of the 2-form vector we are dealing with. It must be 0, 1 or 2.
    """

    for i in range(shape(f_eval_aux)[0]):
        if c == 0:
            computei = abs(basis0[i]) >= 10.0 ** (-16)
        else:
            computei = arezero_a[i] != 0
        if computei:
            for j in range(shape(f_eval_aux)[1]):
                if c == 1:
                    computej = abs(basis1[j]) >= 10.0 ** (-16)
                elif c == 0:
                    computej = arezero_a[j] != 0
                elif c == 2:
                    computej = arezero_b[j] != 0
                if computej:
                    for k in range(shape(f_eval_aux)[2]):
                        if c == 2:
                            computek = abs(basis2[k]) >= 10.0 ** (-16)
                        else:
                            computek = arezero_b[k] != 0
                        if computek:
                            if c == 0:
                                in_start_a = j * args_solve.degree[1]
                                in_start_b = k * args_solve.degree[2]
                                for jj in range(args_solve.degree[1]):
                                    for kk in range(args_solve.degree[2]):
                                        f_eval_aux[i, j, k] += (
                                            fc[i, in_start_a + jj, in_start_b + kk]
                                            * basis0[i]
                                            * basis1[in_start_a + jj]
                                            * basis2[in_start_b + kk]
                                            * args_solve.wts1[args_solve.inv_index_translation1[j], jj]
                                            * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                                        )

                            elif c == 1:
                                in_start_a = i * args_solve.degree[0]
                                in_start_b = k * args_solve.degree[2]
                                for jj in range(args_solve.degree[0]):
                                    for kk in range(args_solve.degree[2]):
                                        f_eval_aux[i, j, k] += (
                                            fc[in_start_a + jj, j, in_start_b + kk]
                                            * basis0[in_start_a + jj]
                                            * basis1[j]
                                            * basis2[in_start_b + kk]
                                            * args_solve.wts0[args_solve.inv_index_translation0[i], jj]
                                            * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                                        )

                            elif c == 2:
                                in_start_a = i * args_solve.degree[0]
                                in_start_b = j * args_solve.degree[1]
                                for jj in range(args_solve.degree[0]):
                                    for kk in range(args_solve.degree[1]):
                                        f_eval_aux[i, j, k] += (
                                            fc[in_start_a + jj, in_start_b + kk, k]
                                            * basis0[in_start_a + jj]
                                            * basis1[in_start_b + kk]
                                            * basis2[k]
                                            * args_solve.wts0[args_solve.inv_index_translation0[i], jj]
                                            * args_solve.wts1[args_solve.inv_index_translation1[j], kk]
                                        )
