"""Kernel for evaluating the degrees of freedom for the c-th component of 1-forms. This function is for local commuting projetors."""

from numpy import shape, zeros

import struphy.kernel_arguments.local_projectors_args_kernels as local_projectors_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments


def get_dofs_local_1_form_ec_component_weighted(
    args_solve: LocalProjectorsArguments,
    fc: "float[:,:,:]",
    basis0: "float[:]",
    basis1: "float[:]",
    basis2: "float[:]",
    arezeroc: "int[:]",
    f_eval_aux: "float[:,:,:]",
    c: int,
):
    """Kernel for evaluating the degrees of freedom for the c-th component of 1-forms. This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        fc : 3d float array
            Evaluation for the c-th component of the 1-form function over all the interpolation points in e_a and e_b (a != b != c), as well as all the Gauss-Legendre quadrature point in e_c.

        basis0 : 1d float array
            Array with the evaluated basis functions for the e1 direction.

        basis1 : 1d float array
            Array with the evaluated basis functions for the e2 direction.

        basis2 : 1d float array
            Array with the evaluated basis functions for the e3 direction.

        arezeroc : 1d int array
            Array of zeros or ones. A one means that for this particular set of quadrature points, in the c-th direction, the basis function is not zero for at least one of them.

        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.

        c : int
            This int tell us whichone of the three components of the 1-form vector we are dealing with. It must be 0, 1 or 2.
    """
    p = args_solve.degree[c]

    for i in range(shape(f_eval_aux)[0]):
        if c == 0:
            computei = arezeroc[i] != 0
        else:
            computei = abs(basis0[i]) >= 10.0 ** (-16)
        if computei:
            for j in range(shape(f_eval_aux)[1]):
                if c == 1:
                    computej = arezeroc[j] != 0
                else:
                    computej = abs(basis1[j]) >= 10.0 ** (-16)
                if computej:
                    for k in range(shape(f_eval_aux)[2]):
                        if c == 2:
                            computek = arezeroc[k] != 0
                        else:
                            computek = abs(basis2[k]) >= 10.0 ** (-16)
                        if computek:
                            if c == 0:
                                in_start = i * p
                            elif c == 1:
                                in_start = j * p
                            elif c == 2:
                                in_start = k * p
                            for ii in range(p):
                                if c == 0:
                                    f_eval_aux[i, j, k] += (
                                        fc[in_start + ii, j, k]
                                        * basis0[in_start + ii]
                                        * basis1[j]
                                        * basis2[k]
                                        * args_solve.wts0[args_solve.inv_index_translation0[i], ii]
                                    )
                                elif c == 1:
                                    f_eval_aux[i, j, k] += (
                                        fc[i, in_start + ii, k]
                                        * basis0[i]
                                        * basis1[in_start + ii]
                                        * basis2[k]
                                        * args_solve.wts1[args_solve.inv_index_translation1[j], ii]
                                    )
                                elif c == 2:
                                    f_eval_aux[i, j, k] += (
                                        fc[i, j, in_start + ii]
                                        * basis0[i]
                                        * basis1[j]
                                        * basis2[in_start + ii]
                                        * args_solve.wts2[args_solve.inv_index_translation2[k], ii]
                                    )
