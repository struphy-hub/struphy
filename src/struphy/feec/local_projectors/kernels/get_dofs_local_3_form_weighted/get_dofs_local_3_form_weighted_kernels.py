"""Kernel for evaluating the degrees of freedom for 3-forms.  This function is for local commuting projetors."""

from numpy import shape, zeros

import struphy.kernel_arguments.local_projectors_args_kernels as local_projectors_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments


def get_dofs_local_3_form_weighted(
    args_solve: LocalProjectorsArguments,
    faux: "float[:,:,:]",
    basis0: "float[:]",
    basis1: "float[:]",
    basis2: "float[:]",
    arezero0: "int[:]",
    arezero1: "int[:]",
    arezero2: "int[:]",
    f_eval: "float[:,:,:]",
):
    """Kernel for evaluating the degrees of freedom for 3-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        faux : 3d float array
            Evaluation for the 3-form function over all the Gauss-Legendre quadrature point in e1, e2 and e3.

        basis0 : 1d float array
            Array with the evaluated basis functions for the e1 direction.

        basis1 : 1d float array
            Array with the evaluated basis functions for the e2 direction.

        basis2 : 1d float array
            Array with the evaluated basis functions for the e3 direction.

        arezero0 : 1d int array
            Array of zeros or ones. A one means that for this particular set of quadrature points, in the first direction, the basis function is not zero for at least one of them.

        arezero1 : 1d int array
            Array of zeros or ones. A one means that for this particular set of quadrature points, in the second direction, the basis function is not zero for at least one of them.

        arezero2 : 1d int array
            Array of zeros or ones. A one means that for this particular set of quadrature points, in the third direction, the basis function is not zero for at least one of them.

        f_eval : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    """

    for i in range(shape(f_eval)[0]):
        if arezero0[i] != 0:
            for j in range(shape(f_eval)[1]):
                if arezero1[j] != 0:
                    for k in range(shape(f_eval)[2]):
                        if arezero2[k] != 0:
                            in_start_1 = i * args_solve.degree[0]
                            in_start_2 = j * args_solve.degree[1]
                            in_start_3 = k * args_solve.degree[2]
                            for ii in range(args_solve.degree[0]):
                                for jj in range(args_solve.degree[1]):
                                    for kk in range(args_solve.degree[2]):
                                        f_eval[i, j, k] += (
                                            faux[
                                                in_start_1 + ii,
                                                in_start_2 + jj,
                                                in_start_3 + kk,
                                            ]
                                            * basis0[in_start_1 + ii]
                                            * basis1[in_start_2 + jj]
                                            * basis2[in_start_3 + kk]
                                            * args_solve.wts0[args_solve.inv_index_translation0[i], ii]
                                            * args_solve.wts1[args_solve.inv_index_translation1[j], jj]
                                            * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                                        )
