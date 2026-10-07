"""Kernel for evaluating the degrees of freedom for 3-forms.  This function is for local commuting projetors."""

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.local_projectors_args_kernels as local_projectors_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments


@stack_array("shp")
def get_dofs_local_3_form(args_solve: LocalProjectorsArguments, faux: "float[:,:,:]", f_eval: "float[:,:,:]"):
    """Kernel for evaluating the degrees of freedom for 3-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        faux : 3d float array
            Evaluation for the 3-form function over all the Gauss-Legendre quadrature point in e1, e2 and e3.

        f_eval : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    """
    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
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
                                * args_solve.wts0[args_solve.inv_index_translation0[i], ii]
                                * args_solve.wts1[args_solve.inv_index_translation1[j], jj]
                                * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                            )
