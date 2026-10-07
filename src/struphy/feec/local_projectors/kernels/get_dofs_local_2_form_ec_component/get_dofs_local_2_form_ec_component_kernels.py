"""Kernel for evaluating the degrees of freedom for the c-th component of 2-forms.  This function is for local commuting projetors."""

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.local_projectors_args_kernels as local_projectors_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments


@stack_array("shp")
def get_dofs_local_2_form_ec_component(
    args_solve: LocalProjectorsArguments,
    fc: "float[:,:,:]",
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

        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.

        c : int
            This integer determines which of the three components of the 2-form vector we are working on. Must be 0, 1 or 2.
    """

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                if c == 0:
                    in_start_a = j * args_solve.degree[1]
                    in_start_b = k * args_solve.degree[2]
                    for jj in range(args_solve.degree[1]):
                        for kk in range(args_solve.degree[2]):
                            f_eval_aux[i, j, k] += (
                                fc[i, in_start_a + jj, in_start_b + kk]
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
                                * args_solve.wts0[args_solve.inv_index_translation0[i], jj]
                                * args_solve.wts1[args_solve.inv_index_translation1[j], kk]
                            )
