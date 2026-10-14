"""Kernel for evaluating the degrees of freedom for the c-th component of 1-forms.  This function is for local commuting projetors."""

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.local_projectors_args_kernels as local_projectors_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments


@stack_array("shp")
def get_dofs_local_1_form_ec_component(
    args_solve: LocalProjectorsArguments,
    f3: "float[:,:,:]",
    f_eval_aux: "float[:,:,:]",
    c: int,
):
    """Kernel for evaluating the degrees of freedom for the c-th component of 1-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        f3 : 3d float array
            Evaluation for the c-th component of the 1-form function over all the interpolation points in e_a and e_b (a!=b!=c), as well as all the Gauss-Legendre quadrature point in e_c.

        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.

        c : int
            This integer determines which of the three components of the 1-form vector we are working on. Must be 0,1 or 2.
    """

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    p = args_solve.degree[c]
    if c == 0:
        wts = args_solve.wts0
        inv_index_translation = args_solve.inv_index_translation0
    elif c == 1:
        wts = args_solve.wts1
        inv_index_translation = args_solve.inv_index_translation1
    elif c == 2:
        wts = args_solve.wts2
        inv_index_translation = args_solve.inv_index_translation2

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                if c == 0:
                    in_start = i * p
                    for ii in range(p):
                        f_eval_aux[i, j, k] += f3[in_start + ii, j, k] * wts[inv_index_translation[i], ii]
                elif c == 1:
                    in_start = j * p
                    for ii in range(p):
                        f_eval_aux[i, j, k] += f3[i, in_start + ii, k] * wts[inv_index_translation[j], ii]
                elif c == 2:
                    in_start = k * p
                    for ii in range(p):
                        f_eval_aux[i, j, k] += f3[i, j, in_start + ii] * wts[inv_index_translation[k], ii]
