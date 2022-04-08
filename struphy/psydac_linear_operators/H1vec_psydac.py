# -*- coding: UTF-8 -*-

from psydac.feec.global_projectors import GlobalProjector

#==============================================================================
class Projector_H1vec(GlobalProjector):
    """
    Projector from H1xH1xH1 to a conforming finite element space, i.e.
    a finite dimensional subspace of H1xH1xH1, constructed with tensor-product
    B-splines in 2 or 3 dimensions.

    This is a global projector constructed over a tensor-product grid in the
    logical domain. The vertices of this grid are obtained as the tensor
    product of the 1D splines' Greville points along each direction.

    Parameters
    ----------
    H1vec : ProductFemSpace
        H1xH1xH1-conforming finite element space, codomain of the projection
        operator.
    """
    def _structure(self, dim):
        if dim == 3:
            return [
                ['I', 'I', 'I'],
                ['I', 'I', 'I'],
                ['I', 'I', 'I']
            ]
        elif dim == 2:
            return [
                ['I', 'I'],
                ['I', 'I']
            ]
        else:
            raise NotImplementedError('The H1vec projector is only available in 2D or 3D.')
    
    def _function(self, dim):
        if dim == 3: return evaluate_dofs_3d_0form_vec
        else:
            raise NotImplementedError('The H1vec projector is only available in 3D.')

    #--------------------------------------------------------------------------
    def __call__(self, fun):
        r"""
        Project vector function onto the H1xH1xH1-conforming finite element
        space. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : list/tuple of callables
            Scalar components of the real-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain. These correspond to the coefficients
            of a 1-form in the canonical basis (dx_1, ..., dx_N).

            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H(curl)-conforming
            finite element space). This is also a real-valued vector function
            in the logical domain.
        """
        return super().__call__(fun)


#------------------------------------------------------------------------------
def evaluate_dofs_3d_0form_vec(
        intp_x1, intp_x2, intp_x3, # interpolation points
        F1, F2, F3,                # arrays of degrees of freedom (intent out)
        f1, f2, f3                 # input scalar functions (callable)
        ):

    n1, n2, n3 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F1[i1, i2, i3] = f1(intp_x1[i1], intp_x2[i2], intp_x3[i3])

    n1, n2, n3 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F2[i1, i2, i3] = f2(intp_x1[i1], intp_x2[i2], intp_x3[i3])

    n1, n2, n3 = F3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F3[i1, i2, i3] = f3(intp_x1[i1], intp_x2[i2], intp_x3[i3])