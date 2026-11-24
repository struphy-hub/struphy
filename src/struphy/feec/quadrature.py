from struphy.feec.kernels.kernel_3d_vec import kernel_3d_vec
from struphy.feec.psydac_derham import Derham


class QuadratureIntegrator:
    r"""Integrals of functions given at the local quadrature points against the basis functions of a space,
    without metric coefficients,

    .. math::

        v_{\mu,ijk} = \int_{[0,1]^3} \Lambda_{\mu,ijk}(\boldsymbol\eta)\, f_\mu(\boldsymbol\eta) \,\textnormal d\boldsymbol\eta\,,

    computed with the Gauss-Legendre quadrature of the mass matrices of ``derham``. The values :math:`f_\mu` are
    given on the tensor-product grid of the local quadrature points (the flattened ``quad_grid_pts``), as returned by
    :meth:`~struphy.feec.psydac_derham.SplineFunction.eval_tp_fixed_loc` on that grid. The result is a tensor-product
    vector (no boundary conditions, no polar extraction), also with polar splines: the integrals against the polar
    basis functions are ``derham.extraction_ops[form].dot(v)``.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.

    space_id : str
        Space of the basis functions ('H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec').
    """

    def __init__(self, derham: Derham, space_id: str):
        attr = derham.spline_attributes[space_id]
        self._fem_spaces = attr.tensor_spaces
        self._wts = attr.quad_grid_wts
        self._spans = attr.quad_grid_spans
        self._bases = attr.quad_grid_bases
        self._space = derham.coeff_spaces[derham.space_to_form[space_id]]

    @property
    def space(self):
        """The coefficient space of the result."""
        return self._space

    def integrate(self, values: list, out=None, clear: bool = True):
        """Integrate the values (one array per component) against the basis functions.

        Parameters
        ----------
        values : list[xp.ndarray]
            Function values at the local quadrature points, one array per component of the space.

        out : StencilVector | BlockVector, optional
            Output vector in :attr:`space`.

        clear : bool
            Whether to first set all data of ``out`` to zero. If False, the integrals are added to the existing data
            in ``out``.

        Returns
        -------
        StencilVector | BlockVector
            The integrals.
        """
        assert len(values) == len(self._fem_spaces)
        if out is None:
            out = self.space.zeros()
        else:
            assert out.space == self.space

        blocks = [out] if len(self._fem_spaces) == 1 else out.blocks
        # with clear=False, keep the owned data: the ghost regions may hold copies of it (after update_ghost_regions),
        # which exchange_assembly_data would add a second time
        kept = [block._data[self._owned(block)].copy() for block in blocks] if not clear else None
        for fem_space, spans, wts, bases, vals, block in zip(
            self._fem_spaces, self._spans, self._wts, self._bases, values, blocks
        ):
            block._data[:] = 0.0
            kernel_3d_vec(
                *spans,
                *fem_space.degree,
                *[int(start) for start in fem_space.coeff_space.starts],
                *fem_space.coeff_space.pads,
                *wts,
                *bases,
                vals,
                block._data,
            )

        out.exchange_assembly_data()
        if not clear:
            for block, data in zip(blocks, kept):
                block._data[self._owned(block)] += data
        out.update_ghost_regions()
        return out

    @staticmethod
    def _owned(block):
        """Index of the data owned by this process (without the ghost regions) in ``block._data``."""
        space = block.space
        return tuple(
            slice(m * p, m * p + e - s + 1) for s, e, p, m in zip(space.starts, space.ends, space.pads, space.shifts)
        )
