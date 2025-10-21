from psydac.ddm.mpi import mpi as MPI
from psydac.linalg.block import BlockVector, BlockVectorSpace
from psydac.linalg.stencil import StencilVector, StencilVectorSpace
from scipy.sparse import csr_matrix, identity

from struphy.feec.linear_operators import LinOpWithTransp
from struphy.linear_algebra.linalg_kron import kron_matvec_2d
from struphy.polar.basic import PolarDerhamSpace, PolarVector
from struphy.utils.arrays import xp


class PolarExtractionOperator(LinOpWithTransp):
    """
    Linear operator mapping from Stencil-/BlockVectorSpace (V) to PolarDerhamSpace (W).

    For fixed third index k, the dot product maps tensor-product (tp) basis functions/DOFs a_ijk

        a) with index i < n_rings ("polar rings") to n_polar polar basis functions/DOFs,
        b) with index i <= n_rings to n2 tp basis functions/DOFs ("first tp ring"),

    and leaves the outer tp zone unchanged (identity map). For notation, see Fig. below.

                     /       /       /               /               /
             k = 1  /       /       /               /               /
           k = 0   /       /       /               /               /
                  ------------------------------- -----------------
          j = 0   |       |       |               |               |
          j = 1   |       |       |               |               |
                  |       |       |               |               |
                  | i = 0 |  ...  |  i = n_rings  |  i > n_rings  |
                  |       |       |               |               |   /
                  |       |       |               |               |  /
        j = n2-1  |       |       |               |               | /
                  -------------------------------------------------
                  |  polar rings  | first tp ring | outer tp zone |

    Fig. : Indexing of 3d spline tensor-product (tp) basis functions/DOFs/coefficients.

    Parameters
    ----------
    V : StencilVectorSpace | BlockVectorSpace
        Domain of the operator (always corresponding to the case transpose=False).

    W : PolarDerhamSpace
        Codomain of the operator (always corresponding to the case transpose=False).

    blocks_ten_to_pol : list
        2D nested list with matrices that map inner most n_rings tp rings to n_polar polar basis functions/DOFs.
            * shape[m][n] = (n_polar[m], n_rings[n]*n2) if transposed=False.
            * shape[m][n] = (n_rings[m]*n2, n_polar[n]) if transposed=True.

    blocks_ten_to_ten : list
        2D nested list with matrices that map inner most n_rings + 1 tp rings to n2 tp basis functions/DOF on "first tp ring".
            * shape[m][n] = (n2, (n_rings[n] + 1)*n2) if transposed=False.
            * shape[m][n] = ((n_rings[m] + 1)*n2, n2) if transposed=True.

    transposed : bool
        Whether the transposed extraction operator shall be constructed.
    """

    def __init__(self, V, W, blocks_ten_to_pol=None, blocks_ten_to_ten=None, transposed=False):
        assert isinstance(V, (StencilVectorSpace, BlockVectorSpace))
        assert isinstance(W, PolarDerhamSpace)
        assert W.parent_space == V

        self._transposed = transposed

        if self.transposed:
            self._domain = W
            self._codomain = V
        else:
            self._domain = V
            self._codomain = W

        self._dtype = V.dtype

        # demanded shapes of blocks_ten_to_pol and blocks_ten_to_ten for checking shapes in setters
        self._blocks_ten_to_pol_shapes = []
        self._blocks_ten_to_ten_shapes = []

        # loop over codomain components
        for m in range(W.n_comps):
            self._blocks_ten_to_pol_shapes += [[]]
            self._blocks_ten_to_ten_shapes += [[]]

            # loop over domain components
            for n in range(W.n_comps):
                # dot product not possible if types of basis in eta_3 are not compatible
                if W.type_of_basis_3[m] != W.type_of_basis_3[n]:
                    self._blocks_ten_to_pol_shapes[-1] += [None]
                    self._blocks_ten_to_ten_shapes[-1] += [None]
                else:
                    if transposed:
                        self._blocks_ten_to_pol_shapes[-1] += [(W.n_rings[m] * W.n2[m], W.n_polar[n])]
                        self._blocks_ten_to_ten_shapes[-1] += [((W.n_rings[m] + 1) * W.n2[m], W.n2[n])]
                    else:
                        self._blocks_ten_to_pol_shapes[-1] += [(W.n_polar[m], W.n_rings[n] * W.n2[n])]
                        self._blocks_ten_to_ten_shapes[-1] += [(W.n2[m], (W.n_rings[n] + 1) * W.n2[n])]

        # set polar blocks (map from first n_rings tensor-product rings to polar basis functions/DOFs)
        self.blocks_ten_to_pol = blocks_ten_to_pol

        # set tensor-product blocks (map from first n_rings + 1 tensor-product rings to "first tp ring")
        self.blocks_ten_to_ten = blocks_ten_to_ten

        # dummy identity blocks in eta_3 direction are needed for usage of low-level dot products
        self._blocks_e3 = []

        for m in range(W.n_comps):
            self._blocks_e3 += [[]]
            for n in range(W.n_comps):
                # dot product not possible if types of basis in eta_3 are not compatible
                if W.type_of_basis_3[m] != W.type_of_basis_3[n]:
                    self._blocks_e3[-1] += [None]
                else:
                    self._blocks_e3[-1] += [identity(W.n3[n])]

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def transposed(self):
        return self._transposed

    @property
    def blocks_ten_to_pol_shapes(self):
        return self._blocks_ten_to_pol_shapes

    @property
    def blocks_ten_to_pol(self):
        return self._blocks_ten_to_pol

    @blocks_ten_to_pol.setter
    def blocks_ten_to_pol(self, blocks):
        """TODO"""

        assert isinstance(blocks, list) or blocks is None

        if blocks is not None:
            check_blocks(blocks, self.blocks_ten_to_pol_shapes)

        self._blocks_ten_to_pol = blocks

    @property
    def blocks_ten_to_ten_shapes(self):
        return self._blocks_ten_to_ten_shapes

    @property
    def blocks_ten_to_ten(self):
        return self._blocks_ten_to_ten

    @blocks_ten_to_ten.setter
    def blocks_ten_to_ten(self, blocks):
        """TODO"""

        assert isinstance(blocks, list) or blocks is None

        if blocks is not None:
            check_blocks(blocks, self.blocks_ten_to_ten_shapes)

        self._blocks_ten_to_ten = blocks

    @property
    def blocks_e3(self):
        return self._blocks_e3

    def dot(self, v, out=None):
        """
        Dot product mapping from Stencil-/BlockVector to PolarVector (or vice versa in case of transposed)

        Parameters
        ----------
        v : StencilVector | BlockVector | (PolarVector, if transposed)
            Input (domain) vector.

        out : PolarVector | (StencilVector | BlockVector, if transposed), optional
            Optional output vector the result will be written to in-place.

        Returns
        -------
        out : PolarVector | (StencilVector | BlockVector, if transposed)
            Output (codomain) vector.
        """

        # transposed operator (polar vector --> tensor-product vector)
        if self.transposed:
            assert isinstance(v, PolarVector)
            assert v.space == self._domain

            # 1. identity operation on outer tp zone
            if out is None:
                out = v.tp.copy()
            else:
                assert isinstance(out, (StencilVector, BlockVector))
                assert out.space == self._codomain
                v.tp.copy(out=out)

            # 2. map "first tp ring" to "polar rings" + "first tp ring"
            if self.blocks_ten_to_ten is not None:
                dot_parts_of_polar(self.blocks_ten_to_ten, self.blocks_e3, v, out)

            # 3. map polar coeffs to "polar rings"
            if self.blocks_ten_to_pol is not None:
                out2 = out.space.zeros()
                dot_parts_of_polar(self.blocks_ten_to_pol, self.blocks_e3, v, out2)

                # add contributions to "polar rings"
                out += out2

        # "standard" operator (tensor-product vector --> polar vector)
        else:
            assert isinstance(v, (StencilVector, BlockVector))
            assert v.space == self._domain

            if out is None:
                out = PolarVector(self._codomain)
            else:
                assert isinstance(out, PolarVector)
                assert out.space == self._codomain

            # 1. identity operation on outer tp zone
            out.tp = v

            # 2. map from "polar rings" to polar coeffs
            if self.blocks_ten_to_pol is not None:
                dot_inner_tp_rings(self.blocks_ten_to_pol, self.blocks_e3, v, out)

            # 3. map to "polar rings" + "first tp ring" to "first tp ring"
            if self.blocks_ten_to_ten is not None:
                dot_inner_tp_rings(self.blocks_ten_to_ten, self.blocks_e3, v, out)

        return out

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """

        if self.transposed:
            V = self.codomain
            W = self.domain
        else:
            V = self.domain
            W = self.codomain

        if self.blocks_ten_to_pol is not None:
            blocks_ten_to_pol = transpose_block_mat(self.blocks_ten_to_pol)
        else:
            blocks_ten_to_pol = None

        if self.blocks_ten_to_ten is not None:
            blocks_ten_to_ten = transpose_block_mat(self.blocks_ten_to_ten)
        else:
            blocks_ten_to_ten = None

        return PolarExtractionOperator(
            V,
            W,
            blocks_ten_to_pol=blocks_ten_to_pol,
            blocks_ten_to_ten=blocks_ten_to_ten,
            transposed=not self.transposed,
        )


class PolarLinearOperator(LinOpWithTransp):
    """
    Linear operator mapping from PolarDerhamSpace (V) to PolarDerhamSpace (W).

    The dot product maps a PolarVector to a PolarVector with
        a) outer tp zone to outer tp zone (stencil),
        b) polar coeffs to polar coeffs (dense),
        c) polar coeffs to "first tp ring" (dense).

    "Polar rings" are alway zero for PolarVectors. For notation, see Fig. below.

                     /       /       /               /               /
             k = 1  /       /       /               /               /
           k = 0   /       /       /               /               /
                  ------------------------------- -----------------
          j = 0   |       |       |               |               |
          j = 1   |       |       |               |               |
                  |       |       |               |               |
                  | i = 0 |  ...  |  i = n_rings  |  i > n_rings  |
                  |       |       |               |               |   /
                  |       |       |               |               |  /
        j = n2-1  |       |       |               |               | /
                  -------------------------------------------------
                  |  polar rings  | first tp ring | outer tp zone |

    Fig. : Indexing of 3d spline tensor-product (tp) basis functions/DOFs/coefficients.

    Parameters
    ----------
    V : PolarDerhamSpace
        Domain of the operator (always corresponding to the case transposed=False).

    W : PolarDerhamSpace
        Codomain of the operator (always corresponding to the case transposed=False).

    tp_operator : LinOpWithTransp
        Standard (stencil) linear operator on the outer tp zone.

    blocks_pol_to_ten : list
        2D nested list with matrices that map polar coeffs to "first tp ring".
            * shape[m][n] = ((n_rings[m] + 1)*n2, n_polar[n]) if transposed=False.
            * shape[m][n] = (n_polar[m], (n_rings[n] + 1)*n2) if transposed=True.

    blocks_pol_to_pol : list
        2D nested list with matrices that map polar coeffs to polar coeffs.
            * shape[m][n] = (n_polar[m], n_polar[n]).

    blocks_e3 : list
        2D nested list with matrices that map in eta_3 direction.

    transposed : bool
        Whether to create the transposed extraction operator.
    """

    def __init__(
        self, V, W, tp_operator=None, blocks_pol_to_ten=None, blocks_pol_to_pol=None, blocks_e3=None, transposed=False
    ):
        assert isinstance(V, PolarDerhamSpace)
        assert isinstance(W, PolarDerhamSpace)

        self._transposed = transposed

        if self.transposed:
            self._domain = W
            self._codomain = V
        else:
            self._domain = V
            self._codomain = W

        self._dtype = V.dtype

        _dom = self.domain
        _codom = self.codomain

        # number of components of domain/codomain elements
        n_comps_dom = _dom.n_comps
        n_comps_codom = _codom.n_comps

        self._tp_operator = tp_operator

        # demanded shape of polar blocks, tiny blocks and eta_3 blocks
        self._blocks_pol_to_ten_shapes = []
        self._blocks_pol_to_pol_shapes = []

        self._blocks_e3_shapes = []

        # loop over codomain components
        for m in range(n_comps_codom):
            self._blocks_pol_to_ten_shapes += [[]]
            self._blocks_pol_to_pol_shapes += [[]]

            self._blocks_e3_shapes += [[]]

            # loop over domain components
            for n in range(n_comps_dom):
                if self.transposed:
                    self._blocks_pol_to_ten_shapes[-1] += [(_codom.n_polar[m], (_dom.n_rings[n] + 1) * _dom.n2[n])]
                else:
                    self._blocks_pol_to_ten_shapes[-1] += [((_codom.n_rings[m] + 1) * _codom.n2[m], _dom.n_polar[n])]

                self._blocks_pol_to_pol_shapes[-1] += [(_codom.n_polar[m], _dom.n_polar[n])]

                self._blocks_e3_shapes[-1] += [(_codom.n3[m], _dom.n3[n])]

        # set polar blocks (map from polar coeffs to "first tp ring")
        self.blocks_pol_to_ten = blocks_pol_to_ten

        # set tiny blocks (map from polar coeffs to polar coeffs)
        self.blocks_pol_to_pol = blocks_pol_to_pol

        # set blocks for Kronecker product in eta_3 direction
        self.blocks_e3 = blocks_e3

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def transposed(self):
        return self._transposed

    @property
    def tp_operator(self):
        return self._tp_operator

    @property
    def blocks_pol_to_ten_shapes(self):
        return self._blocks_pol_to_ten_shapes

    @property
    def blocks_pol_to_ten(self):
        return self._blocks_pol_to_ten

    @blocks_pol_to_ten.setter
    def blocks_pol_to_ten(self, blocks):
        """TODO"""

        assert isinstance(blocks, list) or blocks is None

        if blocks is None:
            self._blocks_pol_to_ten = set_blocks(self.blocks_pol_to_ten_shapes)
        else:
            check_blocks(blocks, self.blocks_pol_to_ten_shapes)
            self._blocks_pol_to_ten = blocks

    @property
    def tp_blocks_shapes(self):
        return self._tp_blocks_shapes

    @property
    def tp_blocks(self):
        return self._tp_blocks

    @tp_blocks.setter
    def tp_blocks(self, blocks):
        """TODO"""

        assert isinstance(blocks, list) or blocks is None

        if blocks is None:
            self._tp_blocks = set_blocks(self.tp_blocks_shapes)
        else:
            check_blocks(blocks, self.tp_blocks_shapes)
            self._tp_blocks = blocks

    @property
    def blocks_pol_to_pol_shapes(self):
        return self._blocks_pol_to_pol_shapes

    @property
    def blocks_pol_to_pol(self):
        return self._blocks_pol_to_pol

    @blocks_pol_to_pol.setter
    def blocks_pol_to_pol(self, blocks):
        """TODO"""

        assert isinstance(blocks, list) or blocks is None

        if blocks is None:
            self._blocks_pol_to_pol = set_blocks(self.blocks_pol_to_pol_shapes)
        else:
            check_blocks(blocks, self.blocks_pol_to_pol_shapes)
            self._blocks_pol_to_pol = blocks

    @property
    def blocks_e3_shapes(self):
        return self._blocks_e3_shapes

    @property
    def blocks_e3(self):
        return self._blocks_e3

    @blocks_e3.setter
    def blocks_e3(self, blocks):
        """TODO"""

        assert isinstance(blocks, list) or blocks is None

        if blocks is None:
            self._blocks_e3 = set_blocks(self.blocks_e3_shapes)
        else:
            check_blocks(blocks, self.blocks_e3_shapes)
            self._blocks_e3 = blocks

    def dot(self, v, out=None):
        """
        Dot product mapping from PolarVector to PolarVector

        Parameters
        ----------
        v : struphy.polar.basic.PolarVector
            Input (domain) vector.

        out : struphy.polar.basic.PolarVector, optional
            If given, the result will be written into this vector in-place.

        Returns
        -------
        out : struphy.polar.basic.PolarVector
            Output (codomain) vector.
        """

        assert isinstance(v, PolarVector)
        assert v.space == self._domain

        if out is None:
            out = PolarVector(self._codomain)
        else:
            assert isinstance(out, PolarVector)
            assert out.space == self._codomain

        # 1. total tp zone to total tp zone (stencil)
        out.tp = self.tp_operator.dot(v.tp)

        # 2. polar to polar (dense)
        dot_pol_pol(self.blocks_pol_to_pol, self.blocks_e3, v, out)

        out2 = PolarVector(self.codomain)

        # transposed operator
        if self.transposed:
            # 3. "first tp ring" to polar
            dot_inner_tp_rings(self.blocks_pol_to_ten, self.blocks_e3, v.tp, out2)

        # "standard" operator
        else:
            # 3. polar to "first tp ring"
            dot_parts_of_polar(self.blocks_pol_to_ten, self.blocks_e3, v, out2.tp)

        # sum up contributions
        out += out2

        return out

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """

        if self.transposed:
            V = self.codomain
            W = self.domain
        else:
            V = self.domain
            W = self.codomain

        blocks_pol_to_ten = transpose_block_mat(self.blocks_pol_to_ten)
        blocks_pol_to_pol = transpose_block_mat(self.blocks_pol_to_pol)
        blocks_e3 = transpose_block_mat(self.blocks_e3)

        return PolarLinearOperator(
            V,
            W,
            tp_operator=self.tp_operator.transpose(),
            blocks_pol_to_ten=blocks_pol_to_ten,
            blocks_pol_to_pol=blocks_pol_to_pol,
            blocks_e3=blocks_e3,
            transposed=not self.transposed,
        )


def dot_inner_tp_rings(blocks_e1_e2, blocks_e3, v, out):
    """
    Maps either

        a) "polar rings" of v to polar coeffs of out (blocks[m][:].shape[0] = n_polar[m] polar coeffs),
        b) "polar rings" + "first tp ring" of v to "first tp ring" of out (blocks[m][:].shape[0] = n2),

    and performs a Kronecker product in eta_3 dirction (k-indices). For notation see Fig. below.

                     /       /       /               /               /
             k = 1  /       /       /               /               /
           k = 0   /       /       /               /               /
                  ------------------------------- -----------------
          j = 0   |       |       |               |               |
          j = 1   |       |       |               |               |
                  |       |       |               |               |
                  | i = 0 |  ...  |  i = n_rings  |  i > n_rings  |
                  |       |       |               |               |   /
                  |       |       |               |               |  /
        j = n2-1  |       |       |               |               | /
                  -------------------------------------------------
                  |  polar rings  | first tp ring | outer tp zone |

    Fig. : Indexing of 3d spline tensor-product (tp) basis functions/DOFs/coefficients.

    Parameters
    ----------
    blocks_e1_e2 : list
        2D nested list with matrices that map inner tp rings to polar coeffs or "first tp ring" depending on shape.

    blocks_e3 : list
        2D nested list with matrices that solely act along eta_3 direction.

    v : StencilVector | BlockVector
        Input vector.

    out : PolarVector
        Output vector that is written to.
    """

    assert isinstance(blocks_e1_e2, list)
    assert isinstance(blocks_e3, list)
    assert isinstance(v, (StencilVector, BlockVector))
    assert isinstance(out, PolarVector)

    polar_space = out.space

    # number of coeffs in eta_2 (always the same (periodic))
    n2 = polar_space.n2[0]

    # extract needed shapes
    n_rows = []
    n_cols = []

    n3_in = []
    n3_out = []

    for m in range(len(blocks_e1_e2)):
        for n in range(len(blocks_e1_e2[0])):
            if blocks_e1_e2[m][n] is not None:
                n_rows += [blocks_e1_e2[m][n].shape[0]]
                n3_out += [blocks_e3[m][n].shape[0]]
                break

    for n in range(len(blocks_e1_e2[0])):
        for m in range(len(blocks_e1_e2)):
            if blocks_e1_e2[m][n] is not None:
                n_cols += [blocks_e1_e2[m][n].shape[1]]
                n3_in += [blocks_e3[m][n].shape[1]]
                break

    # number of rings to be written to
    n_rings_in = [nc // n2 for nc in n_cols]
    n_rings_out = polar_space.n_rings

    # convert needed scalar attributes to lists
    is_scalar_in = isinstance(v, StencilVector)
    in_starts = [v.space.starts] if is_scalar_in else v.space.starts
    in_ends = [v.space.ends] if is_scalar_in else v.space.ends
    in_vec = [v] if is_scalar_in else v

    is_scalar_out = isinstance(out.tp, StencilVector)
    out_starts = [polar_space.starts] if is_scalar_out else polar_space.starts
    out_ends = [polar_space.ends] if is_scalar_out else polar_space.ends
    out_tp = [out.tp] if is_scalar_out else out.tp

    # determine if mapped to polar coeffs or "first tp ring"
    map_to_tp = True if n_rows[0] == n2 else False

    # loop over codomain components
    for m, (row_e1_e2, row_e3) in enumerate(zip(blocks_e1_e2, blocks_e3)):
        res = xp.zeros((n_rows[m], n3_out[m]), dtype=float)

        # loop over domain components
        for n, (block_e1_e2, block_e3) in enumerate(zip(row_e1_e2, row_e3)):
            if in_starts[n][0] == 0:
                s1, s2, s3 = in_starts[n]
                e1, e2, e3 = in_ends[n]

                if block_e1_e2 is not None:
                    tmp = xp.zeros((n_rings_in[n], n2, n3_in[n]), dtype=float)
                    tmp[:, s2 : e2 + 1, s3 : e3 + 1] = in_vec[n][0 : n_rings_in[n], s2 : e2 + 1, s3 : e3 + 1]
                    res += kron_matvec_2d([block_e1_e2, block_e3], tmp.reshape(n_rings_in[n] * n2, n3_in[n]))

        # sum up local dot products
        if polar_space.comm is not None:
            polar_space.comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)

        # write result to output polar vector (in-place)
        if map_to_tp:
            s1, s2, s3 = out_starts[m]
            e1, e2, e3 = out_ends[m]
            out_tp[m][n_rings_out[m], s2 : e2 + 1, s3 : e3 + 1] = res[s2 : e2 + 1, s3 : e3 + 1]
        else:
            out.pol[m][:, :] = res


def dot_parts_of_polar(blocks_e1_e2, blocks_e3, v, out):
    """
    Maps either

        a) polar coeffs of v to "polar rings" of out (blocks[:][n].shape[1] = n_polar[n] polar coeffs),
        b) "first tp ring" of v to "polar rings" + "first tp ring" of out (blocks[:][n].shape[1] = n2),

    and performs a Kronecker product in eta_3 dirction (k-indices). For notation see Fig. below.

                     /       /       /               /               /
             k = 1  /       /       /               /               /
           k = 0   /       /       /               /               /
                  ------------------------------- -----------------
          j = 0   |       |       |               |               |
          j = 1   |       |       |               |               |
                  |       |       |               |               |
                  | i = 0 |  ...  |  i = n_rings  |  i > n_rings  |
                  |       |       |               |               |   /
                  |       |       |               |               |  /
        j = n2-1  |       |       |               |               | /
                  -------------------------------------------------
                  |  polar rings  | first tp ring | outer tp zone |

    Fig. : Indexing of 3d spline tensor-product (tp) basis functions/DOFs/coefficients.

    Parameters
    ----------
    blocks_e1_e2 : list
        2D nested list with matrices that map polar coeffs or "first tp ring" to inner to rings depending on shape.

    blocks_e3 : list
        2D nested list with matrices that solely act along eta_3 direction.

    v : StencilVector | BlockVector
        Input vector.

    out : PolarVector
        Output vector that is written to.
    """

    assert isinstance(blocks_e1_e2, list)
    assert isinstance(blocks_e3, list)
    assert isinstance(v, PolarVector)
    assert isinstance(out, (StencilVector, BlockVector))

    polar_space = v.space

    # number of coeffs in eta_2 (always the same (periodic))
    n2 = polar_space.n2[0]

    # extract needed shapes
    n_rows = []
    n_cols = []

    n3_in = []
    n3_out = []

    for m in range(len(blocks_e1_e2)):
        for n in range(len(blocks_e1_e2[0])):
            if blocks_e1_e2[m][n] is not None:
                n_rows += [blocks_e1_e2[m][n].shape[0]]
                n3_out += [blocks_e3[m][n].shape[0]]
                break

    for n in range(len(blocks_e1_e2[0])):
        for m in range(len(blocks_e1_e2)):
            if blocks_e1_e2[m][n] is not None:
                n_cols += [blocks_e1_e2[m][n].shape[1]]
                n3_in += [blocks_e3[m][n].shape[1]]
                break

    # number of rings to be written to
    n_rings_in = polar_space.n_rings
    n_rings_out = [nr // n2 for nr in n_rows]

    # convert needed scalar attributes to lists
    is_scalar_in = isinstance(v.tp, StencilVector)
    in_starts = [polar_space.starts] if is_scalar_in else polar_space.starts
    in_ends = [polar_space.ends] if is_scalar_in else polar_space.ends
    in_tp = [v.tp] if is_scalar_in else v.tp

    is_scalar_out = isinstance(out, StencilVector)
    out_starts = [out.space.starts] if is_scalar_out else out.space.starts
    out_ends = [out.space.ends] if is_scalar_out else out.space.ends
    out_vec = [out] if is_scalar_out else out

    # determine if mapped from polar coeffs or "first tp ring"
    map_from_tp = True if n_cols[0] == n2 else False

    # loop over codomain components
    for m, (row_e1_e2, row_e3) in enumerate(zip(blocks_e1_e2, blocks_e3)):
        res = xp.zeros((n_rings_out[m], n2, n3_out[m]), dtype=float)

        # loop over domain components
        for n, (block_e1_e2, block_e3) in enumerate(zip(row_e1_e2, row_e3)):
            if block_e1_e2 is not None:
                if map_from_tp:
                    if in_starts[n][0] == 0:
                        s1, s2, s3 = in_starts[n]
                        e1, e2, e3 = in_ends[n]
                        tmp = xp.zeros((n2, n3_in[n]), dtype=float)
                        tmp[s2 : e2 + 1, s3 : e3 + 1] = in_tp[n][n_rings_in[n], s2 : e2 + 1, s3 : e3 + 1]
                        res += kron_matvec_2d([block_e1_e2, block_e3], tmp).reshape(n_rings_out[m], n2, n3_out[m])
                else:
                    res += kron_matvec_2d([block_e1_e2, block_e3], v.pol[n]).reshape(n_rings_out[m], n2, n3_out[m])

        if map_from_tp:
            if polar_space.comm is not None:
                polar_space.comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)

        if out_starts[m][0] == 0:
            s1, s2, s3 = out_starts[m]
            e1, e2, e3 = out_ends[m]
            out_vec[m][0 : n_rings_out[m], s2 : e2 + 1, s3 : e3 + 1] = res[:, s2 : e2 + 1, s3 : e3 + 1]


def dot_pol_pol(blocks_e1_e2, blocks_e3, v, out):
    """
    Maps polar coeffs to polar coeffs.

    Parameters
    ----------
    blocks_e1_e2 : list[list[ndarray]]
        blocks_e1_e2[m][n].shape = (out.pol[m].shape[0], v.pol[n].shape[0])

    v : PolarVector
        Input vector.

    out : PolarVector
        output vector.
    """

    assert isinstance(v, PolarVector)
    assert isinstance(out, PolarVector)

    for out_pol, row_e1_e2, row_e3 in zip(out.pol, blocks_e1_e2, blocks_e3):
        out_pol[:, :] = 0.0
        for v_pol, block_e1_e2, block_e3 in zip(v.pol, row_e1_e2, row_e3):
            if block_e1_e2 is not None:
                out_pol[:, :] += kron_matvec_2d([block_e1_e2, block_e3], v_pol)


def transpose_block_mat(blocks):
    """
    Transpose 2D nested list of numpy arrays.

    Parameters
    ----------
    blocks : list[list[ndarray]]
        Block matrix.

    Returns
    -------
    out : list[list[ndarray]]
        Transposed block matrix.
    """

    n_rows = len(blocks)
    n_cols = len(blocks[0])

    out = []

    for m in range(n_cols):
        out += [[]]
        for n in range(n_rows):
            if blocks[n][m] is not None:
                out[-1] += [blocks[n][m].T]
            else:
                out[-1] += [None]

    return out


def set_blocks(shapes):
    """
    Creates zero blocks list of given shapes.

    Parameters
    ----------
    shapes : list
        2D list of tuples (.,.) holding the shape of individual blocks (shapes[m][n] = (.,.)).

    Returns
    -------
    blocks : list
        2D list of 2D matrices (scipy.sparse or nd.ndarray) filled with zeros.
    """

    assert isinstance(shapes, list)

    blocks = []

    for row in shapes:
        blocks += [[]]
        for shp in row:
            if shp is None:
                blocks[-1] += [None]
            else:
                blocks[-1] += [csr_matrix(shp, dtype=float)]

    return blocks


def check_blocks(blocks, shapes):
    """
    Checks if given blocks have correct shape.

    Parameters
    ----------
    blocks : list
        2D list of 2D matrices (scipy.sparse or nd.ndarray).

    shapes : list
        2D list of tuples (.,.) holding the shape of individual blocks (shapes[m][n] = (.,.)).
    """

    assert isinstance(shapes, list)
    assert isinstance(blocks, list)

    for blk_row, shp_row in zip(blocks, shapes):
        for blk, shp in zip(blk_row, shp_row):
            if shp is None:
                assert blk is None
            else:
                if blk is not None:
                    assert blk.shape == shp
