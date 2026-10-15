"""The high-order polar subcomplex of the 2D de Rham sequence, built with homotopy operators.

Port of ``src/polar.jl`` of PolarFEEC.jl (R. Remmerswaal, *High-order FEEC discretization on 2D and 3D domains with a
disk-like coordinate singularity*). With regularity ``r`` the polar spaces are

* ``V0 = pi0 T^r (+) external 0-forms`` (rings ``> r + 1``),
* ``V1 = grad pi0 T^r (+) curl^dagger Pi~^{2,1} pi2 T^{r-2} (+) external 1-forms``,
* ``V2 = Pi~^{2,1} pi2 T^{r-2} (+) external 2-forms`` (rings ``> r``),

where ``T^r`` are the Cartesian polynomials of total degree ``r``, ``pi^k`` the tensor-product commuting projections
and ``curl^dagger`` the radial Poincare homotopy. Each degree is represented by an extraction matrix ``E`` (columns are
the full tensor-product coefficients of the polar basis functions, central columns first) and dual matrix ``F`` with
``F^T E = I``; ``E F^T`` is the coefficient projector of the commuting polar projections.

All indices are zero-based; ring ``i`` of the paper is index ``i - 1``.
"""

from functools import cached_property

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from struphy.polar.homotopy.forms2d import PhysicalFormSpace2D
from struphy.polar.homotopy.linalg import SparsePlusLowRank, clean, identity_minus, matmul, max_abs
from struphy.polar.homotopy.polynomials import center_shifted_basis, polynomial_dimension, zernike_basis
from struphy.polar.homotopy.spaces1d import (
    ClampedBSplineSpace,
    PeriodicBSplineSpace,
    TrigonometricSpace,
    periodic_primitive_matrix,
)

ENRICHMENTS = ("cochain_chain", "cochain", "none")


# ---------------------------------------------------------------------------------------------------------- helpers
def cumulative_sum_matrix(n):
    """The ``(n + 1) x n`` matrix ``Sigma`` with ``(Sigma v)_i = sum_{i' < i} v_i'`` (``delta Sigma = I``)."""
    rows, columns = np.tril_indices(n + 1, -1)
    return sp.csr_matrix((np.ones(rows.size), (rows, columns)), shape=(n + 1, n))


def radial_indices(nx, ny, internal_rings, offset=0):
    """Linear indices of the internal (radial index ``< internal_rings``) and external coefficients."""
    if not 0 <= internal_rings <= nx:
        raise ValueError(f"the number of internal rings must lie between 0 and {nx}")
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    linear = (offset + i + j * nx).ravel(order="F")
    ring = i.ravel(order="F")
    return linear[ring < internal_rings], linear[ring >= internal_rings]


def identity_columns(n, indices):
    indices = np.asarray(indices, dtype=int)
    return sp.csr_matrix((np.ones(indices.size), (indices, np.arange(indices.size))), shape=(n, indices.size))


def _rank_tolerance(matrix, rank_tolerance):
    if rank_tolerance is not None:
        return rank_tolerance
    return max(matrix.shape) * np.finfo(float).eps


def dual_columns(columns, rank_tolerance=None, context="extraction"):
    """Pseudo-inverse columns ``D`` with ``D^T columns = I`` (thin SVD with a rank check)."""
    if columns.shape[1] == 0:
        return np.zeros((columns.shape[0], 0))
    U, S, Vt = sla.svd(columns, full_matrices=False)
    tolerance = _rank_tolerance(columns, rank_tolerance)
    rank = int(np.sum(S > tolerance * S[0]))
    if rank != columns.shape[1]:
        raise ValueError(f"{context} has rank {rank}, but {columns.shape[1]} independent columns are required")
    return U @ np.diag(1.0 / S) @ Vt


def first_nonzero_rows(A):
    """Row of the first nonzero entry of every column of the sparse matrix ``A`` (``-1`` if empty)."""
    A = sp.csc_matrix(A)
    A.sort_indices()
    result = np.full(A.shape[1], -1)
    for column in range(A.shape[1]):
        start, stop = A.indptr[column], A.indptr[column + 1]
        nonzero = np.nonzero(A.data[start:stop])[0]
        if nonzero.size:
            result[column] = A.indices[start + nonzero[0]]
    return result


# ---------------------------------------------------------------------------------------------------------- spaces
class PolarFormSpace:
    """One degree of a :class:`PolarComplex2D`: extraction ``E`` and duals ``F`` of a physical form space."""

    def __init__(self, physical_space, extraction, duals, central_count):
        self.physical_space = physical_space
        self.extraction = clean(extraction).tocsr()
        self.duals = clean(duals).tocsr()
        self.central_count = central_count

    def __repr__(self):
        return (
            f"PolarFormSpace(degree {self.k}, {self.dimension} of {self.physical_space.n} coefficients, "
            f"{self.central_count} central)"
        )

    @property
    def k(self):
        return self.physical_space.k

    @property
    def dimension(self):
        return self.extraction.shape[1]

    @cached_property
    def projector(self):
        """The coefficient projector ``E F^T``."""
        return (self.extraction @ self.duals.T).tocsr()

    def project_coefficients(self, coefficients):
        return self.extraction @ (self.duals.T @ coefficients)

    def commuting_projection(self, f, additional_order=2, representation="physical"):
        """Full coefficients of the polar commuting projection ``E F^T pi f``."""
        full = self.physical_space.commuting_projection(f, additional_order, representation)
        return self.project_coefficients(full)

    def __call__(self, f, **kwargs):
        return self.commuting_projection(f, **kwargs)

    def central_elements(self):
        """Full coefficients of the central polar basis functions (one column each)."""
        return self.extraction[:, : self.central_count].toarray()


# ---------------------------------------------------------------------------------------------------------- complex
class PolarComplex2D:
    """The high-order polar subcomplex ``V0 -> V1 -> V2`` over a disk-like mapping.

    Parameters
    ----------
    radial : ClampedBSplineSpace
        Radial 0-form space (logical ``x``).
    angular : PeriodicBSplineSpace or TrigonometricSpace
        Angular 0-form space (logical ``y``).
    mapping : Mapping2D
        Disk-like mapping that collapses ``x = 0`` to the pole.
    regularity : int
        The regularity ``r`` at the pole (polynomial degree of the enrichment).
    basis : PolynomialBasis, optional
        Polynomial basis of ``T^r``; default ``zernike_basis(r)`` (it is centre-shifted internally).
    enrichment : {"cochain_chain", "cochain", "none"}
        ``"cochain_chain"`` is the paper's construction; ``"cochain"`` drops the compact 2-forms and curl-like 1-forms
        (the reduced spaces of Section 4.2); ``"none"`` keeps only external functions (no constants in ``V0``).
    boundary : {"free", "dirichlet"}
        ``"dirichlet"`` builds the relative complex (``P(1, y) = 0``, ``A_y(1, y) = 0``).
    polynomial_projection : {"commuting", "l2"}
        Projection of the polynomials onto the tensor-product spaces.
    """

    def __init__(
        self,
        radial,
        angular,
        mapping,
        regularity,
        basis=None,
        enrichment="cochain_chain",
        boundary="free",
        polynomial_projection="commuting",
        additional_quadrature_order=None,
        rank_tolerance=None,
        check_assumptions=True,
    ):
        if enrichment not in ENRICHMENTS:
            raise ValueError(f"enrichment must be one of {ENRICHMENTS}")
        if boundary not in ("free", "dirichlet"):
            raise ValueError("boundary must be 'free' or 'dirichlet'")
        if polynomial_projection not in ("commuting", "l2"):
            raise ValueError("polynomial_projection must be 'commuting' or 'l2'")
        self.regularity = r = int(regularity)
        self.enrichment, self.boundary = enrichment, boundary
        self.polynomial_projection = polynomial_projection
        self.additional_quadrature_order = additional_quadrature_order
        self.rank_tolerance = rank_tolerance
        self.mapping = mapping
        self.radial, self.angular = radial, angular
        self._validate(check_assumptions)

        basis = zernike_basis(r) if basis is None else basis
        if basis.degree != r:
            raise ValueError("the polynomial basis must have the degree of the regularity")
        self.polynomial_basis = basis if enrichment == "none" else center_shifted_basis(basis)

        self.spaces = [PhysicalFormSpace2D.from_factors(k, radial, angular, mapping) for k in range(3)]
        V0, V1, V2 = self.spaces
        nx0, ny0 = radial.n, angular.n
        nx1, ny1 = radial.scaled.n, angular.scaled.n
        self._shapes = (nx0, ny0, nx1, ny1)

        E0, F0 = self._zero_form_matrices()
        E1, E2, central_one, central_two = self._higher_form_extractions(E0)
        F2 = self._two_form_duals(E2, central_two)
        self._P0 = (E0 @ F0.T).tocsr()
        self._P2 = (E2 @ F2.T).tocsr()
        F1 = self._one_form_duals(E1, central_one)

        if boundary == "dirichlet":
            E0, keep0 = _remove_boundary_columns(E0, self.zero_form_dirichlet_rows(), "the zero-form extraction")
            F0 = F0[:, keep0]
            E1, keep1 = _remove_boundary_columns(E1, self.one_form_dirichlet_rows(), "the one-form extraction")
            F1 = F1[:, keep1]

        central_zero = 0 if enrichment == "none" else len(self.polynomial_basis)
        self.forms = [
            PolarFormSpace(V0, E0, F0, central_zero),
            PolarFormSpace(V1, E1, F1, central_one),
            PolarFormSpace(V2, E2, F2, central_two),
        ]
        self._cache = {}

    # ------------------------------------------------------------------ basics
    def __getitem__(self, k):
        return self.forms[k]

    def __repr__(self):
        dims = tuple(self.dimension(k) for k in range(3))
        return (
            f"PolarComplex2D(regularity {self.regularity}, dimensions {dims}, enrichment={self.enrichment!r}, "
            f"boundary={self.boundary!r})"
        )

    top_degree = 2

    def dimension(self, k):
        return self.forms[k].dimension

    def extraction(self, k):
        return self.forms[k].extraction

    def duals(self, k):
        return self.forms[k].duals

    def _validate(self, check_assumptions):
        x, y, r = self.radial, self.angular, self.regularity
        if not isinstance(x, ClampedBSplineSpace):
            raise TypeError("the radial factor must be a ClampedBSplineSpace")
        if not isinstance(y, (PeriodicBSplineSpace, TrigonometricSpace)):
            raise TypeError("the angular factor must be periodic or trigonometric")
        if r + 2 > x.n:
            raise ValueError(
                f"the polar construction requires radial dimension >= regularity + 2 (got {x.n} and regularity {r})"
            )
        if self.enrichment == "none" or not check_assumptions:
            return
        if r > x.degree:
            raise ValueError(
                f"the paper assumes regularity <= radial degree (got regularity {r}, degree {x.degree}); "
                "pass check_assumptions=False to construct the spaces anyway"
            )
        if y.n < 2 * r + 1:
            raise ValueError(
                f"the paper assumes an angular dimension >= 2 regularity + 1 (got {y.n} for regularity {r}); "
                "pass check_assumptions=False to construct the spaces anyway"
            )

    def _polynomial_columns(self, functions, space):
        return space.polynomial_projection_columns(
            list(functions), self.polynomial_projection, self.additional_quadrature_order
        )

    # ------------------------------------------------------------------ degree 0
    def _zero_form_matrices(self):
        nx, ny = self.radial.n, self.angular.n
        n = nx * ny
        rings = self.regularity + 1
        internal, external = radial_indices(nx, ny, rings)
        external_identity = identity_columns(n, external)
        constant = self.angular.constant_coefficients()
        denominator = constant @ constant
        if self.enrichment == "none":
            center_dual = np.zeros(n)
            center_dual[np.arange(ny) * nx] = constant / denominator
            external_constant = constant[external // nx]
            F = external_identity - sp.csr_matrix(np.outer(center_dual, external_constant))
            return external_identity, F
        # project on the full tensor-product space first, then truncate to the internal rings
        projected = self._polynomial_columns(self.polynomial_basis.functions, self.spaces[0])
        internal_columns = projected[internal]
        duals = dual_columns(internal_columns, self.rank_tolerance, "the zero-form polynomial extraction")
        # the first dual reproduces the centre value, which makes the projectors commute with the homotopies
        center_dual = np.zeros(internal.size)
        center_dual[np.arange(ny) * rings] = constant / denominator
        duals[:, 0] = center_dual
        tolerance = max(1e-10, 1000 * _rank_tolerance(internal_columns, self.rank_tolerance)) * np.linalg.cond(
            internal_columns
        )
        if np.max(np.abs(duals.T @ internal_columns - np.eye(internal_columns.shape[1]))) > tolerance:
            raise ValueError("the projected polynomial basis does not reproduce its prescribed values at the centre")
        m = len(self.polynomial_basis)
        polynomial = np.zeros((n, m))
        polynomial[internal] = internal_columns
        dual = np.zeros((n, m))
        dual[internal] = duals
        E = sp.hstack([sp.csr_matrix(polynomial), external_identity]).tocsr()
        F = sp.hstack([sp.csr_matrix(dual), external_identity]).tocsr()
        return E, F

    # ------------------------------------------------------------------ degrees 1 and 2
    def reduced_two_form_projector(self, regularity):
        """``Pi~^{2,r}``: rings ``0..r`` are accumulated in ring ``r`` and later rings are copied (``r = 0``: identity)."""
        nx, ny = self.radial.scaled.n, self.angular.scaled.n
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        columns = (i + j * nx).ravel(order="F")
        rows = (np.maximum(i, regularity) + j * nx).ravel(order="F")
        return sp.csr_matrix((np.ones(rows.size), (rows, columns)), shape=(nx * ny, nx * ny))

    @cached_property
    def gradient_homotopy_matrix(self):
        """Full-space radial homotopy ``G^dagger = [I (x) Sigma, 0]`` from 1-forms to 0-forms."""
        nx0, ny0, nx1, ny1 = self._shapes
        radial = sp.kron(sp.identity(ny0), cumulative_sum_matrix(nx1))
        return sp.hstack([radial, sp.csr_matrix((radial.shape[0], nx0 * ny1))]).tocsr()

    @cached_property
    def curl_homotopy_matrix(self):
        """Full-space radial homotopy ``C^dagger = [0; I (x) Sigma]`` from 2-forms to 1-forms."""
        nx0, ny0, nx1, ny1 = self._shapes
        radial = sp.kron(sp.identity(ny1), cumulative_sum_matrix(nx1))
        return sp.vstack([sp.csr_matrix((nx1 * ny0, radial.shape[1])), radial]).tocsr()

    def _higher_form_extractions(self, E0):
        r = self.regularity
        nx0, ny0, nx1, ny1 = self._shapes
        V1, V2 = self.spaces[1], self.spaces[2]
        G = self.spaces[0].derivative_matrix
        m = len(self.polynomial_basis)
        if self.enrichment == "none":
            gradient_columns = sp.csr_matrix((V1.n, 0))
        else:
            gradient_columns = (G @ E0[:, 1:m]).tocsr()
        if self.enrichment == "cochain_chain" and r >= 2:
            functions = self.polynomial_basis.functions[: polynomial_dimension(r - 2)]
            raw = self._polynomial_columns(functions, V2)
            conforming = self.reduced_two_form_projector(1) @ raw
            compact = np.zeros_like(conforming)
            for j in range(ny1):
                internal = j * nx1 + np.arange(r)
                compact[internal] = conforming[internal]
                # the first external ring cancels the radial sum of each angular column
                compact[j * nx1 + r] = -conforming[internal].sum(axis=0)
            compact = sp.csr_matrix(compact)
        else:
            compact = sp.csr_matrix((V2.n, 0))
        # curl homotopy lifts; their entries vanish exactly from ring r + 2 on (zero radial sums)
        curl_columns = (self.curl_homotopy_matrix @ compact).toarray()
        first = nx1 * ny0
        for j in range(ny1):
            curl_columns[first + j * nx0 + r + 1 : first + (j + 1) * nx0] = 0.0
        curl_columns = sp.csr_matrix(curl_columns)
        _, external_x = radial_indices(nx1, ny0, r)
        _, external_y = radial_indices(nx0, ny1, r + 1, offset=first)
        external_one = identity_columns(V1.n, np.concatenate([external_x, external_y]))
        _, external_two = radial_indices(nx1, ny1, r)
        E1 = sp.hstack([gradient_columns, curl_columns, external_one]).tocsr()
        E2 = sp.hstack([compact, identity_columns(V2.n, external_two)]).tocsr()
        return E1, E2, gradient_columns.shape[1] + compact.shape[1], compact.shape[1]

    def _two_form_duals(self, E2, central_count):
        r = self.regularity
        nx, ny = self.radial.scaled.n, self.angular.scaled.n
        internal, external = radial_indices(nx, ny, r)
        central = np.zeros((E2.shape[0], central_count))
        if central_count > 0:
            central[internal] = dual_columns(
                E2[internal][:, :central_count].toarray(), self.rank_tolerance, "the two-form polynomial extraction"
            )
        external_duals = self.reduced_two_form_projector(r)[external].T
        F = sp.hstack([sp.csr_matrix(central), external_duals]).tocsr()
        tolerance = max(1e-10, 1000 * _rank_tolerance(E2, self.rank_tolerance))
        if max_abs(F.T @ E2 - sp.identity(E2.shape[1])) > tolerance:
            raise ValueError("the two-form extraction is incompatible with the derived polar projector")
        return F

    @cached_property
    def one_form_projector_full(self):
        """``Pi^1 = (G Pi^0) G^dagger + (C^dagger Pi^2) C`` on the full tensor-product 1-form coefficients."""
        G = self.spaces[0].derivative_matrix
        C = self.spaces[1].derivative_matrix
        return ((G @ self._P0) @ self.gradient_homotopy_matrix + (self.curl_homotopy_matrix @ self._P2) @ C).tocsr()

    def _one_form_duals(self, E1, central_count):
        n = E1.shape[0]
        external = first_nonzero_rows(E1[:, central_count:])
        internal = np.setdiff1d(np.arange(n), external)
        E_internal = E1[:, :central_count]
        projector = self.one_form_projector_full
        internal_rows = projector[internal]
        Q = np.zeros((central_count, n))
        conditioning = 1.0
        representation_error = max_abs(internal_rows)
        active = np.unique(sp.csc_matrix(internal_rows).nonzero()[1])
        if central_count > 0 and active.size:
            A = E_internal[internal].toarray()
            if np.linalg.matrix_rank(A) != central_count:
                raise ValueError("the internal one-form extraction columns are linearly dependent")
            conditioning = np.linalg.cond(A)
            rhs = internal_rows[:, active].toarray()
            coordinates = np.linalg.lstsq(A, rhs, rcond=None)[0]
            Q[:, active] = coordinates
            representation_error = np.max(np.abs(A @ coordinates - rhs))
        external_coordinates = projector[external] - sp.csr_matrix(E_internal[external] @ sp.csr_matrix(Q))
        dual_transpose = sp.vstack([sp.csr_matrix(Q), external_coordinates]).tocsr()
        dual_transpose.data[np.abs(dual_transpose.data) < 100 * np.finfo(float).eps] = 0.0
        dual_transpose.eliminate_zeros()
        F = dual_transpose.T.tocsr()
        tolerance = max(1e-10, 1000 * _rank_tolerance(E1, self.rank_tolerance)) * conditioning
        biorthogonality = max_abs(F.T @ E1 - sp.identity(E1.shape[1]))
        if max(representation_error, biorthogonality) > tolerance:
            raise ValueError("the one-form extraction is incompatible with the homotopy projector")
        return F

    # ------------------------------------------------------------------ boundary helpers
    def zero_form_dirichlet_rows(self):
        nx, ny = self.radial.n, self.angular.n
        return nx - 1 + np.arange(ny) * nx

    def one_form_dirichlet_rows(self):
        nx0, ny0, nx1, ny1 = self._shapes
        return nx1 * ny0 + nx0 - 1 + np.arange(ny1) * nx0

    @cached_property
    def center_projection_full(self):
        """Rank-one full-space matrix of ``P u = u(0)`` on continuous 0-forms (left, right factors)."""
        nx, ny = self.radial.n, self.angular.n
        constant = self.angular.constant_coefficients()
        functional = np.zeros(nx * ny)
        functional[np.arange(ny) * nx] = constant / (constant @ constant)
        return np.kron(constant, np.ones(nx)), functional

    @cached_property
    def two_form_integral_vector(self):
        """Integrals of the full tensor-product 2-form basis functions (logical, hence physical)."""
        return np.kron(self.angular.scaled.coefficient_integrals, self.radial.scaled.coefficient_integrals)

    # ------------------------------------------------------------------ operators in polar coefficients
    def derivative(self, k):
        """``D_k = F_{k+1}^T D E_k`` with ``D E_k = E_{k+1} D_k``."""
        if k not in (0, 1):
            raise ValueError("degree must be 0 or 1")
        key = ("derivative", k)
        if key not in self._cache:
            D = self.spaces[k].derivative_matrix
            self._cache[key] = clean(self.forms[k + 1].duals.T @ D @ self.forms[k].extraction)
        return self._cache[key]

    def full_mass_matrix(self, k):
        key = ("full_mass", k)
        if key not in self._cache:
            self._cache[key] = self.spaces[k].mass_matrix()
        return self._cache[key]

    def mass_matrix(self, k):
        """Polar mass matrix ``E^T M E``."""
        key = ("mass", k)
        if key not in self._cache:
            E = self.forms[k].extraction
            M = (E.T @ self.full_mass_matrix(k) @ E).tocsc()
            self._cache[key] = ((M + M.T) / 2).tocsc()
        return self._cache[key]

    def mass_solver(self, k):
        """Sparse LU factorization of the polar mass matrix (``.solve(b)``)."""
        key = ("mass_solver", k)
        if key not in self._cache:
            self._cache[key] = spla.splu(self.mass_matrix(k))
        return self._cache[key]

    def l2_projection(self, k, f, additional_order=0):
        """Full coefficients of the L2 projection into the polar space of degree ``k``."""
        E = self.forms[k].extraction
        load = E.T @ self.spaces[k].inner_product(f, additional_order)
        return E @ self.mass_solver(k).solve(load)

    def commuting_projection(self, k, f, additional_order=2):
        return self.forms[k].commuting_projection(f, additional_order)

    def cohomology_projection(self, k):
        """Projection onto the harmonic representatives: centre value (free, ``k = 0``), mean ``B`` (Dirichlet, ``k = 2``)."""
        n = self.dimension(k)
        if self.boundary == "free":
            if k != 0:
                return SparsePlusLowRank.zero(n)
            left, right = self.center_projection_full
            form = self.forms[0]
            return SparsePlusLowRank.rank_one(form.duals.T @ left, form.extraction.T @ right)
        if k != 2:
            return SparsePlusLowRank.zero(n)
        key = "mean_projection"
        if key not in self._cache:
            integral = self.forms[2].extraction.T @ self.two_form_integral_vector
            representer = self.mass_solver(2).solve(integral)
            normalization = integral @ representer
            if not np.isfinite(normalization) or normalization <= 0:
                raise ValueError("the polar two-form space does not contain a nonzero mean mode")
            self._cache[key] = SparsePlusLowRank.rank_one(representer / normalization, integral)
        return self._cache[key]

    def homotopy(self, k):
        """Polar homotopy ``V^k -> V^{k-1}``: radial (free) or boundary-compatible (Dirichlet)."""
        if k not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        key = ("homotopy", k)
        if key not in self._cache:
            if self.boundary == "dirichlet":
                if self.dimension(1) != self.dimension(0) + self.dimension(2) - 1:
                    raise ValueError("the boundary-restricted complex does not admit the boundary homotopy")
                value = self._boundary_curl_homotopy() if k == 2 else self._boundary_gradient_homotopy()
            else:
                H = self.curl_homotopy_matrix if k == 2 else self.gradient_homotopy_matrix
                value = clean(self.forms[k - 1].duals.T @ H @ self.forms[k].extraction)
            self._cache[key] = value
        return self._cache[key]

    def _boundary_curl_homotopy(self):
        nx0, ny0, nx1, ny1 = self._shapes
        zero_mean = identity_minus(self.cohomology_projection(2))
        radial_lift = matmul(sp.csr_matrix(self.curl_homotopy_matrix @ self.forms[2].extraction), zero_mean)
        # periodic primitive of the tangential trace, extended by the outer radial B-spline
        trace_rows = self.one_form_dirichlet_rows()
        trace = sp.csr_matrix((np.ones(ny1), (np.arange(ny1), trace_rows)), shape=(ny1, self.spaces[1].n))
        extension_rows = self.zero_form_dirichlet_rows()
        extension = sp.csr_matrix((np.ones(ny0), (extension_rows, np.arange(ny0))), shape=(self.spaces[0].n, ny0))
        primitive = periodic_primitive_matrix(self.angular, normalization="mean")
        potential = matmul((extension @ primitive @ trace).tocsr(), radial_lift)
        corrected = radial_lift - matmul(self.spaces[0].derivative_matrix, potential)
        return matmul(self.forms[1].duals.T.tocsr(), corrected)

    def _boundary_gradient_homotopy(self):
        nx, ny = self.radial.n, self.angular.n
        closed_part = identity_minus(self.homotopy(2) @ self.derivative(1))
        radial_potential = matmul(sp.csr_matrix(self.gradient_homotopy_matrix @ self.forms[1].extraction), closed_part)
        constant = np.kron(self.angular.constant_coefficients(), self.radial.constant_coefficients())
        boundary_mean = np.zeros(nx * ny)
        boundary_mean[self.zero_form_dirichlet_rows()] = self.angular.coefficient_integrals
        # the boundary trace of the potential is constant; subtract its mean
        mean_part = SparsePlusLowRank.rank_one(constant, boundary_mean) @ radial_potential
        corrected = radial_potential - mean_part
        return matmul(self.forms[0].duals.T.tocsr(), corrected)

    def homotopy_adjoint(self, k, coefficients):
        """``M_k^{-1} h_k^T M_{k-1}`` applied to polar coefficients of degree ``k - 1``."""
        return self.mass_solver(k).solve(self.homotopy(k).T @ (self.mass_matrix(k - 1) @ coefficients))

    def cohomology_dimensions(self):
        """``dim ker d^k / im d^{k-1}`` from dense ranks (for testing)."""
        ranks = [np.linalg.matrix_rank(self.derivative(k).toarray()) for k in range(2)] + [0]
        return [self.dimension(k) - ranks[k] - (ranks[k - 1] if k > 0 else 0) for k in range(3)]

    def central_counts(self):
        return tuple(form.central_count for form in self.forms)


def _remove_boundary_columns(extraction, boundary_rows, context):
    """Remove the first column with a nonzero in each boundary row (the interpolatory boundary columns)."""
    rows = first_nonzero_rows(sp.csr_matrix(extraction).T)[boundary_rows]
    if np.any(rows < 0):
        raise ValueError(f"{context} does not contain an interpolatory outer-boundary column")
    if np.unique(rows).size != rows.size:
        raise ValueError(f"{context} has duplicate outer-boundary columns")
    keep = np.setdiff1d(np.arange(extraction.shape[1]), rows)
    return extraction[:, keep], keep
