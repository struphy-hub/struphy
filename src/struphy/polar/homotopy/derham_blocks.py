"""High-order polar extraction blocks of the 3D :class:`~struphy.feec.psydac_derham.Derham` complex.

The 3D polar complex is the total complex of the 2D polar complex on the poloidal cross-section (logical
``eta1``-``eta2``, built with :class:`~struphy.polar.homotopy.complex2d.PolarComplex2D`) and the 1D complex in
``eta3`` (paper Section 5)::

    V^m = sum_{p + q = m} V2D^p (x) V1D^q.

The blocks are applied by the polar operators of :mod:`struphy.polar.linear_operators`. Regularity 1 spans the C1
polar splines of Toshniwal et al. (2017) and Holderied & Possanner (2022), which they replace. Per tensor-product component, the first
``n_rings`` rings hold the polar coefficients (the central polar basis functions and the external basis functions of
these rings); all later rings are standard tensor-product coefficients. The ring counts ``r + 2`` (0-forms),
``(r + 1, r + 2)`` (1-forms) and ``r + 1`` (2-forms) of regularity ``r`` are chosen such that

* the extraction ``E`` and the duals ``F`` (``F^T E = I``) are the identity outside the polar rings,
* the exterior derivative maps no tensor-product coefficient into the polar coefficients.

The duals are those of the paper with the non-conforming ring-0 ``y``-component removed, ``F~ = Z F``: on the
coefficients of physical (conforming) fields both coincide, and ``F~^T E = I``, ``D_bar = F~^T D E`` and the homotopies
are unchanged.
"""

import cunumpy as xp
import numpy as np
import scipy.sparse as sp

from struphy.polar.homotopy.complex2d import PolarComplex2D
from struphy.polar.homotopy.derham_spaces import derham_factors
from struphy.polar.homotopy.layout import STRUPHY
from struphy.polar.homotopy.polynomials import PolynomialBasis, zernike_basis


# ------------------------------------------------------------------------------------------------- cross-sections
def _host(values):
    """Domain evaluations (arrays or tuples of arrays of the active backend) as one NumPy array."""
    if isinstance(values, (tuple, list)):
        return np.stack([xp.to_numpy(v) for v in values])
    return xp.to_numpy(values)


class CrossSection:
    """A 2D map ``(x, y) -> (X, Y)`` of the logical square onto a cross-section (``__call__``, ``jacobian_matrix``,
    ``jacobian``), as used by the polar construction."""

    has_polar_axis = True

    def jacobian(self, x, y):
        J = self.jacobian_matrix(x, y)
        return J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]


class UnitDiskCrossSection(CrossSection):
    """The unit disk ``(X, Y) = x (cos 2 pi y, sin 2 pi y)``, the reference cross-section ``"unit_disk"``."""

    def __call__(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return x * np.cos(2 * np.pi * y), x * np.sin(2 * np.pi * y)

    def jacobian_matrix(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        c, s = np.cos(2 * np.pi * y), np.sin(2 * np.pi * y)
        J = np.empty(x.shape + (2, 2))
        J[..., 0, 0], J[..., 0, 1] = c, -2 * np.pi * x * s
        J[..., 1, 0], J[..., 1, 1] = s, 2 * np.pi * x * c
        return J


def _tensor_axes(x, y):
    """The 1D axes ``(x[:, 0], y[0, :])`` if ``x, y`` are an ``"ij"`` meshgrid, else ``None``."""
    if x.ndim == 2 and x.shape[1] > 0 and x.shape[0] > 0:
        if np.all(x == x[:, :1]) and np.all(y == y[:1, :]):
            return x[:, 0], y[0, :]
    return None


class DomainCrossSection(CrossSection):
    """The poloidal cross-section ``eta3 = const`` of a struphy :class:`~struphy.geometry.base.Domain` as a 2D map.

    Physical points are expressed in an orthonormal basis ``(q1, q2)`` of the (planar) cross-section, centred at the
    pole; the Jacobian is ``[q1 q2]^T DF[:, :2]`` from the domain's analytic Jacobian. Tensor grids (``"ij"``
    meshgrids) are evaluated with one call of the domain.
    """

    def __init__(self, domain, eta3=0.0):
        self.domain, self.eta3 = domain, float(eta3)
        e3 = np.array([self.eta3])
        pole = self._map(np.array([0.0]), np.array([0.0]), e3).reshape(3)
        a = self._map(np.array([1.0]), np.array([0.0]), e3).reshape(3) - pole
        b = self._map(np.array([1.0]), np.array([0.25]), e3).reshape(3) - pole
        q1 = a / np.linalg.norm(a)
        b = b - (b @ q1) * q1
        q2 = b / np.linalg.norm(b)
        self.pole, self.basis = pole, np.stack([q1, q2], axis=1)
        J = self.jacobian_matrix(np.array([0.5]), np.array([0.3]))
        if np.linalg.det(J[0]) < 0:
            self.basis[:, 1] *= -1
        # the cross-section must be planar
        e1, e2 = np.linspace(0, 1, 5), np.linspace(0, 1, 7)
        P = self._map(e1, e2, e3).reshape(3, -1) - pole[:, None]
        normal = np.cross(self.basis[:, 0], self.basis[:, 1])
        size = np.max(np.linalg.norm(P, axis=0))
        if np.max(np.abs(normal @ P)) > 1e-8 * size:
            raise ValueError("the polar splines need a planar cross-section eta3 = const of the domain")

    def _map(self, e1, e2, e3):
        """Physical points of the domain on a tensor grid, on the host (the domain evaluates on any backend)."""
        return _host(self.domain(*(xp.asarray(e) for e in (e1, e2, e3))))

    def _domain_jacobian(self, e1, e2):
        return _host(self.domain.jacobian(*(xp.asarray(e) for e in (e1, e2, [self.eta3]))))

    def __call__(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        axes = _tensor_axes(x, y)
        if axes is not None:
            P = self._map(axes[0], axes[1], np.array([self.eta3])).reshape(3, -1)
            Q = (self.basis.T @ (P - self.pole[:, None])).reshape(2, *x.shape)
            return Q[0], Q[1]
        flat_x, flat_y = x.ravel(), y.ravel()
        out = np.empty((2, flat_x.size))
        for unique_y in np.unique(flat_y):
            mask = flat_y == unique_y
            P = self._map(flat_x[mask], np.array([unique_y]), np.array([self.eta3])).reshape(3, -1)
            out[:, mask] = self.basis.T @ (P - self.pole[:, None])
        return out[0].reshape(x.shape), out[1].reshape(x.shape)

    def jacobian_matrix(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        axes = _tensor_axes(x, y)
        if axes is not None:
            DF = self._domain_jacobian(axes[0], axes[1]).reshape(3, 3, -1)[:, :2, :]
            return np.einsum("ka,kbn->nab", self.basis, DF).reshape(x.shape + (2, 2))
        flat_x, flat_y = x.ravel(), y.ravel()
        out = np.empty((flat_x.size, 2, 2))
        for unique_y in np.unique(flat_y):
            mask = flat_y == unique_y
            DF = self._domain_jacobian(flat_x[mask], [unique_y]).reshape(3, 3, -1)[:, :2, :]
            out[mask] = np.einsum("ka,kbn->nab", self.basis, DF)
        return out.reshape(x.shape + (2, 2))


def cross_section_mapping(domain, cross_section="domain"):
    """The 2D reference cross-section of the polar complex: ``"domain"``, ``"unit_disk"`` or a 2D map (an object
    with ``__call__`` and ``jacobian_matrix``)."""
    if not isinstance(cross_section, str):
        return cross_section
    if cross_section == "unit_disk":
        return UnitDiskCrossSection()
    if cross_section == "domain":
        if domain is None:
            raise ValueError("the cross-section 'domain' needs a domain")
        return DomainCrossSection(domain)
    raise ValueError("polar_cross_section must be 'domain', 'unit_disk' or a 2D map")


def _warn_if_cross_section_varies(mapping):
    """The polynomials are projected on the cross-section ``eta3 = 0``; on other cross-sections of a different shape
    (e.g. a rotating ellipse) the polar splines of regularity ``r >= 2`` are only approximately ``C^r``."""
    import warnings

    e1, e2 = np.linspace(0, 1, 5), np.linspace(0, 1, 9)
    X0, Y0 = mapping(*np.meshgrid(e1, e2, indexing="ij"))
    size = np.max(np.hypot(X0, Y0))
    for eta3 in (0.29, 0.61):
        other = DomainCrossSection(mapping.domain, eta3)
        X, Y = other(*np.meshgrid(e1, e2, indexing="ij"))
        if max(np.max(np.abs(X - X0)), np.max(np.abs(Y - Y0))) > 1e-8 * size:
            warnings.warn(
                "the cross-sections eta3 = const of the domain differ in shape: the polar splines of regularity >= 2 "
                "are built on the cross-section eta3 = 0 and are only approximately C^r on the others",
                stacklevel=3,
            )
            return


def polar_radius(mapping, eta1):
    """Mean distance of the circle ``eta1 = const`` of a 2D cross-section from its pole (its physical size)."""
    eta2 = (np.arange(16) + 0.5) / 16
    x, y = mapping(np.full_like(eta2, eta1), eta2)
    x0, y0 = mapping(np.zeros(1), np.zeros(1))
    return float(np.mean(np.hypot(x - x0[0], y - y0[0])))


def scaled_zernike_basis(regularity, radius):
    """The Zernike polynomials of the disk of ``radius`` scaled to values of order one on that disk.

    The central polar basis functions are the projections of these polynomials onto the polar rings; with the radius
    of the polar rings (about ``p / N1`` of the cross-section) they are as large as the tensor-product splines, which
    keeps the polar mass and stiffness matrices as well conditioned as the tensor-product ones under refinement."""
    basis = zernike_basis(regularity, radius=radius)
    return PolynomialBasis([lambda x, y, f=f: radius * f(x, y) for f in basis.functions], regularity)


# ------------------------------------------------------------------------------------------------- blocks
class PolarExtractionBlocksHomotopy:
    """High-order polar extraction blocks for :class:`~struphy.feec.psydac_derham.Derham`.

    Parameters
    ----------
    domain : struphy.geometry.base.Domain
        Mapping with a pole at ``eta1 = 0``.
    derham : struphy.feec.psydac_derham.Derham
        The (partially constructed) tensor-product de Rham complex.
    regularity : int
        Regularity ``r`` at the pole.
    enrichment : str
        ``"cochain_chain"`` (paper), ``"cochain"`` or ``"none"``.
    polynomial_projection : str
        ``"commuting"`` (paper) or ``"l2"``.
    cross_section : str or CrossSection
        Reference cross-section of the 2D polar complex: ``"domain"`` (the domain at ``eta3 = 0``), ``"unit_disk"``
        (as PolarFEEC.jl for general 3D maps) or a 2D mapping.
    """

    method = "homotopy"

    def __init__(
        self,
        domain,
        derham,
        regularity=1,
        enrichment="cochain_chain",
        polynomial_projection="commuting",
        cross_section="domain",
        check_assumptions=True,
    ):
        assert not derham.spl_kind[0], "polar splines need clamped splines in eta1"
        assert derham.spl_kind[1], "polar splines need periodic splines in eta2"
        p1 = derham.degree[0]
        N1, N2 = derham.num_elements[0], derham.num_elements[1]
        self.regularity = r = int(regularity)
        # the 1D spaces of the Derham complex, coefficients in Struphy's ordering
        radial, angular, _ = derham_factors(derham)
        mapping = cross_section_mapping(domain, cross_section)
        if isinstance(mapping, DomainCrossSection) and r >= 2:
            _warn_if_cross_section_varies(mapping)
        self.polar_radius = polar_radius(mapping, min(1.0, p1 / N1))
        self.complex2d = c = PolarComplex2D(
            radial,
            angular,
            mapping,
            r,
            enrichment=enrichment,
            polynomial_projection=polynomial_projection,
            basis=scaled_zernike_basis(r, self.polar_radius),
            check_assumptions=check_assumptions,
            layout=STRUPHY,
        )
        nx0, ny0, nx1, ny1 = c._shapes
        # shapes (rings, angular coefficients) of the 2D tensor-product components
        self.shapes = {"0": (nx0, ny0), "1x": (nx1, ny0), "1y": (nx0, ny1), "2": (nx1, ny1)}
        self.n2 = N2
        self.n3 = {"B": derham.H1_1d_serial[2].spaces[0].nbasis, "M": derham.L2_1d_serial[2].spaces[0].nbasis}
        self.spl_kind3 = derham.spl_kind[2]

        # 2D extraction and duals per 2D tensor-product component (Struphy ordering)
        first = nx1 * ny0
        E, F = [c.extraction(k) for k in range(3)], [c.duals(k).tolil() for k in range(3)]
        # F~ = Z F: remove the non-conforming ring-0 y-component from the 1-form duals
        F[1][first + STRUPHY.index(0, np.arange(ny1), nx0, ny1), :] = 0.0
        F = [f.tocsr() for f in F]
        rows = {"0": slice(0, nx0 * ny0), "1x": slice(0, first), "1y": slice(first, None), "2": slice(0, nx1 * ny1)}
        degree = {"0": 0, "1x": 1, "1y": 1, "2": 2}
        self._E2D = {key: E[degree[key]][rows[key]].tocsc() for key in rows}
        self._F2D = {key: F[degree[key]][rows[key]].tocsc() for key in rows}

        # polar rings per 2D component and the polar coefficient sets (columns of the 2D extractions)
        self.rings = {"0": r + 2, "1x": r + 1, "1y": r + 2, "2": r + 1}
        central = [c[k].central_count for k in range(3)]
        self._sets = {
            "0": self._polar_columns("0", 0, central[0], ["0"]),
            "1x": self._polar_columns("1x", None, None, ["1x"]),
            "1y": self._polar_columns("1y", 1, central[1], ["1y"]),
            "2": self._polar_columns("2", 2, central[2], ["2"]),
        }
        self._check_structure()

        # 3D spaces: tensor-product components (2D component, sign, eta3 basis) and polar sets per component
        # (2D component, sign, eta3 basis, coupling group) per tp component; (polar set, coupling group) per component.
        # Components of one group belong to the same 2D form (e.g. the x- and y-components of 2D 1-forms).
        self.spaces = {
            "0": dict(tp=[("0", 1, "B", 0)], polar=[("0", 0)]),
            "1": dict(
                tp=[("1x", 1, "B", 0), ("1y", 1, "B", 0), ("0", 1, "M", 1)], polar=[("1x", 0), ("1y", 0), ("0", 1)]
            ),
            "2": dict(
                tp=[("1y", 1, "M", 0), ("1x", -1, "M", 0), ("2", 1, "B", 1)], polar=[("1y", 0), ("1x", 0), ("2", 1)]
            ),
            "3": dict(tp=[("2", 1, "M", 0)], polar=[("2", 0)]),
            "v": dict(tp=[("0", 1, "B", g) for g in range(3)], polar=[("0", g) for g in range(3)]),
        }
        self._e_ten_to_pol, self._p_ten_to_pol, self._p_ten_to_ten = {}, {}, {}
        for form in self.spaces:
            self._e_ten_to_pol[form] = self._extraction_blocks(form, self._E2D)
            self._p_ten_to_pol[form] = self._extraction_blocks(form, self._F2D)
            self._p_ten_to_ten[form] = None
        self._build_derivatives()

    # ------------------------------------------------------------------ layout
    def _polar_columns(self, key, degree, central_count, components):
        """Central columns plus the external identity columns whose coefficient lies in the polar rings."""
        c = self.complex2d
        nx0, ny0, nx1, ny1 = c._shapes
        first = nx1 * ny0
        if key == "1x":
            E = c.extraction(1)
            lead = _first_rows(E)
            ring = STRUPHY.ring(lead, nx1, ny0)
            mask = (lead < first) & (ring < self.rings["1x"]) & (np.arange(E.shape[1]) >= c[1].central_count)
            return np.flatnonzero(mask)
        E = c.extraction(degree)
        lead = _first_rows(E)
        if key == "1y":
            ring = np.where(lead >= first, STRUPHY.ring(lead - first, nx0, ny1), -1)
            external = (np.arange(E.shape[1]) >= central_count) & (lead >= first) & (ring < self.rings["1y"])
        else:
            nx, ny = self.shapes[key]
            external = (np.arange(E.shape[1]) >= central_count) & (STRUPHY.ring(lead, nx, ny) < self.rings[key])
        return np.concatenate([np.arange(central_count), np.flatnonzero(external)])

    def _check_structure(self):
        """E and F~ of polar columns live in the polar rings; all other columns are the identity there."""
        for degree, keys in ((0, ["0"]), (1, ["1x", "1y"]), (2, ["2"])):
            polar = np.unique(np.concatenate([self._sets[k] for k in keys]))
            for key in keys:
                limit = self.rings[key] * self.shapes[key][1]
                for M in (self._E2D[key], self._F2D[key]):
                    block = M[:, polar]
                    if block.nnz and block.nonzero()[0].max() >= limit:
                        raise AssertionError(f"polar basis/duals leave the polar rings of component {key}")
            others = np.setdiff1d(np.arange(self.complex2d.dimension(degree)), polar)
            for M in (self._E2D, self._F2D):
                stacked = sp.vstack([M[k][:, others] for k in keys]).tocsc()
                stacked.eliminate_zeros()
                if np.any(np.diff(stacked.indptr) != 1) or not np.allclose(stacked.data, 1.0):
                    raise AssertionError(f"external {degree}-form basis/duals are not the identity")

    def space_layout(self, form):
        """``(n_polar, n_rings)`` per component of the 3D space ``form`` (``"0"``, ``"1"``, ``"2"``, ``"3"``, ``"v"``)."""
        info = self.spaces[form]
        n_polar = tuple(len(self._sets[key]) for key, _ in info["polar"])
        n_rings = tuple(self.rings[key] for key, _, _, _ in info["tp"])
        return n_polar, n_rings

    def _extraction_blocks(self, form, matrices):
        """Blocks ``[m][n]`` (polar of component m <- first rings of tp component n): ``(M[rings, set_m])^T sign``."""
        info = self.spaces[form]
        blocks = []
        for m, (polar_key, group_m) in enumerate(info["polar"]):
            row = []
            type_m = info["tp"][m][2]
            for n, (key, sign, type_n, group_n) in enumerate(info["tp"]):
                if type_m != type_n:
                    row.append(None)
                    continue
                limit = self.rings[key] * self.shapes[key][1]
                if group_m == group_n:
                    block = sign * matrices[key][:limit, self._sets[polar_key]].T
                else:
                    block = sp.csr_matrix((len(self._sets[polar_key]), limit))
                row.append(sp.csr_matrix(block))
            blocks.append(row)
        return blocks

    # ------------------------------------------------------------------ derivatives
    def _derivative_terms(self, name):
        """Terms ``(tp_out, tp_in, sign, A2D, eta3 factor)`` of the 3D derivative ``name``."""
        c = self.complex2d
        nx0, ny0, nx1, ny1 = c._shapes
        first = nx1 * ny0
        G, C = c.spaces[0].derivative_matrix, c.spaces[1].derivative_matrix
        Gx, Gy = G[:first], G[first:]
        Cx, Cy = C[:, :first], C[:, first:]
        I0, Ix, Iy, I2 = (sp.identity(n, format="csr") for n in (nx0 * ny0, first, nx0 * ny1, nx1 * ny1))
        if name == "grad":
            return [(0, 0, 1, Gx, "I"), (1, 0, 1, Gy, "I"), (2, 0, 1, I0, "dz")]
        if name == "curl":
            return [
                (0, 2, 1, Gy, "I"),
                (0, 1, -1, Iy, "dz"),
                (1, 0, 1, Ix, "dz"),
                (1, 2, -1, Gx, "I"),
                (2, 0, 1, Cx, "I"),
                (2, 1, 1, Cy, "I"),
            ]
        if name == "div":
            return [(0, 0, 1, Cy, "I"), (0, 1, -1, Cx, "I"), (0, 2, 1, I2, "dz")]
        raise ValueError(name)

    def _e3(self, kind, type_out, type_in):
        import cunumpy as xp

        from struphy.eigenvalue_solvers.derivatives import grad_1d_matrix

        if kind == "I":
            assert type_out == type_in
            return sp.identity(self.n3[type_in], format="csr")
        return sp.csr_matrix(xp.to_numpy(grad_1d_matrix(self.spl_kind3, self.n3["B"])))

    def _build_derivatives(self):
        forms = {"grad": ("0", "1"), "curl": ("1", "2"), "div": ("2", "3")}
        for name, (source, target) in forms.items():
            V, W = self.spaces[source], self.spaces[target]
            terms = self._derivative_terms(name)
            pol_to_pol, pol_to_ten, e3 = [], [], []
            for m, (polar_out, group_out) in enumerate(W["polar"]):
                rows_pp, rows_pt, rows_e3 = [], [], []
                out_key, _, type_out, _ = W["tp"][m]
                n_rings_out = self.rings[out_key]
                n2 = self.shapes[out_key][1]
                limit = (n_rings_out + 1) * n2
                for n, (polar_in, group_in) in enumerate(V["polar"]):
                    kinds = set()
                    pp = sp.csr_matrix((len(self._sets[polar_out]), len(self._sets[polar_in])))
                    pt = sp.csr_matrix((limit, len(self._sets[polar_in])))
                    for k_out, k_in, sign, A, kind in terms:
                        key_out, sign_out, _, tp_group_out = W["tp"][k_out]
                        key_in, sign_in, _, tp_group_in = V["tp"][k_in]
                        if tp_group_in != group_in:
                            continue
                        # 3D tensor-product coefficients of component k_out of the derivative of the polar basis
                        image = sign * sign_in * (A @ self._E2D[key_in][:, self._sets[polar_in]])
                        image = _clean(image)
                        if image.nnz == 0:
                            continue
                        if tp_group_out == group_out:
                            kinds.add(kind)
                            pp = pp + sign_out * (self._F2D[key_out][:, self._sets[polar_out]].T @ image)
                        if k_out == m:
                            kinds.add(kind)
                            if image.nonzero()[0].max() >= limit:
                                raise AssertionError(f"{name}: polar image leaves the first tensor-product ring")
                            first_ring = sp.lil_matrix(image[:limit])
                            first_ring[: n_rings_out * n2] = 0.0
                            pt = pt + first_ring.tocsr()
                    pp, pt = _clean(pp), _clean(pt)
                    if not kinds or (pp.nnz == 0 and pt.nnz == 0):
                        rows_pp.append(None), rows_pt.append(None), rows_e3.append(None)
                        continue
                    if len(kinds) != 1:
                        raise AssertionError(f"{name}: mixed eta3 factors in block [{m}][{n}]")
                    type_in = V["tp"][n][2]
                    rows_pp.append(pp)
                    rows_pt.append(pt)
                    rows_e3.append(self._e3(kinds.pop(), type_out, type_in))
                pol_to_pol.append(rows_pp)
                pol_to_ten.append(rows_pt)
                e3.append(rows_e3)
            setattr(self, f"_{name}_pol_to_pol", pol_to_pol)
            setattr(self, f"_{name}_pol_to_ten", pol_to_ten)
            setattr(self, f"_{name}_e3", e3)

    # ------------------------------------------------------------------ interface of the polar operators
    @property
    def n_polar(self):
        return [self.space_layout(form)[0] for form in ("0", "1", "2", "3")]

    @property
    def n_rings(self):
        return [self.space_layout(form)[1] for form in ("0", "1", "2", "3")]

    @property
    def e_ten_to_pol(self):
        return self._e_ten_to_pol

    @property
    def p_ten_to_pol(self):
        return self._p_ten_to_pol

    @property
    def p_ten_to_ten(self):
        return self._p_ten_to_ten

    grad_pol_to_pol = property(lambda self: self._grad_pol_to_pol)
    grad_pol_to_ten = property(lambda self: self._grad_pol_to_ten)
    grad_e3 = property(lambda self: self._grad_e3)
    curl_pol_to_pol = property(lambda self: self._curl_pol_to_pol)
    curl_pol_to_ten = property(lambda self: self._curl_pol_to_ten)
    curl_e3 = property(lambda self: self._curl_e3)
    div_pol_to_pol = property(lambda self: self._div_pol_to_pol)
    div_pol_to_ten = property(lambda self: self._div_pol_to_ten)
    div_e3 = property(lambda self: self._div_e3)


def _first_rows(E):
    from struphy.polar.homotopy.complex2d import first_nonzero_rows

    return first_nonzero_rows(E)


def _clean(A):
    A = sp.csr_matrix(A)
    A.data[np.abs(A.data) < 100 * np.finfo(float).eps] = 0.0
    A.eliminate_zeros()
    return A
