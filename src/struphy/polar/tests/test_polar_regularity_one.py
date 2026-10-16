"""The polar splines of regularity 1 are the C1 polar splines of Toshniwal et al. (2017) and Holderied & Possanner
(2022): both span the same spaces in every degree.

The C1 extraction blocks (``PolarExtractionBlocksC1``, removed after commit c0ee1bec) are frozen in
``data/c1_reference.npz`` for spline mappings with the resolution of the grid; the test compares the column spans of
the polar spaces in the tensor-product coefficients of the rings next to the pole (the outer rings are tensor-product
functions in both constructions).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REFERENCE = Path(__file__).resolve().parent / "data" / "c1_reference.npz"


@dataclass(frozen=True)
class Case:
    domain: str
    num_elements: tuple
    degree: int

    @property
    def key(self):
        return f"{self.domain}_{self.num_elements[0]}x{self.num_elements[1]}_p{self.degree}"


CASES = [
    Case(domain, num_elements, degree)
    for domain in ("cylinder", "torus", "shaped")
    for num_elements, degree in (((6, 9), 2), ((8, 12), 3))
]


def shaped_cross_section(eta1, eta2):
    """A non-circular, off-centre cross-section (x, y) used for the spline mapping of the "shaped" case."""
    R = eta1 * (1 + 0.2 * eta1 * np.cos(6 * np.pi * eta2)) * (1 + 0.3 * np.sin(2 * np.pi * eta2))
    return R * np.cos(2 * np.pi * eta2) + 0.1 * eta1**2, 0.7 * R * np.sin(2 * np.pi * eta2)


def build_domain(case):
    from struphy.geometry import domains
    from struphy.geometry.base import PoloidalSplineStraight, interp_mapping

    num_elements, degree = case.num_elements, (case.degree, case.degree)
    if case.domain == "cylinder":
        return domains.IGAPolarCylinder(a=1.0, Lz=1.0, num_elements=num_elements, degree=degree)
    if case.domain == "torus":
        return domains.IGAPolarTorus(num_elements=num_elements, degree=degree)
    cx, cy = interp_mapping(
        num_elements,
        degree,
        (False, True),
        lambda e1, e2: shaped_cross_section(e1, e2)[0],
        lambda e1, e2: shaped_cross_section(e1, e2)[1],
    )
    cx[0], cy[0] = cx[0].mean(), cy[0].mean()
    return PoloidalSplineStraight(num_elements=num_elements, degree=degree, cx=cx, cy=cy, Lz=1.0)


def build_derham(case, method="homotopy", polar=True):
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    options = DerhamOptions(
        degree=(case.degree, case.degree, 1),
        bcs=(("free", "free"), None, None),
        polar_splines=polar,
        polar_method=method,
        polar_regularity=1,
    )
    grid = TensorProductGrid(num_elements=(*case.num_elements, 1))
    return Derham(grid, options, domain=build_domain(case))


# coupled components of the polar spaces (the eta1-eta2 components of 1- and 2-forms are coupled at the pole)
GROUPS = {"0": [(0,)], "1": [(0, 1), (2,)], "2": [(0, 1), (2,)], "3": [(0,)]}


def _rings(blocks, form, component, n2):
    return blocks[form][component][component].shape[1] // n2


def polar_basis(blocks, form, components, rings, n2):
    """Rows: the polar basis functions of the coupled ``components`` in the tensor-product coefficients of the first
    ``rings[c]`` rings of each component, completed by the tensor-product functions of the rings that the polar
    construction leaves untouched."""
    widths = [rings[c] * n2 for c in components]
    offsets = np.concatenate([[0], np.cumsum(widths)])
    out = []
    for r in components:
        row = [blocks[form][r][c] for c in components]
        count = next(b.shape[0] for b in row if b is not None)
        M = np.zeros((count, offsets[-1]))
        for k, block in enumerate(row):
            if block is not None:
                M[:, offsets[k] : offsets[k] + block.shape[1]] = block
        out.append(M)
    for k, c in enumerate(components):
        for index in range(_rings(blocks, form, c, n2) * n2, rings[c] * n2):
            unit = np.zeros(offsets[-1])
            unit[offsets[k] + index] = 1.0
            out.append(unit[None])
    return np.vstack(out)


def frozen_c1_blocks(case):
    data = np.load(REFERENCE)
    blocks = {}
    for form, groups in GROUPS.items():
        components = sorted({c for group in groups for c in group})
        blocks[form] = {r: {c: data.get(f"{case.key}/{form}/{r}{c}") for c in components} for r in components}
    return blocks


def dense_blocks(ck_blocks):
    blocks = {}
    for form, groups in GROUPS.items():
        rows = ck_blocks.e_ten_to_pol[form]
        components = sorted({c for group in groups for c in group})
        blocks[form] = {
            r: {
                c: None
                if rows[r][c] is None
                else (rows[r][c].toarray() if hasattr(rows[r][c], "toarray") else np.asarray(rows[r][c]))
                for c in components
            }
            for r in components
        }
    return blocks


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.key)
def test_regularity_one_spans_the_c1_spaces(case):
    derham = build_derham(case)
    n2 = case.num_elements[1]
    c1, general = frozen_c1_blocks(case), dense_blocks(derham.ck_blocks)
    for form, groups in GROUPS.items():
        for components in groups:
            rings = {c: max(_rings(b, form, c, n2) for b in (c1, general)) for c in components}
            A = polar_basis(c1, form, components, rings, n2)
            B = polar_basis(general, form, components, rings, n2)
            rank_a, rank_b, rank_ab = (np.linalg.matrix_rank(M, tol=1e-9) for M in (A, B, np.vstack([A, B])))
            assert rank_a == A.shape[0] and rank_b == B.shape[0], (form, components)
            assert rank_a == rank_b == rank_ab, f"degree {form}, components {components}: spans differ"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.key)
def test_regularity_one_zero_forms_are_affine_on_the_first_rings(case):
    """The characterization of C1 0-forms that does not depend on the mapping: on the rings 0 and 1, every polar
    0-form is the projection of an affine function a + b x + c y."""
    derham = build_derham(case)
    n2 = case.num_elements[1]
    tensor = build_derham(case, polar=False)
    domain = tensor.domain

    def in_plane(e1, e2, e3):
        """Cartesian coordinates in the plane of the cross-section ((R, Z) on the torus)."""
        x, y, z = domain(e1, e2, e3)
        return (np.hypot(x, y), z) if case.domain == "torus" else (x, y)

    affine = [tensor.P0(lambda e1, e2, e3: 1.0 + 0 * e1)]
    for k in range(2):
        affine.append(tensor.P0(lambda e1, e2, e3, k=k: in_plane(e1, e2, e3)[k]))
    Q = np.array([v.toarray().reshape(-1, n2)[:2].ravel() for v in affine])
    E0 = dense_blocks(derham.ck_blocks)["0"][0][0][:, : 2 * n2]
    assert np.linalg.matrix_rank(Q) == 3
    assert np.linalg.matrix_rank(np.vstack([Q, E0]), tol=1e-9) == 3
    assert np.linalg.matrix_rank(E0, tol=1e-9) == 3
