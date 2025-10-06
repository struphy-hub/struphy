import numpy as np
import pytest

from struphy.fields_background import equils


@pytest.mark.parametrize(
    "equil_domain_pair",
    [
        ("HomogenSlab", {}, "Cuboid", {}),
        ("HomogenSlab", {}, "Colella", {"alpha": 0.06}),
        ("ShearedSlab", {"a": 0.75, "R0": 3.5}, "Cuboid", {"r1": 0.75, "r2": 2 * np.pi * 0.75, "r3": 2 * np.pi * 3.5}),
        (
            "ShearedSlab",
            {"a": 0.75, "R0": 3.5, "q0": "inf", "q1": "inf"},
            "Cuboid",
            {"r1": 0.75, "r2": 2 * np.pi * 0.75, "r3": 2 * np.pi * 3.5},
        ),
        (
            "ShearedSlab",
            {"a": 0.55, "R0": 4.5},
            "Orthogonal",
            {"Lx": 0.55, "Ly": 2 * np.pi * 0.55, "Lz": 2 * np.pi * 4.5},
        ),
        ("ScrewPinch", {"a": 0.45, "R0": 2.5}, "HollowCylinder", {"a1": 0.05, "a2": 0.45, "Lz": 2 * np.pi * 2.5}),
        ("ScrewPinch", {"a": 1.45, "R0": 6.5}, "IGAPolarCylinder", {"a": 1.45, "Lz": 2 * np.pi * 6.5}),
        (
            "ScrewPinch",
            {"a": 0.45, "R0": 2.5, "q0": 1.5, "q1": 1.5},
            "HollowCylinder",
            {"a1": 0.05, "a2": 0.45, "Lz": 2 * np.pi * 2.5},
        ),
        (
            "ScrewPinch",
            {"a": 1.45, "R0": 6.5, "q0": 1.5, "q1": 1.5},
            "IGAPolarCylinder",
            {"a": 1.45, "Lz": 2 * np.pi * 6.5},
        ),
        (
            "ScrewPinch",
            {"a": 0.45, "R0": 2.5, "q0": "inf", "q1": "inf"},
            "HollowCylinder",
            {"a1": 0.05, "a2": 0.45, "Lz": 2 * np.pi * 2.5},
        ),
        (
            "ScrewPinch",
            {"a": 1.45, "R0": 6.5, "q0": "inf", "q1": "inf"},
            "IGAPolarCylinder",
            {"a": 1.45, "Lz": 2 * np.pi * 6.5},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 0, "p_kind": 0},
            "HollowTorus",
            {"a1": 0.05, "a2": 1.45, "R0": 6.5, "sfl": False},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 0, "p_kind": 1},
            "HollowTorus",
            {"a1": 0.05, "a2": 1.45, "R0": 6.5, "sfl": False},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 1, "p_kind": 0},
            "HollowTorus",
            {"a1": 0.05, "a2": 1.45, "R0": 6.5, "sfl": False},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 1, "p_kind": 1},
            "HollowTorus",
            {"a1": 0.05, "a2": 1.45, "R0": 6.5, "sfl": False},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 0, "p_kind": 0},
            "HollowTorus",
            {"a1": 0.05, "a2": 1.45, "R0": 6.5, "sfl": True},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 0, "p_kind": 1},
            "HollowTorus",
            {"a1": 0.05, "a2": 1.45, "R0": 6.5, "sfl": True},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 1, "p_kind": 0},
            "HollowTorus",
            {"a1": 0.05, "a2": 1.45, "R0": 6.5, "sfl": True},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 1, "p_kind": 1},
            "HollowTorus",
            {"a1": 0.05, "a2": 1.45, "R0": 6.5, "sfl": True},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 0, "p_kind": 0},
            "IGAPolarTorus",
            {"a": 1.45, "R0": 6.5, "sfl": True},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 0, "p_kind": 1},
            "IGAPolarTorus",
            {"a": 1.45, "R0": 6.5, "sfl": True},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 1, "p_kind": 0},
            "IGAPolarTorus",
            {"a": 1.45, "R0": 6.5, "sfl": True},
        ),
        (
            "AdhocTorus",
            {"a": 1.45, "R0": 6.5, "q_kind": 1, "p_kind": 1},
            "IGAPolarTorus",
            {"a": 1.45, "R0": 6.5, "sfl": True},
        ),
        ("AdhocTorus", {"a": 1.45, "R0": 6.5, "q_kind": 0, "p_kind": 0}, "Tokamak", {}),
        ("AdhocTorus", {"a": 1.45, "R0": 6.5, "q_kind": 0, "p_kind": 1}, "Tokamak", {}),
        ("AdhocTorus", {"a": 1.45, "R0": 6.5, "q_kind": 1, "p_kind": 0}, "Tokamak", {}),
        ("AdhocTorus", {"a": 1.45, "R0": 6.5, "q_kind": 1, "p_kind": 1}, "Tokamak", {}),
        ("AdhocTorusQPsi", {"a": 0.8, "R0": 3.6}, "HollowTorus", {"a1": 0.05, "a2": 0.8, "R0": 3.6, "sfl": False}),
        ("AdhocTorusQPsi", {"a": 0.8, "R0": 3.6}, "HollowTorus", {"a1": 0.05, "a2": 0.8, "R0": 3.6, "sfl": True}),
        ("AdhocTorusQPsi", {"a": 0.8, "R0": 3.6}, "IGAPolarTorus", {"a": 0.8, "R0": 3.6, "sfl": True}),
        ("AdhocTorusQPsi", {"a": 1.0, "R0": 3.6}, "Tokamak", {}),
        ("EQDSKequilibrium", {}, "Tokamak", {}),
    ],
)
def test_equils(equil_domain_pair):
    """
    Test field evaluations of all implemented MHD equilbria with default parameters.
    """

    from struphy.fields_background import equils
    from struphy.fields_background.base import CartesianMHDequilibrium, NumericalMHDequilibrium
    from struphy.geometry import domains

    # logical evalution point
    pt = (np.random.rand(), np.random.rand(), np.random.rand())

    # logical arrays:
    e1 = np.random.rand(4)
    e2 = np.random.rand(5)
    e3 = np.random.rand(6)

    # 2d slices
    mat_12_1, mat_12_2 = np.meshgrid(e1, e2, indexing="ij")
    mat_13_1, mat_13_3 = np.meshgrid(e1, e3, indexing="ij")
    mat_23_2, mat_23_3 = np.meshgrid(e2, e3, indexing="ij")

    # 3d
    mat_123_1, mat_123_2, mat_123_3 = np.meshgrid(e1, e2, e3, indexing="ij")
    mat_123_1_sp, mat_123_2_sp, mat_123_3_sp = np.meshgrid(e1, e2, e3, indexing="ij", sparse=True)

    # markers
    markers = np.random.rand(33, 10)

    # create MHD equilibrium
    eq_mhd = getattr(equils, equil_domain_pair[0])(**equil_domain_pair[1])

    # for numerical MHD equilibria, no domain is needed
    if isinstance(eq_mhd, NumericalMHDequilibrium):
        assert equil_domain_pair[2] is None

    else:
        if equil_domain_pair[2] == "Tokamak":
            domain = getattr(domains, equil_domain_pair[2])(**equil_domain_pair[3], equilibrium=eq_mhd)
        else:
            domain = getattr(domains, equil_domain_pair[2])(**equil_domain_pair[3])

        eq_mhd.domain = domain

    # --------- point-wise evaluation ---------
    results = []

    # scalar functions
    results.append(eq_mhd.absB0(*pt, squeeze_out=True))
    results.append(eq_mhd.p0(*pt, squeeze_out=True))
    results.append(eq_mhd.p3(*pt, squeeze_out=True))
    results.append(eq_mhd.n0(*pt, squeeze_out=True))
    results.append(eq_mhd.n3(*pt, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(*pt, squeeze_out=True))
    results.append(eq_mhd.b2(*pt, squeeze_out=True))
    results.append(eq_mhd.bv(*pt, squeeze_out=True))
    results.append(eq_mhd.j1(*pt, squeeze_out=True))
    results.append(eq_mhd.j2(*pt, squeeze_out=True))
    results.append(eq_mhd.jv(*pt, squeeze_out=True))
    results.append(eq_mhd.unit_b1(*pt, squeeze_out=True))
    results.append(eq_mhd.unit_b2(*pt, squeeze_out=True))
    results.append(eq_mhd.unit_bv(*pt, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(*pt, squeeze_out=True))
    results.append(eq_mhd.j_cart(*pt, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(*pt, squeeze_out=True))

    # asserts
    kind = "point"

    for i in range(0, 5):
        assert_scalar(results[i], kind, *pt)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, *pt)
            assert_vector(results[i][1], kind, *pt)
        else:
            assert_vector(results[i], kind, *pt)

    print()
    print("   Evaluation type".ljust(30), "|   equilibrium".ljust(20), "|   domain".ljust(20), "|   status".ljust(20))
    print("--------------------------------------------------------------------------------------")

    print(
        "   point-wise".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- markers evaluation ---------
    results = []

    # scalar functions
    results.append(eq_mhd.absB0(markers))
    results.append(eq_mhd.p0(markers))
    results.append(eq_mhd.p3(markers))
    results.append(eq_mhd.n0(markers))
    results.append(eq_mhd.n3(markers))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(markers))
    results.append(eq_mhd.b2(markers))
    results.append(eq_mhd.bv(markers))
    results.append(eq_mhd.j1(markers))
    results.append(eq_mhd.j2(markers))
    results.append(eq_mhd.jv(markers))
    results.append(eq_mhd.unit_b1(markers))
    results.append(eq_mhd.unit_b2(markers))
    results.append(eq_mhd.unit_bv(markers))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(markers))
    results.append(eq_mhd.j_cart(markers))
    results.append(eq_mhd.unit_b_cart(markers))

    # asserts
    kind = "markers"

    for i in range(0, 5):
        assert_scalar(results[i], kind, markers)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, markers)
            assert_vector(results[i][1], kind, markers)
        else:
            assert_vector(results[i], kind, markers)

    print(
        "   markers".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- eta1 evaluation ---------
    results = []

    e2_pt = np.random.rand()
    e3_pt = np.random.rand()

    # scalar functions
    results.append(eq_mhd.absB0(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.p0(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.p3(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.n0(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.n3(e1, e2_pt, e3_pt, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.b2(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.bv(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j1(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j2(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.jv(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b1(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b2(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_bv(e1, e2_pt, e3_pt, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j_cart(e1, e2_pt, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(e1, e2_pt, e3_pt, squeeze_out=True))

    # asserts
    for i in range(0, 5):
        assert_scalar(results[i], kind, e1, e2_pt, e3_pt)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, e1, e2_pt, e3_pt)
            assert_vector(results[i][1], kind, e1, e2_pt, e3_pt)
        else:
            assert_vector(results[i], kind, e1, e2_pt, e3_pt)

    print(
        "   eta1-array".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- eta2 evaluation ---------
    results = []

    e1_pt = np.random.rand()
    e3_pt = np.random.rand()

    # scalar functions
    results.append(eq_mhd.absB0(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.p0(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.p3(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.n0(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.n3(e1_pt, e2, e3_pt, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.b2(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.bv(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j1(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j2(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.jv(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b1(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b2(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_bv(e1_pt, e2, e3_pt, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j_cart(e1_pt, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(e1_pt, e2, e3_pt, squeeze_out=True))

    # asserts
    kind = "e2"

    for i in range(0, 5):
        assert_scalar(results[i], kind, e1_pt, e2, e3_pt)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, e1_pt, e2, e3_pt)
            assert_vector(results[i][1], kind, e1_pt, e2, e3_pt)
        else:
            assert_vector(results[i], kind, e1_pt, e2, e3_pt)

    print(
        "   eta2-array".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- eta3 evaluation ---------
    results = []

    e1_pt = np.random.rand()
    e2_pt = np.random.rand()

    # scalar functions
    results.append(eq_mhd.absB0(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.p0(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.p3(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.n0(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.n3(e1_pt, e2_pt, e3, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.b2(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.bv(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.j1(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.j2(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.jv(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b1(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b2(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.unit_bv(e1_pt, e2_pt, e3, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.j_cart(e1_pt, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(e1_pt, e2_pt, e3, squeeze_out=True))

    # asserts
    kind = "e3"

    for i in range(0, 5):
        assert_scalar(results[i], kind, e1_pt, e2_pt, e3)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, e1_pt, e2_pt, e3)
            assert_vector(results[i][1], kind, e1_pt, e2_pt, e3)
        else:
            assert_vector(results[i], kind, e1_pt, e2_pt, e3)

    print(
        "   eta3-array".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- eta1-eta2 evaluation ---------
    results = []

    e3_pt = np.random.rand()

    # scalar functions
    results.append(eq_mhd.absB0(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.p0(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.p3(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.n0(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.n3(e1, e2, e3_pt, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.b2(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.bv(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j1(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j2(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.jv(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b1(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b2(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_bv(e1, e2, e3_pt, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j_cart(e1, e2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(e1, e2, e3_pt, squeeze_out=True))

    # asserts
    kind = "e1_e2"

    for i in range(0, 5):
        assert_scalar(results[i], kind, e1, e2, e3_pt)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, e1, e2, e3_pt)
            assert_vector(results[i][1], kind, e1, e2, e3_pt)
        else:
            assert_vector(results[i], kind, e1, e2, e3_pt)

    print(
        "   eta1-eta2-array".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- eta1-eta3 evaluation ---------
    results = []

    e2_pt = np.random.rand()

    # scalar functions
    results.append(eq_mhd.absB0(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.p0(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.p3(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.n0(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.n3(e1, e2_pt, e3, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.b2(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.bv(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.j1(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.j2(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.jv(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b1(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b2(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.unit_bv(e1, e2_pt, e3, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.j_cart(e1, e2_pt, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(e1, e2_pt, e3, squeeze_out=True))

    # asserts
    kind = "e1_e3"

    for i in range(0, 5):
        assert_scalar(results[i], kind, e1, e2_pt, e3)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, e1, e2_pt, e3)
            assert_vector(results[i][1], kind, e1, e2_pt, e3)
        else:
            assert_vector(results[i], kind, e1, e2_pt, e3)

    print(
        "   eta1-eta3-array".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- eta2-eta3 evaluation ---------
    results = []

    e1_pt = np.random.rand()

    # scalar functions
    results.append(eq_mhd.absB0(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.p0(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.p3(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.n0(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.n3(e1_pt, e2, e3, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.b2(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.bv(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.j1(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.j2(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.jv(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b1(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b2(e1_pt, e2, e3, squeeze_out=True))
    results.append(eq_mhd.unit_bv(e1_pt, e2, e3, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(e1_pt, e2, e3))
    results.append(eq_mhd.j_cart(e1_pt, e2, e3))
    results.append(eq_mhd.unit_b_cart(e1_pt, e2, e3))

    # asserts
    kind = "e2_e3"

    for i in range(0, 5):
        assert_scalar(results[i], kind, e1_pt, e2, e3)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, e1_pt, e2, e3)
            assert_vector(results[i][1], kind, e1_pt, e2, e3)
        else:
            assert_vector(results[i], kind, e1_pt, e2, e3)

    print(
        "   eta2-eta3-array".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- eta1-eta2-eta3 evaluation ---------
    results = []

    # scalar functions
    results.append(eq_mhd.absB0(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.p0(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.p3(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.n0(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.n3(e1, e2, e3, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.b2(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.bv(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.j1(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.j2(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.jv(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b1(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b2(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.unit_bv(e1, e2, e3, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.j_cart(e1, e2, e3, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(e1, e2, e3, squeeze_out=True))

    # asserts
    kind = "e1_e2_e3"

    for i in range(0, 5):
        assert_scalar(results[i], kind, e1, e2, e3)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, e1, e2, e3)
            assert_vector(results[i][1], kind, e1, e2, e3)
        else:
            assert_vector(results[i], kind, e1, e2, e3)

    print(
        "   eta1-eta2-eta3-array".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- 12 matrix evaluation ---------
    results = []

    e3_pt = np.random.rand()

    # scalar functions
    results.append(eq_mhd.absB0(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.p0(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.p3(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.n0(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.n3(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.b2(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.bv(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j1(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j2(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.jv(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b1(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b2(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_bv(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.j_cart(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(mat_12_1, mat_12_2, e3_pt, squeeze_out=True))

    # asserts
    kind = "e1_e2_m"

    for i in range(0, 5):
        assert_scalar(results[i], kind, mat_12_1, mat_12_2, e3_pt)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, mat_12_1, mat_12_2, e3_pt)
            assert_vector(results[i][1], kind, mat_12_1, mat_12_2, e3_pt)
        else:
            assert_vector(results[i], kind, mat_12_1, mat_12_2, e3_pt)

    print(
        "   12-matrix".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- 13 matrix evaluation ---------
    results = []

    e2_pt = np.random.rand()

    # scalar functions
    results.append(eq_mhd.absB0(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.p0(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.p3(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.n0(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.n3(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.b2(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.bv(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.j1(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.j2(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.jv(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.unit_b1(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.unit_b2(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.unit_bv(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.j_cart(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(mat_13_1, e2_pt, mat_13_3, squeeze_out=True))

    # asserts
    kind = "e1_e3_m"

    for i in range(0, 5):
        assert_scalar(results[i], kind, mat_13_1, e2_pt, mat_13_3)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, mat_13_1, e2_pt, mat_13_3)
            assert_vector(results[i][1], kind, mat_13_1, e2_pt, mat_13_3)
        else:
            assert_vector(results[i], kind, mat_13_1, e2_pt, mat_13_3)

    print(
        "   13-matrix".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- 23 matrix evaluation ---------
    results = []

    e1_pt = np.random.rand()

    # scalar functions
    results.append(eq_mhd.absB0(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.p0(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.p3(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.n0(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.n3(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.b2(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.bv(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.j1(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.j2(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.jv(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.unit_b1(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.unit_b2(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.unit_bv(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.j_cart(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))
    results.append(eq_mhd.unit_b_cart(e1_pt, mat_23_2, mat_23_3, squeeze_out=True))

    # asserts
    kind = "e2_e3_m"

    for i in range(0, 5):
        assert_scalar(results[i], kind, e1_pt, mat_23_2, mat_23_3)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, e1_pt, mat_23_2, mat_23_3)
            assert_vector(results[i][1], kind, e1_pt, mat_23_2, mat_23_3)
        else:
            assert_vector(results[i], kind, e1_pt, mat_23_2, mat_23_3)

    print(
        "   23-matrix".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- 123 matrix evaluation ---------
    results = []

    # scalar functions
    results.append(eq_mhd.absB0(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.p0(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.p3(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.n0(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.n3(mat_123_1, mat_123_2, mat_123_3))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.b2(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.bv(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.j1(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.j2(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.jv(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.unit_b1(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.unit_b2(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.unit_bv(mat_123_1, mat_123_2, mat_123_3))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.j_cart(mat_123_1, mat_123_2, mat_123_3))
    results.append(eq_mhd.unit_b_cart(mat_123_1, mat_123_2, mat_123_3))

    # asserts
    kind = "e1_e2_e3_m"

    for i in range(0, 5):
        assert_scalar(results[i], kind, mat_123_1, mat_123_2, mat_123_3)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, mat_123_1, mat_123_2, mat_123_3)
            assert_vector(results[i][1], kind, mat_123_1, mat_123_2, mat_123_3)
        else:
            assert_vector(results[i], kind, mat_123_1, mat_123_2, mat_123_3)

    print(
        "   123-matrix".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )

    # --------- 123 matrix evaluation (sparse meshgrid) ---------
    results = []

    # scalar functions
    results.append(eq_mhd.absB0(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.p0(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.p3(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.n0(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.n3(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))

    # vector-valued functions (logical)
    results.append(eq_mhd.b1(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.b2(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.bv(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.j1(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.j2(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.jv(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.unit_b1(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.unit_b2(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.unit_bv(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))

    # vector-valued functions (cartesian)
    results.append(eq_mhd.b_cart(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.j_cart(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))
    results.append(eq_mhd.unit_b_cart(mat_123_1_sp, mat_123_2_sp, mat_123_3_sp))

    # asserts
    kind = "e1_e2_e3_m_sparse"

    for i in range(0, 5):
        assert_scalar(results[i], kind, mat_123_1_sp, mat_123_2_sp, mat_123_3_sp)

    for i in range(5, 17):
        if isinstance(results[i], tuple):
            assert_vector(results[i][0], kind, mat_123_1_sp, mat_123_2_sp, mat_123_3_sp)
            assert_vector(results[i][1], kind, mat_123_1_sp, mat_123_2_sp, mat_123_3_sp)
        else:
            assert_vector(results[i], kind, mat_123_1_sp, mat_123_2_sp, mat_123_3_sp)

    print(
        "   123-matrix (sparse)".ljust(30),
        ("|   " + equil_domain_pair[0]).ljust(20),
        ("|   " + equil_domain_pair[2]).ljust(20),
        ("|   passed"),
    )


def assert_scalar(result, kind, *etas):
    if kind == "markers":
        markers = etas[0]
        n_p = markers.shape[0]

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_p,)

        for ip in range(n_p):
            assert isinstance(result[ip], float)
            assert not np.isnan(result[ip])

    else:
        # point-wise
        if kind == "point":
            assert isinstance(result, float)
            assert not np.isnan(result)

        # slices
        else:
            assert isinstance(result, np.ndarray)

            # eta1-array
            if kind == "e1":
                assert result.shape == (etas[0].size,)

            # eta2-array
            elif kind == "e2":
                assert result.shape == (etas[1].size,)

            # eta3-array
            elif kind == "e3":
                assert result.shape == (etas[2].size,)

            # eta1-eta2-array
            elif kind == "e1_e2":
                assert result.shape == (etas[0].size, etas[1].size)

            # eta1-eta3-array
            elif kind == "e1_e3":
                assert result.shape == (etas[0].size, etas[2].size)

            # eta2-eta3-array
            elif kind == "e2_e3":
                assert result.shape == (etas[1].size, etas[2].size)

            # eta1-eta2-eta3-array
            elif kind == "e1_e2_e3":
                assert result.shape == (etas[0].size, etas[1].size, etas[2].size)

            # 12-matrix
            elif kind == "e1_e2_m":
                assert result.shape == (etas[0].shape[0], etas[1].shape[1])

            # 13-matrix
            elif kind == "e1_e3_m":
                assert result.shape == (etas[0].shape[0], etas[2].shape[1])

            # 123-matrix
            elif kind == "e1_e2_e3_m":
                assert result.shape == (etas[0].shape[0], etas[1].shape[1], etas[2].shape[2])

            # 123-matrix (sparse)
            elif kind == "e1_e2_e3_m_sparse":
                assert result.shape == (etas[0].shape[0], etas[1].shape[1], etas[2].shape[2])


def assert_vector(result, kind, *etas):
    if kind == "markers":
        markers = etas[0]
        n_p = markers.shape[0]

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, n_p)

        for c in range(3):
            for ip in range(n_p):
                assert isinstance(result[c, ip], float)
                assert not np.isnan(result[c, ip])

    else:
        # point-wise
        if kind == "point":
            assert isinstance(result, np.ndarray)
            assert result.shape == (3,)

            for c in range(3):
                assert isinstance(result[c], float)
                assert not np.isnan(result[c])

        # slices
        else:
            assert isinstance(result, np.ndarray)

            # eta1-array
            if kind == "e1":
                assert result.shape == (3, etas[0].size)

            # eta2-array
            elif kind == "e2":
                assert result.shape == (3, etas[1].size)

            # eta3-array
            elif kind == "e3":
                assert result.shape == (3, etas[2].size)

            # eta1-eta2-array
            elif kind == "e1_e2":
                assert result.shape == (3, etas[0].size, etas[1].size)

            # eta1-eta3-array
            elif kind == "e1_e3":
                assert result.shape == (3, etas[0].size, etas[2].size)

            # eta2-eta3-array
            elif kind == "e3_e3":
                assert result.shape == (3, etas[1].size, etas[2].size)

            # eta1-eta2-eta3-array
            elif kind == "e1_e2_e3":
                assert result.shape == (3, etas[0].size, etas[1].size, etas[2].size)

            # 12-matrix
            elif kind == "e1_e2_m":
                assert result.shape == (3, etas[0].shape[0], etas[1].shape[1])

            # 13-matrix
            elif kind == "e1_e3_m":
                assert result.shape == (3, etas[0].shape[0], etas[2].shape[1])

            # 123-matrix
            elif kind == "e1_e2_e3_m":
                assert result.shape == (3, etas[0].shape[0], etas[1].shape[1], etas[2].shape[2])

            # 123-matrix (sparse)
            elif kind == "e1_e2_e3_m_sparse":
                assert result.shape == (3, etas[0].shape[0], etas[1].shape[1], etas[2].shape[2])


if __name__ == "__main__":
    # test_equils(('AdhocTorusQPsi', {'a': 1.0, 'R0': 3.6}, 'Tokamak', {'xi_param': 'sfl'}))
    test_equils(("HomogenSlab", {}, "Cuboid", {}))
