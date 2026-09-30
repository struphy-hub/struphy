import cunumpy as xp

from struphy.fields_background.equils import EQDSKequilibrium


def test_axisymm_gradB_vs_finite_differences():
    """AxisymmMHDequilibrium.gradB_xyz must match finite differences of |B|,
    including the g_tor * grad(g_tor) contribution (non-constant g_tor in EQDSK)."""

    eq = EQDSKequilibrium()
    R0, Z0 = eq.psi_axis_RZ

    rng = xp.random.default_rng(0)
    R = R0 + rng.uniform(-0.25, 0.25, 20)
    Z = Z0 + rng.uniform(-0.4, 0.4, 20)
    phi = rng.uniform(0.0, 2 * xp.pi, 20)
    x, y, z = R * xp.cos(phi), R * xp.sin(phi), Z

    def absB(x, y, z):
        bx, by, bz = eq.b_xyz(x, y, z)
        return xp.sqrt(bx**2 + by**2 + bz**2)

    h = 1e-5
    fd = [
        (absB(x + h, y, z) - absB(x - h, y, z)) / (2 * h),
        (absB(x, y + h, z) - absB(x, y - h, z)) / (2 * h),
        (absB(x, y, z + h) - absB(x, y, z - h)) / (2 * h),
    ]
    gradB = eq.gradB_xyz(x, y, z)

    for g, f in zip(gradB, fd):
        assert xp.allclose(g, f, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_axisymm_gradB_vs_finite_differences()
