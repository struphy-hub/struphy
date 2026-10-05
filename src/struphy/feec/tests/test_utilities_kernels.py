import pytest


@pytest.mark.parametrize("mapping", ["Cuboid", "Colella", "HollowTorus"])
def test_hybrid_weight_metric(mapping):
    """hybrid_weight must multiply the data by the metric tensor G = DF^T DF (divided by the density)."""

    import cunumpy as xp

    import struphy.feec.utilities_kernels as kernels
    from struphy.geometry import domains

    if mapping == "Cuboid":
        domain = domains.Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=3.0, l3=0.0, r3=4.0)
    elif mapping == "Colella":
        domain = domains.Colella(Lz=2.0)
    else:
        domain = domains.HollowTorus()

    nq = 2
    rng = xp.random.default_rng(0)
    pts = [rng.uniform(0.1, 0.9, (1, nq)) for _ in range(3)]
    spans = [xp.zeros(1, dtype=int) for _ in range(3)]
    wts = [xp.ones((1, nq)) for _ in range(3)]
    n_data = xp.ones((1, 1, 1, nq, nq, nq))

    # unit-vector input with density 1 returns the columns of G
    G_kernel = xp.zeros((3, 3, nq, nq, nq))
    for col in range(3):
        data = [xp.full((nq, nq, nq), float(col == i)) for i in range(3)]
        kernels.hybrid_weight(0, 0, 0, *pts, *spans, nq, nq, nq, *wts, *data, n_data, domain.args_domain)
        for i in range(3):
            G_kernel[i, col] = data[i]

    G_ref = domain.metric(pts[0][0], pts[1][0], pts[2][0])

    assert xp.allclose(G_kernel, G_ref, atol=1e-12)


if __name__ == "__main__":
    test_hybrid_weight_metric("Colella")
