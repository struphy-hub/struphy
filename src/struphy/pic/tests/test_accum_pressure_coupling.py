import pytest


@pytest.mark.parametrize("kernel_name", ["pc_lin_mhd_6d_full", "pc_lin_mhd_6d"])
def test_pc_accum_uses_time_dependent_weights(kernel_name, Np=200):
    """The pressure-coupling accumulation kernels must fill with the current weights w
    (column ``index["weights"]``), not with the initial weights w0 (column ``index["w0"]``).
    The two differ for delta-f; here w0 is overwritten with garbage and the result must not change.
    """

    import cunumpy as xp

    from struphy import LoadingParameters, domains, maxwellians
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.pic.accumulation.kernels.pc_lin_mhd_6d import pc_lin_mhd_6d
    from struphy.pic.accumulation.kernels.pc_lin_mhd_6d_full import pc_lin_mhd_6d_full
    from struphy.pic.accumulation.particles_to_grid import Accumulator
    from struphy.pic.particles import Particles6D
    from struphy.topology.grids import TensorProductGrid

    domain = domains.Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=3.0, l3=0.0, r3=1.0)
    derham = Derham(TensorProductGrid(num_elements=[4, 3, 1]), DerhamOptions(degree=[2, 2, 1]), comm=None)
    mass_ops = WeightedMassOperators(derham, domain)

    particles = Particles6D(
        loading_params=LoadingParameters(Np=Np, seed=1234),
        domain=domain,
        domain_decomp=(derham.domain_array, derham.domain_decomposition.nprocs),
        background=maxwellians.Maxwellian3D(n=(1.0, None)),
    )
    particles.draw_markers()
    particles.initialize_weights()

    acc = Accumulator(
        particles,
        "Hcurl",
        {"pc_lin_mhd_6d": pc_lin_mhd_6d, "pc_lin_mhd_6d_full": pc_lin_mhd_6d_full}[kernel_name],
        mass_ops,
        domain.args_domain,
        add_vector=True,
        symmetry="pressure",
    )

    def accumulate():
        acc(1.0)
        mats = [op.toarray() for op in acc.operators]
        vecs = [vec.toarray() for vec in acc.vectors]
        return mats, vecs

    idx_w = particles.index["weights"]
    idx_w0 = particles.index["w0"]
    valid = particles.valid_mks

    # make the current weights distinct from w0 (as in delta-f)
    particles.markers[valid, idx_w] *= 1.0 + xp.arange(xp.count_nonzero(valid)) / Np
    mats_ref, vecs_ref = accumulate()

    # garbage in w0 must not change the result
    particles.markers[valid, idx_w0] = -17.0
    mats, vecs = accumulate()

    assert any(xp.max(xp.abs(v)) > 0.0 for v in vecs_ref)
    for a, b in zip(mats_ref + vecs_ref, mats + vecs):
        assert xp.allclose(a, b, rtol=1e-14, atol=0.0)


if __name__ == "__main__":
    test_pc_accum_uses_time_dependent_weights("pc_lin_mhd_6d_full")
