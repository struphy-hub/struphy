import cunumpy as xp
import pytest

from struphy.dispersion_relations.analytic import ExtendedMHDhomogenSlab, MHDhomogenSlab

# analytic dispersion relations are pure numpy, no parallel code involved
pytestmark = pytest.mark.mpi_skip


@pytest.mark.parametrize("name", ["ColdPlasma1D", "CurrentCoupling6DParallel", "PressureCouplingFull6DParallel"])
def test_call_with_defaults(name):
    """All branches are computed and stored when called with default parameters."""

    import cunumpy as xp

    from struphy.dispersion_relations import analytic

    k = xp.linspace(0.1, 2.0, 4)

    disp = getattr(analytic, name)()
    omegas = disp(k)

    assert omegas is disp.branches
    for omega in omegas.values():
        assert omega.shape == k.shape
        assert xp.all(xp.isfinite(omega))


def test_cold_plasma_cutoffs():
    """Parallel cold plasma at k -> 0: R/L cutoffs and plasma frequency (alpha = 1, Omega_c = 1)."""

    import cunumpy as xp

    from struphy.dispersion_relations.analytic import ColdPlasma1D

    omegas = ColdPlasma1D(alpha=1.0)(xp.array([1e-6]))
    cutoffs = xp.array([omega[0] for omega in omegas.values()])

    assert xp.allclose(cutoffs, [0.0, (xp.sqrt(5) - 1) / 2, 1.0, (xp.sqrt(5) + 1) / 2], atol=1e-5)


def test_current_coupling_limits():
    """Sound waves are omega = sqrt(gamma p0) k and shear Alfvén waves are omega = B0 k without energetic particles."""

    import cunumpy as xp

    from struphy.dispersion_relations.analytic import CurrentCoupling6DParallel

    k = xp.linspace(0.1, 2.0, 4)

    disp = CurrentCoupling6DParallel(nuh=0.0, B0=1.5)
    omegas = disp(k)

    assert xp.allclose(omegas["sound"], xp.sqrt(disp.params["gamma"] * disp.params["p0"]) * k)
    assert xp.allclose(omegas["shear_Alfvén_R"], 1.5 * k)
    assert xp.allclose(omegas["shear_Alfvén_L"], 1.5 * k)

    # Newton iteration must stop after max_it iterations (returns initial guess B0 * k)
    omegas = CurrentCoupling6DParallel()(k[:1], max_it=0)
    assert xp.allclose(omegas["shear_Alfvén_R"], k[:1])


@pytest.mark.parametrize("B0", [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0)])
def test_extended_mhd_homogen_slab(B0):
    """Compare with ideal MHD for eps -> 0 and check that the Hall term splits the shear Alfvén branch."""

    k = xp.array([0.5, 1.0, 3.0])
    names = ("slow magnetosonic", "shear Alfvén", "fast magnetosonic")

    ext = ExtendedMHDhomogenSlab(B0x=B0[0], B0y=B0[1], B0z=B0[2], eps=1e-8)(k)
    mhd = MHDhomogenSlab(B0x=B0[0], B0y=B0[1], B0z=B0[2])(k)
    for name in names:
        assert xp.allclose(ext[name], mhd[name])

    # parallel propagation, eps=0.1, k=1: ion cyclotron and whistler branches omega = sqrt(1 + w0^2/4) -/+ w0/2
    ext = ExtendedMHDhomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, eps=0.1)(xp.array([1.0]))
    w0 = 0.1
    assert xp.allclose(ext["compressional Alfvén"], w0)
    assert xp.allclose(ext["shear Alfvén"], xp.sqrt(1.0 + w0**2 / 4) - w0 / 2)
    assert xp.allclose(ext["fast magnetosonic"], xp.sqrt(1.0 + w0**2 / 4) + w0 / 2)
