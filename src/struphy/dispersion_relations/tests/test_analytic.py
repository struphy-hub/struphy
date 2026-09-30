import cunumpy as xp
import pytest

from struphy.dispersion_relations.analytic import ExtendedMHDhomogenSlab, MHDhomogenSlab


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
