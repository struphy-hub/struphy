import pytest


def test_adiabatic_phi_not_available():
    """AdiabaticPhi is not ported to the current Propagator API and must fail clearly at construction."""

    from struphy.propagators import AdiabaticPhi

    with pytest.raises(NotImplementedError, match="AdiabaticPhi"):
        AdiabaticPhi(None)


if __name__ == "__main__":
    test_adiabatic_phi_not_available()
