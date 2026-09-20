from struphy.models import HasegawaWakatani


def test_hasegawa_wakatani_couples_vorticity_to_potential():
    model = HasegawaWakatani()

    assert model.propagators.poisson.rho is model.plasma.vorticity
    assert model.propagators.hw.options.coupling == 1.0
