import pytest

from struphy.propagators.current_coupling_5d_curlb import CurrentCoupling5DCurlb
from struphy.propagators.current_coupling_5d_gradb import CurrentCoupling5DGradB


@pytest.mark.parametrize("u_space", ["Hdiv", "H1vec"])
def test_curlb_supported_u_space(u_space):
    CurrentCoupling5DCurlb.Options(u_space=u_space)


def test_curlb_rejects_hcurl():
    with pytest.raises(ValueError):
        CurrentCoupling5DCurlb.Options(u_space="Hcurl")


@pytest.mark.parametrize(
    "algo, u_space",
    [("explicit", "Hdiv"), ("explicit", "H1vec"), ("discrete_gradient", "Hdiv")],
)
def test_gradb_supported_u_space(algo, u_space):
    CurrentCoupling5DGradB.Options(algo=algo, u_space=u_space)


@pytest.mark.parametrize(
    "algo, u_space",
    [("explicit", "Hcurl"), ("discrete_gradient", "Hcurl"), ("discrete_gradient", "H1vec")],
)
def test_gradb_rejects_unsupported_u_space(algo, u_space):
    with pytest.raises(ValueError):
        CurrentCoupling5DGradB.Options(algo=algo, u_space=u_space)
