from struphy.io.options import BaseUnits
from struphy.models.shear_alfven_curl_curl import ShearAlfvenCurlCurl
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlwithProjPropagator


class ShearAlfvenCurlCurlwithProj(ShearAlfvenCurlCurl):
    """Shear-Alfvén part of linear MHD with zero-flow equilibrium, with the magnetic field in :math:`H(\\textnormal{curl})`
    and the exactly energy conserving time step.

    Same as :class:`~struphy.models.shear_alfven_curl_curl.ShearAlfvenCurlCurl`, but the curl-curl term of the implicit
    time step contains the :math:`\\rho_0`-weighted :math:`L^2` projection of the Lorentz force onto
    :math:`H(\\textnormal{div})`, which conserves the perturbed energy exactly (up to the solver tolerances), at the
    price of an inner mass matrix solve in every application of the Schur operator (and no effective curl-curl
    preconditioner).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the ion species (default: 1.0)
    """

    class Propagators:
        def __init__(self) -> None:
            self.shear_alf = ShearAlfvenCurlCurlwithProjPropagator()

    def __init__(self, base_units: BaseUnits = BaseUnits(), mass_number: float = 1.0):
        super().__init__(base_units=base_units, mass_number=mass_number)

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Perturbed kinetic energy: ``en_U``
        - Perturbed magnetic energy: ``en_B``
        - Background magnetic energy: ``en_B_eq``
        - Total magnetic energy including background: ``en_B_tot``
        - Perturbation total energy: ``en_tot`` (conserved by the implicit time step up to the solver tolerances)
        - Total energy including background contribution: ``en_tot2``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlwithProjPropagator`
        """
        doc = rf"""**1. ShearAlfvenCurlCurlwithProjPropagator:**

{ShearAlfvenCurlCurlwithProjPropagator.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""ShearAlfvenCurlCurlwithProj solves the same equations as
        :class:`~struphy.models.shear_alfven_curl_curl.ShearAlfvenCurlCurl`
        (magnetic field in :math:`H(\textnormal{curl})`, weak induction
        equation, momentum equation projected onto
        :math:`H(\textnormal{div})`), with the projection also in the
        curl-curl term of the implicit time step, such that the perturbed
        energy is conserved exactly. The Schur complement then contains the
        inverse of the density-weighted mass matrix of 2-forms."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the energy conserving shear-Alfvén curl-curl model:

        .. code-block:: python

            from struphy.models import ShearAlfvenCurlCurlwithProj

            model = ShearAlfvenCurlCurlwithProj()
            model.em_fields.b_field
            model.mhd.velocity
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - isolated shear-Alfvén wave propagation with exact energy conservation
        - comparison with :class:`~struphy.models.shear_alfven_curl_curl.ShearAlfvenCurlCurl`"""
