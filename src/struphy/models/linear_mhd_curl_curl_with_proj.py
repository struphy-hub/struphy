from struphy.models.linear_mhd_curl_curl import LinearMHDCurlCurl
from struphy.models.variables import FEECVariable
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator


class LinearMHDCurlCurlwithProj(LinearMHDCurlCurl):
    """Linear ideal MHD with zero-flow equilibrium, with the magnetic field in :math:`H(\\textnormal{curl})` and the
    exactly energy conserving shear-Alfvén step.

    Same as :class:`~struphy.models.linear_mhd_curl_curl.LinearMHDCurlCurl`, but the curl-curl term of the implicit
    shear-Alfvén step contains the :math:`\\rho_0`-weighted :math:`L^2` projection of the Lorentz force onto
    :math:`H(\\textnormal{div})` (as in :class:`~struphy.models.shear_alfven.ShearAlfven` with ``b_space="Hcurl", projected=True``),
    at the price of an inner mass matrix solve in every application of the Schur operator (and no effective curl-curl
    preconditioner).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the ion species (default: 1.0)
    """

    class Propagators:
        def __init__(self, b_field: FEECVariable = None):
            self.shear_alf = ShearAlfvenCurlCurlPropagator(projected=True)
            self.mag_sonic = Magnetosonic(b_field=b_field)

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy (perturbation): :math:`E_U = \frac{1}{2} \int \rho_0 |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V`
        - Magnetic energy (perturbation, magnetic field as a 1-form): :math:`E_B = \frac{1}{2} \int \frac{|\tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`
        - Mean pressure perturbation: :math:`E_p = \int \frac{\tilde{p}}{\gamma - 1} \, \mathrm{d}V` (``en_p``). This is a linear diagnostic, not an energy.
        - Thermal perturbation energy: :math:`E_{\mathrm{thermal}} = \frac{1}{2}\int \frac{\tilde{p}^2}{\gamma p_0} \, \mathrm{d}V` (``en_thermal``), with :math:`\gamma = 5/3`
        - Total perturbed energy: :math:`E_{\mathrm{tot}} = E_U + E_B + E_{\mathrm{thermal}}` (``en_tot``),
          conserved by the shear-Alfvén step up to the solver tolerances
        - Equilibrium magnetic energy: :math:`E_{B0} = \frac{1}{2} \int \frac{|\mathbf{B}_0|^2}{\mu_0} \, \mathrm{d}V` (with :math:`\mathbf B_0` projected as a 1-form)
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V`
        - Total magnetic energy: :math:`E_{B,\mathrm{tot}} = \frac{1}{2} \int \frac{|\mathbf{B}_0 + \tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator` (``projected=True``)
        2. :class:`~struphy.propagators.magnetosonic.Magnetosonic`"""
        doc = rf"""**1. ShearAlfvenCurlCurlPropagator** (``projected=True``):

{ShearAlfvenCurlCurlPropagator.__doc__}

**2. Magnetosonic:**

{Magnetosonic.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""LinearMHDCurlCurlwithProj solves the same equations as
        :class:`~struphy.models.linear_mhd_curl_curl.LinearMHDCurlCurl` (magnetic field in
        :math:`H(\textnormal{curl})`, weak induction equation, momentum equation projected onto
        :math:`H(\textnormal{div})`), with the projection also in the curl-curl term of the implicit shear-Alfvén step,
        such that this step conserves the perturbed energy exactly. The Schur complement then contains the inverse of
        the density-weighted mass matrix of 2-forms. The magnetosonic step is that of
        :class:`~struphy.models.linear_mhd_curl_curl.LinearMHDCurlCurl`. Free boundary conditions are recommended for
        bounded domains (see :class:`~struphy.models.linear_mhd_curl_curl.LinearMHDCurlCurl`)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the energy conserving linear MHD curl-curl model:

        .. code-block:: python

            from struphy.models import LinearMHDCurlCurlwithProj

            model = LinearMHDCurlCurlwithProj()
            model.em_fields.b_field
            model.mhd.velocity
            model.scalars["en_tot"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear MHD wave propagation with exact energy conservation of the shear-Alfvén step
        - comparison with :class:`~struphy.models.linear_mhd_curl_curl.LinearMHDCurlCurl` and
          :class:`~struphy.models.linear_mhd.LinearMHD`"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear MHD dynamics with finite-amplitude perturbations
        - equilibria with non-zero background flow
        - dissipative effects such as resistivity or viscosity
        - kinetic or particle-field effects beyond the fluid MHD description
        - large problems where the inner mass matrix solves of the shear-Alfvén step are too expensive (use
          :class:`~struphy.models.linear_mhd_curl_curl.LinearMHDCurlCurl`)"""
