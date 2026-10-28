from struphy.models.linear_mhd import LinearMHD
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator


class LinearMHDCurlCurl(LinearMHD):
    """Linear ideal MHD with zero-flow equilibrium, with the magnetic field in :math:`H(\\textnormal{curl})`.

    Same equations as :class:`~struphy.models.linear_mhd.LinearMHD`, but with a weak induction equation, such that the
    implicit shear-Alfvén step leads to a curl-curl system for the magnetic field, which is preconditioned by the
    Hiptmair-Xu preconditioner (as in :class:`~struphy.models.shear_alfven.ShearAlfven` with ``b_space="Hcurl"``).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the ion species (default: 1.0)
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hcurl")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self, b_field: FEECVariable = None):
            self.shear_alf = ShearAlfvenCurlCurlPropagator()
            self.mag_sonic = Magnetosonic(b_field=b_field)

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity (mass conservation):

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        Momentum (Lorentz force):

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}

        Energy (adiabatic process):

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        Induction (Faraday's law, solved weakly in :math:`H(\textnormal{curl})`):

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0) = 0.
        """

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the Alfvén velocity:

        .. math::

            \hat{U} = \hat{v}_\mathrm{A} = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}

        The model therefore uses the Alfvén velocity as its characteristic speed scale."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy (perturbation): :math:`E_U = \frac{1}{2} \int \rho_0 |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V`
        - Magnetic energy (perturbation, magnetic field as a 1-form): :math:`E_B = \frac{1}{2} \int \frac{|\tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`
        - Mean pressure perturbation: :math:`E_p = \int \frac{\tilde{p}}{\gamma - 1} \, \mathrm{d}V` (``en_p``). This is a linear diagnostic, not an energy.
        - Thermal perturbation energy: :math:`E_{\mathrm{thermal}} = \frac{1}{2}\int \frac{\tilde{p}^2}{\gamma p_0} \, \mathrm{d}V` (``en_thermal``), with :math:`\gamma = 5/3`
        - Total perturbed energy: :math:`E_{\mathrm{tot}} = E_U + E_B + E_{\mathrm{thermal}}` (``en_tot``). The
          shear-Alfvén step damps it numerically (the damping vanishes with the resolution along the background field;
          exactly conserving shear-Alfvén step in
          :class:`~struphy.models.linear_mhd_curl_curl_with_proj.LinearMHDCurlCurlwithProj`).
        - Equilibrium magnetic energy: :math:`E_{B0} = \frac{1}{2} \int \frac{|\mathbf{B}_0|^2}{\mu_0} \, \mathrm{d}V` (with :math:`\mathbf B_0` projected as a 1-form)
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V`
        - Total magnetic energy: :math:`E_{B,\mathrm{tot}} = \frac{1}{2} \int \frac{|\mathbf{B}_0 + \tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`
        2. :class:`~struphy.propagators.magnetosonic.Magnetosonic`"""
        doc = rf"""**1. ShearAlfvenCurlCurlPropagator:**

{ShearAlfvenCurlCurlPropagator.__doc__}

**2. Magnetosonic:**

{Magnetosonic.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""LinearMHDCurlCurl solves the same equations as :class:`~struphy.models.linear_mhd.LinearMHD`, with the
        perturbed magnetic field in :math:`H(\textnormal{curl})` instead of :math:`H(\textnormal{div})`. The splitting is
        the same: the shear-Alfvén step of
        :class:`~struphy.models.shear_alfven.ShearAlfven` with ``b_space="Hcurl"`` (weak induction equation, field-aligned
        curl-curl system for the magnetic field, Hiptmair-Xu preconditioner by default), followed by the magnetosonic
        step, in which the force :math:`(\nabla \times \mathbf B_0) \times \tilde{\mathbf B}` of the 1-form
        :math:`\tilde{\mathbf B}` enters through the mixed mass matrix
        :attr:`~struphy.feec.mass.WeightedMassOperators.M21J`.

        The shear-Alfvén step is not exactly energy conserving: it damps the waves numerically, with a damping that
        vanishes quickly with the resolution along the background field (see
        :class:`~struphy.models.linear_mhd_curl_curl_with_proj.LinearMHDCurlCurlwithProj` for the exactly energy
        conserving version).

        Boundary conditions: with essential boundary conditions in the de Rham sequence, the tangential components of
        the magnetic field (a 1-form) vanish on that boundary, which is unphysical at a wall with
        :math:`\mathbf B_0 \cdot \mathbf n = 0`. Free boundary conditions are therefore recommended: they impose
        :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` weakly in the shear-Alfvén step and :math:`\tilde p = 0` weakly in
        the magnetosonic step (natural boundary condition of the pressure term).

        All evolved quantities are perturbations around a stationary equilibrium:

        - :math:`\tilde{\rho}` - density perturbation
        - :math:`\tilde{\mathbf{U}}` - velocity perturbation (2-form)
        - :math:`\tilde{p}` - pressure perturbation
        - :math:`\tilde{\mathbf{B}}` - magnetic field perturbation (1-form)"""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a linear MHD curl-curl model:

        .. code-block:: python

            from struphy.models import LinearMHDCurlCurl

            model = LinearMHDCurlCurl()

            # Access fields
            model.em_fields.b_field
            model.mhd.density
            model.mhd.velocity
            model.mhd.pressure

            # Access tracked scalar quantities
            model.scalars["en_U"]
            model.scalars["en_B"]
            model.scalars["en_tot"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear MHD wave propagation with large time steps (fast curl-curl solve of the shear-Alfvén step)
        - comparison with :class:`~struphy.models.linear_mhd.LinearMHD` (magnetic field in :math:`H(\textnormal{div})`)
        - testing curl-curl (Hiptmair-Xu) preconditioning in linear MHD"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear MHD dynamics with finite-amplitude perturbations
        - equilibria with non-zero background flow
        - dissipative effects such as resistivity or viscosity
        - kinetic or particle-field effects beyond the fluid MHD description
        - polar splines with the Hiptmair-Xu preconditioner"""
