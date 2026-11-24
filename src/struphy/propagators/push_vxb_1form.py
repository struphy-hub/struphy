"Only particle variables are updated."

from struphy.models.variables import FEECVariable
from struphy.pic.pushing.kernels.push_vxb_analytic_1form import push_vxb_analytic_1form
from struphy.pic.pushing.kernels.push_vxb_implicit_1form import push_vxb_implicit_1form
from struphy.propagators.push_vxb import PushVxB


class PushVxB1form(PushVxB):
    r""":class:`~struphy.propagators.push_vxb.PushVxB` with the magnetic field as a 1-form (in
    :math:`H(\textnormal{curl})`).

    **Continuous problem.** For each marker :math:`p`, solve

    .. math::

        \frac{\textnormal d \mathbf{v}_p(t)}{\textnormal d t} =  \frac{1}{\varepsilon} \, \mathbf{v}_p(t) \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) (\boldsymbol \eta_p)\,,

    at fixed positions, where :math:`\varepsilon = 1/(\hat{\Omega}_c \hat t)` is a constant scaling factor,
    :math:`\mathbf B_0` is the equilibrium field and :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` an optional
    magnetic field variable (``b1_var``, not updated).

    **Semi-discrete problem.** Both fields are 1-forms: :math:`\mathbf B_0` is the commuting projection
    :math:`\hat \Pi^1` of the equilibrium 1-form :math:`\hat{\mathbf B}^1_0` (``projected_equil.b1``) and
    :math:`\tilde{\mathbf B}` has the FE coefficients :math:`\mathbf b \in \mathbb R^{N_1}`, so that the Cartesian
    rotation vector at the marker is

    .. math::

        \mathbf{B}(\boldsymbol \eta_p) = DF^{-\top}(\boldsymbol \eta_p) \left( \hat \Pi^1 \hat{\mathbf B}^1_0 + \tilde{\mathbf b}^\top \vec \Lambda^1 \right) (\boldsymbol \eta_p)\,,

    with no conversion to 2-forms.

    **Time discretization.** The rotation vector is constant in the step:

    - ``algo="analytic"`` (default): exact rotation of :math:`\mathbf v_p` about :math:`\mathbf B(\boldsymbol \eta_p)`
      by the angle :math:`|\mathbf B| \Delta t / \varepsilon`.
    - ``algo="implicit"``: Crank-Nicolson (implicit mid-point), a 3x3 solve per marker; it conserves
      :math:`|\mathbf v_p|` too.

    Without magnetic field (no equilibrium field or a zero one, and no ``b1_var``) the propagator does nothing.
    """

    _b_form = "1"
    _kernels = {"analytic": push_vxb_analytic_1form, "implicit": push_vxb_implicit_1form}

    def __init__(self, b1_var: FEECVariable = None):
        """
        Parameters
        ----------
        b1_var : FEECVariable, default=None
            Optional additional magnetic-field 1-form (``"Hcurl"`` space) added to the projected equilibrium 1-form
            before pushing.
        """
        assert b1_var is None or b1_var.space == "Hcurl", "b1_var must be in Hcurl."
        self.variables = self.Variables()
        self.b1_var = b1_var

    @property
    def _b_var(self) -> FEECVariable | None:
        return self.b1_var
