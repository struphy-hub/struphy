"Only particle variables are updated."

from struphy.models.variables import FEECVariable
from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar


class PushGuidingCenterBxEstar1form(PushGuidingCenterBxEstar):
    r""":class:`~struphy.propagators.push_guiding_center_bx_estar.PushGuidingCenterBxEstar` with the magnetic
    perturbation as a 1-form (in :math:`H(\textnormal{curl})`).

    **Continuous problem.** For each marker :math:`p`, solve

    .. math::

        \frac{\textnormal d \mathbf{X}_p(t)}{\textnormal d t} = \frac{\mathbf{E}^* \times \mathbf{b}_0}{B_{\parallel}^*} (\mathbf{X}_p(t))   \,,

    where

    .. math::

        \mathbf E^* = -\nabla \phi - \varepsilon \mu_p \nabla |\mathbf B|\,,\qquad
        B^*_\parallel = \mathbf B \cdot \mathbf b_0 + \varepsilon v_\parallel (\nabla \times \mathbf b_0) \cdot \mathbf b_0\,,

    with :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` and the magnetic perturbation
    :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` (``b_tilde``, not updated), approximated to first order by
    :math:`|\mathbf B| \approx \mathbf B \cdot \mathbf b_0 = |\mathbf B_0| + \mathbf b_0 \cdot \tilde{\mathbf B}`.
    The potential ``phi`` and/or ``b_tilde`` can be ignored by passing ``None``; instead of the potential, an electric
    field :math:`\mathbf E \in H(\textnormal{curl})` can be passed (``e_field``), then
    :math:`\mathbf E^* = \mathbf E - \varepsilon \mu_p \nabla |\mathbf B|`.

    **Semi-discrete problem.** The perturbation has the FE coefficients :math:`\mathbf b \in \mathbb R^{N_1}`,
    :math:`\tilde{\mathbf B} = DF^{-\top} \tilde{\mathbf b}^\top \vec \Lambda^1`. With
    :math:`\mathbf b_0 = DF \, \hat{\mathbf b}^\textnormal{v}_0` the parallel component has no metric factor,
    :math:`\mathbf b_0 \cdot \tilde{\mathbf B} = \hat{\mathbf b}^\textnormal{v}_0 \cdot \hat{\mathbf B}^1`, and it is
    projected to :math:`H^1` with the basis operator
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.PB1form`,
    :math:`\mathcal P^{b,1} = \hat \Pi^0 [\hat{\mathbf b}^\textnormal{v}_0 \cdot \vec \Lambda^1]`:

    .. math::

        \hat B_\parallel = \hat \Pi^0 |\hat{\mathbf B}_0| + \mathcal P^{b,1} \mathbf b \in V^0_h\,,\qquad
        \hat \nabla \hat B_\parallel = \hat \Pi^1 \hat \nabla |\hat{\mathbf B}_0| + \mathbb G \mathcal P^{b,1} \mathbf b
        \in V^1_h\,,

    with the discrete gradient :math:`\mathbb G`. In logical coordinates :math:`\mathbf X = F(\boldsymbol \eta)`,

    .. math::

        \frac{\textnormal d \boldsymbol{\eta}_p(t)}{\textnormal d t} = \frac{\hat{\mathbf E}^{*1} \times \hat{\mathbf b}^1_0}{\sqrt{g}\,\hat{B}_{\parallel}^{*}} (\boldsymbol \eta_p(t)) \,,\qquad
        \hat{\mathbf E}^{*1} = -\hat \nabla \hat \phi - \varepsilon \mu_p \hat \nabla \hat B_\parallel\,.

    The kernels see the perturbation only through :math:`\hat B_\parallel` and :math:`\hat \nabla \hat B_\parallel`, so
    they are those of the 2-form version; only the operator :math:`\mathcal P^{b,1}` replaces
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.PB`.

    **Time discretization.** As in
    :class:`~struphy.propagators.push_guiding_center_bx_estar.PushGuidingCenterBxEstar`, selected by ``algo``:

    - ``"discrete_gradient_1st_order"`` (default), ``"discrete_gradient_1st_order_newton"``,
      ``"discrete_gradient_2nd_order"``: discrete-gradient schemes (Picard or Newton iteration up to ``tol``,
      ``maxiter``) conserving the Hamiltonian :math:`\varepsilon v_\parallel^2/2 + \varepsilon \mu_p \hat B_\parallel
      + \hat \phi`;
    - ``"explicit"``: explicit Runge-Kutta with :class:`~struphy.ode.utils.ButcherTableau` ``butcher`` (forced if
      ``e_field`` is passed).
    """

    _b_form = "1"
    _pb_name = "PB1form"

    def __init__(
        self,
        phi: FEECVariable = None,
        b_tilde: FEECVariable = None,
        e_field: FEECVariable = None,
    ):
        r"""
        Parameters
        ----------
        phi : FEECVariable, default=None
            Electric potential in ``"H1"`` space contributing to :math:`E^*`.
            If ``None``, an empty ``FEECVariable(space="H1")`` is created internally.
        b_tilde : FEECVariable, default=None
            Magnetic perturbation in ``"Hcurl"`` space (1-form) contributing to :math:`|B|`.
            If ``None``, the perturbation is ignored.
        e_field : FEECVariable, default=None
            Electric field in ``"Hcurl"`` space, used in :math:`E^* = E - \varepsilon \mu \nabla |B|` instead of
            :math:`-\nabla \phi`. Cannot be combined with ``phi``; forces the explicit algorithm.
        """
        super().__init__(phi=phi, b_tilde=b_tilde, e_field=e_field)
