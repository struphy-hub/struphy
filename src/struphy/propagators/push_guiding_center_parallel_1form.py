"Only particle variables are updated."

from struphy.models.variables import FEECVariable
from struphy.pic.pushing.kernels.bstar_1form import bstar_1form
from struphy.pic.pushing.kernels.push_gc_Bstar_discrete_gradient_2nd_order_1form import (
    push_gc_Bstar_discrete_gradient_2nd_order_1form,
)
from struphy.pic.pushing.kernels.push_gc_Bstar_explicit_multistage_1form import (
    push_gc_Bstar_explicit_multistage_1form,
)
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel


class PushGuidingCenterParallel1form(PushGuidingCenterParallel):
    r""":class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel` with the magnetic
    field as a 1-form (in :math:`H(\textnormal{curl})`).

    **Continuous problem.** For each marker :math:`p`, solve

    .. math::

        \left\{
            \begin{aligned}
                \frac{\textnormal d \mathbf X_p(t)}{\textnormal d t} &= v_{\parallel,p}(t) \frac{\mathbf B^*}{B^*_\parallel}(\mathbf X_p(t)) \,,
                \\
                \frac{\textnormal d v_{\parallel,p}(t)}{\textnormal d t} &= \frac{1}{\varepsilon} \frac{\mathbf B^*}{B^*_\parallel} \cdot \mathbf E^* (\mathbf X_p(t)) \,,
            \end{aligned}
        \right.

    where

    .. math::

        \mathbf E^* = -\nabla \phi - \varepsilon \mu_p \nabla |\mathbf B|\,,\qquad
        \mathbf B^* = \mathbf B + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad
        B^*_\parallel = \mathbf B^* \cdot \mathbf b_0\,,

    with :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` and the magnetic perturbation
    :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` (``b_tilde``, not updated); in :math:`|\mathbf B|` and
    :math:`B^*_\parallel` the field enters through :math:`\mathbf B \cdot \mathbf b_0 = |\mathbf B_0| +
    \mathbf b_0 \cdot \tilde{\mathbf B}`. The potential ``phi`` and/or ``b_tilde`` can be ignored by passing ``None``;
    instead of the potential, an electric field :math:`\mathbf E \in H(\textnormal{curl})` can be passed
    (``e_field``, explicit algorithm), then :math:`\mathbf E^* = \mathbf E - \varepsilon \mu_p \nabla |\mathbf B|`.

    **Semi-discrete problem.** The total field is a 1-form: the commuting projection :math:`\hat \Pi^1` of the
    equilibrium 1-form :math:`\hat{\mathbf B}^1_0` (``projected_equil.b1``) plus the perturbation with FE coefficients
    :math:`\mathbf b \in \mathbb R^{N_1}`,

    .. math::

        \hat{\mathbf B}^1 = \hat \Pi^1 \hat{\mathbf B}^1_0 + \mathbf b^\top \vec \Lambda^1 \in V^1_h\,,\qquad
        \mathbf B = DF^{-\top} \hat{\mathbf B}^1\,,

    never converted to a 2-form. As in
    :class:`~struphy.propagators.push_guiding_center_bx_estar_1form.PushGuidingCenterBxEstar1form`, the parallel
    component is :math:`\hat B_\parallel = \hat \Pi^0 |\hat{\mathbf B}_0| + \mathcal P^{b,1} \mathbf b` with
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.PB1form`, and
    :math:`\hat \nabla \hat B_\parallel = \hat \Pi^1 \hat \nabla |\hat{\mathbf B}_0| + \mathbb G \mathcal P^{b,1}
    \mathbf b`. In logical coordinates :math:`\mathbf X = F(\boldsymbol \eta)`, with the vector-field components
    :math:`\hat{\mathbf B}^\textnormal{v} = G^{-1} \hat{\mathbf B}^1` evaluated at the markers,

    .. math::

        \left\{
            \begin{aligned}
                \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} &= v_{\parallel,p}(t) \frac{\hat{\mathbf B}^{*\textnormal{v}}}{\hat B^{*}_\parallel}(\boldsymbol \eta_p(t)) \,,
                \\
                \frac{\textnormal d v_{\parallel,p}(t)}{\textnormal d t} &= \frac{1}{\varepsilon} \frac{\hat{\mathbf B}^{*\textnormal{v}}}{\hat B^{*}_\parallel} \cdot \hat{\mathbf E}^{*1} (\boldsymbol \eta_p(t)) \,,
            \end{aligned}
        \right.

    .. math::

        \hat{\mathbf B}^{*\textnormal{v}} = G^{-1} \hat{\mathbf B}^1 + \frac{\varepsilon v_\parallel}{\sqrt g}
        \hat \nabla \times \hat{\mathbf b}^1_0\,,\qquad
        \hat B^*_\parallel = \hat B_\parallel + \varepsilon v_\parallel
        \widehat{\left[(\nabla \times \mathbf b_0) \cdot \mathbf b_0\right]}\,,\qquad
        \hat{\mathbf E}^{*1} = -\hat \nabla \hat \phi - \varepsilon \mu_p \hat \nabla \hat B_\parallel\,.

    This equals the 2-form system :math:`\hat{\mathbf B}^{*2} / \hat B^{*3}_\parallel` of the parent class, since
    :math:`\hat{\mathbf B}^2 = \sqrt g \, G^{-1} \hat{\mathbf B}^1` and :math:`\hat B^{*3}_\parallel = \sqrt g \,
    \hat B^*_\parallel`; the factor :math:`\sqrt g` cancels in the magnetic-field term.

    **Time discretization.** As in
    :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel`, selected by ``algo``:

    - ``"discrete_gradient_1st_order"`` (default) and ``"discrete_gradient_1st_order_newton"``: discrete gradient
      with :math:`\hat{\mathbf B}^*` frozen at :math:`\mathbf Z^n_p` (Picard resp. Newton iteration up to ``tol``,
      ``maxiter``). The unchanged pushers
      :func:`~struphy.pic.pushing.kernels.push_gc_Bstar_discrete_gradient_1st_order.push_gc_Bstar_discrete_gradient_1st_order_kernels.push_gc_Bstar_discrete_gradient_1st_order`
      (resp. ``_newton``) read :math:`\sqrt g \hat B^*_\parallel` and
      :math:`\sqrt g \, \hat{\mathbf B}^{*\textnormal{v}}` (written by
      :func:`~struphy.pic.pushing.kernels.bstar_1form.bstar_1form_kernels.bstar_1form` from the 1-form) at
      :math:`\mathbf Z^n_p`;
    - ``"discrete_gradient_2nd_order"``: discrete gradient with :math:`\hat{\mathbf B}^*` at the mid-point,
      :func:`~struphy.pic.pushing.kernels.push_gc_Bstar_discrete_gradient_2nd_order_1form.push_gc_Bstar_discrete_gradient_2nd_order_1form_kernels.push_gc_Bstar_discrete_gradient_2nd_order_1form`;
    - ``"explicit"``: explicit Runge-Kutta with :class:`~struphy.ode.utils.ButcherTableau` ``butcher``,
      :func:`~struphy.pic.pushing.kernels.push_gc_Bstar_explicit_multistage_1form.push_gc_Bstar_explicit_multistage_1form_kernels.push_gc_Bstar_explicit_multistage_1form`
      (forced if ``e_field`` is passed).

    The discrete gradients conserve the Hamiltonian :math:`\varepsilon v_\parallel^2/2 + \varepsilon \mu_p
    \hat B_\parallel + \hat \phi`.
    """

    _b_form = "1"
    _pb_name = "PB1form"
    _kernels = {
        "bstar": bstar_1form,
        "discrete_gradient_2nd_order": push_gc_Bstar_discrete_gradient_2nd_order_1form,
        "explicit": push_gc_Bstar_explicit_multistage_1form,
    }

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
            Magnetic perturbation in ``"Hcurl"`` space (1-form) contributing to :math:`B^*`.
            If ``None``, the perturbation is ignored.
        e_field : FEECVariable, default=None
            Electric field in ``"Hcurl"`` space, used in :math:`E^* = E - \varepsilon \mu \nabla |B|` instead of
            :math:`-\nabla \phi`. Cannot be combined with ``phi``; forces the explicit algorithm.
        """
        super().__init__(phi=phi, b_tilde=b_tilde, e_field=e_field)
