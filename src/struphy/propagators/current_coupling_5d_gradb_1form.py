"Particle and FEEC variables are updated."

from feectools.linalg.block import BlockVector

from struphy.models.variables import FEECVariable
from struphy.pic.accumulation.kernels.cc_lin_mhd_5d_gradB_1form import cc_lin_mhd_5d_gradB_1form
from struphy.pic.accumulation.kernels.cc_lin_mhd_5d_gradB_dg_1form import cc_lin_mhd_5d_gradB_dg_1form
from struphy.pic.accumulation.kernels.cc_lin_mhd_5d_gradB_dg_init_1form import cc_lin_mhd_5d_gradB_dg_init_1form
from struphy.pic.pushing.kernels.push_gc_cc_J2_dg_Hdiv_1form import push_gc_cc_J2_dg_Hdiv_1form
from struphy.pic.pushing.kernels.push_gc_cc_J2_dg_init_Hdiv_1form import push_gc_cc_J2_dg_init_Hdiv_1form
from struphy.pic.pushing.kernels.push_gc_cc_J2_stage_H1vec_1form import push_gc_cc_J2_stage_H1vec_1form
from struphy.pic.pushing.kernels.push_gc_cc_J2_stage_Hdiv_1form import push_gc_cc_J2_stage_Hdiv_1form
from struphy.propagators.current_coupling_5d_gradb import CurrentCoupling5DGradB


class CurrentCoupling5DGradB1form(CurrentCoupling5DGradB):
    r""":class:`~struphy.propagators.current_coupling_5d_gradb.CurrentCoupling5DGradB` with the magnetic field as a
    1-form (in :math:`H(\textnormal{curl})`).

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in \{H(\textnormal{div}), (H^1)^3\}` and the
    guiding-center distribution :math:`f` such that

    .. math::

        \left\{
            \begin{aligned}
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf x
                = - \frac{A_\textnormal{h}}{A_\textnormal{b}} \iint \mu \frac{f^\text{vol}}{B^*_\parallel}
                (\mathbf b_0 \times \nabla B_\parallel) \times \mathbf B \cdot \mathbf V \,\textnormal{d} \mathbf x
                \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \,\mathbf V \in \{H(\textnormal{div}),
                (H^1)^3\}\,,
                \\
                &\frac{\partial f}{\partial t} = \frac{1}{B^*_\parallel} \left[ \mathbf b_0 \times (\tilde{\mathbf U}
                \times \mathbf B) \right] \cdot \nabla f\,,
            \end{aligned}
        \right.

    with :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`, the magnetic perturbation
    :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` (``b_tilde``, not updated), :math:`B_\parallel = \mathbf b_0
    \cdot \mathbf B` and :math:`B^*_\parallel = \mathbf b_0 \cdot (\mathbf B + \epsilon v_\parallel \nabla \times
    \mathbf b_0)`.

    **Semi-discrete problem.** The total field is a 1-form, the commuting projection :math:`\hat \Pi^1` of the
    equilibrium 1-form :math:`\hat{\mathbf B}^1_0` (``projected_equil.b1``) plus the perturbation with FE coefficients
    :math:`\mathbf b \in \mathbb R^{N_1}` (extracted to the tensor-product space),

    .. math::

        \hat{\mathbf B}^1 = \hat \Pi^1 \hat{\mathbf B}^1_0 + \mathbf b^\top \vec \Lambda^1\,,\qquad
        \hat{\mathbf B}^\textnormal{v} = G^{-1} \hat{\mathbf B}^1\,,

    never converted to a 2-form. The parallel component is :math:`\hat B_\parallel = \hat \Pi^0 |\hat{\mathbf B}_0|
    + \mathcal P^{b,1} \mathbf b` with the basis operator
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.PB1form` (:math:`\mathcal P^{b,1}`, the
    1-form version of ``PB``), and its gradient is :math:`\hat \nabla \hat B_\parallel = \hat \Pi^1 \hat \nabla
    |\hat{\mathbf B}_0| + \mathbb G \mathcal P^{b,1} \mathbf b` (``projected_equil.gradB1`` plus the discrete gradient
    :math:`\mathbb G`). With the unit vector :math:`\hat{\mathbf b}^1_0` (``projected_equil.unit_b1``), the 2-form
    :math:`\hat{\mathbf c} = \hat \nabla \times \hat{\mathbf b}^1_0` (``projected_equil.curl_unit_b2``) and
    :math:`\hat B^{*\textnormal{v}}_\parallel = \hat{\mathbf b}^1_0 \cdot (\hat{\mathbf B}^\textnormal{v} + \epsilon
    v_\parallel \hat{\mathbf c} / \sqrt g)`, the FE coefficients :math:`\mathbf u \in \mathbb R^{N_\alpha}` of
    :math:`\tilde{\mathbf U}` and the marker positions :math:`\boldsymbol \eta_p` satisfy

    .. math::

        \left\{
            \begin{aligned}
                \mathbb M^\alpha_{\rho_0} \dot{\mathbf u} &= \sum_p \vec \Lambda^\alpha(\boldsymbol \eta_p)
                \mathbf B_p\,,\qquad \mathbf B_p = w_p \mu_p \frac{A_\textnormal{h}}{A_\textnormal{b}}
                \frac{s_\alpha}{\hat B^{*\textnormal{v}}_\parallel} \hat{\mathbf B}^\textnormal{v} \times
                \left(\hat{\mathbf b}^1_0 \times \hat \nabla \hat B_\parallel\right)\,,
                \\
                \dot{\boldsymbol \eta}_p &= - \frac{1}{\hat B^{*\textnormal{v}}_\parallel} \hat{\mathbf b}^1_0 \times
                \left(\hat{\mathbf B}^\textnormal{v} \times s_\alpha (\mathbf u^\top \vec \Lambda^\alpha)\right)\,,
            \end{aligned}
        \right.

    at :math:`\boldsymbol \eta_p`, where :math:`s_\alpha = 1/\sqrt g` for :math:`H(\textnormal{div})` and
    :math:`s_\alpha = 1` for :math:`(H^1)^3` (``u_space``), and :math:`\mathbb M^\alpha_{\rho_0}` is the
    density-weighted mass matrix. This equals the 2-form system of the parent class, since
    :math:`\hat{\mathbf B}^2 = \sqrt g\, \hat{\mathbf B}^\textnormal{v}` and :math:`\hat{\mathbf b}^1_0 \cdot
    \hat{\mathbf B}^{*2} = \sqrt g\, \hat B^{*\textnormal{v}}_\parallel`: the factor :math:`\sqrt g` cancels in
    their ratio.

    **Time discretization.** Selected by ``algo``:

    - ``"explicit"`` (default): explicit Runge-Kutta with the :class:`~struphy.ode.utils.ButcherTableau` ``butcher``
      for :math:`(\mathbf u, \boldsymbol \eta_p)`, one mass-matrix solve (``solver``, ``precond``) per stage; kernels
      :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_gradB_1form.cc_lin_mhd_5d_gradB_1form_kernels.cc_lin_mhd_5d_gradB_1form`
      and
      :func:`~struphy.pic.pushing.kernels.push_gc_cc_J2_stage_Hdiv_1form.push_gc_cc_J2_stage_Hdiv_1form_kernels.push_gc_cc_J2_stage_Hdiv_1form`
      (resp. ``push_gc_cc_J2_stage_H1vec_1form``).
    - ``"discrete_gradient"`` (only :math:`H(\textnormal{div})`): discrete gradient of the energy
      :math:`\mathbf u^\top \mathbb M^\alpha_{\rho_0} \mathbf u / 2 + \frac{A_\textnormal{h}}{A_\textnormal{b}}
      \frac{1}{N} \sum_p w_p \mu_p \hat B_\parallel(\boldsymbol \eta_p)`, solved by relaxed fixed-point iterations
      (``dg_solver_params``) with the initial guess from a forward Euler step; kernels
      :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_gradB_dg_init_1form.cc_lin_mhd_5d_gradB_dg_init_1form_kernels.cc_lin_mhd_5d_gradB_dg_init_1form`,
      :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_gradB_dg_1form.cc_lin_mhd_5d_gradB_dg_1form_kernels.cc_lin_mhd_5d_gradB_dg_1form`,
      :func:`~struphy.pic.pushing.kernels.push_gc_cc_J2_dg_init_Hdiv_1form.push_gc_cc_J2_dg_init_Hdiv_1form_kernels.push_gc_cc_J2_dg_init_Hdiv_1form`
      and
      :func:`~struphy.pic.pushing.kernels.push_gc_cc_J2_dg_Hdiv_1form.push_gc_cc_J2_dg_Hdiv_1form_kernels.push_gc_cc_J2_dg_Hdiv_1form`.
      The marker energies :math:`\mu_p \hat B_\parallel` use :math:`\mathcal P^{b,1} \mathbf b`.
    """

    _b_form = "1"
    _pb_name = "PB1form"
    _kernels = {
        "accum": cc_lin_mhd_5d_gradB_1form,
        "push_Hdiv": push_gc_cc_J2_stage_Hdiv_1form,
        "push_H1vec": push_gc_cc_J2_stage_H1vec_1form,
        "accum_dg_init": cc_lin_mhd_5d_gradB_dg_init_1form,
        "accum_dg": cc_lin_mhd_5d_gradB_dg_1form,
        "push_dg_init": push_gc_cc_J2_dg_init_Hdiv_1form,
        "push_dg": push_gc_cc_J2_dg_Hdiv_1form,
    }

    def __init__(self, b_tilde: FEECVariable = None):
        """
        Parameters
        ----------
        b_tilde : FEECVariable, default=None
            Magnetic perturbation 1-form (``"Hcurl"`` space) entering the grad-B
            coupling term. If ``None``, only the equilibrium field is used.
        """
        super().__init__(b_tilde=b_tilde)
