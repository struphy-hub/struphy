"Particle and FEEC variables are updated."

from feectools.linalg.block import BlockVector

from struphy.models.variables import FEECVariable
from struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb_1form import cc_lin_mhd_5d_curlb_1form
from struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb_mf_1form import cc_lin_mhd_5d_curlb_mf_1form
from struphy.pic.pushing.kernels.push_gc_cc_J1_H1vec_1form import push_gc_cc_J1_H1vec_1form
from struphy.pic.pushing.kernels.push_gc_cc_J1_Hdiv_1form import push_gc_cc_J1_Hdiv_1form
from struphy.propagators.current_coupling_5d_curlb import CurrentCoupling5DCurlb


class CurrentCoupling5DCurlb1form(CurrentCoupling5DCurlb):
    r""":class:`~struphy.propagators.current_coupling_5d_curlb.CurrentCoupling5DCurlb` with the magnetic field as a
    1-form (in :math:`H(\textnormal{curl})`).

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in \{H(\textnormal{div}), (H^1)^3\}` and the
    guiding-center distribution :math:`f` such that

    .. math::

        \left\{
            \begin{aligned}
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V\, \textnormal{d} \mathbf x
                = - \frac{A_\textnormal{h}}{A_\textnormal{b}} \iint \frac{f^\text{vol}}{B^*_\parallel}
                v_\parallel^2 (\nabla \times \mathbf b_0) \times \mathbf B \cdot \mathbf V \, \textnormal{d} \mathbf x
                \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \,\mathbf V \in \{H(\textnormal{div}),
                (H^1)^3\}\,,
                \\
                &\frac{\partial f}{\partial t} = - \frac{1}{B^*_\parallel} v_\parallel (\nabla \times \mathbf b_0)
                \cdot (\mathbf B \times \tilde{\mathbf U}) \nabla_{v_\parallel}f \,,
            \end{aligned}
        \right.

    with :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`, the magnetic perturbation
    :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` (``b_tilde``, not updated) and :math:`B^*_\parallel =
    \mathbf b_0 \cdot (\mathbf B + \epsilon v_\parallel \nabla \times \mathbf b_0)`.

    **Semi-discrete problem.** The total field is a 1-form, the commuting projection :math:`\hat \Pi^1` of the
    equilibrium 1-form :math:`\hat{\mathbf B}^1_0` (``projected_equil.b1``) plus the perturbation with FE coefficients
    :math:`\mathbf b \in \mathbb R^{N_1}` (extracted to the tensor-product space),

    .. math::

        \hat{\mathbf B}^1 = \hat \Pi^1 \hat{\mathbf B}^1_0 + \mathbf b^\top \vec \Lambda^1\,,\qquad
        \hat{\mathbf B}^\textnormal{v} = G^{-1} \hat{\mathbf B}^1\,,

    never converted to a 2-form. With the unit vector :math:`\hat{\mathbf b}^1_0` (``projected_equil.unit_b1``), the
    2-form :math:`\hat{\mathbf c} = \hat \nabla \times \hat{\mathbf b}^1_0` (``projected_equil.curl_unit_b2``) and
    :math:`\hat B^{*\textnormal{v}}_\parallel = \hat{\mathbf b}^1_0 \cdot (\hat{\mathbf B}^\textnormal{v} + \epsilon
    v_\parallel \hat{\mathbf c} / \sqrt g)`, define per marker (at :math:`\boldsymbol \eta_p`)

    .. math::

        \mathbf e_p = s_\alpha \frac{\hat{\mathbf B}^\textnormal{v} \times \hat{\mathbf c}}{\hat
        B^{*\textnormal{v}}_\parallel}\,,\qquad c_p = w_p v_{\parallel,p}^2 \frac{A_\textnormal{h}}{A_\textnormal{b}}\,,

    with :math:`s_\alpha = 1/\sqrt g` for :math:`H(\textnormal{div})` and :math:`s_\alpha = 1` for :math:`(H^1)^3`
    (``u_space``). The FE coefficients :math:`\mathbf u \in \mathbb R^{N_\alpha}` of :math:`\tilde{\mathbf U}` and
    the parallel velocities :math:`v_{\parallel,p}` satisfy

    .. math::

        \mathbb M^\alpha_{\rho_0} \dot{\mathbf u} = \sum_p c_p\, \vec \Lambda^\alpha(\boldsymbol \eta_p) \cdot
        \mathbf e_p\,,\qquad
        \dot v_{\parallel,p} = - v_{\parallel,p}\, \mathbf e_p \cdot (\mathbf u^\top \vec \Lambda^\alpha)
        (\boldsymbol \eta_p)\,,

    with the density-weighted mass matrix :math:`\mathbb M^\alpha_{\rho_0}`; the sum of :math:`\mathbf u^\top
    \mathbb M^\alpha_{\rho_0} \mathbf u / 2` and :math:`\frac{A_\textnormal{h}}{A_\textnormal{b}} \sum_p w_p
    v_{\parallel,p}^2 / 2` is conserved. This equals the 2-form system of the
    parent class, since :math:`\hat{\mathbf B}^2 = \sqrt g\, \hat{\mathbf B}^\textnormal{v}` and
    :math:`\hat{\mathbf b}^1_0 \cdot \hat{\mathbf B}^{*2} = \sqrt g\, \hat B^{*\textnormal{v}}_\parallel`: the factor
    :math:`\sqrt g` cancels in their ratio.

    **Time discretization.** Crank-Nicolson (implicit mid-point). With the accumulated vector :math:`\mathbf a =
    \sum_p c_p \vec \Lambda^\alpha_p \cdot \mathbf e_p` and matrix :math:`\mathbb A = \sum_p c_p (\vec \Lambda^\alpha_p
    \cdot \mathbf e_p) (\vec \Lambda^\alpha_p \cdot \mathbf e_p)^\top` (:math:`c_p` and :math:`\mathbf e_p` at
    :math:`v^n_{\parallel,p}`), the velocities are eliminated by the :class:`~struphy.linear_algebra.schur_solver.SchurSolver` (``solver``,
    ``precond``),

    .. math::

        \left(\mathbb M^\alpha_{\rho_0} + \frac{\Delta t^2}{4} \mathbb A\right) \mathbf u^{n+1} =
        \left(\mathbb M^\alpha_{\rho_0} - \frac{\Delta t^2}{4} \mathbb A\right) \mathbf u^n + \Delta t\, \mathbf a\,,

    then the markers are pushed with :math:`(\mathbf u^{n+1} + \mathbf u^n)/2` by
    :func:`~struphy.pic.pushing.kernels.push_gc_cc_J1_Hdiv_1form.push_gc_cc_J1_Hdiv_1form_kernels.push_gc_cc_J1_Hdiv_1form`
    (resp. ``push_gc_cc_J1_H1vec_1form``). The accumulation kernels are
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb_1form.cc_lin_mhd_5d_curlb_1form_kernels.cc_lin_mhd_5d_curlb_1form`
    (assembled) and
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb_mf_1form.cc_lin_mhd_5d_curlb_mf_1form_kernels.cc_lin_mhd_5d_curlb_mf_1form`
    (``matrix_free=True``).
    """

    _b_form = "1"
    _kernels = {
        "assembled": cc_lin_mhd_5d_curlb_1form,
        "matrix_free": cc_lin_mhd_5d_curlb_mf_1form,
        "push_Hdiv": push_gc_cc_J1_Hdiv_1form,
        "push_H1vec": push_gc_cc_J1_H1vec_1form,
    }

    def __init__(self, b_tilde: FEECVariable = None):
        """
        Parameters
        ----------
        b_tilde : FEECVariable, default=None
            Magnetic perturbation 1-form (``"Hcurl"`` space) entering the curl-B
            coupling term. If ``None``, only the equilibrium field is used.
        """
        assert b_tilde is None or b_tilde.space == "Hcurl", "b_tilde must be in Hcurl."
        super().__init__(b_tilde=b_tilde)

    def allocate(self):
        # extraction of the 1-form b_tilde to the tensor-product space read by the kernels (built in each
        # allocation: a new allocation of the model may come with a new Derham object)
        self._E1T = self.derham.extraction_ops["1"].transpose()
        self._b_tilde_tp = self.derham.V1.zeros()
        super().allocate()

    def _b_tilde_coeffs(self) -> BlockVector:
        """Tensor-product FE coefficients (extracted, ghost regions updated) of the 1-form ``b_tilde``."""
        b_tilde = self._E1T.dot(self.b_tilde.spline.vector, out=self._b_tilde_tp)
        b_tilde.update_ghost_regions()
        return b_tilde
