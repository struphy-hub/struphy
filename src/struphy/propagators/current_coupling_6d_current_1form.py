"Particle and FEEC variables are updated."

from struphy.models.variables import FEECVariable
from struphy.pic.accumulation.kernels.cc_lin_mhd_6d_2_1form import cc_lin_mhd_6d_2_1form
from struphy.pic.accumulation.kernels.cc_lin_mhd_6d_2_mf_1form import cc_lin_mhd_6d_2_mf_1form
from struphy.pic.pushing.kernels.push_bxu_H1vec_1form import push_bxu_H1vec_1form
from struphy.pic.pushing.kernels.push_bxu_Hcurl_1form import push_bxu_Hcurl_1form
from struphy.pic.pushing.kernels.push_bxu_Hdiv_1form import push_bxu_Hdiv_1form
from struphy.propagators.current_coupling_6d_current import CurrentCoupling6DCurrent


class CurrentCoupling6DCurrent1form(CurrentCoupling6DCurrent):
    r""":class:`~struphy.propagators.current_coupling_6d_current.CurrentCoupling6DCurrent` with the magnetic field as
    a 1-form (in :math:`H(\textnormal{curl})`).

    **Continuous problem.** Find :math:`\tilde{\mathbf{U}} \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}`
    and :math:`f_\textnormal{h}` such that

    .. math::

        \int_\Omega \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V \,\textrm d \mathbf x = - \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon}  \int_\Omega n_\textnormal{h}\mathbf{u}_\textnormal{h} \times(\mathbf{B}_0+\tilde{\mathbf{B}}) \cdot \mathbf V \,\textrm d \mathbf x
        \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &\frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{\varepsilon} \Big[(\mathbf{B}_0+\tilde{\mathbf{B}})\times\tilde{\mathbf{U}} \Big] \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} =0\,,
        \qquad n_\textnormal{h}\mathbf{u}_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}\,\textnormal{d}^3 \mathbf v\,,

    where :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` (``b_tilde``) is fixed in the substep.

    **Semi-discrete problem.** Both magnetic fields are 1-forms: :math:`\mathbf B_0` is the commuting projection
    :math:`\hat \Pi^1` of the equilibrium 1-form :math:`\hat{\mathbf B}^1_0` (``projected_equil.b1``) and
    :math:`\tilde{\mathbf B}` has the FE coefficients :math:`\mathbf b \in \mathbb R^{N_1}`, so that at the marker
    positions :math:`\boldsymbol \eta_p` the Cartesian components of the total field are

    .. math::

        \mathbf B_p = DF^{-\top} \left( \hat \Pi^1 \hat{\mathbf B}^1_0 + \mathbf b^\top \vec \Lambda^1 \right)
        (\boldsymbol \eta_p)\,,

    with no conversion to 2-forms. With the FE coefficients :math:`\mathbf u \in \mathbb R^{N_\alpha}` of
    :math:`\tilde{\mathbf U}` in the space :math:`\alpha \in \{1, 2, v\}` (``u_space``), the marker velocities
    :math:`\mathbf v_p` and weights :math:`w_p`, the semi-discrete system reads

    .. math::

        \mathbb M^\alpha_{\rho_0} \dot{\mathbf u} &= \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon}
        \sum_p w_p\, \vec \Lambda^\alpha(\boldsymbol \eta_p)\, T^\alpha_p\, \mathbf v_p\,,
        \\[2mm]
        \dot{\mathbf v}_p &= - \frac{1}{\varepsilon}\, (T^\alpha_p)^\top \vec \Lambda^\alpha(\boldsymbol \eta_p)^\top
        \mathbf u = \frac{1}{\varepsilon}\, \mathbf B_p \times \tilde{\mathbf U}(\boldsymbol \eta_p)\,,

    where :math:`\mathbb M^\alpha_{\rho_0}` is the mass matrix weighted with the bulk density (``M1n``, ``M2n`` or
    ``Mvn``) and, with :math:`\mathbf a_\times \mathbf c := \mathbf a \times \mathbf c` and
    :math:`\sqrt g = \det DF` at :math:`\boldsymbol \eta_p`,

    .. math::

        T^1_p = DF^{-1} \mathbf B_{p,\times}\,,\qquad
        T^2_p = \frac{1}{\sqrt g}\, DF^\top \mathbf B_{p,\times}\,,\qquad
        T^v_p = DF^\top \mathbf B_{p,\times}\,.

    The vector :math:`\sum_p w_p \vec \Lambda^\alpha_p T^\alpha_p \mathbf v_p` and the symmetric matrix
    :math:`\sum_p w_p \vec \Lambda^\alpha_p T^\alpha_p (T^\alpha_p)^\top (\vec \Lambda^\alpha_p)^\top` are accumulated
    by
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_6d_2_1form.cc_lin_mhd_6d_2_1form_kernels.cc_lin_mhd_6d_2_1form`
    (with ``matrix_free=True`` the matrix fillings are stored per marker by
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_6d_2_mf_1form.cc_lin_mhd_6d_2_mf_1form_kernels.cc_lin_mhd_6d_2_mf_1form`
    and the matrix is applied matrix-free by :class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`).
    The velocity update uses the pusher kernels ``push_bxu_<u_space>_1form``, e.g.
    :func:`~struphy.pic.pushing.kernels.push_bxu_Hcurl_1form.push_bxu_Hcurl_1form_kernels.push_bxu_Hcurl_1form`,
    which evaluate :math:`\mathbf B_p \times \tilde{\mathbf U}(\boldsymbol \eta_p) = DF (\hat{\mathbf B}^1 \times
    \hat{\mathbf U}^1) / \sqrt g` with the 1-form components :math:`\hat{\mathbf U}^1` of :math:`\tilde{\mathbf U}`.

    **Time discretization.** Crank-Nicolson (implicit mid-point), the only option: with
    :math:`\mathbb B = \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \sum_p w_p
    \vec \Lambda^\alpha_p T^\alpha_p (\cdot)_p` the step

    .. math::

        \mathbb M^\alpha_{\rho_0} \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t} = \mathbb B\,
        \frac{\mathbf V^{n+1} + \mathbf V^n}{2}\,,\qquad
        \frac{\mathbf v_p^{n+1} - \mathbf v_p^n}{\Delta t} = - \frac{1}{\varepsilon} (T^\alpha_p)^\top
        \vec \Lambda^\alpha(\boldsymbol \eta_p)^\top \frac{\mathbf u^{n+1} + \mathbf u^n}{2}\,,

    with :math:`\mathbf V = (\mathbf v_p)_p` is reduced to the system

    .. math::

        \left( \mathbb M^\alpha_{\rho_0} + \frac{\Delta t^2}{4} \mathbb C \right) \mathbf u^{n+1} =
        \left( \mathbb M^\alpha_{\rho_0} - \frac{\Delta t^2}{4} \mathbb C \right) \mathbf u^n
        + \Delta t\, \mathbb B \mathbf V^n\,,\qquad
        \mathbb C = \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon^2} \sum_p w_p\,
        \vec \Lambda^\alpha_p T^\alpha_p (T^\alpha_p)^\top (\vec \Lambda^\alpha_p)^\top\,,

    for :math:`\mathbf u^{n+1}` (:class:`~struphy.linear_algebra.schur_solver.SchurSolver`, symmetric solver
    ``solver``, default ``"pcg"``, with the mass-matrix preconditioner ``precond``); then the velocities are updated
    explicitly with :math:`(\mathbf u^{n+1} + \mathbf u^n)/2`. The positions do not change.
    """

    _b_form = "1"
    _kernels = {
        "accum": cc_lin_mhd_6d_2_1form,
        "accum_mf": cc_lin_mhd_6d_2_mf_1form,
        "Hcurl": push_bxu_Hcurl_1form,
        "Hdiv": push_bxu_Hdiv_1form,
        "H1vec": push_bxu_H1vec_1form,
    }

    def __init__(self, b_tilde: FEECVariable = None):
        """
        Parameters
        ----------
        b_tilde : FEECVariable, default=None
            Magnetic perturbation 1-form (``"Hcurl"`` space) entering the current coupling term (not updated).
            If ``None``, only the equilibrium field is used.
        """
        assert b_tilde is None or b_tilde.space == "Hcurl", "b_tilde must be in Hcurl."
        super().__init__(b_tilde=b_tilde)

    def _sum_b_full(self):
        """Sum up the total magnetic field b_eq + E1^T b_tilde in the tensor-product space (in ``self._b_full2``) and
        update its ghost regions; the projected equilibrium is already a tensor-product vector."""
        self._b_eq.copy(out=self._b_full2)

        if self.b_tilde is not None:
            self._b_full2 += self._EbT.dot(self.b_tilde.spline.vector, out=self._b_full1)

        # update ghost regions because of non-local access in accumulation and pusher kernels!
        self._b_full2.update_ghost_regions()
