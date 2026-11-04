"Only FEEC variables are updated."

from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.kernels.cc_lin_mhd_6d_1_1form import cc_lin_mhd_6d_1_1form
from struphy.pic.accumulation.kernels.cc_lin_mhd_6d_1_mf_1form import cc_lin_mhd_6d_1_mf_1form
from struphy.propagators.current_coupling_6d_density import CurrentCoupling6DDensity


class CurrentCoupling6DDensity1form(CurrentCoupling6DDensity):
    r""":class:`~struphy.propagators.current_coupling_6d_density.CurrentCoupling6DDensity` with the magnetic field as
    a 1-form (in :math:`H(\textnormal{curl})`).

    **Continuous problem.** Find :math:`\tilde{\mathbf{U}} \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}`
    such that

    .. math::

        \int_\Omega \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V \,\textrm d \mathbf x = \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \int_\Omega n_\textnormal{h}\tilde{\mathbf{U}} \times(\mathbf{B}_0+\tilde{\mathbf{B}}) \cdot \mathbf V \,\textrm d \mathbf x
        \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \qquad n_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\,\textnormal{d}^3 \mathbf v\,,

    where :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` (``b_tilde``) and the hot density
    :math:`n_\textnormal{h}` are fixed in the substep.

    **Semi-discrete problem.** Both magnetic fields are 1-forms: :math:`\mathbf B_0` is the commuting projection
    :math:`\hat \Pi^1` of the equilibrium 1-form :math:`\hat{\mathbf B}^1_0` (``projected_equil.b1``) and
    :math:`\tilde{\mathbf B}` has the FE coefficients :math:`\mathbf b \in \mathbb R^{N_1}`, so that at the marker
    positions :math:`\boldsymbol \eta_p` the total field has the 1-form components and the Cartesian components

    .. math::

        \hat{\mathbf B}^1 = \hat \Pi^1 \hat{\mathbf B}^1_0 + \mathbf b^\top \vec \Lambda^1\,,\qquad
        \mathbf B = DF^{-\top} \hat{\mathbf B}^1\,,

    with no conversion to 2-forms. With the FE coefficients :math:`\mathbf u \in \mathbb R^{N_\alpha}` of
    :math:`\tilde{\mathbf U}` in the space :math:`\alpha \in \{1, 2, v\}` (``u_space``) and the marker weights
    :math:`w_p`, the semi-discrete equation reads

    .. math::

        \mathbb M^\alpha_{\rho_0} \dot{\mathbf u} = \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon}\,
        \mathbb A^\alpha \mathbf u\,,\qquad
        \mathbb A^\alpha = - \sum_p w_p\, \vec \Lambda^\alpha(\boldsymbol \eta_p)\, F^\alpha_p\,
        \vec \Lambda^\alpha(\boldsymbol \eta_p)^\top\,,

    where :math:`\mathbb M^\alpha_{\rho_0}` is the mass matrix weighted with the bulk density (``M1n``, ``M2n`` or
    ``Mvn``) and the antisymmetric fillings :math:`F^\alpha_p` follow from the pull-back of
    :math:`n_\textnormal{h}\tilde{\mathbf{U}} \times \mathbf B \cdot \mathbf V`, simplified analytically for the
    1-form field (:math:`\hat{\mathbf B}^v = G^{-1} \hat{\mathbf B}^1`, :math:`\sqrt g = \det DF`,
    :math:`\mathbf a_\times \mathbf c := \mathbf a \times \mathbf c`, all at :math:`\boldsymbol \eta_p`):

    .. math::

        F^1_p = \frac{1}{\sqrt g}\, \hat{\mathbf B}^1_\times\,,\qquad
        F^2_p = \frac{1}{\sqrt g}\, \hat{\mathbf B}^v_\times\,,\qquad
        F^v_p = \sqrt g\, \hat{\mathbf B}^v_\times\,,

    see the accumulation kernel
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_6d_1_1form.cc_lin_mhd_6d_1_1form_kernels.cc_lin_mhd_6d_1_1form`.
    With ``matrix_free=True`` the fillings are stored per marker
    (:func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_6d_1_mf_1form.cc_lin_mhd_6d_1_mf_1form_kernels.cc_lin_mhd_6d_1_mf_1form`)
    and :math:`\mathbb A^\alpha` is applied matrix-free
    (:class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`).

    **Time discretization.** Crank-Nicolson (implicit mid-point), the only option: :math:`\mathbb A^\alpha` is
    accumulated once per step and

    .. math::

        \left( \mathbb M^\alpha_{\rho_0} - \frac{\Delta t}{2} \frac{A_\textnormal{h}}{A_\textnormal{b}}
        \frac{1}{\varepsilon} \mathbb A^\alpha \right) \mathbf u^{n+1}
        = \left( \mathbb M^\alpha_{\rho_0} + \frac{\Delta t}{2} \frac{A_\textnormal{h}}{A_\textnormal{b}}
        \frac{1}{\varepsilon} \mathbb A^\alpha \right) \mathbf u^n

    is solved with the non-symmetric iterative solver ``solver`` (default ``"pbicgstab"``) and the mass-matrix
    preconditioner ``precond``. As :math:`\mathbb A^\alpha` is antisymmetric, the step conserves
    :math:`\mathbf u^\top \mathbb M^\alpha_{\rho_0} \mathbf u` exactly (up to the solver tolerance).
    """

    _b_form = "1"
    _kernels = {"accum": cc_lin_mhd_6d_1_1form, "accum_mf": cc_lin_mhd_6d_1_mf_1form}

    def __init__(
        self,
        energetic_ions: PICVariable = None,
        b_tilde: FEECVariable = None,
    ):
        """
        Parameters
        ----------
        energetic_ions : PICVariable, default=None
            Energetic-ion particle distribution providing the density source term.
        b_tilde : FEECVariable, default=None
            Magnetic perturbation 1-form (``"Hcurl"`` space) entering the coupling term (not updated).
            If ``None``, only the equilibrium field is used.
        """
        assert b_tilde is None or b_tilde.space == "Hcurl", "b_tilde must be in Hcurl."
        super().__init__(energetic_ions=energetic_ions, b_tilde=b_tilde)
