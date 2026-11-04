"Only FEEC variables are updated."

from feectools.linalg.block import BlockVector

from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.kernels.cc_lin_mhd_5d_D_1form import cc_lin_mhd_5d_D_1form
from struphy.pic.accumulation.kernels.cc_lin_mhd_5d_D_mf_1form import cc_lin_mhd_5d_D_mf_1form
from struphy.propagators.current_coupling_5d_density import CurrentCoupling5DDensity


class CurrentCoupling5DDensity1form(CurrentCoupling5DDensity):
    r""":class:`~struphy.propagators.current_coupling_5d_density.CurrentCoupling5DDensity` with the magnetic field
    as a 1-form (in :math:`H(\textnormal{curl})`).

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in \{H(\textnormal{curl}), H(\textnormal{div}),
    (H^1)^3\}` such that

    .. math::

        \int \rho_0 \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf{x} =
        \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\epsilon} \iiint f^\text{vol} \left(1 -
        \frac{B_\parallel}{B^*_\parallel}\right) \tilde{\mathbf U} \times \mathbf B \cdot \mathbf V \,
        \textnormal{d} \mathbf{x} \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \,\mathbf V \in
        \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

    with :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`, the magnetic perturbation
    :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` (``b_tilde``, not updated),
    :math:`B_\parallel = \mathbf b_0 \cdot \mathbf B` and :math:`B^*_\parallel = \mathbf b_0 \cdot (\mathbf B +
    \epsilon v_\parallel \nabla \times \mathbf b_0)`.

    **Semi-discrete problem.** The total field is a 1-form, the commuting projection :math:`\hat \Pi^1` of the
    equilibrium 1-form :math:`\hat{\mathbf B}^1_0` (``projected_equil.b1``) plus the perturbation with FE coefficients
    :math:`\mathbf b \in \mathbb R^{N_1}` (extracted to the tensor-product space),

    .. math::

        \hat{\mathbf B}^1 = \hat \Pi^1 \hat{\mathbf B}^1_0 + \mathbf b^\top \vec \Lambda^1\,,\qquad
        \hat{\mathbf B}^\textnormal{v} = G^{-1} \hat{\mathbf B}^1\,,

    never converted to a 2-form. With the unit vector :math:`\hat{\mathbf b}^1_0` (``projected_equil.unit_b1``), the
    2-form :math:`\hat{\mathbf c} = \hat \nabla \times \hat{\mathbf b}^1_0` (``projected_equil.curl_unit_b2``) and
    :math:`\hat{\mathbf B}^{*\textnormal{v}} = \hat{\mathbf B}^\textnormal{v} + \epsilon v_\parallel \hat{\mathbf c}
    / \sqrt g` evaluated at the markers, the FE coefficients :math:`\mathbf u \in \mathbb R^{N_\alpha}` of
    :math:`\tilde{\mathbf U}` satisfy

    .. math::

        \mathbb M^\alpha_{\rho_0} \dot{\mathbf u} = \mathbb A\, \mathbf u\,,\qquad
        \mathbb A = \sum_p \vec \Lambda^\alpha(\boldsymbol \eta_p) F_p\, \vec \Lambda^\alpha(\boldsymbol \eta_p)^\top\,,

    with :math:`\mathbb M^\alpha_{\rho_0}` the density-weighted mass matrix of the space of :math:`\mathbf u`
    (``u_space``) and the antisymmetric 3x3 matrices

    .. math::

        F_p = - w_p \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\epsilon} \left(1 - \frac{\hat{\mathbf
        b}^1_0 \cdot \hat{\mathbf B}^\textnormal{v}}{\hat{\mathbf b}^1_0 \cdot \hat{\mathbf B}^{*\textnormal{v}}}
        \right) \times
        \begin{cases}
            \sqrt g\, \hat{\mathbf B}^\textnormal{v}_\times & (H^1)^3\,,
            \\
            \hat{\mathbf B}^1_\times / \sqrt g & H(\textnormal{curl})\,,
            \\
            \hat{\mathbf B}^\textnormal{v}_\times / \sqrt g & H(\textnormal{div})\,,
        \end{cases}

    at :math:`\boldsymbol \eta_p` (:math:`\mathbf a_\times \mathbf c = \mathbf a \times \mathbf c`), given by the
    kernels
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_D_1form.cc_lin_mhd_5d_D_1form_kernels.cc_lin_mhd_5d_D_1form`
    (assembled) and
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_D_mf_1form.cc_lin_mhd_5d_D_mf_1form_kernels.cc_lin_mhd_5d_D_mf_1form`
    (``matrix_free=True``). This equals the 2-form matrix of the parent class, since :math:`\hat{\mathbf B}^2 =
    \sqrt g\, \hat{\mathbf B}^\textnormal{v}`, :math:`\hat{\mathbf b}^1_0 \cdot \hat{\mathbf B}^{*2} = \sqrt g\,
    \hat{\mathbf b}^1_0 \cdot \hat{\mathbf B}^{*\textnormal{v}}` and :math:`G^{-1} (\sqrt g\, G^{-1} \hat{\mathbf
    B}^1)_\times G^{-1} = \hat{\mathbf B}^1_\times / \sqrt g`.

    **Time discretization.** Crank-Nicolson (implicit mid-point),

    .. math::

        \left(\mathbb M^\alpha_{\rho_0} - \frac{\Delta t}{2} \mathbb A \right) \mathbf u^{n+1} =
        \left(\mathbb M^\alpha_{\rho_0} + \frac{\Delta t}{2} \mathbb A \right) \mathbf u^n\,,

    solved with the iterative solver ``solver`` and the mass-matrix preconditioner ``precond``; :math:`\mathbb A` is
    antisymmetric, so the kinetic energy :math:`\mathbf u^\top \mathbb M^\alpha_{\rho_0} \mathbf u / 2` is conserved.
    """

    _b_form = "1"
    _kernels = {"assembled": cc_lin_mhd_5d_D_1form, "matrix_free": cc_lin_mhd_5d_D_mf_1form}

    def __init__(
        self,
        energetic_ions: PICVariable = None,
        b_tilde: FEECVariable = None,
    ):
        """
        Parameters
        ----------
        energetic_ions : PICVariable, default=None
            Energetic-ion particle distribution (``"Particles5D"`` space) providing
            the density source term.
        b_tilde : FEECVariable, default=None
            Magnetic perturbation 1-form (``"Hcurl"`` space) entering the coupling term.
            If ``None``, only the equilibrium field is used.
        """
        super().__init__(energetic_ions=energetic_ions, b_tilde=b_tilde)
