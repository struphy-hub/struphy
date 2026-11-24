import logging
from dataclasses import dataclass

from line_profiler import profile
from maybempi import MPI

from struphy.io.options import OptionsBase
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")


class BackgroundPressureGradient(Propagator):
    r"""Advection of the background pressure by the velocity perturbation in linear MHD (zero-flow equilibrium): the
    part of the linear pressure equation that is not in the energy conserving
    :class:`~struphy.propagators.sound_wave_linear.SoundWaveLinear` step.

    **Continuous problem.** Find :math:`\tilde p \in L^2` such that

    .. math::

        \frac{\partial \tilde p}{\partial t} + \tilde{\mathbf U} \cdot \nabla p_0 = 0\,,

    in :math:`L^2`, at a fixed velocity perturbation :math:`\tilde{\mathbf U} \in H(\textnormal{div})` (the constructor
    argument ``u_field``) and with the equilibrium pressure :math:`p_0`. Together with the pressure equation
    :math:`\partial_t \tilde p = - \gamma p_0 \nabla \cdot \tilde{\mathbf U}` of
    :class:`~struphy.propagators.sound_wave_linear.SoundWaveLinear`, this gives the full linear pressure equation
    :math:`\partial_t \tilde p + \nabla \cdot (p_0 \tilde{\mathbf U}) + (\gamma - 1) p_0 \nabla \cdot \tilde{\mathbf U} = 0`.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf p` of :math:`\tilde p_h \in V^3_h` and
    :math:`\mathbf u` of :math:`\tilde{\mathbf U}_h \in V^2_h`,

    .. math::

        \dot{\mathbf p} = - \mathbb G \mathbf u\,, \qquad \mathbb G = \mathbb D \mathbb S^2 - \mathbb K^3 \mathbb D\,,

    with the discrete divergence :math:`\mathbb D` (:attr:`~struphy.feec.psydac_derham.Derham.div`) and the basis
    projection operators (:class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators`)
    :math:`\mathbb S^2 \approx \hat \Pi^2[p_0 \,\cdot\,]` (``S2``) and :math:`\mathbb K^3 \approx \hat \Pi^3[p_0 \,\cdot\,]`
    (``K3``). By the commuting property :math:`\mathbb D \hat \Pi^2 = \hat \Pi^3 \nabla \cdot` of the projectors and the
    product rule :math:`\nabla \cdot (p_0 \mathbf U) = \nabla p_0 \cdot \mathbf U + p_0 \nabla \cdot \mathbf U`,

    .. math::

        \mathbb G \mathbf u \approx \hat \Pi^3 \left[ \nabla \cdot (p_0 \tilde{\mathbf U}_h) - p_0 \nabla \cdot \tilde{\mathbf U}_h \right]
        = \hat \Pi^3 \left[ \tilde{\mathbf U}_h \cdot \nabla p_0 \right]\,,

    up to the quadrature in the projectors (exact for exact integration), so no gradient of :math:`p_0` is needed.
    This is the same operator splitting as in :class:`~struphy.propagators.magnetosonic.Magnetosonic`, whose pressure
    operator :math:`\mathbb D \mathbb S^2 + \frac 23 \mathbb K^3 \mathbb D` equals
    :math:`\mathbb G + \gamma \mathbb K^3 \mathbb D`.

    **Time discretization.** Exact in time, the only option, since the right-hand side does not change during the
    step:

    .. math::

        \mathbf p^{n+1} = \mathbf p^n - \Delta t \, \mathbb G \mathbf u\,.
    """

    class Variables:
        """Container for variables advanced by :class:`BackgroundPressureGradient`.

        Attributes
        ----------
        p : FEECVariable
            Pressure perturbation variable in ``"L2"`` space.
        """

        def __init__(self):
            self._p: FEECVariable = None

        @property
        def p(self) -> FEECVariable:
            return self._p

        @p.setter
        def p(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._p = new

    def __init__(self, u_field: FEECVariable = None):
        r"""
        Parameters
        ----------
        u_field : FEECVariable, default=None
            Velocity perturbation in ``"Hdiv"`` space (not updated). If ``None``, an empty
            ``FEECVariable(space="Hdiv")`` is created and allocated internally. Otherwise the given variable is treated
            as shared (e.g. the model's ``mhd.velocity``) and must already be allocated when :meth:`allocate` is
            called; it is not re-allocated.
        """
        assert u_field is None or u_field.space == "Hdiv", "u_field must be in Hdiv."
        self.variables = self.Variables()
        self.u_field = u_field if u_field is not None else FEECVariable(space="Hdiv")
        self._allocate_u_field = u_field is None

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`BackgroundPressureGradient` (none at the moment)."""

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        div = self.derham.div
        self._G = div @ self.basis_ops.S2 - self.basis_ops.K3 @ div

        # only allocate the private default; a shared u_field is allocated (with initial conditions) by the model
        if self._allocate_u_field:
            self.u_field.allocate(self.derham, self.domain)
        self._u = self.u_field.spline.vector

        self._p_tmp1 = self.variables.p.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        pn = self.variables.p.spline.vector

        pn1 = self._G.dot(self._u, out=self._p_tmp1)
        pn1 *= -dt
        pn1 += pn

        diffs = self.update_feec_variables(p=pn1)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.debug(f"Maxdiff p3 for BackgroundPressureGradient: {diffs['p']}")
