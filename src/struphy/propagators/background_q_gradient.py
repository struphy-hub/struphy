import logging
from dataclasses import dataclass

from line_profiler import profile
from maybempi import MPI

from struphy.io.options import OptionsBase
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")


class BackgroundQGradient(Propagator):
    r"""Advection of the background :math:`q_0 = \sqrt{p_0}` by the velocity perturbation in the linear models with the
    variable :math:`q = \sqrt p` (zero-flow equilibrium): the part of the linear equation of :math:`\tilde q` that is not
    in the energy conserving :class:`~struphy.propagators.sound_wave_linear_with_q.SoundWaveLinearWithQ` step.

    **Continuous problem.** Find :math:`\tilde q \in L^2` such that

    .. math::

        \frac{\partial \tilde q}{\partial t} + \tilde{\mathbf U} \cdot \nabla q_0 = 0\,,

    in :math:`L^2`, at a fixed velocity perturbation :math:`\tilde{\mathbf U} \in H(\textnormal{div})` (the constructor
    argument ``u_field``). Together with the equation
    :math:`\partial_t \tilde q = - \frac{\gamma}{2} q_0 \nabla \cdot \tilde{\mathbf U}` of
    :class:`~struphy.propagators.sound_wave_linear_with_q.SoundWaveLinearWithQ`, this gives the full linear equation
    :math:`\partial_t \tilde q + \nabla \cdot (q_0 \tilde{\mathbf U}) + (\frac{\gamma}{2} - 1) q_0 \nabla \cdot
    \tilde{\mathbf U} = 0`, the linearization of :math:`\partial_t q + \mathbf U \cdot \nabla q + \frac{\gamma}{2} q
    \nabla \cdot \mathbf U = 0`.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf q` of :math:`\tilde q_h \in V^3_h` and
    :math:`\mathbf u` of :math:`\tilde{\mathbf U}_h \in V^2_h`,

    .. math::

        \dot{\mathbf q} = - \mathbb G_q \mathbf u\,, \qquad \mathbb G_q = \mathbb D \mathbb S^2_q - \mathbb K^3_q \mathbb D\,,

    with the discrete divergence :math:`\mathbb D` (:attr:`~struphy.feec.psydac_derham.Derham.div`) and the basis
    projection operators (:class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators`)
    :math:`\mathbb S^2_q \approx \hat \Pi^2[q_0 \,\cdot\,]` (``S2q``) and :math:`\mathbb K^3_q \approx \hat \Pi^3[q_0
    \,\cdot\,]` (``K3q``). By the commuting property :math:`\mathbb D \hat \Pi^2 = \hat \Pi^3 \nabla \cdot` of the
    projectors and the product rule,

    .. math::

        \mathbb G_q \mathbf u \approx \hat \Pi^3 \left[ \nabla \cdot (q_0 \tilde{\mathbf U}_h) - q_0 \nabla \cdot \tilde{\mathbf U}_h \right]
        = \hat \Pi^3 \left[ \tilde{\mathbf U}_h \cdot \nabla q_0 \right]\,,

    up to the quadrature in the projectors, so no gradient of :math:`q_0` is needed (as in
    :class:`~struphy.propagators.background_pressure_gradient.BackgroundPressureGradient`, with :math:`q_0` instead of
    :math:`p_0`).

    **Time discretization.** Exact in time, the only option, since the right-hand side does not change during the
    step:

    .. math::

        \mathbf q^{n+1} = \mathbf q^n - \Delta t \, \mathbb G_q \mathbf u\,.
    """

    class Variables:
        """Container for variables advanced by :class:`BackgroundQGradient`.

        Attributes
        ----------
        q : FEECVariable
            Perturbation of :math:`q = \\sqrt p` in ``"L2"`` space.
        """

        def __init__(self):
            self._q: FEECVariable = None

        @property
        def q(self) -> FEECVariable:
            return self._q

        @q.setter
        def q(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._q = new

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
        """Configuration options for :class:`BackgroundQGradient` (none at the moment)."""

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
        self._G = div @ self.basis_ops.S2q - self.basis_ops.K3q @ div

        # only allocate the private default; a shared u_field is allocated (with initial conditions) by the model
        if self._allocate_u_field:
            self.u_field.allocate(self.derham, self.domain)
        self._u = self.u_field.spline.vector

        self._q_tmp1 = self.variables.q.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        qn = self.variables.q.spline.vector

        qn1 = self._G.dot(self._u, out=self._q_tmp1)
        qn1 *= -dt
        qn1 += qn

        diffs = self.update_feec_variables(q=qn1)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.debug(f"Maxdiff q3 for BackgroundQGradient: {diffs['q']}")
