"Only particle variables are updated."

import logging
from dataclasses import dataclass
from typing import Literal

from line_profiler import profile

from struphy.io.options import LiteralOptions, OptionsBase
from struphy.models.variables import FEECVariable, PICVariable
from struphy.ode.utils import ButcherTableau
from struphy.pic.pushing import eval_kernels_gc, pusher_kernels_gc
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class PushGuidingCenterParallel(Propagator):
    r"""For each marker :math:`p`, solves

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

        \mathbf E^* = -\nabla \phi - \varepsilon \mu_p \nabla |\mathbf B|\,,\qquad \mathbf B^* = \mathbf B + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad  B^*_\parallel = \mathbf B^* \cdot \mathbf b_0\,,

    where :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` can be the full magnetic field (equilibrium + perturbation).
    The electric potential ``phi`` and/or the magnetic perturbation ``b_tilde`` 
    can be ignored by passing ``None``.
    In logical space this is given by :math:`\mathbf X = F(\boldsymbol \eta)`:

    .. math::

        \left\{ 
            \begin{aligned} 
                \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} &= v_{\parallel,p}(t) \frac{\hat{\mathbf B}^{*2}}{\hat B^{*3}_\parallel}(\boldsymbol \eta_p(t)) \,,
                \\
                \frac{\textnormal d v_{\parallel,p}(t)}{\textnormal d t} &= \frac{1}{\varepsilon} \frac{\hat{\mathbf B}^{*2}}{\hat B^{*3}_\parallel} \cdot \hat{\mathbf E}^{*1} (\boldsymbol \eta_p(t)) \,.
            \end{aligned}
        \right.

    Available algorithms:

    * Explicit from :class:`~struphy.ode.utils.ButcherTableau`
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order`
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order_newton` 
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_2nd_order`  
    """

    class Variables:
        """Container for variables advanced by :class:`PushGuidingCenterParallel`.

        Attributes
        ----------
        ions : PICVariable
            Guiding-center particle variable in ``"Particles5D"`` space.
        """

        def __init__(self):
            self._ions: PICVariable = None

        @property
        def ions(self) -> PICVariable:
            return self._ions

        @ions.setter
        def ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles5D"
            self._ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`PushGuidingCenterParallel`.

        Parameters
        ----------
        phi : FEECVariable, default=None
            Electrostatic potential variable in ``"H1"`` space.
            If ``None``, defaults to ``FEECVariable(space="H1")``.

        evaluate_e_field : bool, default=False
            If ``True``, evaluate and include electric-field contributions in
            drift-kinetic kernels.

        b_tilde : FEECVariable, default=None
            Optional magnetic perturbation variable added to the equilibrium
            magnetic field.

        algo : {"discrete_gradient_2nd_order", "discrete_gradient_1st_order", "discrete_gradient_1st_order_newton", "explicit"}, default="discrete_gradient_1st_order"
            Guiding-center pushing algorithm.

        butcher : ButcherTableau, default=None
            Butcher tableau used in explicit mode.
            If ``None`` and ``algo="explicit"``, defaults to
            ``ButcherTableau()``.

        maxiter : int, default=20
            Maximum number of fixed-point or Newton iterations in
            discrete-gradient modes.

        tol : float, default=1e-7
            Convergence tolerance for iterative discrete-gradient updates.

        mpi_sort : LiteralOptions.OptsMPIsort, default="each"
            MPI sorting policy for particle exchange.

        verbose : bool, default=False
            Verbosity flag for iterative pusher diagnostics.
        """

        # specific literals
        OptsAlgo = Literal[
            "discrete_gradient_2nd_order",
            "discrete_gradient_1st_order",
            "discrete_gradient_1st_order_newton",
            "explicit",
        ]
        # propagator options
        phi: FEECVariable = None
        evaluate_e_field: bool = False
        b_tilde: FEECVariable = None
        algo: OptsAlgo = "discrete_gradient_1st_order"
        butcher: ButcherTableau = None
        maxiter: int = 20
        tol: float = 1e-7
        mpi_sort: LiteralOptions.OptsMPIsort = "each"
        verbose: bool = False

        def __post_init__(self):
            # checks
            check_option(self.algo, self.OptsAlgo)
            check_option(self.mpi_sort, LiteralOptions.OptsMPIsort)

            # defaults
            if self.phi is None:
                self.phi = FEECVariable(space="H1")

            if self.algo == "explicit" and self.butcher is None:
                self.butcher = ButcherTableau()

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
    def allocate(self, verbose: bool = False):
        # scaling factor
        self._epsilon = self.variables.ions.species.equation_params.epsilon

        # magnetic equilibrium field
        self._gradB1 = self.projected_equil.gradB1
        b2 = self.projected_equil.b2
        curl_unit_b2 = self.projected_equil.curl_unit_b2
        self._absB0 = self.projected_equil.absB0
        curl_unit_b_dot_b0 = self.projected_equil.curl_unit_b_dot_b0

        # magnetic perturbation
        if self.options.b_tilde is not None:
            self._B_dot_b = self.derham.V0.zeros()
            self._grad_b_full = self.derham.V1.zeros()

            self._PB = getattr(self.basis_ops, "PB")

            B_dot_b = self._PB.dot(self.options.b_tilde.spline.vector, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0
        else:
            self._grad_b_full = self._gradB1
            self._B_dot_b = self._absB0

        # allocate electric field
        self.options.phi.allocate(self.derham, domain=self.domain)
        self._phi = self.options.phi.spline.vector
        self._evaluate_e_field = self.options.evaluate_e_field
        self._e_field = self.derham.V1.zeros()

        # choose method
        particles = self.variables.ions.particles

        if "discrete_gradient" in self.options.algo:
            # place for storing data during iteration
            first_free_idx = particles.args_markers.first_free_idx

            if "1st_order" in self.options.algo:
                # init kernels
                self.add_init_kernel(
                    eval_kernels_gc.driftkinetic_hamiltonian,
                    first_free_idx,
                    None,
                    (
                        self.derham.args_derham,
                        self._epsilon,
                        self._B_dot_b._data,
                        self._phi._data,
                        self._evaluate_e_field,
                    ),
                )

                self.add_init_kernel(
                    eval_kernels_gc.bstar_parallel_3form,
                    first_free_idx + 1,
                    None,
                    (
                        self.derham.args_derham,
                        self._epsilon,
                        self._B_dot_b._data,
                        curl_unit_b_dot_b0._data,
                    ),
                )

                self.add_init_kernel(
                    eval_kernels_gc.bstar_2form,
                    first_free_idx + 2,
                    (0, 1, 2),
                    (
                        self.derham.args_derham,
                        self._epsilon,
                        b2[0]._data,
                        b2[1]._data,
                        b2[2]._data,
                        curl_unit_b2[0]._data,
                        curl_unit_b2[1]._data,
                        curl_unit_b2[2]._data,
                    ),
                )

                if "newton" in self.options.algo:
                    # eval kernels
                    self.add_eval_kernel(
                        eval_kernels_gc.driftkinetic_hamiltonian,
                        first_free_idx + 5,
                        None,
                        (
                            self.derham.args_derham,
                            self._epsilon,
                            self._B_dot_b._data,
                            self._phi._data,
                            self._evaluate_e_field,
                        ),
                        alpha=(1.0, 0.0, 0.0, 0.0),
                    )

                    self.add_eval_kernel(
                        eval_kernels_gc.driftkinetic_hamiltonian,
                        first_free_idx + 6,
                        None,
                        (
                            self.derham.args_derham,
                            self._epsilon,
                            self._B_dot_b._data,
                            self._phi._data,
                            self._evaluate_e_field,
                        ),
                        alpha=(1.0, 1.0, 0.0, 0.0),
                    )

                    self.add_eval_kernel(
                        eval_kernels_gc.grad_driftkinetic_hamiltonian,
                        first_free_idx + 7,
                        (0,),
                        (
                            self.derham.args_derham,
                            self._epsilon,
                            self._grad_b_full[0]._data,
                            self._grad_b_full[1]._data,
                            self._grad_b_full[2]._data,
                            self._e_field[0]._data,
                            self._e_field[1]._data,
                            self._e_field[2]._data,
                            self._evaluate_e_field,
                        ),
                        alpha=(1.0, 0.0, 0.0, 0.0),
                    )

                    self.add_eval_kernel(
                        eval_kernels_gc.grad_driftkinetic_hamiltonian,
                        first_free_idx + 8,
                        (0, 1),
                        (
                            self.derham.args_derham,
                            self._epsilon,
                            self._grad_b_full[0]._data,
                            self._grad_b_full[1]._data,
                            self._grad_b_full[2]._data,
                            self._e_field[0]._data,
                            self._e_field[1]._data,
                            self._e_field[2]._data,
                            self._evaluate_e_field,
                        ),
                        alpha=(1.0, 1.0, 0.0, 0.0),
                    )

                    # pusher kernel
                    kernel = Pyccelkernel(pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order_newton)

                    alpha_in_kernel = 1.0  # evaluate at eta^{n+1,k} and save
                    args_kernel = (
                        self.derham.args_derham,
                        self._epsilon,
                        self._grad_b_full[0]._data,
                        self._grad_b_full[1]._data,
                        self._grad_b_full[2]._data,
                        self._B_dot_b._data,
                        self._e_field[0]._data,
                        self._e_field[1]._data,
                        self._e_field[2]._data,
                        self._phi._data,
                        self._evaluate_e_field,
                    )
                else:
                    # eval kernels
                    self.add_eval_kernel(
                        eval_kernels_gc.driftkinetic_hamiltonian,
                        first_free_idx + 5,
                        None,
                        args_eval=(
                            self.derham.args_derham,
                            self._epsilon,
                            self._B_dot_b._data,
                            self._phi._data,
                            self._evaluate_e_field,
                        ),
                        alpha=1.0,
                    )  # evaluate at Z^{n+1,k} and save

                    # pusher kernel
                    kernel = Pyccelkernel(pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order)

                    alpha_in_kernel = 0.5  # evaluate at mid-point
                    args_kernel = (
                        self.derham.args_derham,
                        self._epsilon,
                        self._grad_b_full[0]._data,
                        self._grad_b_full[1]._data,
                        self._grad_b_full[2]._data,
                        self._e_field[0]._data,
                        self._e_field[1]._data,
                        self._e_field[2]._data,
                        self._evaluate_e_field,
                    )

            elif "2nd_order" in self.options.algo:
                # init kernels (evaluate at eta^n and save)
                self.add_init_kernel(
                    eval_kernels_gc.driftkinetic_hamiltonian,
                    first_free_idx,
                    None,
                    (
                        self.derham.args_derham,
                        self._epsilon,
                        self._B_dot_b._data,
                        self._phi._data,
                        self._evaluate_e_field,
                    ),
                )

                # eval kernels
                self.add_eval_kernel(
                    eval_kernels_gc.driftkinetic_hamiltonian,
                    first_free_idx + 1,
                    None,
                    (
                        self.derham.args_derham,
                        self._epsilon,
                        self._B_dot_b._data,
                        self._phi._data,
                        self._evaluate_e_field,
                    ),
                    alpha=1.0,
                )  # evaluate at Z^{n+1,k} and save

                # pusher kernel
                kernel = Pyccelkernel(pusher_kernels_gc.push_gc_Bstar_discrete_gradient_2nd_order)

                alpha_in_kernel = 0.5  # evaluate at mid-point
                args_kernel = (
                    self.derham.args_derham,
                    self._epsilon,
                    self._grad_b_full[0]._data,
                    self._grad_b_full[1]._data,
                    self._grad_b_full[2]._data,
                    b2[0]._data,
                    b2[1]._data,
                    b2[2]._data,
                    curl_unit_b2[0]._data,
                    curl_unit_b2[1]._data,
                    curl_unit_b2[2]._data,
                    self._B_dot_b._data,
                    curl_unit_b_dot_b0._data,
                    self._e_field[0]._data,
                    self._e_field[1]._data,
                    self._e_field[2]._data,
                    self._evaluate_e_field,
                )

            # Pusher instance
            self._pusher = Pusher(
                particles,
                kernel,
                args_kernel,
                self.domain.args_domain,
                alpha_in_kernel=alpha_in_kernel,
                init_kernels=self.init_kernels,
                eval_kernels=self.eval_kernels,
                maxiter=self.options.maxiter,
                tol=self.options.tol,
                mpi_sort=self.options.mpi_sort,
            )

        else:
            if self.options.butcher is None:
                butcher = ButcherTableau()
            else:
                butcher = self.options.butcher
            # temp fix due to refactoring of ButcherTableau:
            import cunumpy as xp

            butcher._a = xp.diag(butcher.a, k=-1)
            butcher._a = xp.array(list(butcher.a) + [0.0])

            kernel = Pyccelkernel(pusher_kernels_gc.push_gc_Bstar_explicit_multistage)

            args_kernel = (
                self.derham.args_derham,
                self._epsilon,
                self._grad_b_full[0]._data,
                self._grad_b_full[1]._data,
                self._grad_b_full[2]._data,
                b2[0]._data,
                b2[1]._data,
                b2[2]._data,
                curl_unit_b2[0]._data,
                curl_unit_b2[1]._data,
                curl_unit_b2[2]._data,
                self._B_dot_b._data,
                curl_unit_b_dot_b0._data,
                self._e_field[0]._data,
                self._e_field[1]._data,
                self._e_field[2]._data,
                self._evaluate_e_field,
                butcher.a,
                butcher.b,
                butcher.c,
            )

            self._pusher = Pusher(
                particles,
                kernel,
                args_kernel,
                self.domain.args_domain,
                alpha_in_kernel=1.0,
                n_stages=butcher.n_stages,
                mpi_sort=self.options.mpi_sort,
            )

    @profile
    def __call__(self, dt):
        # electric field
        # TODO: add out to __neg__ of StencilVector
        if self._evaluate_e_field:
            e_field = self.derham.grad.dot(-self._phi, out=self._e_field)
            e_field.update_ghost_regions()

        # magnetic perturbation
        if self.options.b_tilde is not None:
            B_dot_b = self._PB.dot(self.options.b_tilde.spline.vector, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0

        # call pusher
        self._pusher(dt)

        # update_weights
        if self.variables.ions.species.weights_params.control_variate:
            self.variables.ions.particles.update_weights()
