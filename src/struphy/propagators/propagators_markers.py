"Only particle variables are updated."

from numpy import array, polynomial, random
from typing import Literal, get_args

from psydac.linalg.block import BlockVector
from psydac.linalg.stencil import StencilVector

from struphy.feec.mass import WeightedMassOperators
from struphy.fields_background.base import MHDequilibrium
from struphy.fields_background.equils import set_defaults
from struphy.io.setup import descend_options_dict
from struphy.ode.utils import ButcherTableau
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.base import Particles
from struphy.pic.particles import Particles3D, Particles5D, Particles6D, ParticlesSPH
from struphy.pic.pushing import eval_kernels_gc, pusher_kernels, pusher_kernels_gc
from struphy.pic.pushing.pusher import Pusher
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.models.variables import FEECVariable, PICVariable
from struphy.io.options import check_option
from psydac.linalg.basic import LinearOperator


class PushEta(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf x_p(t)}{\textnormal d t} = \mathbf v_p\,,

    for constant :math:`\mathbf v_p` in logical space given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \,\mathbf v_p\,.

    Available algorithms:

    * Explicit from :class:`~struphy.ode.utils.ButcherTableau`
    """
    # variables to be updated
    var: PICVariable = None

    ## abstract methods
    def set_options(self,
                    butcher: ButcherTableau = None,
                    ):
        
        # use setter for options
        self.options = self.Options(self,
                                    butcher=butcher,
                                    )

    def allocate(self):
        # get kernel
        kernel = Pyccelkernel(pusher_kernels.push_eta_stage)

        # define algorithm
        butcher: ButcherTableau = self.options.butcher
        # temp fix due to refactoring of ButcherTableau:
        from struphy.utils.arrays import xp as np

        butcher._a = np.diag(butcher.a, k=-1)
        butcher._a = np.array(list(butcher.a) + [0.0])

        args_kernel = (
            butcher.a,
            butcher.b,
            butcher.c,
        )

        self._pusher = Pusher(
            self.var.particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=butcher.n_stages,
            mpi_sort="each",
        )

    def __call__(self, dt):
        self._pusher(dt)

        # update_weights
        if self.var.particles.control_variate:
            self.var.particles.update_weights()


class PushVxB(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \kappa \, \mathbf v_p(t) \times (\mathbf B + \mathbf B_{\text{add}}) \,,

    where :math:`\kappa \in \mathbb R` is a constant scaling factor, and for rotation vector :math:`\mathbf B` and optional, additional fixed rotation
    vector :math:`\mathbf B_{\text{add}}`, both given as a 2-form:

    .. math::

        \mathbf B =  \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}\,.

    Available algorithms: ``analytic``, ``implicit``.
    """
    # variables to be updated
    ions: PICVariable = None
    
    # propagator specific options
    OptsAlgo = Literal["analytic", "implicit"]

    ## abstract methods
    def set_options(self,
                    algo: OptsAlgo = "analytic",
                    kappa: float = 1.0,
                    b2_var: FEECVariable = None,
                    ):
    
        # checks
        check_option(algo, self.OptsAlgo)
        
        # use setter for options
        self.options = self.Options(self,
                                    algo=algo, 
                                    kappa=kappa,
                                    b2_var=b2_var,
                                    )

    def allocate(self):
        # TODO: treat PolarVector as well, but polar splines are being reworked at the moment
        self._b2 = self.projected_equil.b2
        assert self._b2.space == self.derham.Vh["2"]
        
        if self.options.b2_var is None:
            self._b2_var = None
        else:
            assert self.options.b2_var.spline.vector.space == self.derham.Vh["2"]
            self._b2_var = self.options.b2_var.spline.vector
        
        # allocate dummy vectors to avoid temporary array allocations
        self._tmp = self.derham.Vh["2"].zeros()
        self._b_full = self.derham.Vh["2"].zeros()

        # define pusher kernel
        if self.options.algo == "analytic":
            kernel = pusher_kernels.push_vxb_analytic
        elif self.options.algo == "implicit":
            kernel = pusher_kernels.push_vxb_implicit
        else:
            raise ValueError(f"{self.options.algo = } not supported.")

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self._b_full[0]._data,
            self._b_full[1]._data,
            self._b_full[2]._data,
        )

        self._pusher = Pusher(
            self.ions.particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T: LinearOperator = self.derham.extraction_ops["2"].transpose()

    def __call__(self, dt):
        # sum up total magnetic field
        tmp = self._b2.copy(out=self._tmp)
        if self._b2_var is not None:
            tmp += self._b2_var

        # extract coefficients to tensor product space
        b_full: BlockVector = self._E2T.dot(tmp, out=self._b_full)
        b_full.update_ghost_regions()

        # call pusher kernel
        self._pusher(self.options.kappa * dt)

        # update_weights
        if self.ions.particles.control_variate:
            self.ions.particles.update_weights()


class PushVinEfield(Propagator):
    r"""Push the velocities according to

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \kappa \, \mathbf{E}(\mathbf{x}_p) \,,

    where :math:`\kappa \in \mathbb R` is a constant and in logical coordinates, given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \kappa \, DF^{-\top} \hat{\mathbf E}^1(\boldsymbol \eta_p)  \,,

    which is solved analytically.
    """

    @staticmethod
    def options():
        pass

    def __init__(
        self,
        particles: Particles6D,
        *,
        e_field: BlockVector | PolarVector,
        kappa: float = 1.0,
    ):
        super().__init__(particles)

        self.kappa = kappa

        assert isinstance(e_field, (BlockVector, PolarVector))
        self._e_field = e_field

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self._e_field[0]._data,
            self._e_field[1]._data,
            self._e_field[2]._data,
            self.kappa,
        )

        self._pusher = Pusher(
            particles,
            Pyccelkernel(pusher_kernels.push_v_with_efield),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(dt)


class PushEtaPC(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf x_p(t)}{\textnormal d t} = \mathbf v_p + \mathbf U (\mathbf x_p(t))\,,

    for constant :math:`\mathbf v_p` and :math:`\mathbf U` in logical space given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \,\mathbf v_p + \textnormal{vec}(\hat{\mathbf U}) \,,

    where

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,, \qquad \textnormal{vec}( \hat{\mathbf U}) = \hat{\mathbf U}\,.

    Available algorithms:

    * ``rk4`` (4th order, default)
    * ``forward_euler`` (1st order)
    * ``heun2`` (2nd order)
    * ``rk2`` (2nd order)
    * ``heun3`` (3rd order)
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct["use_perp_model"] = [True, False]

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(
        self,
        particles: Particles,
        *,
        u: BlockVector | PolarVector,
        use_perp_model: bool = options(default=True)["use_perp_model"],
        u_space: str,
    ):
        super().__init__(particles)

        assert isinstance(u, (BlockVector, PolarVector))

        self._u = u

        # call Pusher class
        if use_perp_model:
            if u_space == "Hcurl":
                kernel = Pyccelkernel(pusher_kernels.push_pc_eta_rk4_Hcurl)
            elif u_space == "Hdiv":
                kernel = Pyccelkernel(pusher_kernels.push_pc_eta_rk4_Hdiv)
            elif u_space == "H1vec":
                kernel = Pyccelkernel(pusher_kernels.push_pc_eta_rk4_H1vec)
            else:
                raise ValueError(
                    f'{u_space = } not valid, choose from "Hcurl", "Hdiv" or "H1vec.',
                )
        else:
            if u_space == "Hcurl":
                kernel = Pyccelkernel(pusher_kernels.push_pc_eta_rk4_Hcurl_full)
            elif u_space == "Hdiv":
                kernel = Pyccelkernel(pusher_kernels.push_pc_eta_rk4_Hdiv_full)
            elif u_space == "H1vec":
                kernel = Pyccelkernel(pusher_kernels.push_pc_eta_rk4_H1vec_full)
            else:
                raise ValueError(
                    f'{u_space = } not valid, choose from "Hcurl", "Hdiv" or "H1vec.',
                )

        args_kernel = (
            self.derham.args_derham,
            self._u[0]._data,
            self._u[1]._data,
            self._u[2]._data,
        )

        self._pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=4,
            mpi_sort="each",
        )

    def __call__(self, dt):
        # check if ghost regions are synchronized
        if not self._u[0].ghost_regions_in_sync:
            self._u[0].update_ghost_regions()
        if not self._u[1].ghost_regions_in_sync:
            self._u[1].update_ghost_regions()
        if not self._u[2].ghost_regions_in_sync:
            self._u[2].update_ghost_regions()

        self._pusher(dt)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()


class PushGuidingCenterBxEstar(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf X_p(t)}{\textnormal d t} = \frac{\mathbf E^* \times \mathbf b_0}{B_\parallel^*} (\mathbf X_p(t))   \,,

    where

    .. math::

        \mathbf E^* = -\nabla \phi - \varepsilon \mu_p \nabla |\mathbf B|\,,\qquad \mathbf B^* = \mathbf B + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad  B^*_\parallel = \mathbf B^* \cdot \mathbf b_0\,,

    where :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` can be the full magnetic field (equilibrium + perturbation).
    The electric potential ``phi`` and/or the magnetic perturbation ``b_tilde``
    can be ignored by passing ``None``.
    In logical space this is given by :math:`\mathbf X = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = \frac{\hat{\mathbf E}^{*1} \times \hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\boldsymbol \eta_p(t)) \,.

    Available algorithms:

    * Explicit from :class:`~struphy.ode.utils.ButcherTableau`
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order`
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order_newton`
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_2nd_order`
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct["algo"] = {
            "method": [
                "discrete_gradient_2nd_order",
                "discrete_gradient_1st_order",
                "discrete_gradient_1st_order_newton",
                "rk4",
                "forward_euler",
                "heun2",
                "rk2",
                "heun3",
            ],
            "maxiter": 20,
            "tol": 1e-7,
            "mpi_sort": "each",
            "verbose": False,
        }
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(
        self,
        particles: Particles5D,
        *,
        phi: StencilVector = None,
        evaluate_e_field: bool = False,
        b_tilde: BlockVector = None,
        epsilon: float = 1.0,
        algo: dict = options(default=True)["algo"],
    ):
        super().__init__(particles)

        # magnetic equilibrium field
        unit_b1 = self.projected_equil.unit_b1
        self._gradB1 = self.projected_equil.gradB1
        self._absB0 = self.projected_equil.absB0
        curl_unit_b_dot_b0 = self.projected_equil.curl_unit_b_dot_b0

        # magnetic perturbation
        self._b_tilde = b_tilde
        if self._b_tilde is not None:
            self._B_dot_b = self.derham.Vh["0"].zeros()
            self._grad_b_full = self.derham.Vh["1"].zeros()

            self._PB = getattr(self.basis_ops, "PB")

            B_dot_b = self._PB.dot(self._b_tilde, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0
        else:
            self._grad_b_full = self._gradB1
            self._B_dot_b = self._absB0

        # allocate electric field
        if phi is None:
            phi = self.derham.Vh["0"].zeros()
            self._evaluate_e_field = False
        self._phi = phi
        self._evaluate_e_field = evaluate_e_field
        self._e_field = self.derham.Vh["1"].zeros()
        self._epsilon = epsilon

        # choose method
        if "discrete_gradient" in algo["method"]:
            # place for storing data during iteration
            first_free_idx = particles.args_markers.first_free_idx

            if "1st_order" in algo["method"]:
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
                    eval_kernels_gc.unit_b_1form,
                    first_free_idx + 2,
                    (0, 1, 2),
                    (
                        self.derham.args_derham,
                        unit_b1[0]._data,
                        unit_b1[1]._data,
                        unit_b1[2]._data,
                    ),
                )

                if "newton" in algo["method"]:
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
                    kernel = Pyccelkernel(pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order_newton)

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
                        (
                            self.derham.args_derham,
                            self._epsilon,
                            self._B_dot_b._data,
                            self._phi._data,
                            self._evaluate_e_field,
                        ),
                        alpha=(1.0, 1.0, 1.0, 0.0),
                    )  # evaluate at eta^{n+1,k} and save

                    # pusher kernel
                    kernel = Pyccelkernel(pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order)

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

            elif "2nd_order" in algo["method"]:
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
                    alpha=(1.0, 1.0, 1.0, 0.0),
                )  # evaluate at eta^{n+1,k} and save)

                # pusher kernel
                kernel = Pyccelkernel(pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_2nd_order)

                alpha_in_kernel = 0.5  # evaluate at mid-point
                args_kernel = (
                    self.derham.args_derham,
                    self._epsilon,
                    unit_b1[0]._data,
                    unit_b1[1]._data,
                    unit_b1[2]._data,
                    self._grad_b_full[0]._data,
                    self._grad_b_full[1]._data,
                    self._grad_b_full[2]._data,
                    self._B_dot_b._data,
                    curl_unit_b_dot_b0._data,
                    self._e_field[0]._data,
                    self._e_field[1]._data,
                    self._e_field[2]._data,
                    self._evaluate_e_field,
                )

            else:
                raise NotImplementedError(
                    f"Chosen method {algo['method']} is not implemented.",
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
                maxiter=algo["maxiter"],
                tol=algo["tol"],
                mpi_sort=algo["mpi_sort"],
                verbose=algo["verbose"],
            )

        else:
            butcher = ButcherTableau(algo["method"])
            # temp fix due to refactoring of ButcherTableau:
            from struphy.utils.arrays import xp as np

            butcher._a = np.diag(butcher.a, k=-1)
            butcher._a = np.array(list(butcher.a) + [0.0])

            kernel = Pyccelkernel(pusher_kernels_gc.push_gc_bxEstar_explicit_multistage)

            args_kernel = (
                self.derham.args_derham,
                self._epsilon,
                unit_b1[0]._data,
                unit_b1[1]._data,
                unit_b1[2]._data,
                self._grad_b_full[0]._data,
                self._grad_b_full[1]._data,
                self._grad_b_full[2]._data,
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
                mpi_sort=algo["mpi_sort"],
                verbose=algo["verbose"],
            )

    def __call__(self, dt):
        # electric field
        # TODO: add out to __neg__ of StencilVector
        if self._evaluate_e_field:
            e_field = self.derham.grad.dot(-self._phi, out=self._e_field)
            e_field.update_ghost_regions()

        # magnetic perturbation
        if self._b_tilde is not None:
            B_dot_b = self._PB.dot(self._b_tilde, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0

        # call pusher
        self._pusher(dt)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()


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

    @staticmethod
    def options(default=False):
        dct = {}
        dct["algo"] = {
            "method": [
                "discrete_gradient_2nd_order",
                "discrete_gradient_1st_order",
                "discrete_gradient_1st_order_newton",
                "rk4",
                "forward_euler",
                "heun2",
                "rk2",
                "heun3",
            ],
            "maxiter": 20,
            "tol": 1e-7,
            "mpi_sort": "each",
            "verbose": False,
        }
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(
        self,
        particles: Particles5D,
        *,
        phi: StencilVector = None,
        evaluate_e_field: bool = False,
        b_tilde: BlockVector = None,
        epsilon: float = 1.0,
        algo: dict = options(default=True)["algo"],
    ):
        super().__init__(particles)

        self._epsilon = epsilon

        # magnetic equilibrium field
        self._gradB1 = self.projected_equil.gradB1
        b2 = self.projected_equil.b2
        curl_unit_b2 = self.projected_equil.curl_unit_b2
        self._absB0 = self.projected_equil.absB0
        curl_unit_b_dot_b0 = self.projected_equil.curl_unit_b_dot_b0

        # magnetic perturbation
        self._b_tilde = b_tilde
        if self._b_tilde is not None:
            self._B_dot_b = self.derham.Vh["0"].zeros()
            self._grad_b_full = self.derham.Vh["1"].zeros()

            self._PB = getattr(self.basis_ops, "PB")

            B_dot_b = self._PB.dot(self._b_tilde, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0
        else:
            self._grad_b_full = self._gradB1
            self._B_dot_b = self._absB0

        # allocate electric field
        if phi is None:
            phi = self.derham.Vh["0"].zeros()
        self._phi = phi
        self._evaluate_e_field = evaluate_e_field
        self._e_field = self.derham.Vh["1"].zeros()
        self._epsilon = epsilon

        # choose method
        if "discrete_gradient" in algo["method"]:
            # place for storing data during iteration
            first_free_idx = particles.args_markers.first_free_idx

            if "1st_order" in algo["method"]:
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

                if "newton" in algo["method"]:
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

            elif "2nd_order" in algo["method"]:
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

            else:
                raise NotImplementedError(
                    f"Chosen method {algo['method']} is not implemented.",
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
                maxiter=algo["maxiter"],
                tol=algo["tol"],
                mpi_sort=algo["mpi_sort"],
                verbose=algo["verbose"],
            )

        else:
            butcher = ButcherTableau(algo["method"])
            # temp fix due to refactoring of ButcherTableau:
            from struphy.utils.arrays import xp as np

            butcher._a = np.diag(butcher.a, k=-1)
            butcher._a = np.array(list(butcher.a) + [0.0])

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
                mpi_sort=algo["mpi_sort"],
                verbose=algo["verbose"],
            )

    def __call__(self, dt):
        # electric field
        # TODO: add out to __neg__ of StencilVector
        if self._evaluate_e_field:
            e_field = self.derham.grad.dot(-self._phi, out=self._e_field)
            e_field.update_ghost_regions()

        # magnetic perturbation
        if self._b_tilde is not None:
            B_dot_b = self._PB.dot(self._b_tilde, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0

        # call pusher
        self._pusher(dt)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()


class PushDeterministicDiffusion(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf x_p(t)}{\textnormal d t} = - D \, \frac{\nabla u}{ u}\mathbf (\mathbf x_p(t))\,,

    in logical space given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = - G\, D \, \frac{\nabla \Pi^0_{L^2}u_h}{\Pi^0_{L^2} u_h}\mathbf (\boldsymbol \eta_p(t))\,,
        \qquad [\Pi^0_{L^2, ijk} u_h](\boldsymbol \eta_p) = \frac 1N \sum_{p} w_p \boldsymbol \Lambda^0_{ijk}(\boldsymbol \eta_p)\,,

    where :math:`D>0` is a positive diffusion coefficient.

    Available algorithms:

    * Explicit from :class:`~struphy.ode.utils.ButcherTableau`
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct["algo"] = ["rk4", "forward_euler", "heun2", "rk2", "heun3"]
        dct["diffusion_coefficient"] = 1.0
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(
        self,
        particles: Particles3D,
        *,
        algo: str = options(default=True)["algo"],
        bc_type: list = ["periodic", "periodic", "periodic"],
        diffusion_coefficient: float = options()["diffusion_coefficient"],
    ):
        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        super().__init__(particles)

        self._bc_type = bc_type
        self._diffusion = diffusion_coefficient

        self._tmp = self.derham.Vh["1"].zeros()

        # choose algorithm
        self._butcher = ButcherTableau(algo)
        # temp fix due to refactoring of ButcherTableau:
        from struphy.utils.arrays import xp as np

        self._butcher._a = np.diag(self._butcher.a, k=-1)
        self._butcher._a = np.array(list(self._butcher.a) + [0.0])

        self._u_on_grid = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self._u_on_grid.vectors[0]._data,
            self._tmp[0]._data,
            self._tmp[1]._data,
            self._tmp[2]._data,
            self._diffusion,
            self._butcher.a,
            self._butcher.b,
            self._butcher.c,
        )

        self._pusher = Pusher(
            particles,
            Pyccelkernel(pusher_kernels.push_deterministic_diffusion_stage),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=self._butcher.n_stages,
            mpi_sort="each",
        )

    def __call__(self, dt):
        """
        TODO
        """

        # accumulate
        self._u_on_grid(self.particles[0].vdim)

        # take gradient
        pi_u = self._u_on_grid.vectors[0]
        grad_pi_u = self.derham.grad.dot(pi_u, out=self._tmp)
        grad_pi_u.update_ghost_regions()

        # push markers
        self._pusher(dt)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()


class PushRandomDiffusion(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \textnormal d \mathbf x_p(t) = \sqrt{2 D} \, \textnormal d \mathbf B_{t}\,,

    where :math:`D>0` is a positive diffusion coefficient and :math:`\textnormal d \mathbf B_{t}` is a Wiener process,

    .. math::

        \mathbf B_{t + \Delta t} - \mathbf B_{t} = \sqrt{\Delta t} \,\mathcal N(0;1)\,,

    with :math:`\mathcal N(0;1)` denoting the standard normal distribution with mean zero and variance one.

    Available algorithms:

    * ``forward_euler`` (1st order)
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct["algo"] = ["forward_euler"]
        dct["diffusion_coefficient"] = 1.0
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(
        self,
        particles: Particles3D,
        algo: str = options(default=True)["algo"],
        bc_type: list = ["periodic", "periodic", "periodic"],
        diffusion_coefficient: float = options()["diffusion_coefficient"],
    ):
        super().__init__(particles)

        self._bc_type = bc_type
        self._diffusion = diffusion_coefficient

        self._noise = array(self.particles[0].markers[:, :3])

        # choose algorithm
        self._butcher = ButcherTableau("forward_euler")
        # temp fix due to refactoring of ButcherTableau:
        from struphy.utils.arrays import xp as np

        self._butcher._a = np.diag(self._butcher.a, k=-1)
        self._butcher._a = np.array(list(self._butcher.a) + [0.0])

        # instantiate Pusher
        args_kernel = (
            self._noise,
            self._diffusion,
            self._butcher.a,
            self._butcher.b,
            self._butcher.c,
        )

        self._pusher = Pusher(
            particles,
            Pyccelkernel(pusher_kernels.push_random_diffusion_stage),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=self._butcher.n_stages,
            mpi_sort="each",
        )

        # self._tmp = self.derham.Vh['1'].zeros()
        self._mean = [0, 0, 0]
        self._cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def __call__(self, dt):
        """
        TODO
        """

        self._noise[:] = random.multivariate_normal(
            self._mean,
            self._cov,
            len(self.particles[0].markers),
        )

        # push markers
        self._pusher(dt)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()


class PushVinSPHpressure(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} = \kappa_p \sum_{i=1}^N w_i \left( \frac{1}{\rho^{N,h}(\boldsymbol \eta_p)} + \frac{1}{\rho^{N,h}(\boldsymbol \eta_i)} \right) DF^{-\top}\nabla W_h(\boldsymbol \eta_p - \boldsymbol \eta_i) \,,

    where :math:`DF^{-\top}` denotes the inverse transpose Jacobian, and with the smoothed density

    .. math::

        \rho^{N,h}(\boldsymbol \eta) = \frac 1N \sum_{j=1}^N w_j \, W_h(\boldsymbol \eta - \boldsymbol \eta_j)\,,

    where :math:`W_h(\boldsymbol \eta)` is a smoothing kernel from :mod:`~struphy.pic.sph_smoothing_kernels`.
    Time stepping:

    * Explicit from :class:`~struphy.ode.utils.ButcherTableau`
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct["kernel_type"] = list(Particles.ker_dct())
        dct["algo"] = [
            "forward_euler",
        ]  # "heun2", "rk2", "heun3", "rk4"]
        dct["gravity"] = (0.0, 0.0, 0.0)
        dct["thermodynamics"] = ["isothermal", "polytropic"]
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(
        self,
        particles: ParticlesSPH,
        *,
        kernel_type: str = "gaussian_2d",
        kernel_width: tuple = None,
        algo: str = options(default=True)["algo"],  # TODO: implement other algos than forward Euler
        gravity: tuple = options(default=True)["gravity"],
        thermodynamics: str = options(default=True)["thermodynamics"],
    ):
        # base class constructor call
        super().__init__(particles)

        # init kernel for evaluating density etc. before each time step.
        init_kernel = Pyccelkernel(eval_kernels_gc.sph_pressure_coeffs)

        first_free_idx = particles.args_markers.first_free_idx
        comps = (0, 1, 2)

        boxes = particles.sorting_boxes.boxes
        neighbours = particles.sorting_boxes.neighbours
        holes = particles.holes
        periodic = [bci == "periodic" for bci in particles.bc]
        kernel_nr = particles.ker_dct()[kernel_type]

        if kernel_width is None:
            kernel_width = tuple([1 / ni for ni in self.particles[0].boxes_per_dim])
        else:
            assert all([hi <= 1 / ni for hi, ni in zip(kernel_width, self.particles[0].boxes_per_dim)])

        # init kernel
        args_init = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *kernel_width,
        )

        self.add_init_kernel(
            init_kernel,
            first_free_idx,
            comps,
            args_init,
        )

        # pusher kernel
        if thermodynamics == "isothermal":
            kernel = Pyccelkernel(pusher_kernels.push_v_sph_pressure)
        elif thermodynamics == "polytropic":
            kernel = Pyccelkernel(pusher_kernels.push_v_sph_pressure_ideal_gas)

        gravity = np.array(gravity, dtype=float)

        args_kernel = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *kernel_width,
            gravity,
        )

        # the Pusher class wraps around all kernels
        self._pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=0.0,
            init_kernels=self.init_kernels,
        )

    def __call__(self, dt):
        self.particles[0].put_particles_in_boxes()
        self._pusher(dt)


class PushVinViscousPotential(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} = \kappa_p \sum_{i=1}^N w_i \left( \frac{1}{\rho^{N,h}(\boldsymbol \eta_p)} + \frac{1}{\rho^{N,h}(\boldsymbol \eta_i)} \right) DF^{-\top}\nabla W_h(\boldsymbol \eta_p - \boldsymbol \eta_i) \,,

    where :math:`DF^{-\top}` denotes the inverse transpose Jacobian, and with the smoothed density

    .. math::

        \rho^{N,h}(\boldsymbol \eta) = \frac 1N \sum_{j=1}^N w_j \, W_h(\boldsymbol \eta - \boldsymbol \eta_j)\,,

    where :math:`W_h(\boldsymbol \eta)` is a smoothing kernel from :mod:`~struphy.pic.sph_smoothing_kernels`.
    Time stepping:

    * Explicit from :class:`~struphy.ode.utils.ButcherTableau`
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct["kernel_type"] = [ker for ker in list(Particles.ker_dct()) if "3d" in ker]
        dct["kernel_width"] = None
        dct["algo"] = [
            "forward_euler",
        ]  # "heun2", "rk2", "heun3", "rk4"]
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(
        self,
        particles: ParticlesSPH,
        *,
        kernel_type: str = "gaussian_3d",
        kernel_width: tuple = None,
        algo: str = options(default=True)["algo"],  # TODO: implement other algos than forward Euler
    ):
        # base class constructor call
        super().__init__(particles)

        # init kernel for evaluating density etc. before each time step.
        init_kernel_1 = eval_kernels_gc.sph_mean_velocity_coeffs
        first_free_idx = particles.args_markers.first_free_idx
        comps = (0, 1, 2)

        init_kernel_2 = eval_kernels_gc.sph_mean_velocity
        # first_free_idx = particles.args_markers.first_free_idx
        # comps = (0, 1, 2)

        init_kernel_3 = eval_kernels_gc.sph_grad_mean_velocity
        comps_tensor = (0, 1, 2, 3, 4, 5, 6, 7, 8)

        init_kernel_4 = eval_kernels_gc.sph_viscosity_tensor

        boxes = particles.sorting_boxes.boxes
        neighbours = particles.sorting_boxes.neighbours
        holes = particles.holes
        periodic = [bci == "periodic" for bci in particles.bc]
        kernel_nr = particles.ker_dct()[kernel_type]

        if kernel_width is None:
            kernel_width = tuple([1 / ni for ni in self.particles[0].boxes_per_dim])
        else:
            assert all([hi <= 1 / ni for hi, ni in zip(kernel_width, self.particles[0].boxes_per_dim)])

        # init kernel
        args_init = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *kernel_width,
        )

        self.add_init_kernel(
            init_kernel_1,
            first_free_idx,
            comps,
            args_init,
        )

        self.add_init_kernel(
            init_kernel_2,
            first_free_idx + 3,  # +3 so that the previous one is not overwritten
            comps,
            args_init,
        )

        self.add_init_kernel(
            init_kernel_3,
            first_free_idx + 6,  # +3 so that the previous one is not overwritten
            comps_tensor,
            args_init,
        )

        self.add_init_kernel(
            init_kernel_4,
            first_free_idx + 15,
            comps_tensor,
            args_init,
        )

        kernel = Pyccelkernel(pusher_kernels.push_v_viscosity)

        args_kernel = (
            boxes,
            neighbours,
            holes,
            *periodic,
            kernel_nr,
            *kernel_width,
        )

        # the Pusher class wraps around all kernels
        self._pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=0.0,
            init_kernels=self.init_kernels,
        )

    def __call__(self, dt):
        self.particles[0].put_particles_in_boxes()
        self._pusher(dt)
