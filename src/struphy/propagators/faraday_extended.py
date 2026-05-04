import cunumpy as xp
from feectools.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from feectools.linalg.solvers import inverse
from numpy import zeros
import struphy.feec.utilities as util
from struphy.feec import preconditioner
from struphy.fields_background.equils import set_defaults
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator

class FaradayExtended(Propagator):
    r"""Equations: Faraday's law

    .. math::
        \begin{align*}
        & \frac{\partial {\mathbf A}}{\partial t} = - \frac{\nabla \times (\nabla \times {\mathbf A} + {\mathbf B}_0) }{n} \times (\nabla \times {\mathbf A} + {\mathbf B}_0) - \frac{\int ({\mathbf A} - {\mathbf p}f \mathrm{d}{\mathbf p})}{n} \times (\nabla \times {\mathbf A} + {\mathbf B}_0), \\
        & n = \int f \mathrm{d}{\mathbf p}.
        \end{align*}

    Mid-point rule:

    .. math::
        \begin{align*}
        & \left[ \mathbb{M}_1 - \frac{\Delta t}{2} \mathbb{F}(\hat{n}^0_h, {\mathbf a}^{n+\frac{1}{2}}) \mathbb{M}_1^{-1} (\mathbb{P}_1^\top \mathbb{W} \mathbb{P}_1 + \mathbb{C}^\top \mathbb{M}_2 \mathbb{C} ) \right] {\mathbf a}^{n+1} \\
        & = \mathbb{M}_1 {\mathbf a}^n + \frac{\Delta t}{2} \mathbb{F}(\hat{n}^0_h, {\mathbf a}^{n+\frac{1}{2}}) \mathbb{M}_1^{-1} (\mathbb{P}_1^\top \mathbb{W} \mathbb{P}_1 + \mathbb{C}^\top \mathbb{M}_2 \mathbb{C} ) {\mathbf a}^{n+1} \\
        & - \Delta t \mathbb{F}(\hat{n}^0_h, {\mathbf a}^{n+\frac{1}{2}})  \mathbb{M}_1^{-1} \mathbb{P}_1^\top \mathbb{W} {\mathbf P}^n\\
        & + \Delta t \mathbb{F}(\hat{n}^0_h, {\mathbf a}^{n+\frac{1}{2}}) \mathbb{M}_1^{-1} \mathbb{C}^\top \mathbb{M}_2 {\mathbf b}_0\\
        & \mathbb{F}_{ij} = - \int \frac{1}{\hat{n}^0_h \sqrt{g}} G (\nabla \times {\mathbf A} + {\mathbf B}_0) \cdot (\Lambda^1_i \times \Lambda^1_j) \mathrm{d}{\boldsymbol \eta}.
        \end{align*}

    Parameters
    ---------- 
        a : feectools.linalg.block.BlockVector
            FE coefficients of vector potential.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, a, **params):
        assert isinstance(a, (BlockVector, PolarVector))

        # parameters
        params_default = {
            "a_space": None,
            "beq": None,
            "particles": None,
            "quad_number": None,
            "shape_degree": None,
            "shape_size": None,
            "solver_params": None,
            "accumulate_density": None,
        }

        params = set_defaults(params, params_default)

        self._a = a
        self._a_old = self._a.copy()

        self._a_space = params["a_space"]
        assert self._a_space in {"Hcurl"}

        self._beq = params["beq"]

        self._particles = params["particles"]

        self._nqs = params["quad_number"]

        self.size1 = int(self.derham.domain_array[self.rank, int(2)])
        self.size2 = int(self.derham.domain_array[self.rank, int(5)])
        self.size3 = int(self.derham.domain_array[self.rank, int(8)])

        self.weight_1 = zeros(
            (self.size1 * self._nqs[0], self.size2 * self._nqs[1], self.size3 * self._nqs[2]),
            dtype=float,
        )
        self.weight_2 = zeros(
            (self.size1 * self._nqs[0], self.size2 * self._nqs[1], self.size3 * self._nqs[2]),
            dtype=float,
        )
        self.weight_3 = zeros(
            (self.size1 * self._nqs[0], self.size2 * self._nqs[1], self.size3 * self._nqs[2]),
            dtype=float,
        )

        self._weight_pre = [self.weight_1, self.weight_2, self.weight_3]

        self._ind = [
            [self.derham.indN[0], self.derham.indD[1], self.derham.indD[2]],
            [self.derham.indD[0], self.derham.indN[1], self.derham.indD[2]],
            [self.derham.indD[0], self.derham.indD[1], self.derham.indN[2]],
        ]

        # Initialize Accumulator object for getting density from particles
        self._pts_x = 1.0 / (2.0 * self.derham.num_elements[0]) * xp.polynomial.legendre.leggauss(
            self._nqs[0],
        )[0] + 1.0 / (2.0 * self.derham.num_elements[0])
        self._pts_y = 1.0 / (2.0 * self.derham.num_elements[1]) * xp.polynomial.legendre.leggauss(
            self._nqs[1],
        )[0] + 1.0 / (2.0 * self.derham.num_elements[1])
        self._pts_z = 1.0 / (2.0 * self.derham.num_elements[2]) * xp.polynomial.legendre.leggauss(
            self._nqs[2],
        )[0] + 1.0 / (2.0 * self.derham.num_elements[2])

        self._p_shape = params["shape_degree"]
        self._p_size = params["shape_size"]
        self._accum_density = params["accumulate_density"]

        # Initialize Accumulator object for getting the matrix and vector related with vector potential
        self._accum_potential = Accumulator(
            self.mass_ops,
            self.domain,
            self._a_space,
            "hybrid_fA_Arelated",
            add_vector=True,
            symmetry="symm",
        )

        self._solver_params = params["solver_params"]
        # preconditioner
        if self._solver_params["pc"] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, self._solver_params["pc"])
            self._pc = pc_class(self.mass_ops.M1)

        self._Minv = inverse(self.mass_ops.M1, tol=1e-8)
        self._CMC = self.derham.curl.T @ self.mass_ops.M2 @ self.derham.curl
        self._M1 = self.mass_ops.M1
        self._M2 = self.mass_ops.M2

    @property
    def variables(self):
        return [self._a]

    def __call__(self, dt):
        # the loop of fixed point iteration, 100 iterations at most.

        self._accum_density.accumulate(
            self._particles,
            xp.array(self.derham.num_elements),
            xp.array(self._nqs),
            xp.array(
                self._pts_x,
            ),
            xp.array(self._pts_y),
            xp.array(self._pts_z),
            xp.array(self._p_shape),
            xp.array(self._p_size),
        )
        self._accum_potential.accumulate(self._particles)

        self._L2 = -dt / 2 * self._Minv @ (self._accum_potential._operators[0].matrix + self._CMC)
        self._RHS = -(self._L2.dot(self._a)) - dt * (
            self._Minv.dot(
                self._accum_potential._vectors[0] - self.derham.curl.T @ self._M2,
            ).dot(self._beq)
        )
        self._rhs = self._M1.dot(self._a)

        for _ in range(10):
            # logger.info('+++++=====++++++', self._accum_density._operators[0].matrix._data)
            # set mid-value used in the fixed iteration
            curla_mid = (
                self.derham.curl.dot(
                    0.5 * (self._a_old + self._a),
                )
                + self._beq
            )
            curla_mid.update_ghost_regions()
            # initialize the curl A
            # remember to check ghost region of curla_mid
            util.create_weight_weightedmatrix_hybrid(
                curla_mid,
                self._weight_pre,
                self.derham,
                self._accum_density,
                self.domain,
            )
            # self._weight = [[None, self._weight_pre[2], -self._weight_pre[1]], [None, None, self._weight_pre[0]], [None, None, None]]
            self._weight = [
                [0.0 * self._weight_pre[k] for k in range(3)],
                [0.0 * self._weight_pre[k] for k in range(3)],
                [0.0 * self._weight_pre[k] for k in range(3)],
            ]
            # self._weight = [[self._weight_pre[0], self._weight_pre[2], self._weight_pre[1]], [self._weight_pre[2], self._weight_pre[1], self._weight_pre[0]], [self._weight_pre[1], self._weight_pre[0], self._weight_pre[2]]]
            HybridM1 = self.mass_ops.create_weighted_mass("Hcurl", "Hcurl", weights=self._weight, assemble=True)

            # next prepare for solving linear system
            _LHS = self._M1 + HybridM1 @ self._L2
            _RHS2 = HybridM1.dot(self._RHS) + self._rhs

            # TODO: unknown function 'pcg', use new solver API
            raise NotImplementedError("pcg solver not available, use new solver API!")
            # a_new, info = pcg(
            #     _LHS,
            #     _RHS2,
            #     self._pc,
            #     x0=self._a,
            #     tol=self._solver_params["tol"],
            #     maxiter=self._solver_params["maxiter"],
            #     verbose=self._solver_params["verbose"],
            # )

            # write new coeffs into Propagator.variables
            # max_da = self.feec_vars_update(a_new)
            # logger.info("++++====check_iteration_error=====+++++", max_da)
            # we can modify the diff function in in_place_update to get another type errors
            # if max_da[0] < 10 ** (-6):
            #     break

    @classmethod
    def options(cls):
        dct = {}
        dct["solver"] = {
            "type": [
                ("pcg", "MassMatrixPreconditioner"),
                ("cg", None),
            ],
            "tol": 1.0e-8,
            "maxiter": 3000,
            "info": False,
            "verbose": False,
            "recycle": True,
        }
        return dct
