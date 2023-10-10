'Only FEEC variables are updated.'


import numpy as np
from numpy import zeros

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.accumulation.particles_to_grid import Accumulator
from struphy.polar.basic import PolarVector
from struphy.kinetic_background.base import Maxwellian
from struphy.kinetic_background.maxwellians import Maxwellian6DUniform, Maxwellian5DUniform
from struphy.fields_background.mhd_equil.equils import set_defaults

from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api.linear_operators import InverseLinearOperator as Inverse
from struphy.psydac_api.linear_operators import IdentityOperator
from struphy.psydac_api import preconditioner
from struphy.psydac_api.mass import WeightedMassOperator
import struphy.linear_algebra.iterative_solvers as it_solvers
from struphy.linear_algebra.iterative_solvers import PConjugateGradient as pcg

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector
import struphy.psydac_api.utilities as util
from mpi4py import MPI


class Maxwell(Propagator):
    r'''Crank-Nicolson step

    .. math::

        \begin{bmatrix} \mathbf e^{n+1} - \mathbf e^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & \mathbb M_1^{-1} \mathbb C^\top \\ - \mathbb C \mathbb M_1^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} \mathbb M_1(\mathbf e^{n+1} + \mathbf e^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ----------
    e : psydac.linalg.block.BlockVector
        FE coefficients of a 1-form.

    b : psydac.linalg.block.BlockVector
        FE coefficients of a 2-form.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, e, b, **params):

        super().__init__(e, b)

        # parameters
        params_default = {'type': ('PConjugateGradient', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1

        # no dt
        self._B = Multiply(-1/2, Compose(self.derham.curl.T, self.mass_ops.M2))
        self._C = Multiply(1/2, self.derham.curl)

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(self.mass_ops.M1)

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'][0],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate place-holder vectors to avoid temporary array allocations in __call__
        self._e_tmp1 = e.space.zeros()
        self._e_tmp2 = e.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    def __call__(self, dt):

        # current variables
        en = self.feec_vars[0]
        bn = self.feec_vars[1]

        # solve for new e coeffs
        self._B.dot(bn, out=self._byn)

        info = self._schur_solver(en, self._byn, dt, out=self._e_tmp1)[1]

        # new b coeffs
        en.copy(out=self._e_tmp2)
        self._e_tmp2 += self._e_tmp1
        self._C.dot(self._e_tmp2, out=self._b_tmp1)
        self._b_tmp1 *= -dt
        self._b_tmp1 += bn

        # write new coeffs into self.feec_vars
        max_de, max_db = self.feec_vars_update(self._e_tmp1, self._b_tmp1)

        if self._info and self._rank == 0:
            print('Status     for Maxwell:', info['success'])
            print('Iterations for Maxwell:', info['niter'])
            print('Maxdiff e1 for Maxwell:', max_de)
            print('Maxdiff b2 for Maxwell:', max_db)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PConjugateGradient', 'MassMatrixPreconditioner'),
                                  ('ConjugateGradient', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class OhmCold(Propagator):
    r'''Crank-Nicolson step

    .. math::

        \begin{bmatrix}
            \mathbf j^{n+1} - \mathbf j^n \\
            \mathbf e^{n+1} - \mathbf e^n
        \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix}
            0 & \frac{1}{\varepsilon_c} \mathbb M_{1,1/n}^{-1} \\
            - \frac{1}{\varepsilon_c} \mathbb M_{1,1/n}^{-1} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbb \alpha^2 M_{1,1/n} (\mathbf j^{n+1} + \mathbf j^{n}) \\
            \mathbb M_1 (\mathbf e^{n+1} + \mathbf e^{n})
        \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ----------
        j : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        e : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        params : dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, j, e, **params):

        super().__init__(e, j)

        # parameters
        params_default = {'type': ('PConjugateGradient', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'alpha': 1.0,
                          'epsilon': 1.0}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._alpha = params['alpha']
        self._epsilon = params['epsilon']

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1ninv

        self._B = Multiply(-1/2 * 1/self._epsilon, self.mass_ops.M1)  # no dt
        self._C = Multiply(1/2 * self._alpha**2 / self._epsilon,
                           IdentityOperator(self.derham._Vh["1"]))  # no dt

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'][0],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

    def __call__(self, dt):

        # current variables
        en = self.feec_vars[0]
        jn = self.feec_vars[1]

        # allocate temporary FemFields _e, _j during solution
        _j, info = self._schur_solver(jn, self._B.dot(en), dt)
        _e = en - dt * self._C.dot(_j + jn)

        # write new coeffs into Propagator.variables
        max_de, max_dj = self.feec_vars_update(_e, _j)

        if self._info:
            print('Status     for OhmCold:', info['success'])
            print('Iterations for OhmCold:', info['niter'])
            print('Maxdiff e1 for OhmCold:', max_de)
            print('Maxdiff j1 for OhmCold:', max_dj)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PConjugateGradient', 'MassMatrixPreconditioner'),
                                  ('ConjugateGradient', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class JxBCold(Propagator):
    r'''Crank-Nicolson step

    .. math::

        \mathbb M_{1,1/n} \left( \mathbf j^{n+1} - \mathbf j^n \right) = \frac{\Delta t}{2} \frac{1}{\varepsilon_c} \mathbb M_{1,B_0} \left( \mathbf j^{n+1} - \mathbf j^n \right).

    Parameters
    ----------
        j : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        params : dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, j, **params):

        super().__init__(j)

        # parameters
        params_default = {'type': ('PConjugateGradient', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'alpha': 1.0,
                          'epsilon': 1.0}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._verbose = params['verbose']
        self._epsc = params['epsilon']

        # mass matrix in system (M - dt/2 * A)*j^(n + 1) = (M + dt/2 * A)*j^n
        self._M = self.mass_ops.M1ninv
        self._A = Multiply(-1/self._epsc, self.mass_ops.M1Bninv)  # no dt

        # Preconditioner
        if params['type'][1] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            self._pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate linear solver
        self._solver = getattr(it_solvers, params['type'][0])(self._M.domain)

        # allocate dummy vectors to avoid temporary array allocations
        self._rhs_j = self._M.codomain.zeros()
        self._j_new = j.space.zeros()

    def __call__(self, dt):

        # current variables
        jn = self.feec_vars[0]

        # define system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        lhs = Sum(self._M, Multiply(-dt/2.0, self._A))
        rhs = Sum(self._M, Multiply(dt/2.0, self._A))

        # solve linear system for updated u coefficients (in-place)
        rhs.dot(jn, out=self._rhs_j)

        info = self._solver.solve(lhs, self._rhs_j, self._pc,
                                  x0=jn, tol=self._tol,
                                  maxiter=self._maxiter, verbose=self._verbose,
                                  out=self._j_new)[1]

        # write new coeffs into Propagator.variables
        max_dj = self.feec_vars_update(self._j_new)[0]

        if self._info:
            print('Status     for FluidCold:', info['success'])
            print('Iterations for FluidCold:', info['niter'])
            print('Maxdiff j1 for FluidCold:', max_dj)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PConjugateGradient', 'MassMatrixPreconditioner'),
                                  ('ConjugateGradient', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class ShearAlfvén(Propagator):
    r'''Crank-Nicolson step for shear Alfvén part in MHD equations,

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_\alpha)^{-1} \mathcal {T^\alpha}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^\alpha} (\mathbb M^\rho_\alpha)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^\rho_\alpha}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    where :math:`\alpha \in \{1, 2, v\}` and :math:`\mathbb M^\rho_\alpha` is a weighted mass matrix in :math:`\alpha`-space, the weight being :math:`\rho_0`,
    the MHD equilibirum density. The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity.

    b : psydac.linalg.block.BlockVector
        FE coefficients of magnetic field as 2-form.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, u, b, **params):

        super().__init__(u, b)

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'type': ('PConjugateGradient', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_M = 'M' + self.derham.spaces_dict[params['u_space']] + 'n'
        id_T = 'T' + self.derham.spaces_dict[params['u_space']]

        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)

        self._B = Multiply(-1/2, Compose(_T.T,
                           self.derham.curl.T, self.mass_ops.M2))
        self._C = Multiply(1/2, Compose(self.derham.curl, _T))

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'][0],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    def __call__(self, dt):

        # current variables
        un = self.feec_vars[0]
        bn = self.feec_vars[1]

        # solve for new u coeffs
        self._B.dot(bn, out=self._byn)

        info = self._schur_solver(un, self._byn, dt, out=self._u_tmp1)[1]

        # new b coeffs
        un.copy(out=self._u_tmp2)
        self._u_tmp2 += self._u_tmp1
        self._C.dot(self._u_tmp2, out=self._b_tmp1)
        self._b_tmp1 *= -dt
        self._b_tmp1 += bn

        # write new coeffs into self.feec_vars
        max_du, max_db = self.feec_vars_update(self._u_tmp1, self._b_tmp1)

        if self._info and self._rank == 0:
            print('Status     for ShearAlfvén:', info['success'])
            print('Iterations for ShearAlfvén:', info['niter'])
            print('Maxdiff up for ShearAlfvén:', max_du)
            print('Maxdiff b2 for ShearAlfvén:', max_db)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('PConjugateGradient', 'MassMatrixPreconditioner'),
                                  ('ConjugateGradient', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class ShearAlfvénB1(Propagator):
    r'''Crank-Nicolson step for shear Alfvén part in Extended MHD equations,

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_2)^{-1} \mathcal {T^2}^\top \mathbb C \mathbb M_1^{-1}\\ - \mathbb M_1^{-1} \mathbb C^\top \mathcal {T^2} (\mathbb M^\rho_2)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^\rho_2}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_1(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    where :math:`\mathbb M^\rho_2` is a weighted mass matrix in 2-space, the weight being :math:`\rho_0`,
    the MHD equilibirum density. The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity as 2-form.

    b : psydac.linalg.block.BlockVector
        FE coefficients of magnetic field as 1-form.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, u, b, **params):

        super().__init__(u, b)

        # parameters
        params_default = {'type': ('PConjugateGradient', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M2n
        self._M1inv = Inverse(self.mass_ops.M1, tol=1e-8)
        self._B = Multiply(1/2, Compose(self.mass_ops.M2B,
                           self.derham.curl))
        # I still have to invert M1
        self._C = Multiply(
            1/2, Compose(self._M1inv, self.derham.curl.T, self.mass_ops.M2B))

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, 'M2n'))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'][0],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    def __call__(self, dt):

        # current variables
        un = self.feec_vars[0]
        bn = self.feec_vars[1]

        # solve for new u coeffs
        self._B.dot(bn, out=self._byn)

        info = self._schur_solver(un, self._byn, dt, out=self._u_tmp1)[1]

        # new b coeffs
        un.copy(out=self._u_tmp2)
        self._u_tmp2 += self._u_tmp1
        self._C.dot(self._u_tmp2, out=self._b_tmp1)
        self._b_tmp1 *= -dt
        self._b_tmp1 += bn

        # write new coeffs into self.feec_vars
        max_du, max_db = self.feec_vars_update(self._u_tmp1, self._b_tmp1)

        if self._info and self._rank == 0:
            print('Status     for ShearAlfvénB1:', info['success'])
            print('Iterations for ShearAlfvénB1:', info['niter'])
            print('Maxdiff up for ShearAlfvénB1:', max_du)
            print('Maxdiff b2 for ShearAlfvénB1:', max_db)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PConjugateGradient', 'MassMatrixPreconditioner'),
                                  ('ConjugateGradient', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class Hall(Propagator):
    r'''Crank-Nicolson step for Hall part in Extended MHD equations,

    .. math::

        \mathbf b^{n+1} - \mathbf b^n 
        = \frac{\Delta t}{2} \mathbb M_1^{-1} \mathbb C^\top  \mathbb M^{\mathcal{T},\rho}_2  \mathbb C  (\mathbf b^{n+1} + \mathbf b^n)  ,

    where :math:`\mathbb M^{\mathcal{T},\rho}_2` is a weighted mass matrix in 2-space, the weight being :math:`\frac{\mathcal{T}}{\rho_0}`,
    the MHD equilibirum density :math:`\rho_0` as a 0-form, and rotation matrix :math:`\mathcal{T} \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,`. 
    The solution of the above system is based on the Pre-conditioned Biconjugate Gradient Stabilized algortihm (PBiConjugateGradientStab).

    Parameters
    ---------- 
    b : psydac.linalg.block.BlockVector
        FE coefficients of magnetic field as 1-form.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, b, **params):

        super().__init__(b)

        # parameters
        params_default = {'type': ('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'kappa': 1.0}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._verbose = params['verbose']
        self._kappa = params['kappa']

        # mass matrix in system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        id_M = 'M1'
        id_M2Bn = 'M2Bn'
        self._M = getattr(self.mass_ops, id_M)
        self._M2Bn = getattr(self.mass_ops, id_M2Bn)
        self._A = Multiply(self._kappa, Compose(
            self.derham.curl.T, self._M2Bn, self.derham.curl))

        # Preconditioner
        if params['type'][1] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            self._pc = pc_class(getattr(self.mass_ops, id_M))

        # Instantiate linear solver
        self._solver = getattr(it_solvers, params['type'][0])(self._M.domain)

        # allocate dummy vectors to avoid temporary array allocations
        self._rhs_b = self._M.codomain.zeros()
        self._b_new = b.space.zeros()

    def __call__(self, dt):

        # current variables
        bn = self.feec_vars[0]

        # define system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        lhs = Sum(self._M, Multiply(-dt/2.0, self._A))
        rhs = Sum(self._M, Multiply(dt/2.0, self._A))

        # solve linear system for updated u coefficients (in-place)
        rhs.dot(bn, out=self._rhs_b)

        info = self._solver.solve(lhs, self._rhs_b, self._pc,
                                  x0=bn, tol=self._tol,
                                  maxiter=self._maxiter, verbose=self._verbose,
                                  out=self._b_new)[1]

        # write new coeffs into self.feec_vars
        max_db = self.feec_vars_update(self._b_new)

        if self._info and self._rank == 0:
            print('Status     for Hall:', info['success'])
            print('Iterations for Hall:', info['niter'])
            print('Maxdiff b1 for Hall:', max_db)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                                  ('BiConjugateGradientStab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class Magnetosonic(Propagator):
    r'''Crank-Nicolson step for magnetosonic part in MHD equations:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1} - \mathbf p^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_\alpha)^{-1} {\mathcal U^\alpha}^\top \mathbb D^\top \mathbb M_3 \\ - \mathbb D \mathcal S^\alpha - (\gamma - 1) \mathcal K^\alpha \mathbb D \mathcal U^\alpha & 0 \end{bmatrix} 
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1} + \mathbf p^n) \end{bmatrix} + \begin{bmatrix} \Delta t (\mathbb M^\rho_\alpha)^{-1} \mathbb M^J_\alpha \mathbf b^n \\ 0 \end{bmatrix},

    where :math:`\alpha \in \{1, 2, v\}` and :math:`\mathcal U^2 = \mathbb Id`; moreover, :math:`\mathbb M^\rho_\alpha` and 
    :math:`\mathbb M^J_\alpha` are weighted mass matrices in :math:`\alpha`-space, 
    the weights being the MHD equilibirum density :math:`\rho_0`
    and the curl of the MHD equilibrium current density :math:`\mathbf J_0 = \nabla \times \mathbf B_0`. 
    The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Decoupled density update:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathbb D \mathcal Q^\alpha (\mathbf u^{n+1} + \mathbf u^n) \,.

    Parameters
    ---------- 
    n : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity.

    p : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, n, u, p, **params):

        super().__init__(n, u, p)

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'b': self.derham.Vh['2'].zeros(),
                          'type': ('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        self._info = params['info']
        self._bc = self.derham.bc
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = 'M' + self.derham.spaces_dict[params['u_space']] + 'n'
        id_MJ = 'M' + self.derham.spaces_dict[params['u_space']] + 'J'

        if params['u_space'] == 'Hcurl':
            id_S, id_U, id_K, id_Q = 'S1', 'U1', 'K3', 'Q1'
        elif params['u_space'] == 'Hdiv':
            id_S, id_U, id_K, id_Q = 'S2', None, 'K3', 'Q2'
        elif params['u_space'] == 'H1vec':
            id_S, id_U, id_K, id_Q = 'Sv', 'Uv', 'K3', 'Qv'

        _A = getattr(self.mass_ops, id_Mn)
        _S = getattr(self.basis_ops, id_S)
        _K = getattr(self.basis_ops, id_K)

        if id_U is None:
            _U, _UT = None, None
        else:
            _U = getattr(self.basis_ops, id_U)
            _UT = _U.T

        self._B = Multiply(-1/2., Compose(_UT,
                           self.derham.div.T, self.mass_ops.M3))
        self._C = Multiply(1/2., Sum(Compose(self.derham.div, _S),
                           Multiply(2/3, Compose(_K, self.derham.div, _U))))

        self._MJ = getattr(self.mass_ops, id_MJ)
        self._DQ = Compose(self.derham.div, getattr(self.basis_ops, id_Q))

        self._b = params['b']

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'][0],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._p_tmp1 = p.space.zeros()
        self._n_tmp1 = n.space.zeros()
        self._b_tmp1 = self._b.space.zeros()

        self._byn1 = self._B.codomain.zeros()
        self._byn2 = self._B.codomain.zeros()

    def __call__(self, dt):

        # current variables
        nn = self.feec_vars[0]
        un = self.feec_vars[1]
        pn = self.feec_vars[2]

        # solve for new u coeffs
        self._B.dot(pn, out=self._byn1)
        self._MJ.dot(self._b, out=self._byn2)
        self._byn2 *= 1/2
        self._byn1 -= self._byn2

        info = self._schur_solver(un, self._byn1, dt, out=self._u_tmp1)[1]

        # new p, n, b coeffs
        un.copy(out=self._u_tmp2)
        self._u_tmp2 += self._u_tmp1
        self._C.dot(self._u_tmp2, out=self._p_tmp1)
        self._p_tmp1 *= -dt
        self._p_tmp1 += pn

        self._DQ.dot(self._u_tmp2, out=self._n_tmp1)
        self._n_tmp1 *= -dt/2
        self._n_tmp1 += nn

        # write new coeffs into self.feec_vars
        max_dn, max_du, max_dp = self.feec_vars_update(self._n_tmp1,
                                                       self._u_tmp1,
                                                       self._p_tmp1)

        if self._info and self._rank == 0:
            print('Status     for Magnetosonic:', info['success'])
            print('Iterations for Magnetosonic:', info['niter'])
            print('Maxdiff n3 for Magnetosonic:', max_dn)
            print('Maxdiff up for Magnetosonic:', max_du)
            print('Maxdiff p3 for Magnetosonic:', max_dp)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                                  ('BiConjugateGradientStab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class SonicIon(Propagator):
    r'''Crank-Nicolson step for Ion sonic part in Extended MHD equations:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1}_i - \mathbf p^n_i \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_2)^{-1} \mathbb D^\top \mathbb M_3 \\ - \gamma \mathcal K^3 \mathbb D & 0 \end{bmatrix} 
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1}_i + \mathbf p^n_i) \end{bmatrix} ,

    where :math:`\mathbb M^\rho_2`  is a weighted mass matrix in 2-space, 
    the weight being the MHD equilibirum density :math:`\rho_0`. Furthermore, :math:`\mathcal K^3` is the basis projection operator given by :

    .. math::

        \mathcal{K}^3_{ijk,mno} := \hat{\Pi}^3_{ijk} \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\Lambda^3_{mno} \right] \,.
    The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Decoupled density update:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathcal Q \mathbb D  (\mathbf u^{n+1} + \mathbf u^n) \,.

    Parameters
    ---------- 
    n : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity 2-form.

    p : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, n, u, p, **params):

        super().__init__(n, u, p)

        # parameters
        params_default = {'type': ('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._bc = self.derham.bc
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = 'M2n'
        id_K, id_Q = 'K3', 'Q3'

        _A = getattr(self.mass_ops, id_Mn)
        _K = getattr(self.basis_ops, id_K)

        self._B = Multiply(-1/2., Compose(self.derham.div.T, self.mass_ops.M3))
        self._C = Multiply(5/6., Compose(_K, self.derham.div))

        self._QD = Compose(getattr(self.basis_ops, id_Q), self.derham.div)

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'][0],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._p_tmp1 = p.space.zeros()
        self._n_tmp1 = n.space.zeros()

        self._byn1 = self._B.codomain.zeros()

    def __call__(self, dt):

        # current variables
        nn = self.feec_vars[0]
        un = self.feec_vars[1]
        pn = self.feec_vars[2]

        # solve for new u coeffs
        self._B.dot(pn, out=self._byn1)

        info = self._schur_solver(un, self._byn1, dt, out=self._u_tmp1)[1]

        # new p, n, b coeffs
        un.copy(out=self._u_tmp2)
        self._u_tmp2 += self._u_tmp1
        self._C.dot(self._u_tmp2, out=self._p_tmp1)
        self._p_tmp1 *= -dt
        self._p_tmp1 += pn

        self._QD.dot(self._u_tmp2, out=self._n_tmp1)
        self._n_tmp1 *= -dt/2.0
        self._n_tmp1 += nn

        # write new coeffs into self.feec_vars
        max_dn, max_du, max_dp = self.feec_vars_update(self._n_tmp1,
                                                       self._u_tmp1,
                                                       self._p_tmp1)

        if self._info and self._rank == 0:
            print('Status     for Magnetosonic:', info['success'])
            print('Iterations for Magnetosonic:', info['niter'])
            print('Maxdiff n3 for Magnetosonic:', max_dn)
            print('Maxdiff up for Magnetosonic:', max_du)
            print('Maxdiff p3 for Magnetosonic:', max_dp)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                                  ('BiConjugateGradientStab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class SonicElectron(Propagator):
    r'''Crank-Nicolson step for Electron sonic part in Extended MHD equations:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1}_i - \mathbf p^n_i \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_2)^{-1} \mathbb D^\top \mathbb M_3 \\ - \gamma \mathcal K^3 \mathbb D & 0 \end{bmatrix} 
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1}_i + \mathbf p^n_i) \end{bmatrix} ,

    where :math:`\mathbb M^\rho_2`  is a weighted mass matrix in 2-space, 
    the weight being the MHD equilibirum density :math:`\rho_0`. Furthermore, :math:`\mathcal K^3` is the basis projection operator given by :

    .. math::

        \mathcal{K}^3_{ijk,mno} := \hat{\Pi}^3_{ijk} \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\Lambda^3_{mno} \right] \,.
    The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Decoupled density update:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathcal Q \mathbb D  (\mathbf u^{n+1} + \mathbf u^n) \,.

    Parameters
    ---------- 
    n : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity 2-form.

    p : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, n, u, p, **params):

        super().__init__(n, u, p)

        # parameters
        params_default = {'type': ('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._bc = self.derham.bc
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = 'M2n'
        # ONCE THE CODE SUPPORTS HAVING A DIFFERENT EQUILIBRIUM ELECTRON PRESSURE TO THE EQUILIBRIUM ION PRESSURE; WE MUST ADD A NEW PROJECTION MATRIX Ke3 THAT TAKES THE EQUILIBRIUM ELECTRON PRESSURE
        # INSTEAD OF THE EQUILIBRIUM ION PRESSURE: AND USE HERE Ke3, NOT K3.
        id_K, id_Q = 'K3', 'Q3'

        _A = getattr(self.mass_ops, id_Mn)
        _K = getattr(self.basis_ops, id_K)

        self._B = Multiply(-1/2., Compose(self.derham.div.T, self.mass_ops.M3))
        self._C = Multiply(5/6., Compose(_K, self.derham.div))

        self._QD = Compose(getattr(self.basis_ops, id_Q), self.derham.div)

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'][0],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._p_tmp1 = p.space.zeros()
        self._n_tmp1 = n.space.zeros()

        self._byn1 = self._B.codomain.zeros()

    def __call__(self, dt):

        # current variables
        nn = self.feec_vars[0]
        un = self.feec_vars[1]
        pn = self.feec_vars[2]

        # solve for new u coeffs
        self._B.dot(pn, out=self._byn1)

        info = self._schur_solver(un, self._byn1, dt, out=self._u_tmp1)[1]

        # new p, n, b coeffs
        un.copy(out=self._u_tmp2)
        self._u_tmp2 += self._u_tmp1
        self._C.dot(self._u_tmp2, out=self._p_tmp1)
        self._p_tmp1 *= -dt
        self._p_tmp1 += pn

        self._QD.dot(self._u_tmp2, out=self._n_tmp1)
        self._n_tmp1 *= -dt/2.0
        self._n_tmp1 += nn

        # write new coeffs into self.feec_vars
        max_dn, max_du, max_dp = self.feec_vars_update(self._n_tmp1,
                                                       self._u_tmp1,
                                                       self._p_tmp1)

        if self._info and self._rank == 0:
            print('Status     for Magnetosonic:', info['success'])
            print('Iterations for Magnetosonic:', info['niter'])
            print('Maxdiff n3 for Magnetosonic:', max_dn)
            print('Maxdiff up for Magnetosonic:', max_du)
            print('Maxdiff p3 for Magnetosonic:', max_dp)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                                  ('BiConjugateGradientStab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class FaradayExtended(Propagator):
    r'''Equations: Faraday's law

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
        a : psydac.linalg.block.BlockVector
            FE coefficients of vector potential.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, a, **params):

        assert isinstance(a, (BlockVector, PolarVector))

        # parameters
        params_default = {'a_space': None,
                          'beq': None,
                          'particles': None,
                          'quad_number': None,
                          'shape_degree': None,
                          'shape_size': None,
                          'solver_params': None,
                          'accumulate_density': None
                          }

        params = set_defaults(params, params_default)

        self._a = a
        self._a_old = self._a.copy()

        self._a_space = params['a_space']
        assert self._a_space in {'Hcurl'}

        self._rank = self.derham.comm.Get_rank()
        self._beq = params['beq']

        self._particles = params['particles']

        self._nqs = params['quad_number']

        self.size1 = int(self.derham.domain_array[self._rank, int(2)])
        self.size2 = int(self.derham.domain_array[self._rank, int(5)])
        self.size3 = int(self.derham.domain_array[self._rank, int(8)])

        self.weight_1 = zeros(
            (self.size1*self._nqs[0], self.size2*self._nqs[1], self.size3*self._nqs[2]), dtype=float)
        self.weight_2 = zeros(
            (self.size1*self._nqs[0], self.size2*self._nqs[1], self.size3*self._nqs[2]), dtype=float)
        self.weight_3 = zeros(
            (self.size1*self._nqs[0], self.size2*self._nqs[1], self.size3*self._nqs[2]), dtype=float)

        self._weight_pre = [self.weight_1, self.weight_2, self.weight_3]

        self._ind = [[self.derham.indN[0], self.derham.indD[1], self.derham.indD[2]],
                     [self.derham.indD[0], self.derham.indN[1], self.derham.indD[2]],
                     [self.derham.indD[0], self.derham.indD[1], self.derham.indN[2]]]

        # Initialize Accumulator object for getting density from particles
        self._pts_x = 1.0 / (2.0*self.derham.Nel[0]) * np.polynomial.legendre.leggauss(
            self._nqs[0])[0] + 1.0 / (2.0*self.derham.Nel[0])
        self._pts_y = 1.0 / (2.0*self.derham.Nel[1]) * np.polynomial.legendre.leggauss(
            self._nqs[1])[0] + 1.0 / (2.0*self.derham.Nel[1])
        self._pts_z = 1.0 / (2.0*self.derham.Nel[2]) * np.polynomial.legendre.leggauss(
            self._nqs[2])[0] + 1.0 / (2.0*self.derham.Nel[2])

        self._p_shape = params['shape_degree']
        self._p_size = params['shape_size']
        self._accum_density = params['accumulate_density']

        # Initialize Accumulator object for getting the matrix and vector related with vector potential
        self._accum_potential = Accumulator(
            self.derham, self.domain, self._a_space, 'hybrid_fA_Arelated', add_vector=True, symmetry='symm')

        self._solver_params = params['solver_params']
        # preconditioner
        if self._solver_params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, self._solver_params['pc'])
            self._pc = pc_class(self.mass_ops.M1)

        self._Minv = Inverse(self.mass_ops.M1, tol=1e-8)
        self._CMC = Compose(self.derham.curl.transpose(),
                            Compose(self.mass_ops.M2, self.derham.curl))
        self._M1 = self.mass_ops.M1
        self._M2 = self.mass_ops.M2

    @property
    def variables(self):
        return [self._a]

    def __call__(self, dt):

        # the loop of fixed point iteration, 100 iterations at most.

        self._accum_density.accumulate(self._particles, np.array(self.derham.Nel), np.array(self._nqs), np.array(
            self._pts_x), np.array(self._pts_y), np.array(self._pts_z), np.array(self._p_shape), np.array(self._p_size))
        self._accum_potential.accumulate(self._particles)

        self._L2 = Multiply(-dt/2, Compose(self._Minv,
                            Sum(self._accum_potential._operators[0].matrix, self._CMC)))
        self._RHS = -(self._L2.dot(self._a)) - dt*(self._Minv.dot(
            self._accum_potential._vectors[0] - Compose(self.derham.curl.transpose(), self._M2).dot(self._beq)))
        self._rhs = self._M1.dot(self._a)

        for _ in range(10):
            # print('+++++=====++++++', self._accum_density._operators[0].matrix._data)
            # set mid-value used in the fixed iteration
            curla_mid = self.derham.curl.dot(
                0.5*(self._a_old + self._a)) + self._beq
            curla_mid.update_ghost_regions()
            # initialize the curl A
            # remember to check ghost region of curla_mid
            util.create_weight_weightedmatrix_hybrid(
                curla_mid, self._weight_pre, self.derham, self._accum_density, self.domain)
            # self._weight = [[None, self._weight_pre[2], -self._weight_pre[1]], [None, None, self._weight_pre[0]], [None, None, None]]
            self._weight = [[0.0*self._weight_pre[0], 0.0*self._weight_pre[2], 0.0*self._weight_pre[1]], [0.0*self._weight_pre[2], 0.0 *
                                                                                                          self._weight_pre[1], 0.0*self._weight_pre[0]], [0.0*self._weight_pre[1], 0.0*self._weight_pre[0], 0.0*self._weight_pre[2]]]
            # self._weight = [[self._weight_pre[0], self._weight_pre[2], self._weight_pre[1]], [self._weight_pre[2], self._weight_pre[1], self._weight_pre[0]], [self._weight_pre[1], self._weight_pre[0], self._weight_pre[2]]]
            HybridM1 = self.mass_ops.assemble_weighted_mass(
                self._weight, 'Hcurl', 'Hcurl')

            # next prepare for solving linear system
            _LHS = Sum(self._M1, Compose(HybridM1, self._L2))
            _RHS2 = HybridM1.dot(self._RHS) + self._rhs

            a_new, info = pcg(_LHS, _RHS2, self._pc, x0=self._a, tol=self._solver_params['tol'],
                              maxiter=self._solver_params['maxiter'], verbose=self._solver_params['verbose'])

            # write new coeffs into Propagator.variables
            max_da = self.feec_vars_update(a_new)
            print('++++====check_iteration_error=====+++++', max_da)
            # we can modify the diff function in in_place_update to get another type errors
            if max_da[0] < 10**(-6):
                break
            
    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PConjugateGradient', 'MassMatrixPreconditioner'),
                                  ('ConjugateGradient', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class CurrentCoupling6DDensity(Propagator):
    """
    Parameters
    ----------
    u : psydac.linalg.block.BlockVector
            FE coefficients of MHD velocity.

    **params : dict
            Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, u, **params):

        from struphy.pic.particles import Particles6D

        super().__init__(u)

        # parameters
        params_default = {'particles': None,
                          'u_space': 'Hdiv',
                          'b_eq': None,
                          'b_tilde': None,
                          'f0': Maxwellian6DUniform(),
                          'type': ('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1,
                          'kappa': 1.}

        params = set_defaults(params, params_default)

        # assert parameters and expose some quantities to self
        assert isinstance(params['particles'], (Particles6D))

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}
        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.spaces_dict[params['u_space']])

        assert isinstance(params['b_eq'], (BlockVector, PolarVector))

        if params['b_tilde'] is not None:
            assert isinstance(params['b_tilde'], (BlockVector, PolarVector))

        self._particles = params['particles']
        self._b_eq = params['b_eq']
        self._b_tilde = params['b_tilde']
        self._f0 = params['f0']

        if self._f0 is not None:

            assert isinstance(self._f0, Maxwellian)

            # evaluate and save nh0*|det(DF)| (H1vec) or nh0/|det(DF)| (Hdiv) at quadrature points for control variate
            quad_pts = [quad_grid[nquad].points.flatten()
                        for quad_grid, nquad in zip(self.derham.Vh_fem['0']._quad_grids, self.derham.Vh_fem['0'].nquads)]

            if params['u_space'] == 'H1vec':
                self._nh0_at_quad = self.domain.pull(
                    [self._f0.n], *quad_pts, kind='3_form', squeeze_out=False, coordinates='logical')
            else:
                self._nh0_at_quad = self.domain.push(
                    [self._f0.n], *quad_pts, kind='3_form', squeeze_out=False)

            # memory allocation of magnetic field at quadrature points
            self._b_quad1 = np.zeros_like(self._nh0_at_quad)
            self._b_quad2 = np.zeros_like(self._nh0_at_quad)
            self._b_quad3 = np.zeros_like(self._nh0_at_quad)

            # memory allocation for self._b_quad x self._nh0_at_quad * self._coupling_const
            self._mat12 = np.zeros_like(self._nh0_at_quad)
            self._mat13 = np.zeros_like(self._nh0_at_quad)
            self._mat23 = np.zeros_like(self._nh0_at_quad)

            self._mat21 = np.zeros_like(self._nh0_at_quad)
            self._mat31 = np.zeros_like(self._nh0_at_quad)
            self._mat32 = np.zeros_like(self._nh0_at_quad)

        self._type = params['type'][0]
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._info = params['info']
        self._verbose = params['verbose']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_const = params['Ah'] * params['kappa'] / params['Ab']
        # load accumulator
        self._accumulator = Accumulator(
            self.derham, self.domain, params['u_space'], 'cc_lin_mhd_6d_1', add_vector=False, symmetry='asym')

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.E['2'].transpose()

        # mass matrix in system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        u_id = self.derham.spaces_dict[params['u_space']]
        self._M = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['type'][1] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            self._pc = pc_class(self._M)

        # linear solver
        self._solver = getattr(it_solvers, params['type'][0])(self._M.domain)

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()

        self._rhs_v = u.space.zeros()
        self._u_new = u.space.zeros()

    def __call__(self, dt):
        """
        TODO
        """

        # pointer to old coefficients
        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        self._b_eq.copy(out=self._b_full1)

        if self._b_tilde is not None:
            self._b_full1 += self._b_tilde

        # extract coefficients to tensor product space (in-place)
        self._E2T.dot(self._b_full1, out=self._b_full2)

        # update ghost regions because of non-local access in accumulation kernel!
        self._b_full2.update_ghost_regions()

        # perform accumulation (either with or without control variate)
        if self._f0 is not None:

            # evaluate magnetic field at quadrature points (in-place)
            WeightedMassOperator.eval_quad(self.derham.Vh_fem['2'], self._b_full2,
                                           out=[self._b_quad1, self._b_quad2, self._b_quad3])

            self._mat12[:, :, :] = self._coupling_const * \
                self._b_quad3 * self._nh0_at_quad
            self._mat13[:, :, :] = -self._coupling_const * \
                self._b_quad2 * self._nh0_at_quad
            self._mat23[:, :, :] = self._coupling_const * \
                self._b_quad1 * self._nh0_at_quad

            self._mat21[:, :, :] = -self._mat12
            self._mat31[:, :, :] = -self._mat13
            self._mat32[:, :, :] = -self._mat23

            self._accumulator.accumulate(self._particles,
                                         self._b_full2[0]._data, self._b_full2[1]._data, self._b_full2[2]._data,
                                         self._space_key_int, self._coupling_const,
                                         control_mat=[[None, self._mat12, self._mat13],
                                                      [self._mat21, None,
                                                          self._mat23],
                                                      [self._mat31, self._mat32, None]])
        else:
            self._accumulator.accumulate(self._particles,
                                         self._b_full2[0]._data, self._b_full2[1]._data, self._b_full2[2]._data,
                                         self._space_key_int, self._coupling_const)

        # define system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        lhs = Sum(self._M, Multiply(-dt/2, self._accumulator.operators[0]))
        rhs = Sum(self._M, Multiply(dt/2, self._accumulator.operators[0]))

        # solve linear system for updated u coefficients (in-place)
        rhs.dot(un, out=self._rhs_v)

        info = self._solver.solve(lhs, self._rhs_v, self._pc,
                                  x0=un, tol=self._tol,
                                  maxiter=self._maxiter, verbose=self._verbose,
                                  out=self._u_new)[1]

        # write new coeffs into Propagator.variables
        max_du = self.feec_vars_update(self._u_new)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling6DDensity:', info['success'])
            print('Iterations for CurrentCoupling6DDensity:', info['niter'])
            print('Maxdiff up for CurrentCoupling6DDensity:', max_du)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                                  ('BiConjugateGradientStab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class ShearAlfvénCurrentCoupling5D(Propagator):
    r'''Crank-Nicolson step for shear Alfvén part in LinearMHDDriftkineticCC,

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_\alpha)^{-1} \mathcal {T^\alpha}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^\alpha} (\mathbb M^\rho_\alpha)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^\rho_\alpha}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) + \mathcal{P}_B^{\top} \sum_k^{N_p} \omega_k \mu_k \Lambda^0(\mathbf{\eta}_k) \ \end{bmatrix} ,

    where :math:`\mathcal{P}_B = \hat \Pi^0 \left[ \frac{\hat b^1}{\sqrt g} \Lambda^2\right]`, :math:`\alpha \in \{1, 2, v\}` and :math:`\mathbb M^\rho_\alpha` is a weighted mass matrix in :math:`\alpha`-space, the weight being :math:`\rho_0`,
    the MHD equilibirum density. The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity.

    b : psydac.linalg.block.BlockVector
        FE coefficients of magnetic field as 2-form.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, u, b, **params):

        from struphy.pic.particles import Particles5D

        super().__init__(u, b)

        # parameters
        params_default = {'particles': None,
                          'u_space': 'Hdiv',
                          'b_eq': None,
                          'f0': Maxwellian5DUniform(),
                          'type': ('PConjugateGradient', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1,
                          'kappa': 1.}

        params = set_defaults(params, params_default)

        assert isinstance(params['particles'], Particles5D)
        self._particles = params['particles']

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        self._f0 = params['f0']
        assert isinstance(params['b_eq'], (BlockVector, PolarVector))
        self._b_eq = params['b_eq']

        self._type = params['type'][0]
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._info = params['info']
        self._verbose = params['verbose']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_const = params['Ah'] / params['Ab']

        self._PB = getattr(self.basis_ops, 'PB')
        self._ACC = Accumulator(self.derham, self.domain,
                                'H1', 'cc_lin_mhd_5d_mu', add_vector=True)

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_M = 'M' + self.derham.spaces_dict[params['u_space']] + 'n'
        id_T = 'T' + self.derham.spaces_dict[params['u_space']]

        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)

        self._B = Multiply(-1/2, Compose(_T.T,
                           self.derham.curl.T, self.mass_ops.M2))
        self._C = Multiply(1/2, Compose(self.derham.curl, _T))
        self._B2 = Multiply(-1/2., Compose(_T.T,
                            self.derham.curl.T, self._PB.T))

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'][0],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    def __call__(self, dt):

        # current variables
        un = self.feec_vars[0]
        bn = self.feec_vars[1]

        # accumulate scalar
        self._ACC.accumulate(self._particles, self._coupling_const)

        # solve for new u coeffs
        self._B.dot(bn, out=self._byn)
        self._byn += self._B2.dot(self._ACC.vectors[0])

        info = self._schur_solver(un, self._byn, dt, out=self._u_tmp1)[1]

        # new b coeffs
        un.copy(out=self._u_tmp2)
        self._u_tmp2 += self._u_tmp1
        self._C.dot(self._u_tmp2, out=self._b_tmp1)
        self._b_tmp1 *= -dt
        self._b_tmp1 += bn

        # write new coeffs into self.feec_vars
        max_du, max_db = self.feec_vars_update(self._u_tmp1, self._b_tmp1)

        if self._info and self._rank == 0:
            print('Status     for ShearAlfvén:', info['success'])
            print('Iterations for ShearAlfvén:', info['niter'])
            print('Maxdiff up for ShearAlfvén:', max_du)
            print('Maxdiff b2 for ShearAlfvén:', max_db)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('PConjugateGradient', 'MassMatrixPreconditioner'),
                                  ('ConjugateGradient', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class MagnetosonicCurrentCoupling5D(Propagator):
    r'''Crank-Nicolson step for Magnetosonic part in LinearMHDDriftkineticCC,

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1} - \mathbf p^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_\alpha)^{-1} {\mathcal U^\alpha}^\top \mathbb D^\top \mathbb M_3 \\ - \mathbb D \mathcal S^\alpha - (\gamma - 1) \mathcal K^\alpha \mathbb D \mathcal U^\alpha & 0 \end{bmatrix} 
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1} + \mathbf p^n) \end{bmatrix} + \begin{bmatrix} \Delta t (\mathbb M^\rho_\alpha)^{-1} (\mathbb M^J_\alpha \mathbf b^n + + \sum_k^{N_p} \omega_k \mu_k \left[ \left\{ \hat \nabla \times \hat{\mathbb b}^1\right\} \times \hat{\mathbb B}^2\right](\mathbb \eta_k)) \\ 0 \end{bmatrix},

    where :math:`\alpha \in \{1, 2, v\}` and :math:`\mathcal U^2 = \mathbb Id`; moreover, :math:`\mathbb M^\rho_\alpha` and 
    :math:`\mathbb M^J_\alpha` are weighted mass matrices in :math:`\alpha`-space, 
    the weights being the MHD equilibirum density :math:`\rho_0`
    and the curl of the MHD equilibrium current density :math:`\mathbf J_0 = \nabla \times \mathbf B_0`. 
    The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Decoupled density update:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathbb D \mathcal Q^\alpha (\mathbf u^{n+1} + \mathbf u^n) \,.

    Parameters
    ---------- 
    n : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity.

    p : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, n, u, p, **params):

        from struphy.pic.particles import Particles5D

        super().__init__(n, u, p)

        # parameters
        params_default = {'b': self.derham.Vh['2'].zeros(),
                          'particles': None,
                          'u_space': 'Hdiv',
                          'unit_b1': None,
                          'f0': Maxwellian5DUniform(),
                          'type': ('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1,
                          'kappa': 1}

        params = set_defaults(params, params_default)

        assert isinstance(params['particles'], Particles5D)
        self._particles = params['particles']

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}
        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.spaces_dict[params['u_space']])

        self._f0 = params['f0']

        assert isinstance(params['b'], (BlockVector, PolarVector))
        self._b = params['b']

        assert isinstance(params['unit_b1'], (BlockVector, PolarVector))
        self._unit_b1 = params['unit_b1']

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        self._curl_norm_b = self.derham.curl.dot(self._unit_b1)
        self._curl_norm_b.update_ghost_regions()
        self._bc = self.derham.bc
        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_const = params['Ah'] / params['Ab']

        self._ACC = Accumulator(self.derham, self.domain,
                                params['u_space'], 'cc_lin_mhd_5d_curlMxB', add_vector=True)

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = 'M' + self.derham.spaces_dict[params['u_space']] + 'n'
        id_MJ = 'M' + self.derham.spaces_dict[params['u_space']] + 'J'

        if params['u_space'] == 'Hcurl':
            id_S, id_U, id_K, id_Q = 'S1', 'U1', 'K3', 'Q1'
        elif params['u_space'] == 'Hdiv':
            id_S, id_U, id_K, id_Q = 'S2', None, 'K3', 'Q2'
        elif params['u_space'] == 'H1vec':
            id_S, id_U, id_K, id_Q = 'Sv', 'Uv', 'K3', 'Qv'

        _A = getattr(self.mass_ops, id_Mn)
        _S = getattr(self.basis_ops, id_S)
        _K = getattr(self.basis_ops, id_K)

        if id_U is None:
            _U, _UT = None, None
        else:
            _U = getattr(self.basis_ops, id_U)
            _UT = _U.T

        self._B = Multiply(-1/2., Compose(_UT,
                           self.derham.div.T, self.mass_ops.M3))
        self._C = Multiply(1/2., Sum(Compose(self.derham.div, _S),
                           Multiply(2/3, Compose(_K, self.derham.div, _U))))

        self._MJ = getattr(self.mass_ops, id_MJ)
        self._DQ = Compose(self.derham.div, getattr(self.basis_ops, id_Q))

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'][0],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._p_tmp1 = p.space.zeros()
        self._n_tmp1 = n.space.zeros()
        self._b_tmp1 = self._b.space.zeros()

        self._byn1 = self._B.codomain.zeros()
        self._byn2 = self._B.codomain.zeros()

    def __call__(self, dt):

        # current variables
        nn = self.feec_vars[0]
        un = self.feec_vars[1]
        pn = self.feec_vars[2]

        # accumulate
        self._ACC.accumulate(self._particles,
                             self._b[0]._data, self._b[1]._data, self._b[2]._data,
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                             self._space_key_int, self._coupling_const)

        # solve for new u coeffs
        self._B.dot(pn, out=self._byn1)
        self._MJ.dot(self._b, out=self._byn2)
        self._byn2 -= self._ACC.vectors[0]
        self._byn2 *= 1/2
        self._byn1 -= self._byn2

        info = self._schur_solver(un, self._byn1, dt, out=self._u_tmp1)[1]

        # new p, n, b coeffs
        un.copy(out=self._u_tmp2)
        self._u_tmp2 += self._u_tmp1
        self._C.dot(self._u_tmp2, out=self._p_tmp1)
        self._p_tmp1 *= -dt
        self._p_tmp1 += pn

        self._DQ.dot(self._u_tmp2, out=self._n_tmp1)
        self._n_tmp1 *= -dt/2
        self._n_tmp1 += nn

        # write new coeffs into self.feec_vars
        max_dn, max_du, max_dp = self.feec_vars_update(self._n_tmp1,
                                                       self._u_tmp1,
                                                       self._p_tmp1)

        if self._info and self._rank == 0:
            print('Status     for Magnetosonic:', info['success'])
            print('Iterations for Magnetosonic:', info['niter'])
            print('Maxdiff n3 for Magnetosonic:', max_dn)
            print('Maxdiff up for Magnetosonic:', max_du)
            print('Maxdiff p3 for Magnetosonic:', max_dp)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('PBiConjugateGradientStab', 'MassMatrixPreconditioner'),
                                  ('BiConjugateGradientStab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class CurrentCoupling5DDensity(Propagator):
    """
    """

    def __init__(self, u, **params):

        from struphy.pic.particles import Particles5D

        super().__init__(u)

        # parameters
        params_default = {'particles': None,
                          'u_space': 'Hdiv',
                          'b_eq': None,
                          'b_tilde': None,
                          'f0': Maxwellian5DUniform(),
                          'type': 'PBiConjugateGradientStab',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1,
                          'kappa': 1.}

        params = set_defaults(params, params_default)

        # assert parameters and expose some quantities to self
        assert isinstance(params['particles'], (Particles5D))

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}
        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.spaces_dict[params['u_space']])

        assert isinstance(params['b_eq'], (BlockVector, PolarVector))

        if params['b_tilde'] is not None:
            assert isinstance(params['b_tilde'], (BlockVector, PolarVector))

        self._particles = params['particles']
        self._b_eq = params['b_eq']
        self._b_tilde = params['b_tilde']
        self._f0 = params['f0']

        self._type = params['type'][0]
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._info = params['info']
        self._verbose = params['verbose']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_const = params['Ah'] * params['kappa'] / params['Ab']
        # load accumulator
        self._accumulator = Accumulator(
            self.derham, self.domain, params['u_space'], 'cc_lin_mhd_5d_D', add_vector=False, symmetry='asym')

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.E['2'].transpose()

        # mass matrix in system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        u_id = self.derham.spaces_dict[params['u_space']]
        self._M = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['type'][1] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            self._pc = pc_class(self._M)

        # linear solver
        self._solver = getattr(it_solvers, params['type'][0])(self._M.domain)

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()

        self._rhs_v = u.space.zeros()
        self._u_new = u.space.zeros()

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):
        """
        TODO
        """

        # pointer to old coefficients
        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        self._b_eq.copy(out=self._b_full1)

        if self._b_tilde is not None:
            self._b_full1 += self._b_tilde

        # extract coefficients to tensor product space (in-place)
        self._E2T.dot(self._b_full1, out=self._b_full2)

        # update ghost regions because of non-local access in accumulation kernel!
        self._b_full2.update_ghost_regions()

        self._accumulator.accumulate(self._particles,
                                     self._b_full2[0]._data, self._b_full2[1]._data, self._b_full2[2]._data,
                                     self._space_key_int, self._coupling_const)

        # define system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        lhs = Sum(self._M, Multiply(-dt/2, self._accumulator.operators[0]))
        rhs = Sum(self._M, Multiply(dt/2, self._accumulator.operators[0]))

        # solve linear system for updated u coefficients (in-place)
        rhs.dot(un, out=self._rhs_v)

        info = self._solver.solve(lhs, self._rhs_v, self._pc,
                                  x0=un, tol=self._tol,
                                  maxiter=self._maxiter, verbose=self._verbose,
                                  out=self._u_new)[1]

        # write new coeffs into Propagator.variables
        max_du = self.feec_vars_update(self._u_new)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling5DDensity:', info['success'])
            print('Iterations for CurrentCoupling5DDensity:', info['niter'])
            print('Maxdiff up for CurrentCoupling5DDensity:', max_du)
            print()
            
    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PConjugateGradient', 'MassMatrixPreconditioner'),
                                  ('ConjugateGradient', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class ImplicitDiffusion(Propagator):
    r"""
    Weak, implicit discretization of the diffusion (or heat) equation (can be used as a Poisson solver too),

    .. math::

        \frac{\partial \phi}{\partial t} - \Delta \phi = 0\,,

    which is discretized as

    .. math::

        (\sigma \mathbb M_0 + \Delta t\,\mathbb G^\top \mathbb M_1 \mathbb G)\, \phi^{n+1} = \int_{(0,1)^3} \Lambda^0 \phi^n\, \textnormal d\eta\,,

    where :math:`\Lambda^0 \in H^1` are the FEEC basis functions and :math:`\sigma \in \mathbb R` is a parameter.
    The solution is :math:`\phi^{n+1}\,\in H^1` and the right-hand side is :math:`\phi^n\,\in H^1`.
    For the choice :math:`\sigma=0` and :math:`\Delta t = 1` this is a Poisson solver,
    where :math:`\phi^n` corresponds to the charge density.
    Boundary terms are assumed to vanish.

    Parameters
    ----------
    phi : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 0-form, the solution.

    sigma : float
        Stabilization parameter: :math:`\sigma=1` for the heat equation and :math:`\sigma=0` for the Poisson equation.

    phi_n : psydac.linalg.stencil.StencilVector
        FE coefficients of a 0-form (optional, can be set with a setter later).

    x0 : psydac.linalg.stencil.StencilVector
        Initial guess for the iterative solver (optional, can be set with a setter later).

    **params : dict
        Parameters for the iteravtive solver.
    """

    def __init__(self, phi, sigma=1., phi_n=None, x0=None, **params):

        super().__init__(phi)

        # parameters
        params_default = {'type': ('PConjugateGradient', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        # allocate memory for solution and rhs
        self._phi_n = StencilVector(self.derham.Vh['0'])

        # check the rhs
        if phi_n is not None:

            assert type(phi_n) == type(self._phi_n)
            self._phi_n[:] = phi_n[:]
            self._phi_n.update_ghost_regions()

            # check solvability condition
            if np.abs(sigma) < 1e-14:
                sigma = 1e-14
                self.check_rhs(phi_n)

        # initial guess and solver params
        self._x0 = x0
        self._params = params

        # Set lhs matrices
        self._A1 = sigma * self.mass_ops.M0
        self._A2 = Compose(self.derham.grad.T,
                           self.mass_ops.M1,
                           self.derham.grad)

        # preconditioner and solver for Ax=b
        if params['type'][1] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            self._pc = pc_class(self.mass_ops.M0)

        # solver for Ax=b with A=const.
        self.solver = getattr(it_solvers, params['type'][0])(self.derham.Vh['0'])

    def check_rhs(self, phi_n):
        '''Checks space of rhs and, for periodic boundary conditions and sigma=0,
        checks whether the integral over phi_n is zero.

        Parameters
        ----------
        phi_n : psydac.linalg.stencil.StencilVector
            FE coefficients of a 0-form.'''

        assert type(phi_n) == type(self._phi_n)

        if np.all(phi_n.space.periods):
            solvability = np.zeros(1)
            self.derham.comm.Allreduce(
                np.sum(phi_n.toarray()), solvability, op=MPI.SUM)
            assert np.abs(
                solvability[0]) <= 1e-11, f'Solvability condition not met: {solvability[0]}'

    @property
    def phi_n(self):
        """
        psydac.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector.
        """
        return self._phi_n

    @phi_n.setter
    def phi_n(self, value):
        """ In-place setter for StencilVector/PolarVector.
        """
        self.check_rhs(value)
        self._phi_n[:] = value[:]

    @property
    def x0(self):
        """
        psydac.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector. First guess of the iterative solver.
        """
        return self._x0

    @x0.setter
    def x0(self, value):
        """ In-place setter for StencilVector/PolarVector. First guess of the iterative solver.
        """
        assert type(value) == type(self._phi_n)
        assert value.space.symbolic_space == 'H1', f'Right-hand side must be in H1, but is in {value.space.symbolic_space}.'

        if self._x0 is None:
            self._x0 = value
        else:
            self._x0[:] = value[:]

    def __call__(self, dt):

        out, info = self.solver.solve(self._A1 + dt * self._A2,
                                      self._phi_n,
                                      pc=self._pc,
                                      x0=self._x0,
                                      tol=self._params['tol'],
                                      maxiter=self._params['maxiter'],
                                      verbose=self._params['verbose']
                                      )

        if self._params['info']:
            print(info)

        self.feec_vars_update(out)
        
    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('PConjugateGradient', 'MassMatrixPreconditioner'),
                                  ('ConjugateGradient', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct
