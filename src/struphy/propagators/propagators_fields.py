'Only FEEC variables are updated.'


import numpy as np
from numpy import zeros

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.accumulation.particles_to_grid import Accumulator
from struphy.polar.basic import PolarVector
from struphy.kinetic_background.base import Maxwellian
from struphy.kinetic_background.maxwellians import Maxwellian6D, Maxwellian5D
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.feec import preconditioner
from struphy.feec.mass import WeightedMassOperator
from struphy.feec.basis_projection_ops import BasisProjectionOperator, CoordinateProjector
from struphy.feec.variational_utilities import BracketOperator


from psydac.linalg.solvers import inverse
from psydac.linalg.basic import IdentityOperator
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector, BlockLinearOperator, BlockVectorSpace
import struphy.feec.utilities as util
from mpi4py import MPI

from copy import deepcopy


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
        params_default = {'type': ('pcg', 'MassMatrixPreconditioner'),
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
        self._B = -1/2 * self.derham.curl.T @ self.mass_ops.M2
        self._C = 1/2 * self.derham.curl

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(self.mass_ops.M1)

        # Instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
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

        en1, info = self._schur_solver(en, self._byn, dt, out=self._e_tmp1)

        # new b coeffs
        _e = en.copy(out=self._e_tmp2)
        _e += en1
        bn1 = self._C.dot(_e, out=self._b_tmp1)
        bn1 *= -dt
        bn1 += bn

        # write new coeffs into self.feec_vars
        max_de, max_db = self.feec_vars_update(en1, bn1)

        if self._info and self._rank == 0:
            print('Status     for Maxwell:', info['success'])
            print('Iterations for Maxwell:', info['niter'])
            print('Maxdiff e1 for Maxwell:', max_de)
            print('Maxdiff b2 for Maxwell:', max_db)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
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
        params_default = {'type': ('pcg', 'MassMatrixPreconditioner'),
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

        self._B = -1/2 * 1/self._epsilon * self.mass_ops.M1  # no dt

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate Schur solver (constant in this case)
        _BC = 1/2 * self._alpha**2 / self._epsilon * self._B

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        self._tmp_j1 = j.space.zeros()
        self._tmp_j2 = j.space.zeros()
        self._tmp_e1 = e.space.zeros()
        self._tmp_e2 = e.space.zeros()

    def __call__(self, dt):

        # current variables
        en = self.feec_vars[0]
        jn = self.feec_vars[1]

        # in-place solution (no tmps created here)
        Ben = self._B.dot(en, out=self._tmp_e1)

        jn1, info = self._schur_solver(jn, Ben, dt, out=self._tmp_j1)

        en1 = jn.copy(out=self._tmp_j2)
        en1 += jn1
        en1 *= 1/2 * self._alpha**2 / self._epsilon
        en1 *= -dt
        en1 += en

        # write new coeffs into Propagator.variables
        max_de, max_dj = self.feec_vars_update(en1, jn1)

        if self._info:
            print('Status     for OhmCold:', info['success'])
            print('Iterations for OhmCold:', info['niter'])
            print('Maxdiff e1 for OhmCold:', max_de)
            print('Maxdiff j1 for OhmCold:', max_dj)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
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
        params_default = {'type': ('pcg', 'MassMatrixPreconditioner'),
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
        self._A = -1/self._epsc * self.mass_ops.M1Bninv  # no dt

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate linear solver
        self._solver = inverse(self._M,
                               params['type'][0],
                               pc=pc,
                               x0=self.feec_vars[0],
                               tol=self._tol,
                               maxiter=self._maxiter,
                               verbose=self._verbose)

        # allocate dummy vectors to avoid temporary array allocations
        self._rhs_j = self._M.codomain.zeros()
        self._j_new = j.space.zeros()

    def __call__(self, dt):

        # current variables
        jn = self.feec_vars[0]

        # define system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        lhs = self._M - dt/2.0 * self._A
        rhs = self._M + dt/2.0 * self._A

        rhsv = rhs.dot(jn, out=self._rhs_j)
        # print(f'{self.derham.comm.Get_rank() = }, after dot')

        self._solver.linop = lhs

        # solve linear system for updated j coefficients (in-place)
        jn1 = self._solver.solve(rhsv, out=self._j_new)
        info = self._solver._info

        # write new coeffs into Propagator.variables
        max_dj = self.feec_vars_update(jn1)[0]

        if self._info:
            print('Status     for FluidCold:', info['success'])
            print('Iterations for FluidCold:', info['niter'])
            print('Maxdiff j1 for FluidCold:', max_dj)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
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
                          'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_M = 'M' + self.derham.space_to_form[params['u_space']] + 'n'
        id_T = 'T' + self.derham.space_to_form[params['u_space']]

        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)

        self._B = -1/2 * _T.T @ self.derham.curl.T @ self.mass_ops.M2
        self._C = 1/2 * self.derham.curl @ _T

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
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
        byn = self._B.dot(bn, out=self._byn)

        un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

        # new b coeffs
        _u = un.copy(out=self._u_tmp2)
        _u += self._u_tmp1
        bn1 = self._C.dot(_u, out=self._b_tmp1)
        bn1 *= -dt
        bn1 += bn

        # write new coeffs into self.feec_vars
        max_du, max_db = self.feec_vars_update(un1, bn1)

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
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
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
        params_default = {'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'type_M1': ('pcg', 'MassMatrixPreconditioner'),
                          'tol_M1': 1e-8,
                          'maxiter_M1': 3000,
                          'info_M1': False,
                          'verbose_M1': False}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()

        # define inverse of M1
        if params['type_M1'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type_M1'][1])
            pc = pc_class(self.mass_ops.M1)

        self._M1inv = inverse(self.mass_ops.M1,
                              params['type_M1'][0],
                              pc=pc,
                              tol=params['tol_M1'],
                              maxiter=params['maxiter_M1'],
                              verbose=params['verbose_M1'])

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M2n
        self._B = 1/2 * self.mass_ops.M2B @ self.derham.curl
        # I still have to invert M1
        self._C = 1/2 * self._M1inv @ self.derham.curl.T @ self.mass_ops.M2B

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, 'M2n'))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
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
        byn = self._B.dot(bn, out=self._byn)

        un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

        # new b coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        bn1 = self._C.dot(_u, out=self._b_tmp1)
        bn1 *= -dt
        bn1 += bn

        # write new coeffs into self.feec_vars
        max_du, max_db = self.feec_vars_update(un1, bn1)

        if self._info and self._rank == 0:
            print('Status     for ShearAlfvénB1:', info['success'])
            print('Iterations for ShearAlfvénB1:', info['niter'])
            print('Maxdiff up for ShearAlfvénB1:', max_du)
            print('Maxdiff b2 for ShearAlfvénB1:', max_db)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        dct['M1_inv'] = {'type_M1': [('pcg', 'MassMatrixPreconditioner'),
                                     ('cg', None)],
                         'tol_M1': 1.e-8,
                         'maxiter_M1': 3000,
                         'info_M1': False,
                         'verbose_M1': False}
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
        params_default = {'type': ('pbicgstab', 'MassMatrixPreconditioner'),
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
        self._A = self._kappa * self.derham.curl.T @ self._M2Bn @ self.derham.curl

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # Instantiate linear solver
        self._solver = inverse(self._M,
                               params['type'][0],
                               pc=pc,
                               x0=self.feec_vars[0],
                               tol=self._tol,
                               maxiter=self._maxiter,
                               verbose=self._verbose)

        # allocate dummy vectors to avoid temporary array allocations
        self._rhs_b = self._M.codomain.zeros()
        self._b_new = b.space.zeros()

    def __call__(self, dt):

        # current variables
        bn = self.feec_vars[0]

        # define system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        lhs = self._M - dt/2.0 * self._A
        rhs = self._M + dt/2.0 * self._A

        # solve linear system for updated b coefficients (in-place)
        rhs = rhs.dot(bn, out=self._rhs_b)
        self._solver.linop = lhs

        bn1 = self._solver.solve(rhs, out=self._b_new)
        info = self._solver._info

        # write new coeffs into self.feec_vars
        max_db = self.feec_vars_update(bn1)

        if self._info and self._rank == 0:
            print('Status     for Hall:', info['success'])
            print('Iterations for Hall:', info['niter'])
            print('Maxdiff b1 for Hall:', max_db)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
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
                          'type': ('pbicgstab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        self._info = params['info']
        self._bc = self.derham.dirichlet_bc
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = 'M' + self.derham.space_to_form[params['u_space']] + 'n'
        id_MJ = 'M' + self.derham.space_to_form[params['u_space']] + 'J'

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
            _U, _UT = IdentityOperator(u.space), IdentityOperator(u.space)
        else:
            _U = getattr(self.basis_ops, id_U)
            _UT = _U.T

        self._B = -1/2. * _UT @ self.derham.div.T @ self.mass_ops.M3
        self._C = 1/2. * self.derham.div @ _S + 2/3 * _K @ self.derham.div @ _U

        self._MJ = getattr(self.mass_ops, id_MJ)
        self._DQ = self.derham.div @ getattr(self.basis_ops, id_Q)

        self._b = params['b']

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
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

        # solve for new u coeffs (no tmps created here)
        byn1 = self._B.dot(pn, out=self._byn1)
        byn2 = self._MJ.dot(self._b, out=self._byn2)
        byn2 *= 1/2
        byn1 -= byn2

        un1, info = self._schur_solver(un, byn1, dt, out=self._u_tmp1)

        # new p, n, b coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        pn1 = self._C.dot(_u, out=self._p_tmp1)
        pn1 *= -dt
        pn1 += pn

        nn1 = self._DQ.dot(_u, out=self._n_tmp1)
        nn1 *= -dt/2
        nn1 += nn

        # write new coeffs into self.feec_vars
        max_dn, max_du, max_dp = self.feec_vars_update(nn1,
                                                       un1,
                                                       pn1)

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
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
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
        params_default = {'type': ('pbicgstab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._bc = self.derham.dirichlet_bc
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = 'M2n'
        id_K, id_Q = 'K3', 'Q3'

        _A = getattr(self.mass_ops, id_Mn)
        _K = getattr(self.basis_ops, id_K)

        self._B = -1/2. * self.derham.div.T @ self.mass_ops.M3
        self._C = 5/6. * _K @ self.derham.div

        self._QD = getattr(self.basis_ops, id_Q) @ self.derham.div

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
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
        byn1 = self._B.dot(pn, out=self._byn1)

        un1, info = self._schur_solver(un, byn1, dt, out=self._u_tmp1)

        # new p, n, b coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        pn1 = self._C.dot(_u, out=self._p_tmp1)
        pn1 *= -dt
        pn1 += pn

        nn1 = self._QD.dot(_u, out=self._n_tmp1)
        nn1 *= -dt/2.0
        nn1 += nn

        # write new coeffs into self.feec_vars
        max_dn, max_du, max_dp = self.feec_vars_update(nn1,
                                                       un1,
                                                       pn1)

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
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
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
        params_default = {'type': ('pbicgstab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._info = params['info']
        self._bc = self.derham.dirichlet_bc
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = 'M2n'
        # ONCE THE CODE SUPPORTS HAVING A DIFFERENT EQUILIBRIUM ELECTRON PRESSURE TO THE EQUILIBRIUM ION PRESSURE; WE MUST ADD A NEW PROJECTION MATRIX Ke3 THAT TAKES THE EQUILIBRIUM ELECTRON PRESSURE
        # INSTEAD OF THE EQUILIBRIUM ION PRESSURE: AND USE HERE Ke3, NOT K3.
        id_K, id_Q = 'K3', 'Q3'

        _A = getattr(self.mass_ops, id_Mn)
        _K = getattr(self.basis_ops, id_K)

        self._B = -1/2. * self.derham.div.T @ self.mass_ops.M3
        self._C = 5/6. * _K @ self.derham.div

        self._QD = getattr(self.basis_ops, id_Q) @ self.derham.div

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
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

        # solve for new u coeffs (no tmps created here)
        byn1 = self._B.dot(pn, out=self._byn1)

        un1, info = self._schur_solver(un, byn1, dt, out=self._u_tmp1)

        # new p, n, b coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        pn1 = self._C.dot(_u, out=self._p_tmp1)
        pn1 *= -dt
        pn1 += pn

        nn1 = self._QD.dot(_u, out=self._n_tmp1)
        nn1 *= -dt/2.0
        nn1 += nn

        # write new coeffs into self.feec_vars
        max_dn, max_du, max_dp = self.feec_vars_update(nn1,
                                                       un1,
                                                       pn1)

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
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
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

        self._Minv = inverse(self.mass_ops.M1, tol=1e-8)
        self._CMC = self.derham.curl.T @ self.mass_ops.M2 @ self.derham.curl
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

        self._L2 = -dt/2 * \
            self._Minv @ (
                self._accum_potential._operators[0].matrix + self._CMC)
        self._RHS = -(self._L2.dot(self._a)) - dt*(self._Minv.dot(
            self._accum_potential._vectors[0] - self.derham.curl.T @ self._M2).dot(self._beq))
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
            _LHS = self._M1 + HybridM1 @ self._L2
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
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
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
                          'f0': Maxwellian6D(),
                          'type': ('pbicgstab', 'MassMatrixPreconditioner'),
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
                self.derham.space_to_form[params['u_space']])

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
                    [self._f0.n], *quad_pts, kind='3', squeeze_out=False, coordinates='logical')
            else:
                self._nh0_at_quad = self.domain.push(
                    [self._f0.n], *quad_pts, kind='3', squeeze_out=False)

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
        self._E2T = self.derham.extraction_ops['2'].transpose()

        # mass matrix in system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        u_id = self.derham.space_to_form[params['u_space']]
        self._M = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(self._M)

        # linear solver
        self._solver = inverse(self._M,
                               params['type'][0],
                               pc=pc,
                               x0=self.feec_vars[0],
                               tol=self._tol,
                               maxiter=self._maxiter,
                               verbose=self._verbose)

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()

        self._rhs_v = u.space.zeros()
        self._u_new = u.space.zeros()

    def __call__(self, dt):

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
        lhs = self._M - dt/2 * self._accumulator.operators[0]
        rhs = self._M + dt/2 * self._accumulator.operators[0]

        # solve linear system for updated u coefficients (in-place)
        rhs = rhs.dot(un, out=self._rhs_v)
        self._solver.linop = lhs

        un1 = self._solver.solve(rhs, out=self._u_new)
        info = self._solver._info

        # write new coeffs into Propagator.variables
        max_du = self.feec_vars_update(un1)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling6DDensity:', info['success'])
            print('Iterations for CurrentCoupling6DDensity:', info['niter'])
            print('Maxdiff up for CurrentCoupling6DDensity:', max_du)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class ShearAlfvénCurrentCoupling5D(Propagator):
    r'''Crank-Nicolson scheme for the shear Alfvén step in :class:`~struphy.models.hybrid.LinearMHDDriftkineticCC`,

    Equation:

    .. math::

        \left\{ 
            \begin{aligned} 
                \int n_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \tilde{\mathbf V} \, \textnormal{d} \mathbf{x} = \int \left(\tilde{\mathbf B} - \frac{A_\textnormal{h}}{A_b} \iint f^\text{vol} \mu \mathbf{b}_0\textnormal{d} \mu \textnormal{d} v_\parallel \right) \cdot \nabla \times (\mathbf B_0 \times \tilde{\mathbf V}) \, \textnormal{d} \mathbf{x} \quad \forall \ \tilde{\mathbf V} \,,
                \\
                &\frac{\partial \tilde{\mathbf B}}{\partial t} = - \nabla \times (\mathbf B_0 \times \tilde{\mathbf U}) \,.
            \end{aligned}
        \right.

    FE coefficients update:

    .. math::

        \begin{bmatrix} 
            \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n
        \end{bmatrix} 
        = \frac{\Delta t}{2} \,.
        \begin{bmatrix} 
            0 & (\mathbb M^{\alpha,n})^{-1} \mathcal {T^\alpha}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^\alpha} (\mathbb M^{\alpha,n})^{-1} & 0 
        \end{bmatrix} 
        \begin{bmatrix}
            {\mathbb M^{\alpha,n}}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) + \sum_k^{N_p} \omega_k \mu_k \hat{\mathbf b}¹_0 (\boldsymbol \eta_k) \cdot \left(\frac{1}{\sqrt{g(\boldsymbol \eta_k)}} \vec \Lambda² (\boldsymbol \eta_k) \right)
        \end{bmatrix} \,,

    where 
    :math:`\mathcal{T}^\alpha` is a :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators` and
    :math:`\mathbb M^{\alpha,n}` is a :class:`~struphy.feec.mass.WeightedMassOperators` being weighted with :math:`\rho_\text{eq}`, the MHD equilibirum density. 
    :math:`\alpha \in \{1, 2, v\}` denotes the :math:`\alpha`-form space where the operators correspond to.
    Moreover, :math:`\sum_k^{N_p} \omega_k \mu_k \hat{\mathbf b}¹_0 (\boldsymbol \eta_k) \cdot \left(\frac{1}{\sqrt{g(\boldsymbol \eta_k)}} \vec \Lambda² (\boldsymbol \eta_k)\right)` is accumulated by the kernel :class:`~struphy.pic.accumulation.accum_kernels_gc.cc_lin_mhd_5d_M`.
    The solution of the above system is based on the :class:`struphy.linear_algebra.schur_solver.SchurSolver`.

    Parameters
    ---------- 
    u : BlockVector
        FE coefficients of MHD velocity.

    b : BlockVector
        FE coefficients of magnetic field as 2-form.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, u, b, **params):

        from struphy.pic.particles import Particles5D

        super().__init__(u, b)

        # parameters
        params_default = {'particles': Particles5D,
                          'u_space': 'Hdiv',
                          'unit_b1': None,
                          'f0': Maxwellian5D(),
                          'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1}

        params = set_defaults(params, params_default)

        self._particles = params['particles']
        self._f0 = params['f0']
        self._type = params['type'][0]
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._info = params['info']
        self._verbose = params['verbose']
        self._rank = self.derham.comm.Get_rank()
        self._scale_vec = params['Ah'] / params['Ab']

        self._unit_b1 = params['unit_b1']

        self._E1T = self.derham.extraction_ops['1'].transpose()
        self._unit_b1 = self._E1T.dot(self._unit_b1)

        self._ACC = Accumulator(self.derham, self.domain,
                                'Hdiv', 'cc_lin_mhd_5d_M', add_vector=True, symmetry='symm')

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_M = 'M' + self.derham.space_to_form[params['u_space']] + 'n'
        id_T = 'T' + self.derham.space_to_form[params['u_space']]

        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)

        self._B = -1/2 * _T.T @ self.derham.curl.T @ self.mass_ops.M2
        self._C = 1/2 * self.derham.curl @ _T
        self._B2 = -1/2 * _T.T @ self.derham.curl.T

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # Instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()
        self._tmp_acc = self._B2.codomain.zeros()

    def __call__(self, dt):

        # current variables
        un = self.feec_vars[0]
        bn = self.feec_vars[1]

        # accumulate vector
        self._ACC.accumulate(self._particles, 
                             self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                             self._scale_vec, 0.1)

        # solve for new u coeffs (no tmps created here)
        byn = self._B.dot(bn, out=self._byn)
        b2acc = self._B2.dot(self._ACC.vectors[0], out=self._tmp_acc)
        byn += b2acc

        un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

        # new b coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        bn1 = self._C.dot(_u, out=self._b_tmp1)
        bn1 *= -dt
        bn1 += bn

        # write new coeffs into self.feec_vars
        max_du, max_db = self.feec_vars_update(un1, bn1)

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
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class MagnetosonicCurrentCoupling5D(Propagator):
    r'''Crank-Nicolson scheme for Magnetosonic step in :class:`~struphy.models.hybrid.LinearMHDDriftkineticCC`,

    Equation:

    .. math::

        \left\{
            \begin{aligned}
                &\frac{\partial \tilde n}{\partial t} = - \nabla \cdot (n_0 \tilde{\mathbf U}) \,,
                \\
                \int n_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \tilde{\mathbf V} \, \textnormal{d} \mathbf{x} = \int (\nabla \times \mathbf B_0) \times \tilde{\mathbf B} \cdot \tilde{\mathbf V} \, \textnormal{d} \mathbf x + \frac{A_\textnormal{h}}{A_b}\iint f^\text{vol} \mu \mathbf b_0 \cdot \nabla \times (\tilde{\mathbf B} \times \tilde{\mathbf V}) \, \textnormal{d} \mathbf x \textnormal{d} v_\parallel \textnormal{d} \mu + \int \tilde p \nabla \cdot \tilde{\mathbf V} \, \textnormal{d} \mathbf x \qquad \forall \ \tilde{\mathbf V}\,,
                \\
                &\frac{\partial \tilde p}{\partial t} = - \nabla \cdot (p_0 \tilde{\mathbf U}) - (\gamma - 1) p_0 \nabla \cdot \tilde{\mathbf U} \,.
            \end{aligned} 
        \right.

    FE coefficients update:

    .. math::

        \boldsymbol{n}^{n+1} - \boldsymbol{n}^n = - \frac{\Delta t}{2} \mathbb D \mathcal Q^\alpha (\mathbf u^{n+1} + \mathbf u^n) \,,

    .. math::

        \begin{bmatrix} 
            \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1} - \mathbf p^n 
        \end{bmatrix} 
        = \frac{\Delta t}{2} 
        \begin{bmatrix} 
            0 & (\mathbb M^{\alpha,n})^{-1} {\mathcal U^\alpha}^\top \mathbb D^\top \mathbb M_3 \\ - \mathbb D \mathcal S^\alpha - (\gamma - 1) \mathcal K^\alpha \mathbb D \mathcal U^\alpha & 0 
        \end{bmatrix} 
        \begin{bmatrix} 
            (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1} + \mathbf p^n) 
        \end{bmatrix} + 
        \begin{bmatrix} 
            \Delta t (\mathbb M^{\alpha,n})^{-1}\left[\mathbb M^{\alpha,J} \mathbf b^n + \frac{A_\textnormal{h}}{A_b}{\mathcal{T}^B}^\top \mathbb{C}^\top \sum_k^{N_p} \omega_k \mu_k \hat{\mathbf b}¹_0 (\boldsymbol \eta_k) \cdot \left(\frac{1}{\sqrt{g(\boldsymbol \eta_k)}} \vec \Lambda² (\boldsymbol \eta_k) \right)\right] \\ 0 
        \end{bmatrix} \,,

    where 
    :math:`\mathcal U^\alpha`, :math:`\mathcal S^\alpha`, :math:`\mathcal K^\alpha` and :math:`\mathcal Q^\alpha` are :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators` and
    :math:`\mathbb M^{\alpha,n}` and :math:`\mathbb M^{\alpha,J}` are :class:`~struphy.feec.mass.WeightedMassOperators` being weighted with :math:`n_0` the MHD equilibrium density. 
    :math:`\alpha \in \{1, 2, v\}` denotes the :math:`\alpha`-form space where the operators correspond to.
    Moreover, :math:`\sum_k^{N_p} \omega_k \mu_k \hat{\mathbf b}¹_0 (\boldsymbol \eta_k) \cdot \left(\frac{1}{\sqrt{g(\boldsymbol \eta_k)}} \vec \Lambda² (\boldsymbol \eta_k)\right)` is accumulated by the kernel :class:`~struphy.pic.accumulation.accum_kernels_gc.cc_lin_mhd_5d_M` and
    the time-varying projection operator :math:`\mathcal{T}^B` is defined as

    .. math::

        \mathcal{T}^B_{(\mu,ijk),(\nu,mno)} := \hat \Pi¹_{(\mu,ijk)} \left[ \epsilon_{\mu \alpha \nu} \frac{\tilde{B}^2_\alpha}{\sqrt{g}} \Lambda²_{\nu,mno} \right] \,.

    The solution of the above system is based on the :class:`struphy.linear_algebra.schur_solver.SchurSolver`.

    Parameters
    ---------- 
    n : StencilVector
        FE coefficients of a discrete 3-form.

    u : BlockVector
        FE coefficients of MHD velocity.

    p : StencilVector
        FE coefficients of a discrete 3-form.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, n, u, p, **params):

        from struphy.pic.particles import Particles5D

        super().__init__(n, u, p)

        # parameters
        params_default = {'b': self.derham.Vh['2'].zeros(),
                          'particles': Particles5D,
                          'u_space': 'Hdiv',
                          'unit_b1': None,
                          'f0': Maxwellian5D(),
                          'type': ('pbicgstab', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1}

        params = set_defaults(params, params_default)

        assert isinstance(params['particles'], Particles5D)
        self._particles = params['particles']

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}
        self._u_id=  self.derham.space_to_form[params['u_space']]
        if self._u_id == 'v':
            self._space_key_int = 0
        else:
            self._space_key_int = int(self._u_id)

        self._f0 = params['f0']
        self._b = params['b']
        self._bc = self.derham.dirichlet_bc
        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()
        self._scale_vec = params['Ah'] / params['Ab']

        self._unit_b1 = params['unit_b1']

        self._E1T = self.derham.extraction_ops['1'].transpose()
        self._unit_b1 = self._E1T.dot(self._unit_b1)

        self._ACC = Accumulator(self.derham, self.domain,
                                'Hdiv', 'cc_lin_mhd_5d_M', add_vector=True, symmetry='symm')

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = 'M' + self._u_id + 'n'
        id_MJ = 'M' + self._u_id + 'J'

        if self._u_id == '1':
            id_S, id_U, id_K, id_Q = 'S1', 'U1', 'K3', 'Q1'
        elif self._u_id == '2':
            id_S, id_U, id_K, id_Q = 'S2', None, 'K3', 'Q2'
        elif self._u_id == 'v':
            id_S, id_U, id_K, id_Q = 'Sv', 'Uv', 'K3', 'Qv'

        self._E2T = self.derham.extraction_ops['2'].transpose()

        _A = getattr(self.mass_ops, id_Mn)
        _S = getattr(self.basis_ops, id_S)
        _K = getattr(self.basis_ops, id_K)

        # initialize projection operator TB
        self._initialize_projection_operator_TB()

        if id_U is None:
            _U, _UT = IdentityOperator(u.space), IdentityOperator(u.space)
        else:
            _U = getattr(self.basis_ops, id_U)
            _UT = _U.T

        self._B = -1/2. * _UT @ self.derham.div.T @ self.mass_ops.M3
        self._C = 1/2. * (self.derham.div @ _S + 2 /
                          3. * _K @ self.derham.div @ _U)

        self._MJ = getattr(self.mass_ops, id_MJ)
        self._DQ = self.derham.div @ getattr(self.basis_ops, id_Q)

        self._TC = self._TB.T @ self.derham.curl.T

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._p_tmp1 = p.space.zeros()
        self._n_tmp1 = n.space.zeros()
        self._byn1 = self._B.codomain.zeros()
        self._byn2 = self._B.codomain.zeros()
        self._tmp_acc = self._TC.codomain.zeros()

    def __call__(self, dt):

        # current variables
        nn = self.feec_vars[0]
        un = self.feec_vars[1]
        pn = self.feec_vars[2]

        # accumulate vector
        self._ACC.accumulate(self._particles, 
                             self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                             self._scale_vec, 0.1)


        # update time-dependent operator
        self._b.update_ghost_regions()
        self._update_weights_TB()

        # solve for new u coeffs (no tmps created here)
        byn1 = self._B.dot(pn, out=self._byn1)
        byn2 = self._MJ.dot(self._b, out=self._byn2)
        b2acc = self._TC.dot(self._ACC.vectors[0], out=self._tmp_acc)
        byn2 += b2acc
        byn2 *= 1/2
        byn1 -= byn2

        un1, info = self._schur_solver(un, byn1, dt, out=self._u_tmp1)

        # new p, n, b coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        pn1 = self._C.dot(_u, out=self._p_tmp1)
        pn1 *= -dt
        pn1 += pn

        nn1 = self._DQ.dot(_u, out=self._n_tmp1)
        nn1 *= -dt/2
        nn1 += nn

        # write new coeffs into self.feec_vars
        max_dn, max_du, max_dp = self.feec_vars_update(nn1,
                                                       un1,
                                                       pn1)

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
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct

    def _initialize_projection_operator_TB(self):
        """Initialize BasisProjectionOperator TB with the time-varying weight.

        .. math::

            \mathcal{T}^B_{(\mu,ijk),(\nu,mno)} := \hat \Pi¹_{(\mu,ijk)} \left[ \epsilon_{\mu \alpha \nu} \frac{\tilde{B}²_\alpha}{\sqrt{g}} \Lambda²_{\nu,mno} \right] \,.

        """

        # Call the projector and the space
        P1 = self.derham.P['1']
        Vh = self.derham.Vh_fem[self._u_id]

        # Femfield for the field evaluation
        self._bf = self.derham.create_field("bf", "Hdiv")

        # define temp callable
        def tmp(x,y,z): return 0*x

        # Initialize BasisProjectionOperator
        self._TB = BasisProjectionOperator(P1, Vh, [[tmp, tmp, tmp]])

    def _update_weights_TB(self):
        """Updats time-dependent weights of the BasisProjectionOperator TB
        """

        # Update Femfield
        self._bf.vector = self._b
        self._bf.vector.update_ghost_regions()

        # define callable weights
        def bf1(x,y,z): return self._bf(x,y,z,local=True)[0]
        def bf2(x,y,z): return self._bf(x,y,z,local=True)[1]
        def bf3(x,y,z): return self._bf(x,y,z,local=True)[2]

        from struphy.feec.utilities import RotationMatrix

        rot_B = RotationMatrix(bf1, bf2, bf3)

        fun = []

        if self._u_id == 'v':
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n:
                                rot_B(e1, e2, e3)[:, :, :, m, n]]

        elif self._u_id == '1':
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n:
                                (rot_B(e1, e2, e3) @ self.domain.metric_inv(e1, e2, e3, change_out_order=True, squeeze_out=False))[:, :, :, m, n]]

        else:
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n:
                                rot_B(e1, e2, e3)[:, :, :, m, n] / abs(self.domain.jacobian_det(e1, e2, e3, squeeze_out=False))]

        # Initialize BasisProjectionOperator
        self._TB.update_weights(fun)


class CurrentCoupling5DDensity(Propagator):
    r'''Crank-Nicolson scheme for the CC-Density step in :class:`~struphy.models.hybrid.LinearMHDDriftkineticCC`,

    Equation:

    .. math::

        \int n_0 \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \tilde{\mathbf V} \, \textnormal{d} \mathbf{x} = \frac{A_\textnormal{h}}{A_b} \frac{1}{\epsilon} \iiint f^\text{vol} \left(1 - \frac{B_\parallel}{B^*_\parallel}\right) \tilde{\mathbf U} \times \mathbf B_f \cdot \tilde{\mathbf V} \, \textnormal{d} \mathbf{x} \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \ \tilde{\mathbf V} \,,

    FE coefficients update:

    .. math::

        \mathbf u^{n+1} - \mathbf u^n = -\frac{A_\textnormal{h}}{A_b} \frac{1}{\epsilon} \mathbb{L}²{\mathbb{B}}^\times_f \mathbb{N}(1/g) \mathbb{W} \mathbb{N}\left(1- \frac{\hat B^0_\parallel}{\hat B^{*0} _\parallel}\right) (\mathbb{L}²)^\top \frac{\Delta t}{2} \cdot (\mathbf u^{n+1} + \mathbf u^n) \,.

    For the detail explanation of the notations, see `2022_DriftKineticCurrentCoupling <https://gitlab.mpcdf.mpg.de/struphy/struphy-projects/-/blob/main/running-projects/2022_DriftKineticCurrentCoupling.md?ref_type=heads>`_.

    Parameters
    ---------- 
    u : BlockVector
        FE coefficients of MHD velocity.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, u, **params):

        from struphy.pic.particles import Particles5D

        super().__init__(u)

        # parameters
        params_default = {'particles': None,
                          'u_space': 'Hdiv',
                          'b_eq': None,
                          'b_tilde': None,
                          'unit_b1': None,
                          'curl_unit_b2': None,
                          'f0': Maxwellian5D(),
                          'type': 'pbicgstab',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1,
                          'epsilon': 1.}

        params = set_defaults(params, params_default)

        # assert parameters and expose some quantities to self
        assert isinstance(params['particles'], (Particles5D))

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}
        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[params['u_space']])

        self._particles = params['particles']
        self._b_eq = params['b_eq']
        self._b_tilde = params['b_tilde']
        self._unit_b1 = params['unit_b1']
        self._curl_norm_b = params['curl_unit_b2']
        self._epsilon = params['epsilon']
        self._f0 = params['f0']

        self._type = params['type'][0]
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._info = params['info']
        self._verbose = params['verbose']
        self._rank = self.derham.comm.Get_rank()

        self._scale_mat = params['Ah'] / params['Ab']
        self._accumulator = Accumulator(
            self.derham, self.domain, params['u_space'], 'cc_lin_mhd_5d_D', add_vector=False, symmetry='asym')

        u_id = self.derham.space_to_form[params['u_space']]
        self._M = getattr(self.mass_ops, 'M' + u_id + 'n')

        self._E0T = self.derham.extraction_ops['0'].transpose()
        self._EuT = self.derham.extraction_ops[u_id].transpose()
        self._E1T = self.derham.extraction_ops['1'].transpose()
        self._E2T = self.derham.extraction_ops['2'].transpose()

        self._PB = getattr(self.basis_ops, 'PB')
        self._unit_b1 = self._E1T.dot(self._unit_b1)

        # preconditioner
        if params['type'][1] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            self._pc = pc_class(self._M)

        # linear solver
        self._solver = inverse(self._M,
                               params['type'][0],
                               pc=self._pc,
                               x0=self.feec_vars[0],
                               tol=self._tol,
                               maxiter=self._maxiter,
                               verbose=self._verbose)

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()
        self._rhs_v = u.space.zeros()
        self._u_new = u.space.zeros()

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):
        # pointer to old coefficients
        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b_eq.copy(out=self._b_full1)

        if self._b_tilde is not None:
            b_full += self._b_tilde

        Eb_full = self._E2T.dot(b_full, out=self._b_full2)
        Eb_full.update_ghost_regions()

        self._accumulator.accumulate(self._particles, self._epsilon,
                                     Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
                                     self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                     self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                     self._space_key_int, self._scale_mat, 0.1)

        # define system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        lhs = self._M - dt/2 * self._accumulator.operators[0]
        rhs = self._M + dt/2 * self._accumulator.operators[0]

        # solve linear system for updated u coefficients (in-place)
        rhs = rhs.dot(un, out=self._rhs_v)
        self._solver.linop = lhs

        un1 = self._solver.solve(rhs, out=self._u_new)
        info = self._solver._info

        # write new coeffs into Propagator.variables
        max_du = self.feec_vars_update(un1)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling5DDensity:', info['success'])
            print('Iterations for CurrentCoupling5DDensity:', info['niter'])
            print('Maxdiff up for CurrentCoupling5DDensity:', max_du)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class ImplicitDiffusion(Propagator):
    r"""
    Weak, implicit discretization of the diffusion (or heat) equation (can be used as a Poisson solver too).

    Find :math:`\phi \in H^1` such that

    .. math::

        \int_\Omega \psi\, n_0(\mathbf x)\frac{\partial \phi}{\partial t}\,\textrm d \mathbf x + \int_\Omega \nabla \psi^\top D_0(\mathbf x) \nabla \phi \,\textrm d \mathbf x = \int_\Omega \psi\, \rho(\mathbf x)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`n_0, \rho:\Omega \to \mathbb R` are real-valued functions and 
    :math:`D_0:\Omega \to \mathbb R^{3\times 3}`
    is a positive diffusion matrix. 
    Boundary terms from integration by parts are assumed to vanish.
    The equation is discretized as

    .. math::

        \left( \frac{\sigma_1}{\Delta t} \mathbb M^0_{n_0} + \mathbb G^\top \mathbb M^1_{D_0} \mathbb G \right)\, \boldsymbol\phi^{n+1} = \frac{\sigma_2}{\Delta t} ( \Lambda^0, n_0 \phi^n )_{L^2} + \frac{\sigma_3}{\Delta t}(\Lambda^0, \rho  )_{L^2}\,,

    where :math:`M^0_{n_0}` and :math:`M^1_{D_0}` are :class:`WeightedMassOperators <struphy.feec.mass.WeightedMassOperators>`
    and :math:`\sigma_1, \sigma_2, \sigma_3 \in \mathbb R` are artificial parameters that can be tuned to
    change the model (see Notes).

    Notes
    -----

    * :math:`\sigma_1=\sigma_2=0` and :math:`\sigma_3 = 1`: **Poisson solver** with a given charge density :math:`\rho/\Delta t`. 
    * :math:`\sigma_2=0` and :math:`\sigma_1 = \sigma_3 = \Delta t` : Poisson with **adiabatic electrons**.
    * :math:`\sigma_1=\sigma_2=1` and :math:`\sigma_3 = 0`: **Implicit heat equation**. 

    Parameters
    ----------
    phi : StencilVector
        FE coefficients of the solution as a discrete 0-form.

    sigma : float
        Stabilization parameter: :math:`\sigma=1` for the heat equation and :math:`\sigma=0` for the Poisson equation.

    rho : StencilVector
        Right-hand side FE coefficients of a 0-form (optional, can be set with a setter later).

    x0 : StencilVector
        Initial guess for the iterative solver (optional, can be set with a setter later).

    **params : dict
        Parameters for the iteravtive solver.
    """

    def __init__(self, phi: StencilVector,
                 sigma_1=0., sigma_2=1., sigma_3=0.,
                 A1_mat='M0', A2_mat='M1',
                 rho=None,
                 x0=None,
                 **params):

        assert phi.space == self.derham.Vh['0']

        super().__init__(phi)

        # model parameters
        self._sigma_1 = sigma_1
        self._sigma_2 = sigma_2
        self._sigma_3 = sigma_3

        # solver parameters
        params_default = {'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        # check the rhs
        if rho is not None:
            assert rho.space == phi.space
            self._rho = rho
        else:
            self._rho = phi.space.zeros()

        # check solvability condition
        if np.abs(self._sigma_1) < 1e-14:
            self._sigma_1 = 1e-14
            #self.check_rhs(self._rho)

        # initial guess and solver params
        self._x0 = x0
        self._params = params
        A1_mat = getattr(self.mass_ops, A1_mat)
        A2_mat = getattr(self.mass_ops, A2_mat)
        # Set lhs matrices (without dt)
        self._A1 = A1_mat
        self._A2 = self.derham.grad.T @ A2_mat @ self.derham.grad

        # preconditioner and solver for Ax=b
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(A1_mat)

        # solver just with A_2, but will be set during call with dt
        self.solver = inverse(self._A2,
                              params['type'][0],
                              pc=pc,
                              x0=self._x0,
                              tol=self._params['tol'],
                              maxiter=self._params['maxiter'],
                              verbose=self._params['verbose'])

        # allocate memory for solution
        self._tmp = phi.space.zeros()
        self._rhs = phi.space.zeros()
        self._rhs2 = phi.space.zeros()

    def check_rhs(self, rho):
        '''Checks space of rhs and, for periodic boundary conditions and sigma=0,
        checks whether the integral over rho is zero.

        Parameters
        ----------
        rho : StencilVector
            FE coefficients of a 0-form.'''

        assert type(rho) == type(self._rho)

        if np.all(rho.space.periods):
            solvability = np.zeros(1)
            self.derham.comm.Allreduce(
                np.sum(rho.toarray()), solvability, op=MPI.SUM)
            assert np.abs(
                solvability[0]) <= 1e-10, f'Solvability condition not met: {solvability[0]}'

    @property
    def rho(self):
        """
        psydac.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector.
        """
        return self._rho

    @rho.setter
    def rho(self, value):
        """ In-place setter for StencilVector/PolarVector.
        """
        self.check_rhs(value)
        self._rho[:] = value[:]

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
        assert type(value) == type(self._rho)
        assert value.space.symbolic_space == 'H1', f'Right-hand side must be in H1, but is in {value.space.symbolic_space}.'

        if self._x0 is None:
            self._x0 = value
        else:
            self._x0[:] = value[:]

    def __call__(self, dt):

        # current variables
        phin = self.feec_vars[0]

        # compute rhs
        rhs = self._A1.dot(phin, out=self._rhs)
        rhs *= self._sigma_2 / dt

        self._rhs2 = self._sigma_3 / dt * self._rho

        rhs += self._rhs2

        # compute lhs
        self.solver.linop = self._sigma_1/dt * self._A1 + self._A2

        # solve
        out = self.solver.solve(rhs, out=self._tmp)
        info = self.solver._info

        if self._params['info']:
            print(info)

        self.feec_vars_update(out)

    @classmethod
    def options(cls):
        dct = {}
        dct['model'] = {'sigma_1': 0.,
                        'sigma_2': 0.,
                        'sigma_3': 1.,
                        'A1_mat': ['M0', 'M0ad'],
                        'A2_mat': ['M1', 'M1perp']}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class VariationalVelocityAdvection(Propagator):
    r'''Crank-Nicolson step for self-advection term in Burger equation,

    .. math::

        \int_{\Omega} \partial_t \mathbf u \cdot \mathbf v \, \textnormal d^3 \mathbf x - \frac{1}{3} \int_{\Omega} \mathbf u \cdot [\mathbf u, \mathbf v] \, \textnormal d^3 \mathbf x = 0 ~ ,

    which is discretized as

    .. math::

        \mathbb M_v \frac{\mathbf u^{n+1}- \mathbf u^n}{\Delta t} - \frac{1}{3}(\sum_i (\hat{\Pi}^{1->0}[\mathbf u^{n+1/2}] \mathbb G P_i - \hat{\Pi}^{X->0}[\nabla\mathbf u^{n+1/2}_i])^\top P_i) \mathbb M_v \mathbf u^{n+1/2} = 0 ~ .

    Parameters
    ----------
    u : psydac.linalg.stencil.BlockVector
        FE coefficients of a discrete vector field, the solution.

    **params : dict
        Parameters for the iterative solver.

    '''

    def __init__(self, u, **params):

        super().__init__(u)

        # parameters
        params_default = {'tol': 1e-8,
                          'maxiter': 100,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._params = params

        # Femfields for the projectors
        self.uf = self.derham.create_field("gu1f", "H1vec")
        self.gu1f = self.derham.create_field("gu1f", "Hcurl")  # grad(u[0])
        self.gu2f = self.derham.create_field("gu2f", "Hcurl")  # grad(u[1])
        self.gu3f = self.derham.create_field("gu3f", "Hcurl")  # grad(u[2])

        self._initialize_projectors()

        # gradient of the component of the vector field
        grad = self.derham.grad
        self.gp1 = grad @ self.Pcoord1
        self.gp2 = grad @ self.Pcoord2
        self.gp3 = grad @ self.Pcoord3

        # v-> int(Pi(grad v_i . u)m_i)
        m1ugv1 = self.gp1.T @ self.PiuT @ self.Pcoord1
        m2ugv2 = self.gp2.T @ self.PiuT @ self.Pcoord2
        m3ugv3 = self.gp3.T @ self.PiuT @ self.Pcoord3

        # v-> int(Pi(grad u_i . v)m_i)
        m1vgu1 = self.PiguT_1 @ self.Pcoord1
        m2vgu2 = self.PiguT_2 @ self.Pcoord2
        m3vgu3 = self.PiguT_3 @ self.Pcoord3

        # v-> int(Pi([u,v]) . m)
        self.mbrackuv = 1/3 * \
            (m1vgu1 + m2vgu2 + m3vgu3 - m1ugv1 - m2ugv2 - m3ugv3)

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_diff = u.space.zeros()
        self._tmp_weak_diff = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn12 = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self.gp1u = self.gp1.dot(u)
        self.gp2u = self.gp2.dot(u)
        self.gp3u = self.gp3.dot(u)

        # mass matrix to compute L2 norm of error
        self._Mv = self.mass_ops.Mv
        self._Mvinv = inverse(self._Mv, 'cg', tol=1e-16)

    def __call__(self, dt):

        # Initialize variable for Picard iteration
        un = self.feec_vars[0]
        mn = self._Mv.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)
        tol = self._params['tol']
        err = tol+1
        for it in range(self._params['maxiter']):
            # Picard iteration
            if err < tol:
                break

            # half time step approximation
            mn12 = mn.copy(out=self._tmp_mn12)
            mn12 += mn1
            mn12 *= 0.5
            mn12.update_ghost_regions()

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            un12.update_ghost_regions()

            # gradients of un12 components
            grad_1_u = self.gp1.dot(un12, out=self.gp1u)
            grad_2_u = self.gp2.dot(un12, out=self.gp2u)
            grad_3_u = self.gp3.dot(un12, out=self.gp3u)

            # To avoid tmp we need to update the fields we created.
            self.gu1f.vector = grad_1_u
            self.gu2f.vector = grad_2_u
            self.gu3f.vector = grad_3_u
            self.uf.vector = un12

            # Update the BasisProjectionOperators
            self._update_all_weights()

            # Compute the advection term
            advection = self.mbrackuv.dot(mn12, out=self._tmp_advection)
            advection *= dt
            advection.update_ghost_regions()

            # Difference with the previous approximation :
            # diff = m^{n+1,r}-m^{n+1,r+1} = m^{n+1,r}-m^{n}+advection
            diff = mn1.copy(out=self._tmp_diff)
            diff -= mn
            diff += advection

            # Compute the norm of the difference
            weak_diff = self._Mvinv.dot(
                self._tmp_diff, out=self._tmp_weak_diff)
            err = self._tmp_diff.dot(weak_diff)

            # Update : m^{n+1,r+1} = m^n-advection
            mn1 = mn.copy(out=self._tmp_mn1)
            mn1 -= advection

            # Inverse the mass matrix to get the velocity
            un1 = self._Mvinv.dot(mn1, out=self._tmp_un1)

        self.feec_vars_update(un1)

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'tol': 1e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct

    def _initialize_projectors(self):
        """Initialization of all the `BasisProjectionOperator` and `CoordinateProjector` needed to compute the bracket term"""

        # Get the projector and the spaces
        P0 = self.derham.P['0']

        Xh = self.derham.Vh_fem['v']
        V0h = self.derham.Vh_fem['0']
        V1h = self.derham.Vh_fem['1']

        # Initialize the CoordinateProjectors
        self.Pcoord1 = CoordinateProjector(0, Xh, V0h)
        self.Pcoord2 = CoordinateProjector(1, Xh, V0h)
        self.Pcoord3 = CoordinateProjector(2, Xh, V0h)

        # Lambda for the initializations
        def uf1(x, y, z): return self.uf(x, y, z)[0]

        # Initialize the BasisProjectionOperators
        self.PiuT = BasisProjectionOperator(
            P0, V1h, [[uf1, uf1, uf1]], transposed=True, use_cache=True)

        self.PiguT_1 = BasisProjectionOperator(
            P0,  Xh, [[uf1, uf1, uf1]], transposed=True, use_cache=True)
        self.PiguT_2 = BasisProjectionOperator(
            P0,  Xh, [[uf1, uf1, uf1]], transposed=True, use_cache=True)
        self.PiguT_3 = BasisProjectionOperator(
            P0,  Xh, [[uf1, uf1, uf1]], transposed=True, use_cache=True)

        # Store the interpolation grid for later use in _update_all_weights
        self._interpolation_grid = [pts.flatten()
                                    for pts in self.derham.proj_grid_pts['0']]

    def _update_all_weights(self,):
        """Update the wieghts of all the `BasisProjectionOperators` appearing in the bracket term"""
        uf_values = self.uf(*self._interpolation_grid)

        guf1_values = self.gu1f(*self._interpolation_grid)
        guf2_values = self.gu2f(*self._interpolation_grid)
        guf3_values = self.gu3f(*self._interpolation_grid)

        self.PiuT.update_weights([[uf_values[0], uf_values[1], uf_values[2]]])

        self.PiguT_1.update_weights(
            [[guf1_values[0], guf1_values[1], guf1_values[2]]])
        self.PiguT_2.update_weights(
            [[guf2_values[0], guf2_values[1], guf2_values[2]]])
        self.PiguT_3.update_weights(
            [[guf3_values[0], guf3_values[1], guf3_values[2]]])


class VariationalMomentumAdvection(Propagator):
    r'''Crank-Nicolson step for self-advection term in fluids model,

    .. math::

        \int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \,\textrm d \boldsymbol \eta - 
        \int_{\hat{\Omega}}( \hat{\rho}^3 \hat{\mathbf{u}}) \cdot G [\hat{\mathbf{u}}, \hat{\mathbf{v}}] \, \textrm d \boldsymbol \eta = 0 ~ ,

    which is discretized as

    .. math::

        \mathbb M^v[\hat{\rho}_h^n] \frac{\mathbf u^{n+1}- \mathbf u^n}{\Delta t} - (\sum_{\mu} (\hat{\Pi}^{0}[\hat{\mathbf u}_h^{n+1/2} \cdot \vec{\boldsymbol \Lambda}^1] \mathbb G P_{\mu} - \hat{\Pi}^0[\hat{\mathbf A}^1_{\mu,h} \cdot \vec{\boldsymbol \Lambda}^v])^\top P_i) \mathbb M^v[\hat{\rho}_h^n] \mathbf u^{n} = 0 ~ .

    where :math:`P_\mu` stand for the :class:`~struphy.feec.basis_projection_ops.CoordinateProjector` and the weights
    in the the two :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{n+1/2} = (\mathbf{u}^{n+1/2})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \,, \qquad \hat{\mathbf A}^1_{\mu,h} = \nabla P_\mu((\mathbf u^{n+1/2})^\top \vec{\boldsymbol \Lambda}^v)] \in V_h^1\,, \qquad \hat{\rho}_h^{n} = (\rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3.
    Parameters
    ----------
    rho : psydac.linalg.stencil.Vector
        FE coefficients of a discrete field, density of the solution.

    u : psydac.linalg.stencil.BlockVector
        FE coefficients of a discrete vector field,velocity of the solution.

    **params : dict
        Parameters for the iterative solvers.

    '''

    def __init__(self, u, **params):

        super().__init__(u)

        # parameters
        params_default = {'linear_tol': 1e-12,
                          'non_linear_tol': 1e-8,
                          'linear_maxiter': 500,
                          'non_linear_maxiter': 100,
                          'type_linear_solver': ('pcg', 'MassMatrixDiagonalPreconditioner'),
                          'non_linear_solver': 'Newton',
                          'info': False,
                          'verbose': False,
                          'mass_ops': None}

        assert 'mass_ops' in params

        params = set_defaults(params, params_default)

        self._params = params

        self.WMM = params['mass_ops']

        self._initialize_mass()

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_diff = u.space.zeros()
        self._tmp_update = u.space.zeros()
        self._tmp_weak_diff = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_advection = u.space.zeros()

        self.brack = BracketOperator(self.derham, self._tmp_mn)

    def __call__(self, dt):
        if self._params['non_linear_solver'] == 'Newton':
            self.__call_newton(dt)
        elif self._params['non_linear_solver'] == 'Picard':
            self.__call_picard(dt)

    def __call_newton(self, dt):

        # Initialize variable for Newton iteration
        un = self.feec_vars[0]
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)
        tol = self._params['non_linear_tol']
        err = tol+1
        # Jacobian matrix for Newton solve
        self.derivative = self.WMM + dt/2*self.brack
        inv_derivative = inverse(self.derivative,
                                 'gmres',
                                 tol=self._params['linear_tol'],
                                 maxiter=self._params['linear_maxiter'],
                                 verbose=self._params['verbose'],
                                 recycle=True)

        for it in range(self._params['non_linear_maxiter']):

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Compute the advection term
            advection = self.brack.dot(un12, out=self._tmp_advection)
            advection *= dt

            # Difference with the previous approximation :
            # diff = m^{n+1,r}-m^{n+1,r+1} = m^{n+1,r}-m^{n}+advection
            diff = mn1.copy(out=self._tmp_diff)
            diff -= mn
            diff += advection

            # Get error and stop if small enough
            err = self._get_error_newton(diff)

            if err < tol**2:
                break

            # Newton step

            update = inv_derivative.dot(diff, out=self._tmp_update)
            un1 -= update
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self._params['non_linear_maxiter']-1:
            print(
                f'!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err = } \n {tol**2 = }')

        self.feec_vars_update(un1)

    def __call_picard(self, dt):

        # Initialize variable for Picard iteration
        un = self.feec_vars[0]
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)
        tol = self._params['non_linear_tol']
        err = tol+1
        # Jacobian matrix for Newton solve
        
        for it in range(self._params['non_linear_maxiter']):

            # Picard iteration
            if err < tol**2:
                break
            # half time step approximation
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Compute the advection term
            advection = self.brack.dot(un12, out=self._tmp_advection)
            advection *= dt

            # Difference with the previous approximation :
            # diff = m^{n+1,r}-m^{n+1,r+1} = m^{n+1,r}-m^{n}+advection
            diff = mn1.copy(out=self._tmp_diff)
            diff -= mn
            diff += advection

            # Compute the norm of the difference
            weak_diff = self._Mrhoinv.dot(
                self._tmp_diff, out=self._tmp_weak_diff)
            err = self._tmp_diff.dot(weak_diff)

            # Update : m^{n+1,r+1} = m^n-advection
            mn1 = mn.copy(out=self._tmp_mn1)
            mn1 -= advection

            # Inverse the mass matrix to get the velocity
            un1 = self._Mrhoinv.dot(mn1, out=self._tmp_un1)

        if it == self._params['non_linear_maxiter']-1:
            print(f'!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err = } \n {tol**2 = }')

        self.feec_vars_update(un1)

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'linear_tol': 1e-12,
                         'non_linear_tol': 1e-8,
                         'linear_maxiter': 500,
                         'non_linear_maxiter': 100,
                         'type_linear_solver': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                                ('cg', None)],
                         'non_linear_solver': ['Newton', 'Picard'],
                         'info': False,
                         'verbose': False}
        return dct

    def _initialize_mass(self):
        """Initialization of the mass matrix solver"""
        # weighted mass matrix to go from m to u
        self._Mrho = self.WMM

        # Inverse weighted mass matrix
        if self._params['type_linear_solver'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._params['type_linear_solver'][1])
            self.pc = pc_class(self.mass_ops.Mv)

        self._Mrhoinv = inverse(self._Mrho,
                                self._params['type_linear_solver'][0],
                                pc=self.pc,
                                tol=self._params['linear_tol'],
                                maxiter=self._params['linear_maxiter'],
                                verbose=self._params['verbose'])
        
    def _get_error_newton(self, mn_diff):
        inv_Mv = inverse(self.mass_ops.Mv, 'cg', tol=1e-16, maxiter = 1000)
        weak_un_diff = inv_Mv.dot(
            mn_diff, out=self._tmp_weak_diff)
        err_u = weak_un_diff.dot(mn_diff)
        return err_u


class VariationalDensityEvolve(Propagator):
    r'''Crank-Nicolson step for the evolution of the density terms in fluids models,

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta  
        + \int_{\hat{\Omega}} \big( \frac{| DF \hat{\mathbf{u}} |^2}{2} - \frac{\partial \hat{\rho}^3 \hat{e}}{\partial \hat{\rho}^3} \big) \nabla \cdot (\hat{\rho}^3 \hat{\mathbf{v}}) \, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{\rho}^3 + \nabla \cdot ( \hat{\rho}^3 \hat{\mathbf{u}} ) = 0 ~ ,
        \end{align}

    where :math:`\hat{e}` depends on the chosen model. It is discretized as

    .. math::

        \begin{align}
        &\frac{\mathbb M^v[\hat{\rho}_h^{n+1}] \mathbf u^{n+1}- \mathbb M^v[\hat{\rho}_h^n] \mathbf u^n}{\Delta t} 
        + (\mathbb D \hat{\Pi}^{2}[\hat{\rho}_h^{n+1/2} \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3\Big( \big(\frac{DF \hat{\mathbf{u}}_h^{n+1} \cdot DF \hat{\mathbf{u}}_h^{n}}{2} 
        - \frac{\hat{\rho}_h^{n+1}\hat{e}(\hat{\rho}_h^{n+1})-\hat{\rho}_h^{n}\hat{e}(\hat{\rho}_h^{n})}{\hat{\rho}_h^{n+1}-\hat{\rho}_h^n} \big)\Big) = 0 ~ ,
        \\[2mm]
        &\frac{\boldsymbol \rho^{n+1}- \boldsymbol \rho^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{\rho}_h^{n+1/2} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n+1/2} = 0 ~ ,
        \end{align}

    where :math:`\hat{l}^3(f)` denotes the vector representing the linear form :math:`v_h \mapsto \int_{\hat{\Omega}} f(\boldsymbol \eta) v_h(\boldsymbol \eta) d \boldsymbol \eta`, that is the vector with components

    .. math::
        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    and the weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{k} = (\mathbf{u}^{k})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \, \text{for k in} \{n, n+1/2, n+1\}, \qquad \hat{\rho}_h^{k} = (\rho^{k})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \, \text{for k in} \{n, n+1/2, n+1\} .

    Parameters
    ----------
    rho : psydac.linalg.stencil.Vector
        FE coefficients of a discrete field, density of the solution.

    u : psydac.linalg.stencil.BlockVector
        FE coefficients of a discrete vector field,velocity of the solution.

    **params : dict
        Parameters for the iterative solver, the linear solver and the model.

    '''

    def __init__(self, rho, u, **params):

        super().__init__(rho, u)

        # parameters
        params_default = {'linear_tol': 1e-12,
                          'non_linear_tol': 1e-8,
                          'linear_maxiter': 500,
                          'non_linear_maxiter': 100,
                          'type_linear_solver': ('pcg', 'MassMatrixDiagonalPreconditioner'),
                          'non_linear_solver': 'Newton',
                          'info': False,
                          'verbose': False,
                          'model': None,
                          'gamma': 5/3,
                          's': None,
                          'mass_ops': None}

        assert 'model' in params, 'model must be provided for VariationalDensityEvolve'
        assert params['model'] in ['pressureless', 'barotropic', 'full']
        if params['model'] == 'full':
            assert 's' in params
        assert 'mass_ops' in params
        params = set_defaults(params, params_default)

        self._params = params

        self.WMM = params['mass_ops']

        # Femfields for the projector
        self.sf = self.derham.create_field("sf", "L2")
        self.rhof = self.derham.create_field("rhof", "L2")
        self.rhof1 = self.derham.create_field("rhof1", "L2")
        self.uf = self.derham.create_field("uf", "H1vec")
        self.uf1 = self.derham.create_field("uf1", "H1vec")

        # Projector
        self._initialize_projectors_and_mass()
        self.rhof1.vector = rho
        self._update_weighted_MM()

        # gradient of the component of the vector field
        self.div = self.derham.div

        # Initialize the transport operator and transposed
        self.divPirho = self.div @ self.Pirho
        self.divPirhoT = self.PirhoT @ self.div.T

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_rhon1 = rho.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_rhon_diff = rho.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_rhon_weak_diff = rho.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_rho_advection = rho.space.zeros()
        self._linear_form_dl_drho = rho.space.zeros()

    def __call__(self, dt):
        if self._params['non_linear_solver'] == 'Newton':
            self.__call_newton(dt)
        elif self._params['non_linear_solver'] == 'Picard':
            self.__call_picard(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # Initialize variable for Newton iteration
        if self._params['model'] == 'full':
            s = self._params['s']
            self.sf.vector = s
        rhon = self.feec_vars[0]
        rhon1 = rhon.copy(out=self._tmp_rhon1)
        self.rhof.vector = rhon
        self.rhof1.vector = rhon1
        self._update_all_weights()
        self._update_weighted_MM()
        un = self.feec_vars[1]
        self.uf.vector = un
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        un1 = un.copy(out=self._tmp_un1)
        mn1 = mn.copy(out=self._tmp_mn1)
        tol = self._params['non_linear_tol']
        err = tol+1

        inv_derivative = inverse(self._Jacobian,
                                 'gmres',
                                 tol=self._params['linear_tol'],
                                 maxiter=self._params['linear_maxiter'],
                                 verbose=self._params['verbose'],
                                 recycle = True)

        for it in range(self._params['non_linear_maxiter']):

            # Newton iteration
            
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self.uf1.vector = un1
            self._update_linear_form_u2()

            # Compute the advection terms
            advection = self.divPirhoT.dot(
                self._linear_form_dl_drho, out=self._tmp_advection)
            advection *= dt

            rho_advection = self.divPirho.dot(
                un12, out=self._tmp_rho_advection)
            rho_advection *= dt

            # Get diff 
            rhon_diff = rhon1.copy(out=self._tmp_rhon_diff)
            rhon_diff -= rhon
            rhon_diff += rho_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error_newton(mn_diff, rhon_diff)

            if err < tol**2:
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = self.mass_ops.M3.dot(rhon_diff)

            incr = inv_derivative.dot(self._tmp_f, out = self._tmp_incr)

            un1 -= incr[0]
            rhon1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            self.rhof1.vector = rhon1
            self._update_weighted_MM()
            mn1 = self._Mrho.dot(un1, out = self._tmp_mn1)


        if it == self._params['non_linear_maxiter']-1:
            print(
                f'!!!Warning: Maximum iteration in VariationalDensityEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(rhon1, un1)

    def __call_picard(self, dt):
        """Solve the non linear system for updating the variables using Picard iteration method"""

        # Initialize variable for Picard iteration
        if self._params['model'] == 'full':
            s = self._params['s']
            self.sf.vector = s
        rhon = self.feec_vars[0]
        rhon1 = rhon.copy(out=self._tmp_rhon1)
        self.rhof.vector = rhon
        self.rhof1.vector = rhon1
        self._update_weighted_MM()
        self._update_all_weights()
        un = self.feec_vars[1]
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        un1 = un.copy(out=self._tmp_un1)
        un2 = un1.copy(out=self._tmp_un2)
        mn1 = mn.copy(out=self._tmp_mn1)
        tol = self._params['non_linear_tol']
        err = tol+1
        for it in range(self._params['non_linear_maxiter']):

            # Picard iteration
            if err < tol**2:
                break
            # half time step approximation
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self.uf.vector = un
            self.uf1.vector = un1
            self.rhof.vector = rhon
            self._update_linear_form_u2()

            # Compute the advection terms
            advection = self.divPirhoT.dot(
                self._linear_form_dl_drho, out=self._tmp_advection)
            advection *= dt

            rho_advection = self.divPirho.dot(
                un12, out=self._tmp_rho_advection)
            rho_advection *= dt

            # Get diff before update
            rhon_diff = rhon1.copy(out=self._tmp_rhon_diff)
            rhon_diff -= rhon
            rhon_diff += rho_advection

            # Update : m^{n+1,r+1} = m^n-advection
            mn1 = mn.copy(out=self._tmp_mn1)
            mn1 -= advection

            # Update : rho^{n+1,r+1} = rho^n-rho_avection
            rhon1 = rhon.copy(out=self._tmp_rhon1)
            rhon1 -= rho_advection

            # Inverse the mass matrix to get the velocity
            self.rhof1.vector = rhon1
            self._update_weighted_MM()
            self.pc.update_mass_operator(self._Mrho)
            self._Mrhoinv._options['x0'] = un1
            un1 = self._Mrhoinv.dot(mn1, out=self._tmp_un1)

            # get the error
            un_diff = un1.copy(out=self._tmp_un_diff)
            un_diff -= un2
            un2 = un1.copy(out=self._tmp_un2)

            err = self._get_error_picard(un_diff, rhon_diff)

        if it == self._params['non_linear_maxiter']-1:
            print(f'!!!Warning: Maximum iteration in VariationalDensityEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(rhon1, un1)

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'linear_tol': 1e-12,
                         'non_linear_tol': 1e-8,
                         'linear_maxiter': 500,
                         'non_linear_maxiter': 100,
                         'type_linear_solver': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                                ('cg', None)],
                        'non_linear_solver': ['Newton', 'Picard'],
                         'info': False,
                         'verbose': False}
        dct['physics'] = {'gamma': 5/3}
        return dct

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and `CoordinateProjector` needed to compute the bracket term"""

        from struphy.feec.projectors import L2Projector

        # Get the projector and the spaces
        P2 = self.derham.P['2']

        Xh = self.derham.Vh_fem['v']
        V3h = self.derham.Vh_fem['3']

        # Initialize the BasisProjectionOperators
        self.Pirho = BasisProjectionOperator(
            P2, Xh, [[None, None, None],
                     [None, None, None],
                     [None, None, None]],
            transposed=False, use_cache=True)

        self.PirhoT = self.Pirho.T

        hist_grid = self.derham.proj_grid_pts['2']

        hist_grid_0 = [pts.flatten()
                       for pts in hist_grid[0]]
        hist_grid_1 = [pts.flatten()
                       for pts in hist_grid[1]]
        hist_grid_2 = [pts.flatten()
                       for pts in hist_grid[2]]

        self.hist_grid_0_spans, self.hist_grid_0_bn, self.hist_grid_0_bd = self.derham.prepare_eval_tp_fixed(
            hist_grid_0)
        self.hist_grid_1_spans, self.hist_grid_1_bn, self.hist_grid_1_bd = self.derham.prepare_eval_tp_fixed(
            hist_grid_1)
        self.hist_grid_2_spans, self.hist_grid_2_bn, self.hist_grid_2_bd = self.derham.prepare_eval_tp_fixed(
            hist_grid_2)

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_0])
        self._rhof_0_values = np.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_1])
        self._rhof_1_values = np.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_2])
        self._rhof_2_values = np.zeros(grid_shape, dtype=float)

        # weighted mass matrix to go from m to u
        self._Mrho = self.WMM

        # Inverse weighted mass matrix
        if self._params['type_linear_solver'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._params['type_linear_solver'][1])
            self.pc = pc_class(self.mass_ops.Mv)

        self._Mrhoinv = inverse(self._Mrho,
                                self._params['type_linear_solver'][0],
                                pc=self.pc,
                                tol=self._params['linear_tol'],
                                maxiter=self._params['linear_maxiter'],
                                verbose=self._params['verbose'])

        integration_grid_X = [grid_1d.flatten()
                              for grid_1d in self.derham.quad_grid_pts['0']]

        self.integration_grid_X_spans, self.integration_grid_X_bn, self.integration_grid_X_bd = self.derham.prepare_eval_tp_fixed(
            integration_grid_X)

        metric = self.domain.metric(*integration_grid_X)
        self._mass_metric_term = deepcopy(metric)

        # tmps
        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid_X])
        self._rhof_values = np.zeros(grid_shape, dtype=float)

        self._full_term_mass = deepcopy(metric)

        # prepare for integration of linear form
        integration_grid_V3 = [grid_1d.flatten()
                               for grid_1d in self.derham.quad_grid_pts['3']]

        self.integration_grid_V3_spans, self.integration_grid_V3_bn, self.integration_grid_V3_bd = self.derham.prepare_eval_tp_fixed(
            integration_grid_V3)

        metric = self.domain.metric(
            *integration_grid_V3)*self.domain.jacobian_det(*integration_grid_V3)
        self._proj_u2_metric_term = deepcopy(metric)

        # Other mass matrices for newton solve
        self._M_un = WeightedMassOperator(
            self.derham.Vh_fem['v'], self.derham.Vh_fem['3'])

        self._M_un1 = WeightedMassOperator(
            self.derham.Vh_fem['3'], self.derham.Vh_fem['v'])

        self._M_drho = WeightedMassOperator(
            self.derham.Vh_fem['3'], self.derham.Vh_fem['3'])

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid_V3])

        self._Guf_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        
        self._Jd2rho_values = np.zeros(grid_shape, dtype=float)
        
        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid_X])

        self._Guf1_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]

        metric = self.domain.metric(*integration_grid_V3)
        self._mass_u_metric_term = deepcopy(metric)

        metric = self.domain.metric(*integration_grid_X)
        self._mass_u1_metric_term = deepcopy(metric)

        Jacs = BlockVectorSpace(self.derham.Vh['v'],self.derham.Vh['3'])

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)
        
        self._I3 = IdentityOperator(self.derham.Vh['3'])

        # L2-projector for V3
        self._get_L2dofs_V3 = L2Projector('L2', self.mass_ops).get_dofs

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid_V3])

        # tmps
        self._eval_dl_drho = np.zeros(grid_shape, dtype=float)

        self._uf_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
        self._uf1_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        
        self._uf_d_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid_X])
        
        self._uf1_d_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid_V3])
        if self._params['model'] == 'barotropic':
            self._rhof_values_V3 = np.zeros(grid_shape, dtype=float)
            self._rhof1_values_V3 = np.zeros(grid_shape, dtype=float)

        elif self._params['model'] == 'full':
            self._rhof_values_V3 = np.zeros(grid_shape, dtype=float)
            self._rhof1_values_V3 = np.zeros(grid_shape, dtype=float)
            self._sf_values_V3 = np.zeros(grid_shape, dtype=float)
            self._rhof_values_d_V3 = np.zeros(grid_shape, dtype=float)
            self._rhof1_values_d_V3 = np.zeros(grid_shape, dtype=float)
            self._sf_values_d_V3 = np.zeros(grid_shape, dtype=float)
            
            gam = self._params['gamma']
            metric = np.power(self.domain.jacobian_det(
                *integration_grid_V3), 2-gam)
            self._proj_rho2_metric_term = deepcopy(metric)

            metric = np.power(self.domain.jacobian_det(
                *integration_grid_V3), 1-gam)
            self._proj_drho_metric_term = deepcopy(metric)

    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._params['gamma']
        return np.power(rho, gam)*np.exp(s/rho)
    
    def __dener_drho(self, rho, s):
        """Derivative with respect to rho of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        dE(rho, s)/drho = (gamma*rho^{gamma-1} - s*rho^{gamma-2})*exp(s/rho)"""
        gam = self._params['gamma']
        return (gam *np.power(rho, gam-1)-
                s * np.power(rho, gam-2))*np.exp(s/rho)
    
    def __d2ener_drho2(self, rho, s):
        """Second derivative with respect to (rho, rho) of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        d^2E(rho, s)/drho^2 = (gamma*(gamma-1) rho^{gamma-2}- 2*s*(gamma-1)*rho^{gamma-3}+ s^2*rho^{gamma-4})*exp(s/rho)"""
        gam = self._params['gamma']
        return (gam * (gam-1) * np.power(rho, gam-2)
                - s * 2 * (gam-1) * np.power(rho, gam-3)
                + s**2 * np.power(rho, gam-4))*np.exp(s/rho)
    
    def __eta(self, delta_x):
        return 1.-np.exp(-(delta_x/1e-5)**2)

    def _update_all_weights(self,):
        """Update the weights of the `BasisProjectionOperator` appearing in the equations"""

        rhof0_values = self.rhof.eval_tp_fixed_loc(
            self.hist_grid_0_spans, self.hist_grid_0_bd, out=self._rhof_0_values)
        rhof1_values = self.rhof.eval_tp_fixed_loc(
            self.hist_grid_1_spans, self.hist_grid_1_bd, out=self._rhof_1_values)
        rhof2_values = self.rhof.eval_tp_fixed_loc(
            self.hist_grid_2_spans, self.hist_grid_2_bd, out=self._rhof_2_values)

        self.Pirho.update_weights([[rhof0_values, None, None],
                                   [None, rhof1_values, None],
                                   [None, None, rhof2_values]])

        self.PirhoT.update_weights([[rhof0_values, None, None],
                                    [None, rhof1_values, None],
                                    [None, None, rhof2_values]])

    def _update_weighted_MM(self,):
        """update the weighted mass matrix operator"""

        rhof_values = self.rhof1.eval_tp_fixed_loc(
            self.integration_grid_X_spans, self.integration_grid_X_bd, out=self._rhof_values)
        for i in range(3):
            for j in range(3):
                self._full_term_mass[i, j] = rhof_values * \
                    self._mass_metric_term[i, j]

        self.WMM.assemble([[self._full_term_mass[0, 0], self._full_term_mass[0, 1], self._full_term_mass[0, 2]],
                           [self._full_term_mass[1, 0], self._full_term_mass[1,
                                                                             1], self._full_term_mass[1, 2]],
                           [self._full_term_mass[2, 0], self._full_term_mass[2, 1], self._full_term_mass[2, 2]]],
                          verbose=False)

    def _update_linear_form_u2(self,):
        """Update the linearform representing integration in V3 against kynetic energy"""

        uf_values = self.uf.eval_tp_fixed_loc(self.integration_grid_V3_spans, [
                                              self.integration_grid_V3_bn]*3, out=self._uf_values)
        uf1_values = self.uf1.eval_tp_fixed_loc(self.integration_grid_V3_spans, [
                                                self.integration_grid_V3_bn]*3, out=self._uf1_values)

        # TODO : probably could be faster, tmp (mabe use a kernel?)
        self._eval_dl_drho *= 0.
        for i in range(3):
            for j in range(3):
                self._eval_dl_drho += uf_values[i] * \
                    self._proj_u2_metric_term[i, j]*uf1_values[j]

        self._eval_dl_drho *= 0.5

        if self._params['model'] == 'barotropic':

            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._rhof_values_V3)
            rhof1_values = self.rhof1.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._rhof1_values_V3)

            self._eval_dl_drho -= (rhof_values + rhof1_values)/2

        if self._params['model'] == 'full':
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._rhof_values_V3)
            rhof1_values = self.rhof1.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._rhof1_values_V3)

            sf_values = self.sf.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._sf_values_V3)

            delta_rho_values = rhof1_values-rhof_values
            rho_mid_values = (rhof1_values+rhof_values)/2
            e = self.__ener
            de = self.__dener_drho
            eta = self.__eta(delta_rho_values)
            self._eval_dl_drho -= self._proj_rho2_metric_term*(eta*delta_rho_values*(e(rhof1_values, sf_values)-e(rhof_values, sf_values)) /
                                                               (delta_rho_values**2+1e-40)+(1-eta)*de(rho_mid_values, sf_values))

        self._get_L2dofs_V3(self._eval_dl_drho, dofs=self._linear_form_dl_drho)

    def _get_jacobian(self,dt):
        uf_values = self.uf.eval_tp_fixed_loc(self.integration_grid_V3_spans, [
                                              self.integration_grid_V3_bn]*3, out=self._uf_d_values)
        uf1_values = self.uf1.eval_tp_fixed_loc(self.integration_grid_X_spans, [
                                                self.integration_grid_X_bn]*3, out=self._uf1_d_values)

        for i in range(3):
            self._Guf_values[i] *= 0.
            self._Guf1_values[i] *= 0.
            for j in range(3):
                self._Guf_values[i] += self._mass_u_metric_term[i,
                                                                j]*uf_values[j]
                self._Guf1_values[i] += self._mass_u1_metric_term[i,
                                                                 j]*uf1_values[j]

        if self._params['model'] == 'barotropic':
            self._M_drho = -self.mass_ops.M3/2.

        if self._params['model'] == 'full':
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._rhof_values_d_V3)
            rhof1_values = self.rhof1.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._rhof1_values_d_V3)
            sf_values = self.sf.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._sf_values_d_V3)
            delta_rho_values = rhof1_values-rhof_values
            e = self.__ener
            de = self.__dener_drho
            d2e = self.__d2ener_drho2
            eta = self.__eta(delta_rho_values)
            self._Jd2rho_values = -self._proj_drho_metric_term*(eta*(de(rhof1_values, sf_values)*delta_rho_values
                                                                     -e(rhof1_values, sf_values)+e(rhof_values, sf_values))
                                                                     /(delta_rho_values**2+1e-40)
                                                                + (1-eta)*d2e(rhof1_values, sf_values))
            self._M_drho.assemble([[self._Jd2rho_values]], verbose=False)

        self._M_un.assemble(
            [[self._Guf_values[0],   self._Guf_values[1],   self._Guf_values[2]]], verbose=False)
        self._M_un1.assemble(
            [[self._Guf1_values[0]],[self._Guf1_values[1]],[self._Guf1_values[2]]], verbose=False)
        
        self._Jacobian[0,0] = self._Mrho + dt * self.divPirhoT@self._M_un/2.
        self._Jacobian[0,1] = self._M_un1 + dt * self.divPirhoT@self._M_drho
        self._Jacobian[1,0] = dt * self.mass_ops.M3 @ self.divPirho/2.
        self._Jacobian[1,1] = self.mass_ops.M3

    def _get_error_newton(self, mn_diff, rhon_diff):
        inv_Mv = inverse(self.mass_ops.Mv, 'cg', tol=1e-16, maxiter = 1000)
        weak_un_diff = inv_Mv.dot(
            mn_diff, out=self._tmp_un_weak_diff)
        weak_rhon_diff = self.mass_ops.M3.dot(
            rhon_diff, out=self._tmp_rhon_weak_diff)
        err_rho = weak_rhon_diff.dot(rhon_diff)
        err_u = weak_un_diff.dot(mn_diff)
        return max(err_rho, err_u)
    
    def _get_error_picard(self, un_diff, rhon_diff):
        weak_un_diff = self.mass_ops.Mv.dot(
            un_diff, out=self._tmp_un_weak_diff)
        weak_rhon_diff = self.mass_ops.M3.dot(
            rhon_diff, out=self._tmp_rhon_weak_diff)
        err_rho = weak_rhon_diff.dot(rhon_diff)
        err_u = weak_un_diff.dot(un_diff)
        return max(err_rho, err_u)


class VariationalEntropyEvolve(Propagator):
    r'''Crank-Nicolson step for the evolution of the entropy terms in fluids models,

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta  
        - \int_{\hat{\Omega}} \big(\frac{\partial \hat{\rho}^3 \hat{e}}{\partial \hat{s}} \big) \nabla \cdot (\hat{s} \hat{\mathbf{v}}) \, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{s} + \nabla \cdot ( \hat{s} \hat{\mathbf{u}} ) = 0 ~ ,
        \end{align}

    where :math:`\hat{e}` depends on the chosen model. It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t} - 
        (\mathbb D \hat{\Pi}^{2}[\hat{s}_h^{n} \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3\Big( \big(\frac{\hat{\rho}_h^{n}\hat{e}(\hat{\rho}_h^{n},\hat{s}_h^{n+1})-\hat{\rho}_h^{n}\hat{e}(\hat{\rho}_h^{n},\hat{s}_h^{n})}{\hat{s}_h^{n+1}-\hat{s}_h^n} \big)\Big) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf s^{n+1}- \mathbf s^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{s}_h^{n} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n+1/2} = 0 ~ ,
        \end{align}

    where :math:`\hat{l}^3(f)` denotes the vector representing the linear form :math:`v_h \mapsto \int_{\hat{\Omega}} f(\boldsymbol \eta) v_h(\boldsymbol \eta) d \boldsymbol \eta`, that is the vector with components

    .. math::
        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    and the weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{k} = (\mathbf{u}^{k})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \, \text{for k in} \{n, n+1/2, n+1\}, \qquad \hat{s}_h^{k} = (s^{k})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \, \text{for k in} \{n, n+1/2, n+1\} \qquad \hat{\rho}_h^{n} = (\rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \.

    Parameters
    ----------
    s : psydac.linalg.stencil.Vector
        FE coefficients of a discrete field, entropy of the solution.

    u : psydac.linalg.stencil.BlockVector
        FE coefficients of a discrete vector field,velocity of the solution.

    **params : dict
        Parameters for the iterative solver, the linear solver and the model.

    '''

    def __init__(self, s, u, **params):

        super().__init__(s, u)

        # parameters
        params_default = {'linear_tol': 1e-12,
                          'non_linear_tol': 1e-8,
                          'linear_maxiter': 500,
                          'non_linear_maxiter': 100,
                          'type_linear_solver': ('pcg', 'MassMatrixDiagonalPreconditioner'),
                          'non_linear_solver': 'Newton',
                          'info': False,
                          'verbose': False,
                          'model': None,
                          'rho': None,
                          'gamma': 5/3,
                          'mass_ops': None}

        assert 'model' in params, 'model must be provided for VariationalDensityEvolve'
        assert params['model'] in ['full']
        assert 'rho' in params
        assert 'mass_ops' in params

        params = set_defaults(params, params_default)

        self._params = params

        self.WMM = params['mass_ops']

        # Femfields for the projector
        self.rhof = self.derham.create_field("rhof", "L2")
        self.sf = self.derham.create_field("sf", "L2")
        self.sf1 = self.derham.create_field("sf1", "L2")
        self.uf = self.derham.create_field("uf", "H1vec")
        self.uf1 = self.derham.create_field("uf1", "H1vec")

        # Projector
        self._initialize_projectors_and_mass()

        # gradient of the component of the vector field
        self.div = self.derham.div

        # Initialize the transport operator and transposed
        self.divPis = self.div @ self.Pis
        self.divPisT = self.PisT @ self.div.T

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn12 = s.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_sn_diff = s.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_sn_weak_diff = s.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn12 = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_s_advection = s.space.zeros()
        self._linear_form_dl_ds = s.space.zeros()

    def __call__(self, dt):
        if self._params['non_linear_solver'] == 'Newton':
            self.__call_newton(dt)
        elif self._params['non_linear_solver'] == 'Picard':
            self.__call_picard(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # Initialize variable for Newton iteration
        rho = self._params['rho']
        self.rhof.vector = rho

        sn = self.feec_vars[0]
        sn1 = sn.copy(out=self._tmp_sn1)
        self.sf.vector = sn
        self.sf1.vector = sn1

        self._update_all_weights()

        un = self.feec_vars[1]
        self.uf.vector = un
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        un1 = un.copy(out=self._tmp_un1)
        mn1 = mn.copy(out=self._tmp_mn1)
        tol = self._params['non_linear_tol']
        err = tol+1

        inv_derivative = inverse(self._Jacobian,
                                 'gmres',
                                 tol=self._params['linear_tol'],
                                 maxiter=self._params['linear_maxiter'],
                                 verbose=self._params['verbose'],
                                 recycle = True)

        for it in range(self._params['non_linear_maxiter']):

            # Newton iteration
            
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self.uf1.vector = un1
            self._update_linear_form_u2()

            # Compute the advection terms
            advection = self.divPisT.dot(
                self._linear_form_dl_ds, out=self._tmp_advection)
            advection *= dt

            s_advection = self.divPis.dot(
                un12, out=self._tmp_s_advection)
            s_advection *= dt

            # Get diff 
            sn_diff = sn1.copy(out=self._tmp_sn_diff)
            sn_diff -= sn
            sn_diff += s_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error_newton(mn_diff, sn_diff)

            if err < tol**2:
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = self.mass_ops.M3.dot(sn_diff)

            incr = inv_derivative.dot(self._tmp_f, out = self._tmp_incr)

            un1 -= incr[0]
            sn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            self.sf1.vector = sn1
            mn1 = self._Mrho.dot(un1, out = self._tmp_mn1)


        if it == self._params['non_linear_maxiter']-1:
            print(
                f'!!!Warning: Maximum iteration in VariationalDensityEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(sn1, un1)

    def __call_picard(self, dt):

        # Initialize variable for Picard iteration
        rho = self._params['rho']
        self.rhof.vector = rho

        sn = self.feec_vars[0]
        sn1 = sn.copy(out=self._tmp_sn1)
        self.sf.vector = sn
        self.sf1.vector = sn1

        self._update_all_weights()

        un = self.feec_vars[1]
        un1 = un.copy(out=self._tmp_un1)
        un2 = un1.copy(out=self._tmp_un2)
        self.uf.vector = un

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)

        self.pc.update_mass_operator(self._Mrho)

        tol = self._params['non_linear_tol']
        err = tol+1
        for it in range(self._params['non_linear_maxiter']):

            # Picard iteration
            if err < tol**2:
                break
            # half time step approximation
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self.sf.vector = sn
            self.sf1.vector = sn1

            self._update_linear_form_u2()

            # Compute the advection terms
            advection = self.divPisT.dot(
                self._linear_form_dl_ds, out=self._tmp_advection)
            advection *= dt

            s_advection = self.divPis.dot(
                un12, out=self._tmp_s_advection)
            s_advection *= dt

            # Get diff before update
            sn_diff = sn1.copy(out=self._tmp_sn_diff)
            sn_diff -= sn
            sn_diff += s_advection

            # Update : m^{n+1,r+1} = m^n-advection
            mn1 = mn.copy(out=self._tmp_mn1)
            mn1 -= advection

            # Update : rho^{n+1,r+1} = rho^n-rho_avection
            sn1 = sn.copy(out=self._tmp_sn1)
            sn1 -= s_advection

            # Inverse the mass matrix to get the velocity
            self._Mrhoinv._options['x0'] = un1
            un1 = self._Mrhoinv.dot(mn1, out=self._tmp_un1)

            # get the error
            un_diff = un1.copy(out=self._tmp_un_diff)
            un_diff -= un2
            un2 = un1.copy(out=self._tmp_un2)

            err = self._get_error_picard(un_diff, sn_diff)

            if it == self._params['non_linear_maxiter']-1:
                print(
                    f'!!!Warning: Maximum iteration in VariationalEntropyEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(sn1, un1)

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'linear_tol': 1e-12,
                         'non_linear_tol': 1e-8,
                         'linear_maxiter': 500,
                         'non_linear_maxiter': 100,
                         'type_linear_solver': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                                ('cg', None)],
                         'non_linear_solver': ['Newton', 'Picard'],
                         'info': False,
                         'verbose': False}
        dct['physics'] = {'gamma': 5/3}
        return dct

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and `CoordinateProjector` needed to compute the bracket term"""

        from struphy.feec.projectors import L2Projector

        # Get the projector and the spaces
        P2 = self.derham.P['2']

        Xh = self.derham.Vh_fem['v']

        # Initialize the BasisProjectionOperators
        self.Pis = BasisProjectionOperator(
            P2, Xh, [[None, None, None],
                     [None, None, None],
                     [None, None, None]],
            transposed=False, use_cache=True)

        self.PisT = self.Pis.T

        hist_grid = self.derham.proj_grid_pts['2']

        hist_grid_0 = [pts.flatten()
                       for pts in hist_grid[0]]
        hist_grid_1 = [pts.flatten()
                       for pts in hist_grid[1]]
        hist_grid_2 = [pts.flatten()
                       for pts in hist_grid[2]]

        self.hist_grid_0_spans, self.hist_grid_0_bn, self.hist_grid_0_bd = self.derham.prepare_eval_tp_fixed(
            hist_grid_0)
        self.hist_grid_1_spans, self.hist_grid_1_bn, self.hist_grid_1_bd = self.derham.prepare_eval_tp_fixed(
            hist_grid_1)
        self.hist_grid_2_spans, self.hist_grid_2_bn, self.hist_grid_2_bd = self.derham.prepare_eval_tp_fixed(
            hist_grid_2)

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_0])
        self._sf_0_values = np.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_1])
        self._sf_1_values = np.zeros(grid_shape, dtype=float)

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_2])
        self._sf_2_values = np.zeros(grid_shape, dtype=float)

        # weighted mass matrix to go from m to u
        self._Mrho = self.WMM

        # Inverse weighted mass matrix
        if self._params['type_linear_solver'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._params['type_linear_solver'][1])
            self.pc = pc_class(self.mass_ops.Mv)

        self._Mrhoinv = inverse(self._Mrho,
                                self._params['type_linear_solver'][0],
                                pc=self.pc,
                                tol=self._params['linear_tol'],
                                maxiter=self._params['linear_maxiter'],
                                verbose=self._params['verbose'])
        
        # For Newton solve
        self._M_ds = WeightedMassOperator(
            self.derham.Vh_fem['3'], self.derham.Vh_fem['3'])

        Jacs = BlockVectorSpace(self.derham.Vh['v'],self.derham.Vh['3'])

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)
        
        self._I3 = IdentityOperator(self.derham.Vh['3'])

        # prepare for integration of linear form
        # L2-projector for V3
        self._get_L2dofs_V3 = L2Projector('L2', self.mass_ops).get_dofs

        integration_grid_V3 = [grid_1d.flatten()
                               for grid_1d in self.derham.quad_grid_pts['3']]

        self.integration_grid_V3_spans, self.integration_grid_V3_bn, self.integration_grid_V3_bd = self.derham.prepare_eval_tp_fixed(
            integration_grid_V3)

        if self._params['model'] == 'full':
            grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid_V3])
            self._sf_values_V3 = np.zeros(grid_shape, dtype=float)
            self._sf1_values_V3 = np.zeros(grid_shape, dtype=float)
            self._rhof_values_V3 = np.zeros(grid_shape, dtype=float)
            self._rhof_values_d_V3 = np.zeros(grid_shape, dtype=float)
            self._sf1_values_d_V3 = np.zeros(grid_shape, dtype=float)
            self._sf_values_d_V3 = np.zeros(grid_shape, dtype=float)

            gam = self._params['gamma']
            metric = np.power(self.domain.jacobian_det(
                *integration_grid_V3), 2-gam)
            self._proj_rho2_metric_term = deepcopy(metric)

            metric = np.power(self.domain.jacobian_det(
                *integration_grid_V3), 1-gam)
            self._proj_ds_metric_term = deepcopy(metric)

    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._params['gamma']
        return np.power(rho, gam)*np.exp(s/rho)
    
    def __dener_ds(self, rho, s):
        """Derivative with respect to s of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        dE(rho, s)/ds = (rho^{gamma-1})*exp(s/rho)"""
        gam = self._params['gamma']
        return (np.power(rho, gam-1)) * np.exp(s/rho)
    
    def __d2ener_ds2(self, rho, s):
        """Second derivative with respect to (s, s) of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        d^2E(rho, s)/ds^2 = (rho^{gamma-2})*exp(s/rho)"""
        gam = self._params['gamma']
        return (np.power(rho, gam-2)) * np.exp(s/rho)
    
    def __eta(self, delta_x):
        return 1.-np.exp(-(delta_x/1e-5)**2)

    def _update_all_weights(self,):
        """Update the weights of the `BasisProjectionOperator` appearing in the equations"""

        sf0_values = self.sf.eval_tp_fixed_loc(
            self.hist_grid_0_spans, self.hist_grid_0_bd, out=self._sf_0_values)
        sf1_values = self.sf.eval_tp_fixed_loc(
            self.hist_grid_1_spans, self.hist_grid_1_bd, out=self._sf_1_values)
        sf2_values = self.sf.eval_tp_fixed_loc(
            self.hist_grid_2_spans, self.hist_grid_2_bd, out=self._sf_2_values)

        self.Pis.update_weights([[sf0_values, None, None],
                                 [None, sf1_values, None],
                                 [None, None, sf2_values]])

        self.PisT.update_weights([[sf0_values, None, None],
                                  [None, sf1_values, None],
                                  [None, None, sf2_values]])

    def _update_linear_form_u2(self,):
        """Update the linearform representing integration in V3 against kynetic energy"""

        if self._params['model'] == 'full':
            sf_values = self.sf.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._sf_values_V3)
            sf1_values = self.sf1.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._sf1_values_V3)

            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._rhof_values_V3)

            delta_s_values = sf1_values-sf_values
            s_mid_values = (sf1_values+sf_values)/2
            e = self.__ener
            de = self.__dener_ds
            eta = self.__eta(delta_s_values)
            eval_dl_ds = -self._proj_rho2_metric_term * (eta*delta_s_values*(e(rhof_values, sf1_values)-e(rhof_values, sf_values)) /
                                                         (delta_s_values**2+1e-40)+(1-eta)*de(rhof_values, s_mid_values))

        self._get_L2dofs_V3(eval_dl_ds, dofs=self._linear_form_dl_ds)

    def _get_jacobian(self,dt):

        if self._params['model'] == 'full':
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._rhof_values_d_V3)
            sf_values = self.sf.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._sf_values_d_V3)
            sf1_values = self.sf1.eval_tp_fixed_loc(
                self.integration_grid_V3_spans, self.integration_grid_V3_bd, out=self._sf1_values_d_V3)
            delta_s_values = sf1_values-sf_values
            e = self.__ener
            de = self.__dener_ds
            d2e = self.__d2ener_ds2
            eta = self.__eta(delta_s_values)
            self._Jd2rho_values = -self._proj_ds_metric_term*(eta*(de(rhof_values, sf1_values)*delta_s_values
                                                                     -e(rhof_values, sf1_values)+e(rhof_values, sf_values))
                                                                     /(delta_s_values**2+1e-40)
                                                                + (1-eta)*d2e(rhof_values, sf1_values))
            self._M_ds.assemble([[self._Jd2rho_values]], verbose=False)
        
        self._Jacobian[0,0] = self._Mrho
        self._Jacobian[0,1] = dt * self.divPisT@self._M_ds
        self._Jacobian[1,0] = dt * self.mass_ops.M3 @ self.divPis/2
        self._Jacobian[1,1] = self.mass_ops.M3


    def _get_error_newton(self, mn_diff, sn_diff):
        inv_Mv = inverse(self.mass_ops.Mv, 'cg', tol=1e-16, maxiter = 1000)
        weak_un_diff = inv_Mv.dot(
            mn_diff, out=self._tmp_un_weak_diff)
        weak_sn_diff = self.mass_ops.M3.dot(
            sn_diff, out=self._tmp_sn_weak_diff)
        err_rho = weak_sn_diff.dot(sn_diff)
        err_u = weak_un_diff.dot(mn_diff)
        return max(err_rho, err_u)

    def _get_error_picard(self, un_diff, sn_diff):
        weak_un_diff = self.mass_ops.Mv.dot(
            un_diff, out=self._tmp_un_weak_diff)
        weak_sn_diff = self.mass_ops.M3.dot(
            sn_diff, out=self._tmp_sn_weak_diff)
        err_s = weak_sn_diff.dot(sn_diff)
        err_u = weak_un_diff.dot(un_diff)
        return max(err_s, err_u)



class VariationalMagFieldEvolve(Propagator):
    r'''Crank-Nicolson step for the evolution of the magnetic field terms in fluids models,

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta  
        - \int_{\hat{\Omega}} \hat{\mathbf{B}}^2 \cdot G \,\nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{\mathbf{B}}^2 + \nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{u}}) = 0 ~ .
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        - (\mathbb C \hat{\Pi}^{1}[\hat{\mathbf{B}}_h^{n} \cdot \vec{\boldsymbol \Lambda}^v])^\top \mathbb M^2 B^{n+\frac{1}{2}} \big) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf b^{n+1}- \mathbf b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{\mathbf{B}}_h^{n} \cdot \vec{\boldsymbol \Lambda}^v]] \mathbf u^{n+1/2} = 0 ~ ,
        \end{align}

    and the weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{B}}_h^{n+1/2} = (\mathbf{b}^{n+1/2})^\top \vec{\boldsymbol \Lambda}^2 \in V_h^2 \, \qquad \hat{\rho}_h^{n} = (\boldsymbol \rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.


    Parameters
    ----------
    b : psydac.linalg.stencil.Vector
        FE coefficients of a discrete field, magnetic field of the solution.

    u : psydac.linalg.stencil.BlockVector
        FE coefficients of a discrete vector field,velocity of the solution.

    **params : dict
        Parameters for the iterative solver, the linear solver and the model.

    '''

    def __init__(self, b, u, **params):

        super().__init__(b, u)

        # parameters
        params_default = {'linear_tol': 1e-12,
                          'non_linear_tol': 1e-8,
                          'linear_maxiter': 500,
                          'non_linear_maxiter': 100,
                          'type_linear_solver': ('pcg', 'MassMatrixDiagonalPreconditioner'),
                          'non_linear_solver': 'Newton',
                          'info': False,
                          'verbose': False,
                          'mass_ops': None}

        assert 'mass_ops' in params

        params = set_defaults(params, params_default)

        self._params = params

        self.WMM = params['mass_ops']

        # Femfields for the projector
        self.bf = self.derham.create_field("bf", "Hdiv")

        self.uf = self.derham.create_field("uf", "H1vec")
        self.uf1 = self.derham.create_field("uf1", "H1vec")

        # Projector
        self._initialize_projectors_and_mass()

        # gradient of the component of the vector field
        self.curl = self.derham.curl

        # Initialize the transport operator and transposed
        self.curlPib = self.curl @ self.Pib
        self.curlPibT = self.PibT @ self.curl.T

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_bn_diff = b.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_bn_weak_diff = b.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_b_advection = b.space.zeros()
        self._linear_form_dl_db = b.space.zeros()

    def __call__(self, dt):
        if self._params['non_linear_solver'] == 'Newton':
            self.__call_newton(dt)
        elif self._params['non_linear_solver'] == 'Picard':
            self.__call_picard(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # Initialize variable for Newton iteration

        bn = self.feec_vars[0]
        bn1 = bn.copy(out=self._tmp_bn1)
        self.bf.vector = bn
        self._update_all_weights()

        un = self.feec_vars[1]
        self.uf.vector = un
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        un1 = un.copy(out=self._tmp_un1)
        mn1 = mn.copy(out=self._tmp_mn1)
        tol = self._params['non_linear_tol']
        err = tol+1

        inv_derivative = inverse(self._Jacobian,
                                 'gmres',
                                 tol=self._params['linear_tol'],
                                 maxiter=self._params['linear_maxiter'],
                                 verbose=self._params['verbose'],
                                 recycle = True)

        for it in range(self._params['non_linear_maxiter']):

            # Newton iteration
            # half time step approximation
            bn12 = bn.copy(out=self._tmp_bn12)
            bn12 += bn1
            bn12 *= 0.5

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self.uf1.vector = un1
            self._update_linear_form_u2()

            # Compute the advection terms
            advection = self.curlPibT.dot(
                self._linear_form_dl_db, out=self._tmp_advection)
            advection *= dt

            b_advection = self.curlPib.dot(
                un12, out=self._tmp_b_advection)
            b_advection *= dt

            # Get diff 
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error_newton(mn_diff, bn_diff)

            if err < tol**2:
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = self.mass_ops.M2.dot(bn_diff)

            incr = inv_derivative.dot(self._tmp_f, out = self._tmp_incr)

            un1 -= incr[0]
            bn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.dot(un1, out = self._tmp_mn1)

        if it == self._params['non_linear_maxiter']-1:
            print(
                f'!!!Warning: Maximum iteration in VariationalDensityEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(bn1, un1)

    def __call_picard(self, dt):

        # Initialize variable for Picard iteration

        bn = self.feec_vars[0]
        bn1 = bn.copy(out=self._tmp_bn1)
        self.bf.vector = bn
        self._update_all_weights()

        un = self.feec_vars[1]
        un1 = un.copy(out=self._tmp_un1)
        un2 = un1.copy(out=self._tmp_un2)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)

        self.pc.update_mass_operator(self._Mrho)

        tol = self._params['non_linear_tol']
        err = tol+1
        for it in range(self._params['non_linear_maxiter']):

            # Picard iteration
            if err < tol**2:
                break
            # half time step approximation
            bn12 = bn.copy(out=self._tmp_bn12)
            bn12 += bn1
            bn12 *= 0.5

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            self._update_linear_form_u2()

            # Compute the advection terms
            advection = self.curlPibT.dot(
                self._linear_form_dl_db, out=self._tmp_advection)
            advection *= dt

            b_advection = self.curlPib.dot(
                un12, out=self._tmp_b_advection)
            b_advection *= dt

            # Get diff before update
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection

            # Update : m^{n+1,r+1} = m^n-advection
            mn1 = mn.copy(out=self._tmp_mn1)
            mn1 -= advection

            # Update : b^{n+1,r+1} = b^n-b_avection
            bn1 = bn.copy(out=self._tmp_bn1)
            bn1 -= b_advection

            # Inverse the mass matrix to get the velocity
            self._Mrhoinv._options['x0'] = un1
            un1 = self._Mrhoinv.dot(mn1, out=self._tmp_un1)

            # get the error
            un_diff = un1.copy(out=self._tmp_un_diff)
            un_diff -= un2
            un2 = un1.copy(out=self._tmp_un2)

            err = self._get_error_picard(un_diff, bn_diff)

            if it == self._params['non_linear_maxiter']-1:
                print(
                    f'!!!Warning: Maximum iteration in VariationalMagFieldEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(bn1, un1)

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'linear_tol': 1e-12,
                         'non_linear_tol': 1e-8,
                         'linear_maxiter': 500,
                         'non_linear_maxiter': 100,
                         'type_linear_solver': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                                ('cg', None)],
                         'non_linear_solver': ['Newton', 'Picard'],
                         'info': False,
                         'verbose': False, }
        return dct

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        # Get the projector and the spaces
        P1 = self.derham.P['1']

        Xh = self.derham.Vh_fem['v']

        # Initialize the BasisProjectionOperators
        self.Pib = BasisProjectionOperator(
            P1, Xh, [[None, None, None],
                     [None, None, None],
                     [None, None, None]],
            transposed=False, use_cache=True)

        self.PibT = self.Pib.T

        hist_grid = self.derham.proj_grid_pts['1']

        hist_grid_0 = [pts.flatten()
                       for pts in hist_grid[0]]
        hist_grid_1 = [pts.flatten()
                       for pts in hist_grid[1]]
        hist_grid_2 = [pts.flatten()
                       for pts in hist_grid[2]]

        self.hist_grid_0_spans, self.hist_grid_0_bn, self.hist_grid_0_bd = self.derham.prepare_eval_tp_fixed(
            hist_grid_0)
        self.hist_grid_1_spans, self.hist_grid_1_bn, self.hist_grid_1_bd = self.derham.prepare_eval_tp_fixed(
            hist_grid_1)
        self.hist_grid_2_spans, self.hist_grid_2_bn, self.hist_grid_2_bd = self.derham.prepare_eval_tp_fixed(
            hist_grid_2)

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_0])
        self._bf0_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        self.hist_grid_0_b = [[self.hist_grid_0_bn[0], self.hist_grid_0_bd[1], self.hist_grid_0_bd[2]],
                              [self.hist_grid_0_bd[0], self.hist_grid_0_bn[1],
                                  self.hist_grid_0_bd[2]],
                              [self.hist_grid_0_bd[0], self.hist_grid_0_bd[1], self.hist_grid_0_bn[2]]]
        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_1])
        self._bf1_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        self.hist_grid_1_b = [[self.hist_grid_1_bn[0], self.hist_grid_1_bd[1], self.hist_grid_1_bd[2]],
                              [self.hist_grid_1_bd[0], self.hist_grid_1_bn[1],
                                  self.hist_grid_1_bd[2]],
                              [self.hist_grid_1_bd[0], self.hist_grid_1_bd[1], self.hist_grid_1_bn[2]]]

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_2])
        self._bf2_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        self.hist_grid_2_b = [[self.hist_grid_2_bn[0], self.hist_grid_2_bd[1], self.hist_grid_2_bd[2]],
                              [self.hist_grid_2_bd[0], self.hist_grid_2_bn[1],
                                  self.hist_grid_2_bd[2]],
                              [self.hist_grid_2_bd[0], self.hist_grid_2_bd[1], self.hist_grid_2_bn[2]]]

        # weighted mass matrix to go from m to u
        self._Mrho = self.WMM

        # Inverse weighted mass matrix
        if self._params['type_linear_solver'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._params['type_linear_solver'][1])
            self.pc = pc_class(self.mass_ops.Mv)

        self._Mrhoinv = inverse(self._Mrho,
                                self._params['type_linear_solver'][0],
                                pc=self.pc,
                                tol=self._params['linear_tol'],
                                maxiter=self._params['linear_maxiter'],
                                verbose=self._params['verbose'])
        
        # For Newton solve
        self._M_db = WeightedMassOperator(
            self.derham.Vh_fem['2'], self.derham.Vh_fem['2'])

        Jacs = BlockVectorSpace(self.derham.Vh['v'],self.derham.Vh['2'])

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)
        
        self._I2 = IdentityOperator(self.derham.Vh['2'])

    def _update_all_weights(self,):
        """Update the weights of the `BasisProjectionOperator` appearing in the equations"""

        bf0_values = self.bf.eval_tp_fixed_loc(
            self.hist_grid_0_spans, self.hist_grid_0_b, out=self._bf0_values)
        bf1_values = self.bf.eval_tp_fixed_loc(
            self.hist_grid_1_spans, self.hist_grid_1_b, out=self._bf1_values)
        bf2_values = self.bf.eval_tp_fixed_loc(
            self.hist_grid_2_spans, self.hist_grid_2_b, out=self._bf2_values)

        self.Pib.update_weights([[None, -bf0_values[2], bf0_values[1]],
                                 [bf1_values[2], None, -bf1_values[0]],
                                 [-bf2_values[1], bf2_values[0], None]])

        self.PibT.update_weights([[None, -bf0_values[2], bf0_values[1]],
                                  [bf1_values[2], None, -bf1_values[0]],
                                  [-bf2_values[1], bf2_values[0], None]])

    def _update_linear_form_u2(self,):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        M2 = -self.mass_ops.M2
        wb = M2.dot(self._tmp_bn12, out=self._linear_form_dl_db)

    def _get_error_newton(self, mn_diff, bn_diff):
        inv_Mv = inverse(self.mass_ops.Mv, 'cg', tol=1e-16, maxiter = 1000)
        weak_un_diff = inv_Mv.dot(
            mn_diff, out=self._tmp_un_weak_diff)
        weak_bn_diff = self.mass_ops.M2.dot(
            bn_diff, out=self._tmp_bn_weak_diff)
        err_b = weak_bn_diff.dot(bn_diff)
        err_u = weak_un_diff.dot(mn_diff)
        return max(err_b, err_u)

    def _get_error_picard(self, un_diff, bn_diff):
        weak_un_diff = self.mass_ops.Mv.dot(
            un_diff, out=self._tmp_un_weak_diff)
        weak_bn_diff = self.mass_ops.M2.dot(
            bn_diff, out=self._tmp_bn_weak_diff)
        err_b = weak_bn_diff.dot(bn_diff)
        err_u = weak_un_diff.dot(un_diff)
        return max(err_b, err_u)
    
    def _get_jacobian(self,dt):

        self._M_db = -self.mass_ops.M2/2.
        
        self._Jacobian[0,0] = self._Mrho
        self._Jacobian[0,1] = dt * self.curlPibT@self._M_db
        self._Jacobian[1,0] = dt * self.mass_ops.M2 @ self.curlPib/2
        self._Jacobian[1,1] = self.mass_ops.M2


class TimeDependentSource(Propagator):
    r'''Propagates a source term :math:`S(t) \in V_h^n` of the form

    .. math::

        S(t) = \sum_{ijk} c_{ijk} \Lambda^n_{ijk} * h(\omega t)\,,

    where :math:`h(\omega t)` is one of the functions in Notes.

    Notes
    -----

    * :math:`h(\omega t) = \cos(\omega t)` (default)
    * :math:`h(\omega t) = \sin(\omega t)` 

    Parameters
    ----------
    c : psydac.linalg.stencil.StencilVector or psydac.linalg.block.BlockVector
        FE coefficients at t=0.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, c, omega=1., hfun='cos'):

        super().__init__(c)

        if hfun == 'cos':
            def hfun(t):
                return np.cos(omega*t)
        elif hfun == 'sin':
            def hfun(t):
                return np.sin(omega*t)
        else:
            raise NotImplementedError(f'{hfun = } not implemented.')

        self._hfun = hfun

    def __call__(self, dt):

        print(f'{self.time_state[0] = }')
        if self.time_state[0] == 0.:
            self._c0 = self.feec_vars[0].copy()
            print('Initial source coeffs set.')

        # new coeffs
        cn1 = self._c0 * self._hfun(self.time_state[0])

        # write new coeffs into self.feec_vars
        max_dc = self.feec_vars_update(cn1)

    @classmethod
    def options(cls):
        dct = {}
        dct['omega'] = 1.
        dct['hfun'] = ['cos', 'sin']
        return dct
