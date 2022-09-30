from numpy import array

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

import numpy as np

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.particles_to_grid import Accumulator
from struphy.pic.pusher import Pusher

from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert
from struphy.psydac_api import preconditioner
from struphy.psydac_api.linear_operators import LinOpWithTransp

from struphy.psydac_api.utilities import apply_essential_bc_to_array
    

class StepMaxwell(Propagator):
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

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, e, b, derham, mass_ops, params):

        assert isinstance(e, BlockVector)
        assert isinstance(b, BlockVector)

        self._e = e
        self._b = b
        self._info = params['info']
        
        # Define block matrix [[A B], [C I]] (without time step size dt in the diangonals)
        _A = mass_ops.M1
        
        self._B = Multiply(-1./2., Compose(derham.curl.transpose(), mass_ops.M2)) # no dt
        self._C = Multiply( 1./2., derham.curl) # no dt
        
        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(derham, 'V1', mass_ops._fun_M1)

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)
        
        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_type=params['type'], 
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

    @property
    def variables(self):
        return [self._e, self._b]

    def __call__(self, dt):

        # current variables
        en = self.variables[0]
        bn = self.variables[1]

        # allocate temporary FemFields _e, _b during solution
        _e, info = self._schur_solver(en, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_e + en)

        # write new coeffs into Propagator.variables
        de, db = self.in_place_update(_e, _b)

        if self._info:
            print('Status     for Push_maxwell_psydac:', info['success'])
            print('Iterations for Push_maxwell_psydac:', info['niter'])
            print('Maxdiff e1 for Push_maxwell_psydac:', max(de))
            print('Maxdiff b2 for Push_maxwell_psydac:', max(db))
            print()

        
class StepShearAlfvénHcurl(Propagator):
    r'''Crank-Nicolson step for shear Alfvén part in MHD equations.

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_1)^{-1} \mathcal {T^1}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^1} (\mathbb M^n_1)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_1}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 1-form.

        b : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.
            
        mhd_ops : struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators
            Linear MHD operators from struphy.psydac_api.mhd_ops_pure_psydac.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, u, b, derham, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)
        assert isinstance(b, BlockVector)

        self._u = u
        self._b = b
        self._bc = derham.bc
        self._info = params['info']
        self._rank = derham.comm.Get_rank()

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M1n

        self._B = Multiply(-1/2., Compose(mhd_ops.T1T, derham.curl.transpose(), mass_ops.M2))
        self._C = Multiply( 1/2., Compose(derham.curl, mhd_ops.T1))
        
        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(derham, 'V1', mass_ops._fun_M1n)
        
        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)
        
        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_type=params['type'], 
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

    @property
    def variables(self):
        return self._u, self._b

    def __call__(self, dt):

        # current variables
        un = self.variables[0]
        bn = self.variables[1]

        # allocate temporary FemFields _u, _b during solution
        _u, self._info = self._schur_solver(un, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_u + un)

        # write new coeffs into Propagator.variables
        du, db = self.in_place_update(_u, _b)

        if self._info and self._rank ==0:
            print('Status     for Push_shear_alfvén:', self._info['success'])
            print('Iterations for Push_shear_alfvén:', self._info['niter'])
            print('Maxdiff u1 for Push_shear_alfvén:', max(du))
            print('Maxdiff b2 for Push_shear_alfvén:', max(db))
            print()


class StepShearAlfvénHdiv(Propagator):
    r'''Crank-Nicolson step for shear Alfvén part in MHD equations:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_2)^{-1} \mathcal {T^2}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^2} (\mathbb M^n_2)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_2}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        b : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.
            
        mhd_ops : struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators
            Linear MHD operators from struphy.psydac_api.mhd_ops_pure_psydac.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, u, b, derham, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)
        assert isinstance(b, BlockVector)

        self._u = u
        self._b = b
        self._bc = derham.bc
        self._info = params['info']
        self._rank = derham.comm.Get_rank()
        
        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M2n

        self._B = Multiply(-1/2., Compose(mhd_ops.T2T, derham.curl.transpose(), mass_ops.M2))
        self._C = Multiply( 1/2., Compose(derham.curl, mhd_ops.T2))
        
        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(derham, 'V2', mass_ops._fun_M2n)

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)
        
        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_type=params['type'], 
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])
        
    @property
    def variables(self):
        return self._u, self._b

    def __call__(self, dt):

        # current variables
        un = self.variables[0]
        bn = self.variables[1]

        # allocate temporary FemFields _u, _b during solution
        _u, self._info = self._schur_solver(un, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_u + un)

        # write new coeffs into Propagator.variables
        du, db = self.in_place_update(_u, _b)

        if self._info and self._rank == 0:
            print('Status     for Push_shear_alfvén:', self._info['success'])
            print('Iterations for Push_shear_alfvén:', self._info['niter'])
            print('Maxdiff u2 for Push_shear_alfvén:', max(du))
            print('Maxdiff b2 for Push_shear_alfvén:', max(db))
            print()


class StepShearAlfvénH1vec(Propagator):
    r'''Crank-Nicolson step for shear Alfvén part in MHD equations:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_v)^{-1} \mathcal {T^0}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^0} (\mathbb M^n_v)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_v}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete vector field (0-form discretization in each component).

        b : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.
            
        mhd_ops : struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators
            Linear MHD operators from struphy.psydac_api.mhd_ops_pure_psydac.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, u, b, derham, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)
        assert isinstance(b, BlockVector)

        self._u = u
        self._b = b
        self._bc = derham.bc
        self._info = params['info']
        self._rank = derham.comm.Get_rank()
        
        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.Mvn

        self._B = Multiply(-1/2., Compose(mhd_ops.T0T, derham.curl.transpose(), mass_ops.M2))
        self._C = Multiply( 1/2., Compose(derham.curl, mhd_ops.T0))
        
        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(derham, 'V0vec', mass_ops._fun_Mvn)
        
        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)
        
        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_type=params['type'], 
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

    @property
    def variables(self):
        return self._u, self._b

    def __call__(self, dt):

        # current variables
        un = self.variables[0]
        bn = self.variables[1]

        # allocate temporary FemFields _u, _b during solution
        _u, self._info = self._schur_solver(un, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_u + un)

        # write new coeffs into Propagator.variables
        du, db = self.in_place_update(_u, _b)

        if self._info and self._rank == 0:
            print('Status     for Push_shear_alfvén:', self._info['success'])
            print('Iterations for Push_shear_alfvén:', self._info['niter'])
            print('Maxdiff uv for Push_shear_alfvén:', max(du))
            print('Maxdiff b2 for Push_shear_alfvén:', max(db))
            print()


class StepMagnetosonicHcurl(Propagator):
    r'''Crank-Nicolson step for magnetosonic part in MHD equations:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1} - \mathbf p^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_1)^{-1} {\mathcal U^1}^\top \mathbb D^\top \mathbb M_3 \\ - \mathbb D \mathcal S^1 - (\gamma - 1) \mathcal K^1 \mathbb D \mathcal U^1 & 0 \end{bmatrix} 
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1} + \mathbf p^n) \end{bmatrix} + \begin{bmatrix} \Delta t (\mathbb M^n_1)^{-1} \mathbb M^J_1 \mathbf b^n \\ 0 \end{bmatrix},

    based on the :ref:`Schur complement <schur_solver>`.
    
    Decoupled density update:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathbb D \mathcal Q^1 (\mathbf u^{n+1} + \mathbf u^n) \,.

    Parameters
    ---------- 
        n : psydac.linalg.block.StencilVector
            FE coefficients of a discrete 3-form.
        
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete vector field 1-form.

        p : psydac.linalg.block.StencilVector
            FE coefficients of a discrete 3-form.
            
        b : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.
            
        mhd_ops : struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators
            Linear MHD operators from struphy.psydac_api.mhd_ops_pure_psydac.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, n, u, p, b, derham, mass_ops, mhd_ops, params):

        assert isinstance(n, StencilVector)
        assert isinstance(u, BlockVector)
        assert isinstance(p, StencilVector)
        assert isinstance(b, BlockVector)

        self._n = n
        self._u = u
        self._p = p
        self._b = b
        self._bc = derham.bc
        self._info = params['info']
        self._rank = derham.comm.Get_rank()
        
        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M1n

        self._B = Multiply(-1/2., Compose(mhd_ops.U1T, derham.div.transpose(), mass_ops.M3))
        self._C = Multiply( 1/2., Sum(Compose(derham.div, mhd_ops.S1), Multiply(2/3, Compose(mhd_ops.K1, derham.div, mhd_ops.U1))))
        self._MJ = mass_ops.M1J
        self._Q  = mhd_ops.Q1
            
        self._DIV = derham.div
        
        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(derham, 'V1', mass_ops._fun_M1n)

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)
        
        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_type=params['type'], 
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

    @property
    def variables(self):
        return self._n, self._u, self._p, self._b

    def __call__(self, dt):

        # current variables
        nn = self.variables[0]
        un = self.variables[1]
        pn = self.variables[2]
        bn = self.variables[3]

        # allocate temporary FemFields _u, _b during solution
        _u, self._info = self._schur_solver(un, self._B.dot(pn) - self._MJ.dot(bn)/2, dt)
        _p = pn - dt*self._C.dot(_u + un)
        _n = nn - dt/2*self._DIV.dot(self._Q.dot(_u + un))
        _b = 1*bn
        
        # write new coeffs into Propagator.variables
        dn, du, dp, db = self.in_place_update(_n, _u, _p, _b)

        if self._info and self._rank == 0:
            print('Status     for Push_magnetosonic:', self._info['success'])
            print('Iterations for Push_magnetosonic:', self._info['niter'])
            print('Maxdiff n3 for Push_magnetosonic:', max(dn))
            print('Maxdiff u1 for Push_magnetosonic:', max(du))
            print('Maxdiff p3 for Push_magnetosonic:', max(dp))
            print()


class StepMagnetosonicHdiv(Propagator):
    r'''Crank-Nicolson step for magnetosonic part in MHD equations:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1} - \mathbf p^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_2)^{-1} \mathbb D^\top \mathbb M_3 \\ - \mathbb D \mathcal S^2 - (\gamma - 1) \mathcal K^2 \mathbb D & 0 \end{bmatrix} 
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1} + \mathbf p^n) \end{bmatrix} + \begin{bmatrix} \Delta t (\mathbb M^n_2)^{-1} \mathbb M^J_2 \mathbf b^n \\ 0 \end{bmatrix},

    based on the :ref:`Schur complement <schur_solver>`.
    
    Decoupled density update:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathbb D \mathcal Q^2 (\mathbf u^{n+1} + \mathbf u^n) \,.

    Parameters
    ---------- 
        n : psydac.linalg.block.StencilVector
            FE coefficients of a discrete 3-form.
        
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        p : psydac.linalg.block.StencilVector
            FE coefficients of a discrete 3-form.
            
        b : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.
            
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.
            
        mhd_ops : struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators
            Linear MHD operators from struphy.psydac_api.mhd_ops_pure_psydac.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, n, u, p, b, derham, mass_ops, mhd_ops, params):

        assert isinstance(n, StencilVector)
        assert isinstance(u, BlockVector)
        assert isinstance(p, StencilVector)
        assert isinstance(b, BlockVector)

        self._n = n
        self._u = u
        self._p = p
        self._b = b
        self._bc = derham.bc
        self._info = params['info']
        
        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M2n

        self._B = Multiply(-1/2., Compose(derham.div.transpose(), mass_ops.M3))
        self._C = Multiply( 1/2., Sum(Compose(derham.div, mhd_ops.S2), Multiply(2/3, Compose(mhd_ops.K2, derham.div))))

        self._MJ = mass_ops.M2J
        self._Q  = mhd_ops.Q2
            
        self._DIV = derham.div
        
        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(derham, 'V2', mass_ops._fun_M2n)
        
        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)
        
        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_type=params['type'], 
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

    @property
    def variables(self):
        return self._n, self._u, self._p, self._b

    def __call__(self, dt):

        # current variables
        nn = self.variables[0]
        un = self.variables[1]
        pn = self.variables[2]
        bn = self.variables[3]

        # allocate temporary FemFields _u, _b during solution
        _u, info = self._schur_solver(un, self._B.dot(pn) - self._MJ.dot(bn)/2, dt)
        _p = pn - dt*self._C.dot(_u + un)
        _n = nn - dt/2*self._DIV.dot(self._Q.dot(_u + un))
        _b = 1*bn

        # write new coeffs into Propagator.variables
        dn, du, dp, db = self.in_place_update(_n, _u, _p, _b)

        if self._info and self._rank == 0:
            print('Status     for Push_magnetosonic:', info['success'])
            print('Iterations for Push_magnetosonic:', info['niter'])
            print('Maxdiff n3 for Push_magnetosonic:', max(dn))
            print('Maxdiff u2 for Push_magnetosonic:', max(du))
            print('Maxdiff p3 for Push_magnetosonic:', max(dp))
            print()


class StepMagnetosonicH1vec(Propagator):
    r'''Crank-Nicolson step for magnetosonic part in MHD equations:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1} - \mathbf p^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_v)^{-1} {\mathcal J^0}^\top \mathbb D^\top \mathbb M_3 \\ - \mathbb D \mathcal S^0 - (\gamma - 1) \mathcal K^0 \mathbb D \mathcal J^0 & 0 \end{bmatrix} 
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1} + \mathbf p^n) \end{bmatrix} + \begin{bmatrix} \Delta t (\mathbb M^n_v)^{-1} \mathbb M^J_v \mathbf b^n \\ 0 \end{bmatrix},

    based on the :ref:`Schur complement <schur_solver>`.
    
    Decoupled density update:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathbb D \mathcal Q^0 (\mathbf u^{n+1} + \mathbf u^n) \,.

    Parameters
    ---------- 
        n : psydac.linalg.block.StencilVector
            FE coefficients of a discrete 3-form.
        
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete vector field (0-form discretization in each component).

        p : psydac.linalg.block.StencilVector
            FE coefficients of a discrete 3-form.
            
        b : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.
            
        mhd_ops : struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators
            Linear MHD operators from struphy.psydac_api.mhd_ops_pure_psydac.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, n, u, p, b, derham, mass_ops, mhd_ops, params):

        assert isinstance(n, StencilVector)
        assert isinstance(u, BlockVector)
        assert isinstance(p, StencilVector)
        assert isinstance(b, BlockVector)

        self._n = n
        self._u = u
        self._p = p
        self._b = b
        self._bc = derham.bc
        self._info = params['info']
        
        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.Mvn

        self._B = Multiply(-1/2., Compose(mhd_ops.J0T, derham.div.transpose(), mass_ops.M3))
        self._C = Multiply( 1/2., Sum(Compose(derham.div, mhd_ops.S0), Multiply(2/3, Compose(mhd_ops.K0, derham.div, mhd_ops.J0))))
        self._MJ = mass_ops.MvJ
        self._Q  = mhd_ops.Q0
            
        self._DIV = derham.div
        
        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(derham, 'V0vec', mass_ops._fun_Mvn)

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)
        
        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_type=params['type'], 
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

    @property
    def variables(self):
        return self._n, self._u, self._p, self._b

    def __call__(self, dt):

        # current variables
        nn = self.variables[0]
        un = self.variables[1]
        pn = self.variables[2]
        bn = self.variables[3]

        # allocate temporary FemFields _u, _b during solution
        _u, info = self._schur_solver(un, self._B.dot(pn) - self._MJ.dot(bn)/2, dt)
        _p = pn - dt*self._C.dot(_u + un)
        _n = nn - dt/2*self._DIV.dot(self._Q.dot(_u + un))
        _b = 1*bn
        
        # write new coeffs into Propagator.variables
        dn, du, dp, db = self.in_place_update(_n, _u, _p, _b)

        if self._info and self._rank == 0:
            print('Status     for Push_magnetosonic:', info['success'])
            print('Iterations for Push_magnetosonic:', info['niter'])
            print('Maxdiff n3 for Push_magnetosonic:', max(dn))
            print('Maxdiff uv for Push_magnetosonic:', max(du))
            print('Maxdiff p3 for Push_magnetosonic:', max(dp))
            print()

class StepEfieldWeights(Propagator):
    r'''Solve the following Crank-Nicolson step

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right) \\
            \mathbf{W}^{n+1} - \mathbf{W}^n
        \end{bmatrix}
        = 
        \begin{bmatrix}
            0 & - \mathbb{H} \\
            - \mathbb{A} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n
            \mathbf{W}^{n+1} + \mathbf{W}^n
        \end{bmatrix}

    based on the :ref:`Schur complement <schur_solver>` where

    .. math::
        \mathbb{A} & = - \frac{\Delta t}{2} \hat{\mathbf{F}}_0 \mathbb{V} \overline{DL}^T \left( \mathbb{P}^1 \right)^T \,,
        \mathbb{H} & = \frac{\Delta t}{2} \mathbb{P}^1 \overline{G} \left( \mathbb{V} \right)^T \hat{\mathbf{F}}_0 \,.

    make up the accumulation matrix :math:`\mathbb{H} \mathbb{A}` .

    Parameters
    ---------- 
        e : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        particles : struphy.pic.particles.Particles6D
            Particles object.

        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.

        params : dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, domain, derham, e, particles, mass_ops, params_bs, params_solver):
        
        assert isinstance(e, BlockVector)

        # Read out relevant parameters for Accumulator object
        self.f0_spec   = params_bs['type']
        self.moms_spec = params_bs['moms_spec']
        self.f0_params = params_bs['moms_params']
        # raise NotImplementedError('Parameters are not correct yet!')

        # Initialize Accumulator object
        self._accum = Accumulator(domain, derham, 'Hcurl', 'linear_vlasov_maxwell',
                                  self.f0_spec, array(self.moms_spec), array(self.f0_params), do_vector=True)

        self._e = e
        self._particles = particles
        self._info = params_solver['info']

        self._domain = domain
        self._derham = derham

        self._accum.accumulate(particles.markers)

        # ================================
        # ========= Schur Solver =========
        # ================================

        # Preconditioner
        if params_solver['pc'] == None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params_solver["pc"])
            self._pc = pc_class(derham, 'V1', mass_ops._fun_M1)

        r'''
        .. math::

            \begin{bmatrix}
                \mathbb{M}_1 & \mathbb{H} \\
                \mathbb{A} & \mathbb{1}
            \end{bmatrix}
            \begin{bmatrix}
                \mathbf{e}^{n+1} \\
                \mathbf{W}^{n+1}
            \end{bmatrix}
            =
            \begin{bmatrix}
                \mathbb{M}_1 & -\mathbb{H} \\
                -\mathbb{A} & \mathbb{1}
            \end{bmatrix}
            \begin{bmatrix}
                \mathbf{e}^{n} \\
                \mathbf{W}^{n}
            \end{bmatrix}
        '''
        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M1
        _BC = self._accum.matrix

        # Instantiate Schur solver
        self._schur_solver = SchurSolver(_A, _BC, pc=self._pc, solver_type=params_solver['type'], 
                                         tol=params_solver['tol'], maxiter=params_solver['maxiter'],
                                         verbose=params_solver['verbose'])

        self._pusher = Pusher(derham, domain, 'push_weights_with_efield')

    @property
    def variables(self):
        return [self._e, self._particles]

    def __call__(self, dt):

        # current variables
        en = self.variables[0]

        self._accum.accumulate(self._particles.markers)

        # Update Schur solver
        self._schur_solver.BC = self._accum.matrix

        # allocate temporary FemFields _e during solution
        _e, info = self._schur_solver(en, self._accum.vector, dt)

        # write new coeffs into Propagator.variables
        de = self.in_place_update(_e)
        self._pusher(self._particles, dt,
                     en.blocks[0]._data, en.blocks[1]._data, en.blocks[2]._data,
                     self.f0_spec, array(self.moms_spec), array(self.f0_params))

        # TODO: Implement info for weights as well
        if self._info:
            print('Status     for StepEfieldWeights:', info['success'])
            print('Iterations for StepEfieldWeights:', info['niter'])
            print('Maxdiff e1 for StepEfieldWeights:', max(de))
            print()


class StepStaticEfield(Propagator):
    r'''Solve the following system

    .. math::

        \frac{\text{d} \mathbf{\eta}_p}{\text{d} t} & = DL^{-1} \mathbf{v}_p \,,
        \frac{\text{d} \mathbf{v}_p}{\text{d} t} & = DL^{-T} \mathbf{E}_0

    which is solved by an average discrete gradient method, implicitly iterating
    over :math:`k` (for every particle :math:`p`):

    .. math::
    
        \mathbf{\eta}^{n+1}_{k+1} = \mathbf{\eta}^n + \frac{\Delta t}{2} DL^{-1}
        \left( \frac{\mathbf{\eta}^{n+1}_k + \mathbf{\eta}^n }{2} \right) \left( \mathbf{v}^{n+1}_k + \mathbf{v}^n \right) \,,
        \mathbf{v}^{n+1}_{k+1} = \mathbf{v}^n + \Delta t DL^{-1}\left(\mathbf{\eta}^n\right)
        \int_0^1 \left[ \mathbb{\Lambda}\left( \eta^n + \tau (\mathbf{\eta}^{n+1}_k - \mathbf{\eta}^n) \right) \right]^T \mathbf{e}_0 \, \text{d} \tau

    Parameters
    ---------- 
        e : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        particles : struphy.pic.particles.Particles6D
            Particles object.

        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.

        params : dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, domain, derham, particles, e_background):
        from numpy import polynomial, floor

        self._domain = domain
        self._derham = derham
        self._particles = particles
        self._e_bg = e_background

        pn1 = derham.p[0]
        pd1 = pn1 - 1
        pn2 = derham.p[1]
        pd2 = pn2 - 1
        pn3 = derham.p[2]
        pd3 = pn3 - 1

        # number of quadrature points in direction 1
        n_quad1 = int(floor(pd1 * pn2 * pn3 / 2 + 1))
        # number of quadrature points in direction 2
        n_quad2 = int(floor(pn1 * pd2 * pn3 / 2 + 1))
        # number of quadrature points in direction 3
        n_quad3 = int(floor(pn1 * pn2 * pd3 / 2 + 1))

        # get quadrature weights and locations
        self._loc1, self._weight1 = polynomial.legendre.leggauss(n_quad1)
        self._loc2, self._weight2 = polynomial.legendre.leggauss(n_quad2)
        self._loc3, self._weight3 = polynomial.legendre.leggauss(n_quad3)

        self._pusher = Pusher(derham, domain, 'push_x_v_static_efield')


    @property
    def variables(self):
        return [self._particles]

    def __call__(self, dt):
        self._pusher(self._particles, dt,
                     self._loc1, self._loc2, self._loc3, self._weight1, self._weight2, self._weight3,
                     self._e_bg.blocks[0]._data, self._e_bg.blocks[1]._data, self._e_bg.blocks[2]._data,
                     array([1e-10, 1e-10]), 100)


class StepStaticBfield(Propagator):
    r'''Solve the following system

    .. math::

        \frac{\text{d} \mathbf{\eta}_p}{\text{d} t} & = 0 \,,
        \frac{\text{d} \mathbf{v}_p}{\text{d} t} & = \mathbf{v}_p \times \left[ \frac{1}{\text{det}(DL)} DL \mathbf{B}_0 \right]

    Parameters
    ---------- 
        e : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        particles : struphy.pic.particles.Particles6D
            Particles object.

        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.

        params : dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, domain, derham, particles, b_background):

        self._domain = domain
        self._derham = derham
        self._particles = particles
        self._b_bg = b_background
        
        self._pusher = Pusher(derham, domain, 'push_vxb_analytic')

    @property
    def variables(self):
        return [self._particles]

    def __call__(self, dt):
        self._pusher.push(self._particles, dt,
                          self._b_bg.blocks[0]._data, self._b_bg.blocks[1]._data, self._b_bg.blocks[2]._data)
    

class StepPressurecouplingHcurl(Propagator):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_1)^{-1} V^\top (\bar {\mathcal X}^1)^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X}^1 V (\mathbb M^n_1)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_1}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 1-form.

        particles : struphy.pic.particles.Particles6D

        domain : struphy.geometry.base.Domain
            Infos regarding mapping.
            
        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.
            
        mhd_ops : struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators
            Linear MHD operators from struphy.psydac_api.mhd_ops_pure_psydac.

        params: dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, u, particles, derham, domain, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)

        self._u = u
        self._particles = particles
        self._derham = derham
        self._G = derham.grad
        self._GT = derham.grad.transpose()
        self._domain = domain
        self._params = params
        self._info = params['info']
        self._mass_ops = mass_ops
        self._mhd_ops = mhd_ops
        self._rank = derham.comm.Get_rank()

        # Preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(derham, 'V1', mass_ops._fun_M1n)

        # call the accumulation class
        args = []
        self._ACC = Accumulator(self._domain, self._derham, 'Hcurl', 'pc_lin_mhd_6d', *args, do_vector = True, symmetry = 'pressure')

        # Define A of the Schur block matrix [[A, B], [C, I]] (B and C are needed to be defined every time the propagater is called since they include accumulated values)
        self._A = mass_ops.M1n

        # call Pusher class
        self._pusher = Pusher(self._derham, self._domain, 'push_pc_Xu_full')

    @property
    def variables(self):
        return [self._u]
    
    def __call__(self, dt):
        un = self.variables[0]

        # reorganize particles
        self._derham.comm.Barrier()
        self._particles.send_recv_markers()
        self._derham.comm.Barrier()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles.markers, self._particles.n_mks)

        MAT = [[self._ACC.matrix11, self._ACC.matrix12, self._ACC.matrix13], 
               [self._ACC.matrix12, self._ACC.matrix22, self._ACC.matrix23], 
               [self._ACC.matrix13, self._ACC.matrix23, self._ACC.matrix33]]
        VEC =  [self._ACC.vector1, self._ACC.vector2, self._ACC.vector3]

        GT_VEC = BlockVector(self._derham.V0vec.vector_space, blocks=[self._GT.dot(VEC[0]),
                                                                      self._GT.dot(VEC[1]),
                                                                      self._GT.dot(VEC[2])])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._mhd_ops.X1T, self.GT_MAT_G(self._derham, MAT), self._mhd_ops.X1))

        BV = Multiply(-1/2, self._mhd_ops.X1T).dot(GT_VEC)
        
        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._params['type'], 
                                         tol=self._params['tol'], maxiter=self._params['maxiter'],
                                         verbose=self._params['verbose'])

        # allocate temporary FemFields _u during solution
        _u, self._info = schur_solver(un, BV, dt) 

        # calculate GXu
        GXu_1 = self._G.dot(self._mhd_ops.X1.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._mhd_ops.X1.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._mhd_ops.X1.dot(un + _u)[2])
            
        # push particles
        # check if ghost regions are synchronized
        for i in range(3):
            if not GXu_1[i].ghost_regions_in_sync: GXu_1[i].update_ghost_regions()
            if not GXu_2[i].ghost_regions_in_sync: GXu_2[i].update_ghost_regions()
            if not GXu_3[i].ghost_regions_in_sync: GXu_3[i].update_ghost_regions()

        self._pusher(self._particles, dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        du = self.in_place_update(_u)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling1:', self._info['success'])
            print('Iterations for StepPressurecoupling1:', self._info['niter'])
            print('Maxdiff u1 for StepPressurecoupling1:', max(du))
            print()

    class GT_MAT_G(LinOpWithTransp):
        r'''
        Class for defining LinearOperator corresponding to :math:`G^\top (\text{MAT}) G \in \mathbb{R}^{3N^0 \times 3N^0}` 
        where :math:`\text{MAT} = V^\top (\bar {\mathbf \Lambda}^1)^\top \bar{DF}^{-1} \bar{W} \bar{DF}^{-\top} \bar{\mathbf \Lambda}^1 V \in \mathbb{R}^{3N^1 \times 3N^1}`.
        
        Parameters
        ----------
            derham : struphy.psydac_api.psydac_derham.Derham
                Discrete de Rham sequence on the logical unit cube.

            MAT : List of StencilMatrices
                List with six of accumulated pressure terms
        '''

        def __init__(self, derham, MAT, transposed=False):

            self._derham = derham
            self._G = derham.grad
            self._GT = derham.grad.transpose()

            self._domain = derham.V0vec.vector_space
            self._codomain = derham.V0vec.vector_space
            self._MAT = MAT

            v1 = StencilVector(derham.V0vec.vector_space.spaces[0])
            v2 = StencilVector(derham.V0vec.vector_space.spaces[1])
            v3 = StencilVector(derham.V0vec.vector_space.spaces[2])
            list_blocks = [v1, v2, v3]
            self._vector = BlockVector(derham.V0vec.vector_space, blocks=list_blocks)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.V0vec.vector_space.dtype

        @property
        def transposed(self):
            return self._transposed

        def transpose(self):
            return self.GT_MAT_G(self._derham, self._MAT, True)

        def dot(self, v, out=None):
            '''dot product between GT_MAT_G and v.

            Parameters
            ----------
                v : StencilVector or BlockVector
                    Input FE coefficients from V.vector_space.

            Returns
            -------
                A StencilVector or BlockVector from W.vector_space.'''

            assert v.space == self.domain

            v.update_ghost_regions()

            temp = [None, None, None]

            for i in range(3):
                for j in range(3):
                    temp[j] = self._MAT[i][j].dot(self._G.dot(v[j]))
                self._vector[i] = self._GT.dot(temp[0] + temp[1] + temp[2])

            self._vector.update_ghost_regions()

            assert self._vector.space == self.codomain

            return self._vector  


class StepPressurecouplingHdiv(Propagator):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_2)^{-1} V^\top (\bar {\mathcal X}^2)^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X}^2 V (\mathbb M^n_2)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_2}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        particles : struphy.pic.particles.Particles6D
        domain : struphy.geometry.base.Domain
            Infos regarding mapping.
            
        mass_ops : struphy.psydac_api.mass_psydac.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass_psydac.
            
        mhd_ops : struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators
            Linear MHD operators from struphy.psydac_api.mhd_ops_pure_psydac.

        params: dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, u, particles, derham, domain, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)

        self._u = u
        self._particles = particles
        self._derham = derham
        self._G = derham.grad
        self._GT = derham.grad.transpose()
        self._domain = domain
        self._params = params
        self._info = params['info']
        self._mass_ops = mass_ops
        self._mhd_ops = mhd_ops
        self._rank = derham.comm.Get_rank()

        # Preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(derham, 'V2', mass_ops._fun_M2n)

        # call the accumulation class
        args = []
        self._ACC = Accumulator(self._domain, self._derham, 'Hcurl', 'pc_lin_mhd_6d', *args, do_vector = True, symmetry = 'pressure')

        # Define A of the Schur block matrix [[A, B], [C, I]] (B and C are needed to be defined every time the propagater is called since they include accumulated values)
        self._A = mass_ops.M2n

        # call Pusher class
        self._pusher = Pusher(self._derham, self._domain, 'push_pc_Xu_full')

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):

        # current variables
        un = self.variables[0]

        # reorganize particles
        self._derham.comm.Barrier()
        self._particles.send_recv_markers()
        self._derham.comm.Barrier()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles.markers, self._particles.n_mks)

        MAT = [[self._ACC.matrix11, self._ACC.matrix12, self._ACC.matrix13], 
               [self._ACC.matrix12, self._ACC.matrix22, self._ACC.matrix23], 
               [self._ACC.matrix13, self._ACC.matrix23, self._ACC.matrix33]]
        VEC =  [self._ACC.vector1, self._ACC.vector2, self._ACC.vector3]

        # assemble G^T dot VEC
        GT_VEC = BlockVector(self._derham.V0vec.vector_space, blocks=[self._GT.dot(VEC[0]),
                                                                      self._GT.dot(VEC[1]),
                                                                      self._GT.dot(VEC[2])])


        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._mhd_ops.X2T, self.GT_MAT_G(self._derham, MAT), self._mhd_ops.X2))

        BV = Multiply(-1/2, self._mhd_ops.X2T).dot(GT_VEC)
        
        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._params['type'], 
                                         tol=self._params['tol'], maxiter=self._params['maxiter'],
                                         verbose=self._params['verbose'])

        # allocate temporary FemFields _u during solution
        _u, self._info = schur_solver(un, BV, dt)

        # calculate GXu
        GXu_1 = self._G.dot(self._mhd_ops.X2.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._mhd_ops.X2.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._mhd_ops.X2.dot(un + _u)[2])

        # push particles
        # check if ghost regions are synchronized
        for i in range(3):
            if not GXu_1[i].ghost_regions_in_sync: GXu_1[i].update_ghost_regions()
            if not GXu_2[i].ghost_regions_in_sync: GXu_2[i].update_ghost_regions()
            if not GXu_3[i].ghost_regions_in_sync: GXu_3[i].update_ghost_regions()

        self._pusher(self._particles, dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        du = self.in_place_update(_u)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling1:', self._info['success'])
            print('Iterations for StepPressurecoupling1:', self._info['niter'])
            print('Maxdiff u1 for StepPressurecoupling1:', max(du))
            print()

    class GT_MAT_G(LinOpWithTransp):
        r'''
        Class for defining LinearOperator corresponding to :math:`G^\top (\text{MAT}) G \in \mathbb{R}^{3N^0 \times 3N^0}` 
        where :math:`\text{MAT} = V^\top (\bar {\mathbf \Lambda}^1)^\top \bar{DF}^{-1} \bar{W} \bar{DF}^{-\top} \bar{\mathbf \Lambda}^1 V \in \mathbb{R}^{3N^1 \times 3N^1}`.
        
        Parameters
        ----------
            derham : struphy.psydac_api.psydac_derham.Derham
                Discrete de Rham sequence on the logical unit cube.

            MAT : List of StencilMatrices
                List with six of accumulated pressure terms
        '''

        def __init__(self, derham, MAT, transposed=False):

            self._derham = derham
            self._G = derham.grad
            self._GT = derham.grad.transpose()

            self._domain = derham.V0vec.vector_space
            self._codomain = derham.V0vec.vector_space
            self._MAT = MAT

            v1 = StencilVector(derham.V0vec.vector_space.spaces[0])
            v2 = StencilVector(derham.V0vec.vector_space.spaces[1])
            v3 = StencilVector(derham.V0vec.vector_space.spaces[2])
            list_blocks = [v1, v2, v3]
            self._vector = BlockVector(derham.V0vec.vector_space, blocks=list_blocks)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.V0vec.vector_space.dtype

        @property
        def transposed(self):
            return self._transposed

        def transpose(self):
            return self.GT_MAT_G(self._derham, self._MAT, True)

        def dot(self, v, out=None):
            '''dot product between GT_MAT_G and v.

            Parameters
            ----------
                v : StencilVector or BlockVector
                    Input FE coefficients from V.vector_space.

            Returns
            -------
                A StencilVector or BlockVector from W.vector_space.'''

            assert v.space == self.domain

            v.update_ghost_regions()

            temp = [None, None, None]

            for i in range(3):
                for j in range(3):
                    temp[j] = self._MAT[i][j].dot(self._G.dot(v[j]))
                self._vector[i] = self._GT.dot(temp[0] + temp[1] + temp[2])

            self._vector.update_ghost_regions()

            assert self._vector.space == self.codomain

            return self._vector 


class StepPushEtaPC(Propagator):
    r'''Step for the update of particles' positions with the RK4 method which solves

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and 

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 0-form, 1-form or 2-form.

        particles : struphy.pic.particles.Particles6D
        domain : struphy.geometry.base.Domain
            Infos regarding mapping.

        u_space : dic
            params['fields']['mhd_u_space']
    '''
    def __init__(self, u, particles, derham, domain, u_space):

        assert isinstance(u, BlockVector)

        self._derham = derham
        self._domain = domain
        self._u = u
        self._particles = particles
        self._u_space = u_space

        # call Pusher class
        if self._u_space == 'Hcurl':
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_eta_rk4_Hcurl_full', stage_num=4)

        elif self._u_space == 'Hdiv':
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_eta_rk4_Hdiv_full', stage_num=4)

        else:
            print('push_pc_eta_rk4_H1vec is not yet implemented.')

    @property
    def variables(self):
        return


    def __call__(self, dt):

        # push particles
        # check if ghost regions are synchronized
        if not self._u[0].ghost_regions_in_sync: self._u[0].update_ghost_regions()
        if not self._u[1].ghost_regions_in_sync: self._u[1].update_ghost_regions()
        if not self._u[2].ghost_regions_in_sync: self._u[2].update_ghost_regions()

        self._pusher(self._particles, dt,
                     self._u[0]._data, self._u[1]._data, self._u[2]._data,
                     do_mpi_sort=True)


class StepPushVxB(Propagator):
    r"""Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.
    
    Parameters
    ----------
        particles : struphy.pic.particles.Particles6D
            Holdes the markers to push.
            
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        b : psydac.linalg.block.BlockVector
            FE coefficients of a dynamical magnetic field (2-form).
            
        b_static : psydac.linalg.block.BlockVector (optional)
            FE coefficients of a static (background) magnetic field (2-form).
    """
    
    def __init__(self, particles, derham, b, b_static=None):
        
        self._particles = particles
        
        # load pusher
        from struphy.pic.pusher import Pusher
        
        self._pusher = Pusher(derham, particles.domain, 'push_vxb_analytic')
        
        assert isinstance(b, BlockVector)
        
        self._b = b
        
        if b_static is None:
            self._b_static = b.space.zeros()
        else:
            assert isinstance(b_static, BlockVector)
            self._b_static = b_static
        
    
    @property
    def variables(self):
        return self._particles
    
    def __call__(self, dt):
        
        # check if ghost regions are synchronized
        if not self._b[0].ghost_regions_in_sync: self._b[0].update_ghost_regions()
        if not self._b[1].ghost_regions_in_sync: self._b[1].update_ghost_regions()
        if not self._b[2].ghost_regions_in_sync: self._b[2].update_ghost_regions()
            
        if not self._b_static[0].ghost_regions_in_sync: self._b_static[0].update_ghost_regions()
        if not self._b_static[1].ghost_regions_in_sync: self._b_static[1].update_ghost_regions()
        if not self._b_static[2].ghost_regions_in_sync: self._b_static[2].update_ghost_regions()
        
        self._pusher(self._particles, dt, 
                     self._b_static[0]._data + self._b[0]._data,
                     self._b_static[1]._data + self._b[1]._data,
                     self._b_static[2]._data + self._b[2]._data)
        
        
class StepPushEtaRk4(Propagator):
    r"""Fourth order Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    
    Parameters
    ----------
        particles : struphy.pic.particles.Particles6D
            Holdes the markers to push.
            
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
    """
    
    def __init__(self, particles, derham):
        
        self._particles = particles
        
        # load pusher
        from struphy.pic.pusher import Pusher
        
        self._pusher = Pusher(derham, particles.domain, 'push_eta_rk4')
        
    @property
    def variables(self):
        return self._particles

    def __call__(self, dt):
        self._pusher(self._particles, dt, do_mpi_sort=True)