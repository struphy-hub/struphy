from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert
from struphy.psydac_api.preconditioner import MassMatrixPreConditioner as MassPre
from struphy.psydac_api.preconditioner import MassMatrixPreconditioner3D as MassPreNew

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

        # Preconditioner
        if params['pc'] == None:
            pc = None
        elif params['pc'] == 'fft':
            pc = MassPre(derham.V1)
        else:
            raise ValueError(f'Preconditioner "{params["pc"]}" not implemented.')

        # Define block matrix [[A B], [C I]] (without time step size dt in the diangonals)
        _A = mass_ops.M1
        self._B = Multiply(-1./2., Compose(derham.curl.transpose(), mass_ops.M2)) # no dt
        self._C = Multiply(1./2., derham.curl) # no dt
        _BC = Compose(self._B, self._C)

        # Instantiate Schur solver (constant in this case)
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
            
            
class StepShearAlfven1(Propagator):
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

        # Preconditioner
        if params['pc'] == None:
            pc = None
        elif params['pc'] == 'fft':
            pc = MassPre(derham.V1)
        elif params['pc'] == 'new':
            pc = MassPreNew(derham, 'V1', mass_ops._fun_M1n)
        else:
            raise ValueError(f'Preconditioner "{params["pc"]}" not implemented.')

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M1n

        self._B = Multiply(-1/2., Compose(mhd_ops.T1T, derham.curl.transpose(), mass_ops.M2))
        self._C = Multiply( 1/2., Compose(derham.curl, mhd_ops.T1))
        
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
        _u, info = self._schur_solver(un, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_u + un)

        # write new coeffs into Propagator.variables
        du, db = self.in_place_update(_u, _b)

        if self._info:
            print('Status     for Push_shear_alfvén:', info['success'])
            print('Iterations for Push_shear_alfvén:', info['niter'])
            print('Maxdiff u1 for Push_shear_alfvén:', max(du))
            print('Maxdiff b2 for Push_shear_alfvén:', max(db))
            print()
            
            
class StepShearAlfven2(Propagator):
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

        # Preconditioner
        if params['pc'] == None:
            pc = None
        elif params['pc'] == 'fft':
            pc = MassPre(derham.V2)
        elif params['pc'] == 'new':
            pc = MassPreNew(derham, 'V2', mass_ops._fun_M2n)
        else:
            raise ValueError(f'Preconditioner "{params["pc"]}" not implemented.')

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M2n

        self._B = Multiply(-1/2., Compose(mhd_ops.T2T, derham.curl.transpose(), mass_ops.M2))
        self._C = Multiply( 1/2., Compose(derham.curl, mhd_ops.T2))
        
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
        _u, info = self._schur_solver(un, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_u + un)

        # write new coeffs into Propagator.variables
        du, db = self.in_place_update(_u, _b)

        if self._info:
            print('Status     for Push_shear_alfvén:', info['success'])
            print('Iterations for Push_shear_alfvén:', info['niter'])
            print('Maxdiff u2 for Push_shear_alfvén:', max(du))
            print('Maxdiff b2 for Push_shear_alfvén:', max(db))
            print()


class StepShearAlfven3(Propagator):
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

        # Preconditioner
        if params['pc'] == None:
            pc = None
        elif params['pc'] == 'fft':
            pc = MassPre(derham.V0vec)
        elif params['pc'] == 'new':
            pc = MassPreNew(derham, 'V0vec', mass_ops._fun_Mvn)
        else:
            raise ValueError(f'Preconditioner "{params["pc"]}" not implemented.')

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.Mvn

        self._B = Multiply(-1/2., Compose(mhd_ops.T0T, derham.curl.transpose(), mass_ops.M2))
        self._C = Multiply( 1/2., Compose(derham.curl, mhd_ops.T0))
        
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
        _u, info = self._schur_solver(un, self._B.dot(bn), dt)
        _b = bn - dt*self._C.dot(_u + un)

        # write new coeffs into Propagator.variables
        du, db = self.in_place_update(_u, _b)

        if self._info:
            print('Status     for Push_shear_alfvén:', info['success'])
            print('Iterations for Push_shear_alfvén:', info['niter'])
            print('Maxdiff uv for Push_shear_alfvén:', max(du))
            print('Maxdiff b2 for Push_shear_alfvén:', max(db))
            print()
            
                
class StepMagnetosonic2(Propagator):
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

        # Preconditioner
        if params['pc'] == None:
            pc = None
        elif params['pc'] == 'fft':
            pc = MassPre(derham.V2)
        elif params['pc'] == 'new':
            pc = MassPreNew(derham, 'V2', mass_ops._fun_M2n)
        else:
            raise ValueError(f'Preconditioner "{params["pc"]}" not implemented.')

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M2n

        self._B = Multiply(-1/2., Compose(derham.div.transpose(), mass_ops.M3))
        self._C = Multiply( 1/2., Sum(Compose(derham.div, mhd_ops.S2), Multiply(2/3, Compose(mhd_ops.K2, derham.div))))

        self._MJ = mass_ops.M2J
        self._Q  = mhd_ops.Q2
            
        self._DIV = derham.div
        
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

        if self._info:
            print('Status     for Push_magnetosonic:', info['success'])
            print('Iterations for Push_magnetosonic:', info['niter'])
            print('Maxdiff n3 for Push_magnetosonic:', max(dn))
            print('Maxdiff u2 for Push_magnetosonic:', max(du))
            print('Maxdiff p3 for Push_magnetosonic:', max(dp))
            print()
            

class StepMagnetosonic3(Propagator):
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

        # Preconditioner
        if params['pc'] == None:
            pc = None
        elif params['pc'] == 'fft':
            pc = MassPre(derham.V0vec)
        elif params['pc'] == 'new':
            pc = MassPreNew(derham, 'V0vec', mass_ops._fun_Mvn)
        else:
            raise ValueError(f'Preconditioner "{params["pc"]}" not implemented.')

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.Mvn

        self._B = Multiply(-1/2., Compose(mhd_ops.J0T, derham.div.transpose(), mass_ops.M3))
        self._C = Multiply( 1/2., Sum(Compose(derham.div, mhd_ops.S0), Multiply(2/3, Compose(mhd_ops.K0, derham.div, mhd_ops.J0))))
        self._MJ = mass_ops.MvJ
        self._Q  = mhd_ops.Q0
            
        self._DIV = derham.div
        
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

        if self._info:
            print('Status     for Push_magnetosonic:', info['success'])
            print('Iterations for Push_magnetosonic:', info['niter'])
            print('Maxdiff n3 for Push_magnetosonic:', max(dn))
            print('Maxdiff uv for Push_magnetosonic:', max(du))
            print('Maxdiff p3 for Push_magnetosonic:', max(dp))
            print()
            