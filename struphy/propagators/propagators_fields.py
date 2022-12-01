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


class Maxwell( Propagator ):
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
            pc = pc_class(mass_ops.M1)

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
            print('Status     for Maxwell:', info['success'])
            print('Iterations for Maxwell:', info['niter'])
            print('Maxdiff e1 for Maxwell:', max(de))
            print('Maxdiff b2 for Maxwell:', max(db))
            print()


class OhmCold( Propagator ):
    r'''Analytical solution

    .. math::

        \begin{bmatrix}
            \mathbf j^{n+1} \\
            \mathbf e^{n+1}
        \end{bmatrix}
        = \begin{bmatrix}
            \cos{\alpha \Delta t} & \sin{\alpha \Delta t} \\
            - \sin{\alpha \Delta t} & \cos{\alpha \Delta t}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf j^n \\
            \mathbf e^n
        \end{bmatrix}

    of the rotation problem

    .. math::

        \frac{\partial}{\partial t}
        \begin{bmatrix}
            \mathbf j \\
            \mathbf e
        \end{bmatrix}
            = \begin{bmatrix}
            0 & \alpha \mathbb M_1^{-1} \\
            -\alpha \mathbb M_1^{-1} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbb M_1 \mathbf j \\
            \mathbb M_1 \mathbf e
        \end{bmatrix}\,, \qquad \begin{bmatrix}
            \mathbf j \\
            \mathbf e
        \end{bmatrix}(0) = 
        \begin{bmatrix}
            \mathbf j^n \\
            \mathbf e^n
        \end{bmatrix}\,,

    where :math:`\alpha \in \mathbb R` denotes the angular frequency of the rotation.

    Parameters
    ----------
        j : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        e : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        a : float
            plasma frequency measured in unit of electron cyclotron frequency
    '''

    def __init__(self, j, e, a):

        assert isinstance(j, BlockVector)
        assert isinstance(e, BlockVector)
        assert isinstance(a, float)

        self._j = j
        self._e = e
        self._a = a

    @property
    def variables(self):
        return [self._j, self._e]

    def __call__(self, dt):

        # current variables
        jn = self.variables[0]
        en = self.variables[1]

        # allocate temporary FemFields _j, _e during solution
        _j = np.cos(self._a * dt) * jn + np.sin(self._a * dt) * en
        _e = -np.sin(self._a * dt) * jn + np.cos(self._a * dt) * en

        # write new coeffs into Propagator.variables
        dj, de = self.in_place_update(_j, _e)

        print('Maxdiff j1 for OhmCold:', max(dj))
        print('Maxdiff e2 for OhmCold:', max(de))
        print()

        
class ShearAlfvén( Propagator ):
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

        u_space : str
            Space identifier of MHD velocity from parameters/fields/init/mhd_u_space: 'Hcurl, 'Hdiv' or 'H1vec'.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        mass_ops : struphy.psydac_api.mass.WeightedMassOperators
            Weighted mass matrices from struphy.psydac_api.mass.
            
        mhd_ops : struphy.psydac_api.basis_projection_ops.MHDOperators
            Linear MHD operators from struphy.psydac_api.basis_projection_ops.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, u, b, u_space, derham, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)
        assert isinstance(b, BlockVector)
        assert u_space in {'Hcurl', 'Hdiv', 'H1vec'}

        self._u = u
        self._b = b
        self._info = params['info']
        self._rank = derham.comm.Get_rank()

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        if u_space == 'Hcurl':
            id_Mn = 'M1n'
            id_T = 'T1'
            id_fun = '_fun_M1n'
        elif u_space == 'Hdiv':
            id_Mn = 'M2n'
            id_T = 'T2'
            id_fun = '_fun_M2n'
        elif u_space == 'H1vec':
            id_Mn = 'Mvn'
            id_T = 'Tv'
            id_fun = '_fun_Mvn'

        _A = getattr(mass_ops, id_Mn)
        _T = getattr(mhd_ops, id_T)
        self._B = Multiply(-1/2., Compose(_T.transpose(), derham.curl.transpose(), mass_ops.M2))
        self._C = Multiply( 1/2., Compose(derham.curl, _T))
        
        # Preconditioner
        _pc_fun = getattr(mass_ops, id_fun)
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(getattr(mass_ops, id_Mn))
        
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

        if self._info and self._rank ==0:
            print('Status     for ShearAlfvén:', info['success'])
            print('Iterations for ShearAlfvén:', info['niter'])
            print('Maxdiff up for ShearAlfvén:', max(du))
            print('Maxdiff b2 for ShearAlfvén:', max(db))
            print()


class Magnetosonic( Propagator ):
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
        n : psydac.linalg.block.StencilVector
            FE coefficients of a discrete 3-form.
        
        u : psydac.linalg.block.BlockVector
            FE coefficients of MHD velocity.

        p : psydac.linalg.block.StencilVector
            FE coefficients of a discrete 3-form.
            
        b : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        u_space : str
            Space identifier of MHD velocity from parameters/fields/init/mhd_u_space: 'Hcurl, 'Hdiv' or 'H1vec'.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        mass_ops : struphy.psydac_api.mass.WeightedMassOperators
            Weighted mass matrices from struphy.psydac_api.mass.
            
        mhd_ops : struphy.psydac_api.basis_projection_ops.MHDOperators
            Linear MHD operators from struphy.psydac_api.basis_projection_ops.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, n, u, p, b, u_space, derham, mass_ops, mhd_ops, params):

        assert isinstance(n, StencilVector)
        assert isinstance(u, BlockVector)
        assert isinstance(p, StencilVector)
        assert isinstance(b, BlockVector)
        assert u_space in {'Hcurl', 'Hdiv', 'H1vec'}

        self._n = n
        self._u = u
        self._p = p
        self._b = b
        self._bc = derham.bc
        self._info = params['info']
        self._rank = derham.comm.Get_rank()
        
        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        if u_space == 'Hcurl':
            id_Mn = 'M1n'
            id_MJ = 'M1J'
            id_S = 'S1'
            id_U = 'U1'
            id_K = 'K1'
            id_Q = 'Q1'
            id_fun = '_fun_M1n'
        elif u_space == 'Hdiv':
            id_Mn = 'M2n'
            id_MJ = 'M2J'
            id_S = 'S2'
            id_U = None
            id_K = 'K2'
            id_Q = 'Q2'
            id_fun = '_fun_M1n'
        elif u_space == 'H1vec':
            id_Mn = 'Mvn'
            id_MJ = 'MvJ'
            id_S = 'S0'
            id_U = 'Uv'
            id_K = 'K0'
            id_Q = 'Q0'
            id_fun = '_fun_M1n'

        _A = getattr(mass_ops, id_Mn)
        _S = getattr(mhd_ops, id_S)
        _U = getattr(mhd_ops, id_U) if id_U is not None else None
        _UT = _U.transpose() if _U is not None else None
        _K = getattr(mhd_ops, id_K)
        self._B = Multiply(-1/2., Compose(_UT, derham.div.transpose(), mass_ops.M3))
        self._C = Multiply( 1/2., Sum(Compose(derham.div, _S), Multiply(2/3, Compose(_K, derham.div, _U))))
        
        self._MJ = getattr(mass_ops, id_MJ)
        self._Q  = getattr(mhd_ops, id_Q)
        self._DIV = derham.div
        
        # Preconditioner
        _pc_fun = getattr(mass_ops, id_fun)
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(getattr(mass_ops, id_Mn))

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
            print('Status     for Magnetosonic:', info['success'])
            print('Iterations for Magnetosonic:', info['niter'])
            print('Maxdiff n3 for Magnetosonic:', max(dn))
            print('Maxdiff up for Magnetosonic:', max(du))
            print('Maxdiff p3 for Magnetosonic:', max(dp))
            print('Maxdiff b2 for Magnetosonic:', max(db))
            print()
