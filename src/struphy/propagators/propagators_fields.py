import numpy as np
from numpy import zeros

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.particles_to_grid import Accumulator
from struphy.polar.basic import PolarVector
from struphy.kinetic_background.analytical import Maxwellian, Maxwellian6DUniform, Maxwellian5DUniform
from struphy.fields_background.mhd_equil.equils import set_defaults

from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api.linear_operators import InverseLinearOperator as Inverse
from struphy.psydac_api import preconditioner
from struphy.psydac_api.mass import WeightedMassOperator
import struphy.linear_algebra.iterative_solvers as it_solvers
from psydac.linalg.iterative_solvers import pcg

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector
import struphy.psydac_api.utilities as util


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

        # pointers to variables
        assert isinstance(e, (BlockVector, PolarVector))
        assert isinstance(b, (BlockVector, PolarVector))
        self._e = e
        self._b = b

        # parameters
        params_default = {'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._info = params['info']

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1

        # no dt
        self._B = Multiply(-1/2, Compose(self.derham.curl.T, self.mass_ops.M2))
        self._C = Multiply(1/2, self.derham.curl)

        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(self.mass_ops.M1)

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate place-holder vectors to avoid temporary array allocations in __call__
        self._e_tmp1 = e.space.zeros()
        self._e_tmp2 = e.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    @property
    def variables(self):
        return [self._e, self._b]

    def __call__(self, dt):

        # current variables
        en = self.variables[0]
        bn = self.variables[1]

        # solve for new e coeffs
        self._B.dot(bn, out=self._byn)

        info = self._schur_solver(en, self._byn, dt, out=self._e_tmp1)[1]

        # new b coeffs
        en.copy(out=self._e_tmp2)
        self._e_tmp2 += self._e_tmp1
        self._C.dot(self._e_tmp2, out=self._b_tmp1)
        self._b_tmp1 *= -dt
        self._b_tmp1 += bn

        # write new coeffs into self.variables
        max_de, max_db = self.in_place_update(self._e_tmp1, self._b_tmp1)

        if self._info:
            print('Status     for Maxwell:', info['success'])
            print('Iterations for Maxwell:', info['niter'])
            print('Maxdiff e1 for Maxwell:', max_de)
            print('Maxdiff b2 for Maxwell:', max_db)
            print()


class OhmCold(Propagator):
    r'''Crank-Nicolson step

    .. math::

        \begin{bmatrix}
            \mathbf e^{n+1} - \mathbf e^n \\
            \mathbf j^{n+1} - \mathbf j^n
        \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix}
            0 & - \mathbb M_1^{-1} M_{1, \alpha} \\
            \mathbb M_1^{-1} M_{1, \alpha} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf e^n \\
            \mathbf j^n
        \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`, of the rotation problem

    .. math::

        \frac{\partial}{\partial t}
        \begin{bmatrix}
            \mathbf e \\
            \mathbf j
        \end{bmatrix}
        = \begin{bmatrix}
            0 & - \mathbb M_1^{-1} \mathbb M_{1, \alpha} \\
            \mathbb M_1^{-1} \mathbb M_{1, \alpha} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf e \\
            \mathbf j
        \end{bmatrix}\,, \qquad \begin{bmatrix}
            \mathbf e \\
            \mathbf j
        \end{bmatrix}(0) = 
        \begin{bmatrix}
            \mathbf e^n \\
            \mathbf j^n
        \end{bmatrix}\,,


    where :math:`\mathbb M_{1, \alpha}` denotes the mass matrix weighted by :math:`\alpha`,
    which represents the plasma frequency in units of the electron cyclotron frequency.

    Parameters
    ----------
        j : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        e : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        params : dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, e, j, mass_ops, params):

        assert isinstance(e, (BlockVector, PolarVector))
        assert isinstance(j, (BlockVector, PolarVector))

        self._e = e
        self._j = j
        self._info = params['info']

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = 1

        self._B = Multiply(-1./2., Compose(mass_ops.M1.invert(),
                           mass_ops.M1alpha))  # no dt
        self._C = Multiply(
            1./2., Compose(mass_ops.M1.invert(), mass_ops.M1alpha))  # no dt

        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            # TODO ???
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(mass_ops.M1)

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

    @property
    def variables(self):
        return [self._e, self._j]

    def __call__(self, dt):

        # current variables
        en = self.variables[0]
        jn = self.variables[1]

        # allocate temporary FemFields _e, _j during solution
        _e, info = self._schur_solver(en, self._B.dot(jn), dt)
        _j = jn - dt*self._C.dot(_e + en)

        # write new coeffs into Propagator.variables
        max_de, max_dj = self.in_place_update(_e, _j)

        if self._info:
            print('Status     for OhmCold:', info['success'])
            print('Iterations for OhmCold:', info['niter'])
            print('Maxdiff e1 for OhmCold:', max_de)
            print('Maxdiff j1 for OhmCold:', max_dj)
            print()


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

        # pointers to variables
        assert isinstance(u, (BlockVector, PolarVector))
        assert isinstance(b, (BlockVector, PolarVector))
        self._u = u
        self._b = b

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
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
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    @property
    def variables(self):
        return [self._u, self._b]

    def __call__(self, dt):

        # current variables
        un = self.variables[0]
        bn = self.variables[1]

        # solve for new u coeffs
        self._B.dot(bn, out=self._byn)

        info = self._schur_solver(un, self._byn, dt, out=self._u_tmp1)[1]

        # new b coeffs
        un.copy(out=self._u_tmp2)
        self._u_tmp2 += self._u_tmp1
        self._C.dot(self._u_tmp2, out=self._b_tmp1)
        self._b_tmp1 *= -dt
        self._b_tmp1 += bn

        # write new coeffs into self.variables
        max_du, max_db = self.in_place_update(self._u_tmp1, self._b_tmp1)

        if self._info and self._rank == 0:
            print('Status     for ShearAlfvén:', info['success'])
            print('Iterations for ShearAlfvén:', info['niter'])
            print('Maxdiff up for ShearAlfvén:', max_du)
            print('Maxdiff b2 for ShearAlfvén:', max_db)
            print()


class ShearAlfvénB1(Propagator):
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
        FE coefficients of MHD velocity as 2-form.

    b : psydac.linalg.block.BlockVector
        FE coefficients of magnetic field as 1-form.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, u, b, **params):

        # pointers to variables
        assert isinstance(u, (BlockVector, PolarVector))
        assert isinstance(b, (BlockVector, PolarVector))
        self._u = u
        self._b = b

        # parameters
        params_default = {'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
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
        #I still have to invert M1
        self._C = Multiply(1/2, Compose(self._M1inv,self.derham.curl.T, self.mass_ops.M2B))

        # Preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(getattr(self.mass_ops, 'M2n'))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    @property
    def variables(self):
        return [self._u, self._b]

    def __call__(self, dt):

        # current variables
        un = self.variables[0]
        bn = self.variables[1]

        # solve for new u coeffs
        self._B.dot(bn, out=self._byn)

        info = self._schur_solver(un, self._byn, dt, out=self._u_tmp1)[1]

        # new b coeffs
        un.copy(out=self._u_tmp2)
        self._u_tmp2 += self._u_tmp1
        self._C.dot(self._u_tmp2, out=self._b_tmp1)
        self._b_tmp1 *= -dt
        self._b_tmp1 += bn

        # write new coeffs into self.variables
        max_du, max_db = self.in_place_update(self._u_tmp1, self._b_tmp1)

        if self._info and self._rank == 0:
            print('Status     for ShearAlfvén:', info['success'])
            print('Iterations for ShearAlfvén:', info['niter'])
            print('Maxdiff up for ShearAlfvén:', max_du)
            print('Maxdiff b2 for ShearAlfvén:', max_db)
            print()


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

        # pointers to variables
        assert isinstance(n, (StencilVector, PolarVector))
        assert isinstance(u, (BlockVector, PolarVector))
        assert isinstance(p, (StencilVector, PolarVector))
        self._n = n
        self._u = u
        self._p = p

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'b': self.derham.Vh['2'].zeros(),
                          'type': 'PBiConjugateGradientStab',
                          'pc': 'MassMatrixPreconditioner',
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
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'],
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

    @property
    def variables(self):
        return [self._n, self._u, self._p]

    def __call__(self, dt):

        # current variables
        nn = self.variables[0]
        un = self.variables[1]
        pn = self.variables[2]

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

        # write new coeffs into self.variables
        max_dn, max_du, max_dp = self.in_place_update(self._n_tmp1,
                                                      self._u_tmp1,
                                                      self._p_tmp1)

        if self._info and self._rank == 0:
            print('Status     for Magnetosonic:', info['success'])
            print('Iterations for Magnetosonic:', info['niter'])
            print('Maxdiff n3 for Magnetosonic:', max_dn)
            print('Maxdiff up for Magnetosonic:', max_du)
            print('Maxdiff p3 for Magnetosonic:', max_dp)
            print()


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

        for loop in range(10):
            #print('+++++=====++++++', self._accum_density._operators[0].matrix._data)
            # set mid-value used in the fixed iteration
            curla_mid = self.derham.curl.dot(
                0.5*(self._a_old + self._a)) + self._beq
            curla_mid.update_ghost_regions()
            # initialize the curl A
            # remember to check ghost region of curla_mid
            util.create_weight_weightedmatrix_hybrid(
                curla_mid, self._weight_pre, self.derham, self._accum_density, self.domain)
            #self._weight = [[None, self._weight_pre[2], -self._weight_pre[1]], [None, None, self._weight_pre[0]], [None, None, None]]
            self._weight = [[0.0*self._weight_pre[0], 0.0*self._weight_pre[2], 0.0*self._weight_pre[1]], [0.0*self._weight_pre[2], 0.0 *
                                                                                                          self._weight_pre[1], 0.0*self._weight_pre[0]], [0.0*self._weight_pre[1], 0.0*self._weight_pre[0], 0.0*self._weight_pre[2]]]
            #self._weight = [[self._weight_pre[0], self._weight_pre[2], self._weight_pre[1]], [self._weight_pre[2], self._weight_pre[1], self._weight_pre[0]], [self._weight_pre[1], self._weight_pre[0], self._weight_pre[2]]]
            HybridM1 = self.mass_ops.assemble_weighted_mass(
                self._weight, 'Hcurl', 'Hcurl')

            # next prepare for solving linear system
            _LHS = Sum(self._M1, Compose(HybridM1, self._L2))
            _RHS2 = HybridM1.dot(self._RHS) + self._rhs

            a_new, info = pcg(_LHS, _RHS2, self._pc, x0=self._a, tol=self._solver_params['tol'],
                              maxiter=self._solver_params['maxiter'], verbose=self._solver_params['verbose'])

            # write new coeffs into Propagator.variables
            max_da = self.in_place_update(a_new)
            print('++++====check_iteration_error=====+++++', max_da)
            # we can modify the diff function in in_place_update to get another type errors
            if max_da[0] < 10**(-6):
                break


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

        from struphy.pic.particles import Particles6D, Particles5D

        # pointers to variables
        assert isinstance(u, (BlockVector, PolarVector))
        self._u = u

        # parameters
        params_default = {'particles': None,
                          'u_space': 'Hdiv',
                          'b_eq': None,
                          'b_tilde': None,
                          'f0': Maxwellian6DUniform(),
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
        assert isinstance(params['particles'], (Particles6D, Particles5D))

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
            quad_pts = [quad_grid.points.flatten()
                        for quad_grid in self.derham.Vh_fem['0'].quad_grids]

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

        self._type = params['type']
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
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(self._M)

        # linear solver
        self._solver = getattr(it_solvers, params['type'])(self._M.domain)

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()

        self._rhs_v = self._u.space.zeros()
        self._u_new = self._u.space.zeros()

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):
        """
        TODO
        """

        # pointer to old coefficients
        un = self.variables[0]

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
        max_du = self.in_place_update(self._u_new)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling6DDensity:', info['success'])
            print('Iterations for CurrentCoupling6DDensity:', info['niter'])
            print('Maxdiff up for CurrentCoupling6DDensity:', max_du)
            print()


class ShearAlfvénCurrentCoupling5D(Propagator):
    r'''Crank-Nicolson step for shear Alfvén part in LinearMHDDriftkineticCC,

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_\alpha)^{-1} \mathcal {T^\alpha}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^\alpha} (\mathbb M^\rho_\alpha)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^\rho_\alpha}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) + \mathcal{P}^{B\top} \sum_k^{N_p} \omega_k \mu_k \Lambda^0(\mathbf{\eta}_k) \ \end{bmatrix} ,

    where :math:`\mathcal{P}^B = \hat \Pi^0 \left[ \frac{\hat b^1}{\sqrt g} \Lambda^2\right]`, :math:`\alpha \in \{1, 2, v\}` and :math:`\mathbb M^\rho_\alpha` is a weighted mass matrix in :math:`\alpha`-space, the weight being :math:`\rho_0`,
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

        # pointers to variables
        assert isinstance(u, (BlockVector, PolarVector))
        assert isinstance(b, (BlockVector, PolarVector))
        self._u = u
        self._b = b

        # parameters
        params_default = {'particles': None,
                          'u_space': 'Hdiv',
                          'b_eq': None,
                          'f0': Maxwellian5DUniform(),
                          'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
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

        self._type = params['type']
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
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # Instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    @property
    def variables(self):
        return [self._u, self._b]

    def __call__(self, dt):

        # current variables
        un = self.variables[0]
        bn = self.variables[1]

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

        # write new coeffs into self.variables
        max_du, max_db = self.in_place_update(self._u_tmp1, self._b_tmp1)

        if self._info and self._rank == 0:
            print('Status     for ShearAlfvén:', info['success'])
            print('Iterations for ShearAlfvén:', info['niter'])
            print('Maxdiff up for ShearAlfvén:', max_du)
            print('Maxdiff b2 for ShearAlfvén:', max_db)
            print()


class MagnetosonicCurrentCoupling5D(Propagator):
    r'''TODO'''

    def __init__(self, n, u, p, **params):

        from struphy.pic.particles import Particles5D

        assert isinstance(n, (StencilVector, PolarVector))
        assert isinstance(u, (BlockVector, PolarVector))
        assert isinstance(p, (StencilVector, PolarVector))
        self._n = n
        self._u = u
        self._p = p

        # parameters
        params_default = {'b': self.derham.Vh['2'].zeros(),
                          'particles': None,
                          'u_space': 'Hdiv',
                          'unit_b1': None,
                          'f0': Maxwellian5DUniform(),
                          'type': 'PBiConjugateGradientStab',
                          'pc': 'MassMatrixPreconditioner',
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
                                params['u_space'], 'cc_lin_mhd_5d_curlM', add_vector=False)

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
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = Compose(self._B, self._C)

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'],
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

    @property
    def variables(self):
        return [self._n, self._u, self._p]

    def __call__(self, dt):

        # current variables
        nn = self.variables[0]
        un = self.variables[1]
        pn = self.variables[2]

        # accumulate
        self._ACC.accumulate(self._particles,
                             self._b[0]._data, self._b[1]._data, self._b[2]._data,
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                             self._space_key_int, self._coupling_const)

        # solve for new u coeffs
        self._B.dot(pn, out=self._byn1)
        self._MJ.dot(self._b, out=self._byn2)
        self._byn2 -= self._ACC.vectors[0].dot(self._b)
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

        # write new coeffs into self.variables
        max_dn, max_du, max_dp = self.in_place_update(self._n_tmp1,
                                                      self._u_tmp1,
                                                      self._p_tmp1)

        if self._info and self._rank == 0:
            print('Status     for Magnetosonic:', info['success'])
            print('Iterations for Magnetosonic:', info['niter'])
            print('Maxdiff n3 for Magnetosonic:', max_dn)
            print('Maxdiff up for Magnetosonic:', max_du)
            print('Maxdiff p3 for Magnetosonic:', max_dp)
            print()
