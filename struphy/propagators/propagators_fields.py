import numpy as np

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.particles_to_grid import Accumulator
from struphy.polar.basic import PolarVector
from struphy.linear_algebra.iterative_solvers import pbicgstab
from struphy.kinetic_background.analytical import Maxwellian6D

from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api import preconditioner
from struphy.psydac_api.mass import WeightedMassOperator
from struphy.psydac_api.Hybrid_linear_operator import HybridOperators

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.iterative_solvers import pcg
from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector


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

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        params : dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, e, b, derham, mass_ops, params):

        assert isinstance(e, (BlockVector, PolarVector))
        assert isinstance(b, (BlockVector, PolarVector))

        self._e = e
        self._b = b
        self._info = params['info']

        # Define block matrix [[A B], [C I]] (without time step size dt in the diangonals)
        _A = mass_ops.M1

        # no dt
        self._B = Multiply(-1./2.,
                           Compose(derham.curl.transpose(), mass_ops.M2))
        self._C = Multiply(1./2., derham.curl)  # no dt

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
        max_de, max_db = self.in_place_update(_e, _b)

        if self._info:
            print('Status     for Maxwell:', info['success'])
            print('Iterations for Maxwell:', info['niter'])
            print('Maxdiff e1 for Maxwell:', max_de)
            print('Maxdiff b2 for Maxwell:', max_db)
            print()


class OhmCold(Propagator):
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

        assert isinstance(j, (BlockVector, PolarVector))
        assert isinstance(e, (BlockVector, PolarVector))
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
        max_dj, max_de = self.in_place_update(_j, _e)

        print('Maxdiff j1 for OhmCold:', max_dj)
        print('Maxdiff e2 for OhmCold:', max_de)
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

        assert isinstance(u, (BlockVector, PolarVector))
        assert isinstance(b, (BlockVector, PolarVector))
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
        self._B = Multiply(-1/2., Compose(_T.transpose(),
                           derham.curl.transpose(), mass_ops.M2))
        self._C = Multiply(1/2., Compose(derham.curl, _T))

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
        max_du, max_db = self.in_place_update(_u, _b)

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

        assert isinstance(n, (StencilVector, PolarVector))
        assert isinstance(u, (BlockVector, PolarVector))
        assert isinstance(p, (StencilVector, PolarVector))
        assert isinstance(b, (BlockVector, PolarVector))
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
        self._B = Multiply(-1/2., Compose(_UT,
                           derham.div.transpose(), mass_ops.M3))
        self._C = Multiply(1/2., Sum(Compose(derham.div, _S),
                           Multiply(2/3, Compose(_K, derham.div, _U))))

        self._MJ = getattr(mass_ops, id_MJ)
        self._Q = getattr(mhd_ops, id_Q)
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
        _u, info = self._schur_solver(
            un, self._B.dot(pn) - self._MJ.dot(bn)/2, dt)
        _p = pn - dt*self._C.dot(_u + un)
        _n = nn - dt/2*self._DIV.dot(self._Q.dot(_u + un))
        _b = 1*bn

        # write new coeffs into Propagator.variables
        max_dn, max_du, max_dp, max_db = self.in_place_update(_n, _u, _p, _b)

        if self._info and self._rank == 0:
            print('Status     for Magnetosonic:', info['success'])
            print('Iterations for Magnetosonic:', info['niter'])
            print('Maxdiff n3 for Magnetosonic:', max_dn)
            print('Maxdiff up for Magnetosonic:', max_du)
            print('Maxdiff p3 for Magnetosonic:', max_dp)
            print('Maxdiff b2 for Magnetosonic:', max_db)
            print()


class Hybrid_potential( Propagator ):
    r'''Crank-Nicolson step for the Faraday's law.

    math::

        \begin{align}
        \textnormal{Faraday's law}\qquad& \frac{\partial {\mathbf A}}{\partial t} = - \frac{\nabla \times \nabla \times A}{n} \times \nabla \times {\mathbf A} - \frac{\int ({\mathbf A} - {\mathbf p}f \mathrm{d}{\mathbf p})}{n} \times \nabla \times {\mathbf A}, \quad n = \int f \mathrm{d}{\mathbf p}.
        \end{align}

    Parameters
    ---------- 
        a : psydac.linalg.block.BlockVector
            FE coefficients of vector potential as 1-form

        a_space : str
            Space identifier of vector potential: 'Hcurl.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
            
        mass_ops : struphy.psydac_api.mass.WeightedMassOperators
            Weighted mass matrices from struphy.psydac_api.mass. 
    '''

    def __init__(self, a, a_space, beq, derham, mass_ops, domain, particles, nqs, p_shape, p_size):

        

        assert isinstance(a, (BlockVector, PolarVector))
        assert a_space in {'Hcurl', 'Hdiv', 'H1vec'}

        self._a = a
        self._rank = derham.comm.Get_rank()
        self._beq = beq

        self._particles = particles

        self._domain = domain
        self._derham = derham

        # Initialize Accumulator object for getting density from particles
        self._pts_x = 1.0 / (2.0*derham.Nel[0]) * np.polynomial.legendre.leggauss(nqs[0])[0] + 1.0 / (2.0*derham.Nel[0])
        self._pts_y = 1.0 / (2.0*derham.Nel[1]) * np.polynomial.legendre.leggauss(nqs[1])[0] + 1.0 / (2.0*derham.Nel[1])
        self._pts_z = 1.0 / (2.0*derham.Nel[2]) * np.polynomial.legendre.leggauss(nqs[2])[0] + 1.0 / (2.0*derham.Nel[2])
        self._nqs   = nqs 
        self._p_shape = p_shape
        self._p_size = p_size
        self._accum_density = Accumulator(derham, domain, 'H1', 'hybrid_fA_density',
                                  do_vector=False, symmetry='None')

        self._accum_density.accumulate(self._particles, np.array(self._derham.Nel), np.array(self._nqs), np.array(self._pts_x), np.array(self._pts_y), np.array(self._pts_z), np.array(self._p_shape), np.array(self._p_size))

        # Initialize Accumulator object for getting the matrix and vector related with vector potential 
        self._accum_potential = Accumulator(derham, domain, 'Hcurl', 'hybrid_fA_Arelated',  
                                  do_vector=True, symmetry='symm')

        self._accum_potential.accumulate(self._particles)
        
        # for testing of hybrid linear operators 
        self._density = StencilMatrix(self._derham.Vh[self._derham.spaces_dict['H1']], self._derham.Vh[self._derham.spaces_dict['H1']], backend=PSYDAC_BACKEND_GPYCCEL)
        self._hybrid_ops = HybridOperators(self._derham, self._domain, self._density, self._a, self._beq)


    @property
    def variables(self):
        return self._a

    def __call__(self, dt):

        # for getting density from particles. 
        self._accum_density.accumulate(self._particles, np.array(self._derham.Nel), np.array(self._nqs), np.array(self._pts_x), np.array(self._pts_y), np.array(self._pts_z), np.array(self._p_shape), np.array(self._p_size))
        # for getting the matrix and vector related with vector potential 
        self._accum_potential.accumulate(self._particles)
        # Iniitialize hybrid linear operators 
        self._hybrid_ops.HybridM1
        # current variables
        an = self.variables[0]

        # allocate temporary FemFields _u, _b during solution
        #_a, info = self._schur_solver(un, self._B.dot(bn), dt)

        # write new coeffs into Propagator.variables
        #max_du, max_db = self.in_place_update(_u, _b)


class CurrentCoupling6DDensity(Propagator):
    """
    TODO
    """

    def __init__(self, particles, derham, domain, mass_ops, solver_params, coupling_params, u, u_space, *b_vectors, f0=None):

        assert isinstance(u, (BlockVector, PolarVector))

        for b in b_vectors:
            assert isinstance(b, (BlockVector, PolarVector))

        assert u_space in {'Hcurl', 'Hdiv', 'H1vec'}

        if u_space == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(derham.spaces_dict[u_space])

        # needed variables
        self._particles = particles
        self._u = u
        self._b_vectors = b_vectors

        # load accumulator
        self._accumulator = Accumulator(
            derham, domain, u_space, 'cc_lin_mhd_6d_1', do_vector=False, symmetry='asym')

        nuh = coupling_params['nuh']
        kap = coupling_params['kappa']
        Ab = coupling_params['Ab']
        Ah = coupling_params['Ah']
        Zh = coupling_params['Zh']

        self._coupling_mat = nuh*kap*Zh/Ab

        # distribution function (control variate, without control variate f0=None)
        self._f0 = f0

        # evaluate and save nh0*|det(DF)| (H1vec) or nh0/|det(DF)| (Hdiv) at quadrature points for control variate
        if f0 is not None:

            # f0 must be a 6d Maxwellian
            assert isinstance(f0, Maxwellian6D)

            quad_pts = [quad_grid.points.flatten()
                        for quad_grid in derham.Vh_fem['0'].quad_grids]

            if u_space == 'H1vec':
                self._nh0_at_quad = domain.pull(
                    [f0.n], *quad_pts, kind='3_form', squeeze_out=False, coordinates='logical')
            else:
                self._nh0_at_quad = domain.push(
                    [f0.n], *quad_pts, kind='3_form', squeeze_out=False)

        # FEM spaces and basis extraction operators for u and b
        self._fem_space_u = derham.Vh_fem[derham.spaces_dict[u_space]]
        self._fem_space_b = derham.Vh_fem['2']

        self._Eu = derham.E[derham.spaces_dict[u_space]]
        self._Eb = derham.E['2']

        self._EuT = derham.E[derham.spaces_dict[u_space]].transpose()
        self._EbT = derham.E['2'].transpose()

        # mass matrix in system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        self._M = getattr(mass_ops, 'M' + derham.spaces_dict[u_space] + 'n')

        # preconditioner
        if solver_params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, solver_params['pc'])
            self._pc = pc_class(self._M)

        self._solver_params = solver_params
        self._info = solver_params['info']
        self._rank = derham.comm.Get_rank()

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):
        """
        TODO
        """

        # old coefficients
        u_old = self.variables[0]

        # sum up total magnetic field
        b_full = self._b_vectors[0].space.zeros()

        for b in self._b_vectors:
            b_full += b

        # extract coefficients to tensor product space
        b_full = self._EbT.dot(b_full)

        # update ghost regions because of non-local access in pusher kernel
        b_full.update_ghost_regions()

        # perform accumulation  (either with or without control variate)
        if self._f0 is not None:

            # evaluate magnetic field at quadrature points
            b_quad = WeightedMassOperator.eval_quad(self._fem_space_b, b_full)

            mat12 = self._coupling_mat*b_quad[2]*self._nh0_at_quad
            mat13 = -self._coupling_mat*b_quad[1]*self._nh0_at_quad
            mat23 = self._coupling_mat*b_quad[0]*self._nh0_at_quad

            control_mat_at_quad = [[None, mat12, mat13],
                                   [None,  None, mat23],
                                   [None,  None,  None]]

            self._accumulator.accumulate(self._particles,
                                         b_full[0]._data, b_full[1]._data, b_full[2]._data,
                                         self._space_key_int, self._coupling_mat,
                                         control_mat=control_mat_at_quad)
        else:
            self._accumulator.accumulate(self._particles,
                                         b_full[0]._data, b_full[1]._data, b_full[2]._data,
                                         self._space_key_int, self._coupling_mat)

        # define system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        lhs = Sum(self._M, Multiply(-dt/2, self._accumulator.A0))
        rhs = Sum(self._M, Multiply(dt/2, self._accumulator.A0)).dot(u_old)

        solver_type = self._solver_params['type']

        # solve linear system for updated u coefficients
        if solver_type == 'pcg':

            u_new, info = pcg(lhs, rhs, self._pc, x0=u_old, tol=self._solver_params['tol'],
                              maxiter=self._solver_params['maxiter'], verbose=self._solver_params['verbose'])

        elif solver_type == 'pbicgstab':

            u_new, info = pbicgstab(lhs, rhs, self._pc, x0=u_old, tol=self._solver_params['tol'],
                                    maxiter=self._solver_params['maxiter'], verbose=self._solver_params['verbose'])

        else:
            raise NotImplementedError(
                f'Solver type {solver_type} is not implemented.')

        # write new coeffs into Propagator.variables
        max_du = self.in_place_update(u_new)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling6DDensity:', info['success'])
            print('Iterations for CurrentCoupling6DDensity:', info['niter'])
            print('Maxdiff up for CurrentCoupling6DDensity:', max_du)
            print()
            

class ShearAlfvén_CurrentCoupling5D( Propagator ):
    r'''TODO'''

    def __init__(self, particles, derham, domain, mass_ops, mhd_ops, u, u_space, b, beq, params):

        assert isinstance(u, (BlockVector, PolarVector))
        assert isinstance(b, (BlockVector, PolarVector))
        assert u_space in {'Hcurl', 'Hdiv', 'H1vec'}

        self._particles = particles
        self._derham = derham
        self._u = u
        self._b = b
        self._beq = beq
        self._info = params['info']
        self._rank = derham.comm.Get_rank()

        self._PB = getattr(mhd_ops, 'PB')
        self._ACC = Accumulator(self._derham, domain, 'H1', 'cc_lin_mhd_5d_mu', do_vector=True)

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
        self._B2 = Multiply(-1/2., Compose(_T.transpose(), derham.curl.transpose(), self._PB.transpose()))
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

        # accumulate scalar
        self._ACC.accumulate(self._particles)

        # allocate temporary FemFields _u, _b during solution
        _u, info = self._schur_solver(un, self._B.dot(bn) + self._B2.dot(self._ACC.vector), dt)
        _b = bn - dt*self._C.dot(_u + un)

        # write new coeffs into Propagator.variables
        max_du, max_db = self.in_place_update(_u, _b)

        self._particles.save_magnetic_energy(self._derham, self._PB.dot(_b + self._beq))

        if self._info and self._rank ==0:
            print('Status     for ShearAlfvén:', info['success'])
            print('Iterations for ShearAlfvén:', info['niter'])
            print('Maxdiff up for ShearAlfvén:', max_du)
            print('Maxdiff b2 for ShearAlfvén:', max_db)
            print()

class Magnetosonic_CurrentCoupling5D( Propagator ):
    r'''TODO'''

    def __init__(self, particles, derham, domain, mass_ops, mhd_ops,n, u, p, b, unit_b1, u_space, params):

        assert isinstance(n, (StencilVector, PolarVector))
        assert isinstance(u, (BlockVector, PolarVector))
        assert isinstance(p, (StencilVector, PolarVector))
        assert isinstance(b, (BlockVector, PolarVector))
        assert u_space in {'Hcurl', 'Hdiv', 'H1vec'}

        self._particles = particles
        self._derham = derham
        self._domain = domain
        self._curl_norm_b = derham.curl.dot(unit_b1)
        self._curl_norm_b.update_ghost_regions()
        self._n = n
        self._u = u
        self._p = p
        self._b = b
        self._bc = derham.bc
        self._info = params['info']
        self._rank = derham.comm.Get_rank()

        #TODO
        self._scale_vec = 1.

        self._ACC = Accumulator(self._derham, domain, u_space, 'cc_lin_mhd_5d_M', do_vector=True)
        
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

        if u_space == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(derham.spaces_dict[u_space])

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

        # accumulate
        self._ACC.accumulate(self._particles, 
                             self._b[0]._data, self._b[1]._data, self._b[2]._data, 
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data, 
                             self._space_key_int, self._scale_vec)

        self._ACC.vector.shape

        # allocate temporary FemFields _u, _b during solution
        _u, info = self._schur_solver(un, self._B.dot(pn) - self._MJ.dot(bn)/2 - self._ACC.vector/2, dt)
        _p = pn - dt*self._C.dot(_u + un)
        _n = nn - dt/2*self._DIV.dot(self._Q.dot(_u + un))
        _b = 1*bn
        
        # write new coeffs into Propagator.variables
        max_dn, max_du, max_dp, max_db = self.in_place_update(_n, _u, _p, _b)

        if self._info and self._rank == 0:
            print('Status     for Magnetosonic:', info['success'])
            print('Iterations for Magnetosonic:', info['niter'])
            print('Maxdiff n3 for Magnetosonic:', max_dn)
            print('Maxdiff up for Magnetosonic:', max_du)
            print('Maxdiff p3 for Magnetosonic:', max_dp)
            print('Maxdiff b2 for Magnetosonic:', max_db)
            print()
