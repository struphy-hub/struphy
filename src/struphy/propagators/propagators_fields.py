'Only FEEC variables are updated.'


import numpy as np
from numpy import zeros

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.base import Particles
from struphy.pic.particles import Particles6D, Particles5D
from struphy.polar.basic import PolarVector
from struphy.kinetic_background.base import Maxwellian
from struphy.kinetic_background.maxwellians import Maxwellian3D, GyroMaxwellian2D
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.feec import preconditioner
from struphy.feec.mass import WeightedMassOperator
from struphy.feec.basis_projection_ops import BasisProjectionOperator, CoordinateProjector
from struphy.feec.variational_utilities import BracketOperator
from struphy.io.setup import descend_options_dict

from psydac.linalg.solvers import inverse
from psydac.linalg.basic import IdentityOperator
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector, BlockLinearOperator, BlockVectorSpace
import struphy.feec.utilities as util

from copy import deepcopy


class Maxwell(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf E \in H(\textnormal{curl})` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \frac{\partial \mathbf E}{\partial t} \cdot \mathbf F \, \textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times \mathbf F \,\textrm d \mathbf x = 0\,, \qquad \forall \, \mathbf F \in H(\textnormal{curl}) \,.
        \\[2mm]
        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`.
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 e: BlockVector,
                 b: BlockVector,
                 *,
                 solver: dict = options(default=True)['solver']):

        super().__init__(e, b)

        self._info = solver['info']
        self._rank = self.derham.comm.Get_rank()

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1

        # no dt
        self._B = -1/2 * self.derham.curl.T @ self.mass_ops.M2
        self._C = 1/2 * self.derham.curl

        # Preconditioner
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(self.mass_ops.M1)

        # Instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         solver['type'][0],
                                         pc=pc,
                                         tol=solver['tol'],
                                         maxiter=solver['maxiter'],
                                         verbose=solver['verbose'])

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


class OhmCold(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf j \in H(\textnormal{curl})` and :math:`\mathbf E \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \frac{1}{n_0} \frac{\partial \mathbf j}{\partial t} \cdot \mathbf F \,\textrm d \mathbf x &= \frac{1}{\varepsilon} \int_\Omega \mathbf E \cdot \mathbf F \,\textrm d \mathbf x \qquad \forall \,\mathbf F \in H(\textnormal{curl})\,,
        \\[2mm]
        -\frac{\partial \mathbf E}{\partial t} &= \frac{\alpha^2}{\varepsilon} \mathbf j \,,

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`, such that

    .. math::

        \begin{bmatrix}
            \mathbf j^{n+1} - \mathbf j^n \\
            \mathbf e^{n+1} - \mathbf e^n
        \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix}
            0 & \frac{1}{\varepsilon} \mathbb M_{1/n_0}^{-1} \\
            - \frac{1}{\varepsilon} \mathbb M_{1/n_0}^{-1} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \alpha^2 \mathbb M_{1/n_0} (\mathbf j^{n+1} + \mathbf j^{n}) \\
            \mathbb M_1 (\mathbf e^{n+1} + \mathbf e^{n})
        \end{bmatrix} \,.
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 j: BlockVector,
                 e: BlockVector,
                 *,
                 alpha: float = 1.,
                 epsilon: float = 1.,
                 solver: dict = options(default=True)['solver']):

        super().__init__(e, j)

        self._info = solver['info']
        self._alpha = alpha
        self._epsilon = epsilon

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1ninv

        self._B = -1/2 * 1/self._epsilon * self.mass_ops.M1  # no dt

        # Preconditioner
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate Schur solver (constant in this case)
        _BC = 1/2 * self._alpha**2 / self._epsilon * self._B

        self._schur_solver = SchurSolver(_A, _BC,
                                         solver['type'][0],
                                         pc=pc,
                                         tol=solver['tol'],
                                         maxiter=solver['maxiter'],
                                         verbose=solver['verbose'])

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


class JxBCold(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf j \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \frac{1}{n_0} \frac{\partial \mathbf j}{\partial t} \cdot \mathbf F \,\textrm d \mathbf x 
        = \frac{1}{\varepsilon} \int_\Omega \frac{1}{n_0} (\mathbf j \times \mathbf B_0) \cdot \mathbf F \,\textrm d \mathbf x \qquad \forall \,\mathbf F \in H(\textnormal{curl})\,,

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point), such that

    .. math::

        \mathbb M_{1/n_0} \left( \mathbf j^{n+1} - \mathbf j^n \right) = \frac{\Delta t}{2} \frac{1}{\varepsilon} \mathbb M_{B_0/n_0} \left( \mathbf j^{n+1} - \mathbf j^n \right)\,.
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 j: BlockVector,
                 *,
                 epsilon: float = 1.,
                 solver: dict = options(default=True)['solver']):

        super().__init__(j)

        self._info = solver['info']

        # mass matrix in system (M - dt/2 * A)*j^(n + 1) = (M + dt/2 * A)*j^n
        self._M = self.mass_ops.M1ninv
        self._A = -1/epsilon * self.mass_ops.M1Bninv  # no dt

        # Preconditioner
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate linear solver
        self._solver = inverse(self._M,
                               solver['type'][0],
                               pc=pc,
                               x0=self.feec_vars[0],
                               tol=solver['tol'],
                               maxiter=solver['maxiter'],
                               verbose=solver['verbose'])

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


class ShearAlfven(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x
        =\int_\Omega \tilde{\mathbf B } \cdot \nabla \times (\mathbf{B}_0 \times \mathbf V) \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_\alpha)^{-1} \mathcal {T^\alpha}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^\alpha} (\mathbb M^\rho_\alpha)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^\rho_\alpha}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    where :math:`\alpha \in \{1, 2, v\}` and :math:`\mathbb M^\rho_\alpha` is a weighted mass matrix in :math:`\alpha`-space, the weight being :math:`\rho_0`,
    the MHD equilibirum density. The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 u: BlockVector,
                 b: BlockVector,
                 *,
                 u_space: str,
                 solver: dict = options(default=True)['solver']):

        super().__init__(u, b)

        assert u_space in {'Hcurl', 'Hdiv', 'H1vec'}

        self._info = solver['info']
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_M = 'M' + self.derham.space_to_form[u_space] + 'n'
        id_T = 'T' + self.derham.space_to_form[u_space]

        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)

        self._B = -1/2 * _T.T @ self.derham.curl.T @ self.mass_ops.M2
        self._C = 1/2 * self.derham.curl @ _T

        # Preconditioner
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         solver['type'][0],
                                         pc=pc,
                                         tol=solver['tol'],
                                         maxiter=solver['maxiter'],
                                         verbose=solver['verbose'])

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
            print('Status     for ShearAlfven:', info['success'])
            print('Iterations for ShearAlfven:', info['niter'])
            print('Maxdiff up for ShearAlfven:', max_du)
            print('Maxdiff b2 for ShearAlfven:', max_db)
            print()


class ShearAlfvenB1(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{curl})` such that

    .. math::

        &\int_\Omega \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x
        =\int_\Omega \tilde{\mathbf B } \cdot \nabla \times (\mathbf{B}_0 \times \mathbf V) \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &\int_\Omega \frac{\partial \tilde{\mathbf{B}}}{\partial t} \cdot \mathbf C\,\textnormal d \mathbf x - \int_\Omega \mathbf C \cdot\nabla\times   \left( \tilde{\mathbf{U}} \times \mathbf{B}_0 \right) \textrm d \mathbf x = 0 \qquad \forall \, \mathbf C \in H(\textrm{curl})\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_2)^{-1} \mathcal {T^2}^\top \mathbb C \mathbb M_1^{-1}\\ - \mathbb M_1^{-1} \mathbb C^\top \mathcal {T^2} (\mathbb M^\rho_2)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^\rho_2}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_1(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    where :math:`\mathbb M^\rho_2` is a weighted mass matrix in 2-space, the weight being :math:`\rho_0`,
    the MHD equilibirum density. 
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        dct['solver_M1'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                     ('cg', None)],
                            'tol': 1.e-8,
                            'maxiter': 3000,
                            'info': False,
                            'verbose': False,
                            'recycle': True}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 u: BlockVector,
                 b: BlockVector,
                 *,
                 solver: dict = options(default=True)['solver'],
                 solver_M1: dict = options(default=True)['solver_M1']):

        super().__init__(u, b)

        self._info = solver['info']
        self._rank = self.derham.comm.Get_rank()

        # define inverse of M1
        if solver_M1['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver_M1['type'][1])
            pc = pc_class(self.mass_ops.M1)

        M1_inv = inverse(self.mass_ops.M1,
                         solver_M1['type'][0],
                         pc=pc,
                         tol=solver_M1['tol'],
                         maxiter=solver_M1['maxiter'],
                         verbose=solver_M1['verbose'])

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M2n
        self._B = 1/2 * self.mass_ops.M2B @ self.derham.curl
        # I still have to invert M1
        self._C = 1/2 * M1_inv @ self.derham.curl.T @ self.mass_ops.M2B

        # Preconditioner
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(getattr(self.mass_ops, 'M2n'))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         solver['type'][0],
                                         pc=pc,
                                         tol=solver['tol'],
                                         maxiter=solver['maxiter'],
                                         verbose=solver['verbose'])

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
            print('Status     for ShearAlfvenB1:', info['success'])
            print('Iterations for ShearAlfvenB1:', info['niter'])
            print('Maxdiff up for ShearAlfvenB1:', max_du)
            print('Maxdiff b2 for ShearAlfvenB1:', max_db)
            print()


class Hall(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf B \in H(\textnormal{curl})` such that

    .. math:: 

        \int_\Omega \frac{\partial \tilde{\mathbf{B}}}{\partial t} \cdot \mathbf C\,\textnormal d \mathbf x + \frac{1}{\varepsilon} \int_\Omega \nabla\times \mathbf C \cdot  \left( \frac{\nabla\times \tilde{\mathbf{B}}}{\rho_0}\times \mathbf{B}_0 \right) \textrm d \mathbf x = 0 \qquad \forall \, \mathbf C \in H(\textrm{curl})

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point):

    .. math::

        \mathbf b^{n+1} - \mathbf b^n 
        = \frac{\Delta t}{2} \mathbb M_1^{-1} \mathbb C^\top  \mathbb M^{\mathcal{T},\rho}_2  \mathbb C  (\mathbf b^{n+1} + \mathbf b^n)  ,

    where :math:`\mathbb M^{\mathcal{T},\rho}_2` is a weighted mass matrix in 2-space, the weight being :math:`\frac{\mathcal{T}}{\rho_0}`,
    the MHD equilibirum density :math:`\rho_0` as a 0-form, and rotation matrix :math:`\mathcal{T} \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,`. 
    The solution of the above system is based on the Pre-conditioned Biconjugate Gradient Stabilized algortihm (PBiConjugateGradientStab).
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 b: BlockVector,
                 *,
                 epsilon: float = 1.,
                 solver: dict = options(default=True)['solver']):

        super().__init__(b)

        self._info = solver['info']
        self._rank = self.derham.comm.Get_rank()
        self._tol = solver['tol']
        self._maxiter = solver['maxiter']
        self._verbose = solver['verbose']

        # mass matrix in system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        id_M = 'M1'
        id_M2Bn = 'M2Bn'
        self._M = getattr(self.mass_ops, id_M)
        self._M2Bn = getattr(self.mass_ops, id_M2Bn)
        self._A = 1./epsilon * self.derham.curl.T @ self._M2Bn @ self.derham.curl

        # Preconditioner
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # Instantiate linear solver
        self._solver = inverse(self._M,
                               solver['type'][0],
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


class Magnetosonic(Propagator):
    r'''
    :ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\tilde \rho \in L^2, \tilde{\mathbf U} \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}, \tilde p \in L^2` such that

    .. math::
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 

        \int \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textrm d \mathbf x  - \int \tilde p\, \nabla \cdot \mathbf V \,\textrm d \mathbf x
        =\int (\nabla\times\mathbf{B}_0)\times \tilde{\mathbf{B}} \cdot \mathbf V\,\textrm d \mathbf x
        \qquad \forall \ \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1} - \mathbf p^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_\alpha)^{-1} {\mathcal U^\alpha}^\top \mathbb D^\top \mathbb M_3 \\ - \mathbb D \mathcal S^\alpha - (\gamma - 1) \mathcal K^\alpha \mathbb D \mathcal U^\alpha & 0 \end{bmatrix} 
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1} + \mathbf p^n) \end{bmatrix} + \begin{bmatrix} \Delta t (\mathbb M^\rho_\alpha)^{-1} \mathbb M^J_\alpha \mathbf b^n \\ 0 \end{bmatrix},

    where :math:`\alpha \in \{1, 2, v\}` and :math:`\mathcal U^2 = \mathbb Id`; moreover, :math:`\mathbb M^\rho_\alpha` and 
    :math:`\mathbb M^J_\alpha` are weighted mass matrices in :math:`\alpha`-space, 
    the weights being the MHD equilibirum density :math:`\rho_0`
    and the curl of the MHD equilibrium current density :math:`\mathbf J_0 = \nabla \times \mathbf B_0`. 
    Density update is decoupled:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathbb D \mathcal Q^\alpha (\mathbf u^{n+1} + \mathbf u^n) \,.
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 n: StencilVector,
                 u: BlockVector,
                 p: StencilVector,
                 *,
                 u_space: str,
                 b: BlockVector,
                 solver: dict = options(default=True)['solver']):

        super().__init__(n, u, p)

        assert u_space in {'Hcurl', 'Hdiv', 'H1vec'}

        self._info = solver['info']
        self._bc = self.derham.dirichlet_bc
        self._rank = self.derham.comm.Get_rank()

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = 'M' + self.derham.space_to_form[u_space] + 'n'
        id_MJ = 'M' + self.derham.space_to_form[u_space] + 'J'

        if u_space == 'Hcurl':
            id_S, id_U, id_K, id_Q = 'S1', 'U1', 'K3', 'Q1'
        elif u_space == 'Hdiv':
            id_S, id_U, id_K, id_Q = 'S2', None, 'K3', 'Q2'
        elif u_space == 'H1vec':
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

        self._b = b

        # preconditioner
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         solver['type'][0],
                                         pc=pc,
                                         tol=solver['tol'],
                                         maxiter=solver['maxiter'],
                                         verbose=solver['verbose'])

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


class MagnetosonicUniform(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\tilde \rho \in L^2, \tilde{\mathbf U} \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}, \tilde p \in L^2` such that

    .. math::
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 

        \int \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textrm d \mathbf x  - \int \tilde p\, \nabla \cdot \mathbf V \,\textrm d \mathbf x
        = 0
        \qquad \forall \ \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

        &\frac{\partial \tilde p}{\partial t}
        + \frac{5}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

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

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 n: StencilVector,
                 u: BlockVector,
                 p: StencilVector,
                 *,
                 solver: dict = options(default=True)['solver']):

        super().__init__(n, u, p)

        self._info = solver['info']
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
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         solver['type'][0],
                                         pc=pc,
                                         tol=solver['tol'],
                                         maxiter=solver['maxiter'],
                                         verbose=solver['verbose'])

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
            self._weight = [
                [0.*self._weight_pre[k] for k in range(3)],
                [0.*self._weight_pre[k] for k in range(3)],
                [0.*self._weight_pre[k] for k in range(3)],
            ]
            # self._weight = [[self._weight_pre[0], self._weight_pre[2], self._weight_pre[1]], [self._weight_pre[2], self._weight_pre[1], self._weight_pre[0]], [self._weight_pre[1], self._weight_pre[0], self._weight_pre[2]]]
            HybridM1 = self.mass_ops.assemble_weighted_mass(
                self._weight, 'Hcurl', 'Hcurl')

            # next prepare for solving linear system
            _LHS = self._M1 + HybridM1 @ self._L2
            _RHS2 = HybridM1.dot(self._RHS) + self._rhs

            # TODO: unknown function 'pcg', use new solver API
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
                         'verbose': False,
                         'recycle': True}
        return dct


class CurrentCoupling6DDensity(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\tilde{\mathbf{U}}  \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` such that

    .. math::

        &\int_\Omega \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V \,\textrm d \mathbf x = \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \int_\Omega n_\textnormal{h}\tilde{\mathbf{U}} \times(\mathbf{B}_0+\tilde{\mathbf{B}}) \cdot \mathbf V \,\textrm d \mathbf x 
        \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &n_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\,\textnormal{d}^3 \mathbf v\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point).
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 u: BlockVector,
                 *,
                 particles: Particles6D,
                 u_space: str,
                 b_eq: BlockVector | PolarVector,
                 b_tilde: BlockVector | PolarVector,
                 Ab: int = 1,
                 Ah: int = 1,
                 epsilon: float = 1.,
                 solver: dict = options(default=True)['solver']):

        super().__init__(u)

        # assert parameters and expose some quantities to self
        if u_space == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[u_space])

        self._particles = particles
        self._b_eq = b_eq
        self._b_tilde = b_tilde

        if self._particles.control_variate:

            # control variate method is only valid with Maxwellian distributions
            assert isinstance(self._particles.f0, Maxwellian)
            assert u_space == 'Hdiv'

            # evaluate and save nh0/|det(DF)| (push-forward) at quadrature points for control variate
            quad_pts = [quad_grid[nquad].points.flatten()
                        for quad_grid, nquad in zip(self.derham.Vh_fem['0']._quad_grids, self.derham.Vh_fem['0'].nquads)]

            self._nh0_at_quad = self.domain.push(
                self._particles.f0.n, *quad_pts, kind='3', squeeze_out=False)

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

        self._type = solver['type'][0]
        self._tol = solver['tol']
        self._maxiter = solver['maxiter']
        self._info = solver['info']
        self._verbose = solver['verbose']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_const = Ah / Ab / epsilon
        # load accumulator
        self._accumulator = Accumulator(particles,
                                        u_space,
                                        accum_kernels.cc_lin_mhd_6d_1,
                                        self.derham,
                                        self.domain.args_domain,
                                        add_vector=False,
                                        symmetry='asym')

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.extraction_ops['2'].transpose()

        # mass matrix in system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        u_id = self.derham.space_to_form[u_space]
        self._M = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(self._M)

        # linear solver
        self._solver = inverse(self._M,
                               solver['type'][0],
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
        if self._particles.control_variate:

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

            self._accumulator(self._b_full2[0]._data,
                              self._b_full2[1]._data,
                              self._b_full2[2]._data,
                              self._space_key_int,
                              self._coupling_const,
                              control_mat=[[None, self._mat12, self._mat13],
                                           [self._mat21, None,
                                            self._mat23],
                                           [self._mat31, self._mat32, None]])
        else:
            self._accumulator(self._b_full2[0]._data,
                              self._b_full2[1]._data,
                              self._b_full2[2]._data,
                              self._space_key_int,
                              self._coupling_const)

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


class ShearAlfvenCurrentCoupling5D(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \left\{ 
            \begin{aligned} 
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf{x} = \int \left(\tilde{\mathbf B} - \frac{A_\textnormal{h}}{A_b} \iint f^\text{vol} \mu \mathbf{b}_0\textnormal{d} \mu \textnormal{d} v_\parallel \right) \cdot \nabla \times (\mathbf B_0 \times \mathbf V) \, \textnormal{d} \mathbf{x} \quad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,, \,,
                \\
                &\frac{\partial \tilde{\mathbf B}}{\partial t} = - \nabla \times (\mathbf B_0 \times \tilde{\mathbf U}) \,.
            \end{aligned}
        \right.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

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
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        dct['filter'] = {'use_filter': None,
                         'modes': (0, 1),
                         'repeat': 3,
                         'alpha': 0.5}
        dct['turn_off'] = False

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 u: BlockVector,
                 b: BlockVector,
                 *,
                 particles: Particles5D,
                 absB0: StencilVector,
                 unit_b1: BlockVector,
                 u_space: str,
                 solver: dict = options(default=True)['solver'],
                 filter: dict = options(default=True)['filter'],
                 coupling_params: dict,
                 accumulated_magnetization: BlockVector):

        super().__init__(u, b)

        self._particles = particles
        self._unit_b1 = unit_b1
        self._absB0 = absB0

        self._info = solver['info']
        self._rank = self.derham.comm.Get_rank()

        self._scale_vec = coupling_params['Ah'] / coupling_params['Ab']

        self._E1T = self.derham.extraction_ops['1'].transpose()
        self._unit_b1 = self._E1T.dot(self._unit_b1)

        self._accumulated_magnetization = accumulated_magnetization

        self._ACC = Accumulator(particles,
                                u_space,
                                accum_kernels_gc.cc_lin_mhd_5d_M,
                                self.derham,
                                self.domain.args_domain,
                                add_vector=True,
                                symmetry='symm',
                                filter_params=filter)

        # if self._particles.control_variate:

        #     # control variate method is only valid with Maxwellian distributions with "zero perp mean velocity".
        #     assert isinstance(self._particles.f0, Maxwellian)

        #     self._ACC.init_control_variate(self.mass_ops)

        #     # evaluate and save f0.n at quadrature points
        #     quad_pts = [quad_grid[nquad].points.flatten()
        #                 for quad_grid, nquad in zip(self.derham.Vh_fem['0']._quad_grids, self.derham.Vh_fem['0'].nquads)]

        #     n0_at_quad = self.domain.push(
        #         self._particles.f0.n, *quad_pts, kind='0', squeeze_out=False)

        #     # evaluate M0 = unit_b1 (1form) / absB0 (0form) * 2 * vth_perp² at quadrature points
        #     quad_pts_array = self.domain.prepare_eval_pts(*quad_pts)[:3]

        #     vth_perp = self.particles.f0.vth(*quad_pts_array)[1]

        #     absB0_at_quad = WeightedMassOperator.eval_quad(self.derham.Vh_fem['0'], self._absB0)

        #     unit_b1_at_quad = WeightedMassOperator.eval_quad(self.derham.Vh_fem['1'], self._unit_b1)

        #     self._M0_at_quad = unit_b1_at_quad / absB0_at_quad * vth_perp**2 * n0_at_quad * self._scale_vec

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_M = 'M' + self.derham.space_to_form[u_space] + 'n'
        id_T = 'T' + self.derham.space_to_form[u_space]

        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)

        self._B = -1/2 * _T.T @ self.derham.curl.T @ self.mass_ops.M2
        self._C = 1/2 * self.derham.curl @ _T
        self._B2 = -1/2 * _T.T @ self.derham.curl.T

        # Preconditioner
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_M))

        # Instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         solver['type'][0],
                                         pc=pc,
                                         tol=solver['tol'],
                                         maxiter=solver['maxiter'],
                                         verbose=solver['verbose'])

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

        # perform accumulation (either with or without control variate)
        # if self._particles.control_variate:

        #     self._ACC.accumulate(self._particles,
        #                          self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
        #                          self._scale_vec, 0.,
        #                          control_vec=[self._M0_at_quad[0], self._M0_at_quad[1], self._M0_at_quad[2]])
        # else:
        #     self._ACC.accumulate(self._particles,
        #                          self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
        #                          self._scale_vec, 0.)

        self._ACC(self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                  self._scale_vec, 0.)

        self._ACC.vectors[0].copy(out=self._accumulated_magnetization)

        # solve for new u coeffs (no tmps created here)
        byn = self._B.dot(bn, out=self._byn)
        b2acc = self._B2.dot(self._ACC.vectors[0], out=self._tmp_acc)
        byn += b2acc

        # b2acc.copy(out=self._accumulated_magnetization)

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
            print('Status     for ShearAlfven:', info['success'])
            print('Iterations for ShearAlfven:', info['niter'])
            print('Maxdiff up for ShearAlfven:', max_du)
            print('Maxdiff b2 for ShearAlfven:', max_db)
            print()


class MagnetosonicCurrentCoupling5D(Propagator):
    r'''
    :ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\tilde \rho \in L^2, \tilde{\mathbf U} \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}, \tilde p \in L^2` such that

    .. math::

        \left\{
            \begin{aligned}
                &\frac{\partial \tilde{\rho}}{\partial t} = - \nabla \cdot (\rho_0 \tilde{\mathbf U}) \,,
                \\
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf{x} = \int (\nabla \times \mathbf B_0) \times \tilde{\mathbf B} \cdot \mathbf V \, \textnormal{d} \mathbf x + \frac{A_\textnormal{h}}{A_b}\iint f^\text{vol} \mu \mathbf b_0 \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf V) \, \textnormal{d} \mathbf x \textnormal{d} v_\parallel \textnormal{d} \mu + \int \tilde p \nabla \cdot \mathbf V \, \textnormal{d} \mathbf x \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
                \\
                &\frac{\partial \tilde p}{\partial t} = - \nabla \cdot (p_0 \tilde{\mathbf U}) - (\gamma - 1) p_0 \nabla \cdot \tilde{\mathbf U} \,.
            \end{aligned} 
        \right.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \boldsymbol{\rho}^{n+1} - \boldsymbol{\rho}^n = - \frac{\Delta t}{2} \mathbb D \mathcal Q^\alpha (\mathbf u^{n+1} + \mathbf u^n) \,,

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
    :math:`\mathbb M^{\alpha,n}` and :math:`\mathbb M^{\alpha,J}` are :class:`~struphy.feec.mass.WeightedMassOperators` being weighted with :math:`\rho_0` the MHD equilibrium density. 
    :math:`\alpha \in \{1, 2, v\}` denotes the :math:`\alpha`-form space where the operators correspond to.
    Moreover, :math:`\sum_k^{N_p} \omega_k \mu_k \hat{\mathbf b}¹_0 (\boldsymbol \eta_k) \cdot \left(\frac{1}{\sqrt{g(\boldsymbol \eta_k)}} \vec \Lambda² (\boldsymbol \eta_k)\right)` is accumulated by the kernel :class:`~struphy.pic.accumulation.accum_kernels_gc.cc_lin_mhd_5d_M` and
    the time-varying projection operator :math:`\mathcal{T}^B` is defined as

    .. math::

        \mathcal{T}^B_{(\mu,ijk),(\nu,mno)} := \hat \Pi¹_{(\mu,ijk)} \left[ \epsilon_{\mu \alpha \nu} \frac{\tilde{B}^2_\alpha}{\sqrt{g}} \Lambda²_{\nu,mno} \right] \,.
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        dct['filter'] = {'use_filter': None,
                         'modes': (0, 1),
                         'repeat': 3,
                         'alpha': 0.5}
        dct['turn_off'] = False

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 n: StencilVector,
                 u: BlockVector,
                 p: StencilVector,
                 *,
                 particles: Particles5D,
                 b: BlockVector,
                 absB0: StencilVector,
                 unit_b1: BlockVector,
                 u_space: str,
                 solver: dict = options(default=True)['solver'],
                 filter: dict = options(default=True)['filter'],
                 coupling_params: dict):

        super().__init__(n, u, p)

        self._particles = particles
        self._b = b
        self._unit_b1 = unit_b1
        self._absB0 = absB0

        self._info = solver['info']
        self._rank = self.derham.comm.Get_rank()

        self._scale_vec = coupling_params['Ah'] / coupling_params['Ab']

        self._E1T = self.derham.extraction_ops['1'].transpose()
        self._unit_b1 = self._E1T.dot(self._unit_b1)

        self._u_id = self.derham.space_to_form[u_space]
        if self._u_id == 'v':
            self._space_key_int = 0
        else:
            self._space_key_int = int(self._u_id)

        self._ACC = Accumulator(particles,
                                u_space,
                                accum_kernels_gc.cc_lin_mhd_5d_M,
                                self.derham,
                                self.domain.args_domain,
                                add_vector=True,
                                symmetry='symm',
                                filter_params=filter)

        # if self._particles.control_variate:

        #     # control variate method is only valid with Maxwellian distributions with "zero perp mean velocity".
        #     assert isinstance(self._particles.f0, Maxwellian)

        #     self._ACC.init_control_variate(self.mass_ops)

        #     # evaluate and save f0.n at quadrature points
        #     quad_pts = [quad_grid[nquad].points.flatten()
        #                 for quad_grid, nquad in zip(self.derham.Vh_fem['0']._quad_grids, self.derham.Vh_fem['0'].nquads)]

        #     n0_at_quad = self.domain.push(
        #         self._particles.f0.n, *quad_pts, kind='0', squeeze_out=False)

        #     # evaluate M0 = unit_b1 (1form) / absB0 (0form) * 2 * vth_perp² at quadrature points
        #     quad_pts_array = self.domain.prepare_eval_pts(*quad_pts)[:3]

        #     vth_perp = self.particles.f0.vth(*quad_pts_array)[1]

        #     absB0_at_quad = WeightedMassOperator.eval_quad(self.derham.Vh_fem['0'], self._absB0)

        #     unit_b1_at_quad = WeightedMassOperator.eval_quad(self.derham.Vh_fem['1'], self._unit_b1)

        #     self._M0_at_quad = unit_b1_at_quad / absB0_at_quad * vth_perp**2 * n0_at_quad * self._scale_vec

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
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(_A, _BC,
                                         solver['type'][0],
                                         pc=pc,
                                         tol=solver['tol'],
                                         maxiter=solver['maxiter'],
                                         verbose=solver['verbose'])

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

        # perform accumulation (either with or without control variate)
        # if self._particles.control_variate:

        #     self._ACC.accumulate(self._particles,
        #                          self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
        #                          self._scale_vec, 0.,
        #                          control_vec=[self._M0_at_quad[0], self._M0_at_quad[1], self._M0_at_quad[2]])
        # else:
        #     self._ACC.accumulate(self._particles,
        #                          self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
        #                          self._scale_vec, 0.)

        self._ACC(self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                  self._scale_vec, 0.)

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
        def tmp(x, y, z): return 0*x

        # Initialize BasisProjectionOperator
        self._TB = BasisProjectionOperator(P1, Vh, [[tmp, tmp, tmp]])

    def _update_weights_TB(self):
        """Updats time-dependent weights of the BasisProjectionOperator TB
        """

        # Update Femfield
        self._bf.vector = self._b
        self._bf.vector.update_ghost_regions()

        # define callable weights
        def bf1(x, y, z): return self._bf(x, y, z, local=True)[0]
        def bf2(x, y, z): return self._bf(x, y, z, local=True)[1]
        def bf3(x, y, z): return self._bf(x, y, z, local=True)[2]

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
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \int \rho_0 \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf{x} = \frac{A_\textnormal{h}}{A_b} \frac{1}{\epsilon} \iiint f^\text{vol} \left(1 - \frac{B_\parallel}{B^*_\parallel}\right) \tilde{\mathbf U} \times \mathbf B_f \cdot \mathbf V \, \textnormal{d} \mathbf{x} \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

    FE coefficients update:

    .. math::

        \mathbf u^{n+1} - \mathbf u^n = -\frac{A_\textnormal{h}}{A_b} \frac{1}{\epsilon} \mathbb{L}²{\mathbb{B}}^\times_f \mathbb{N}(1/g) \mathbb{W} \mathbb{N}\left(1- \frac{\hat B^0_\parallel}{\hat B^{*0} _\parallel}\right) (\mathbb{L}²)^\top \frac{\Delta t}{2} \cdot (\mathbf u^{n+1} + \mathbf u^n) \,.

    For the detail explanation of the notations, see `2022_DriftKineticCurrentCoupling <https://gitlab.mpcdf.mpg.de/struphy/struphy-projects/-/blob/main/running-projects/2022_DriftKineticCurrentCoupling.md?ref_type=heads>`_.
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['solver'] = {'type': [('pbicgstab', 'MassMatrixPreconditioner'),
                                  ('bicgstab', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        dct['turn_off'] = False

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 u: BlockVector,
                 *,
                 particles: Particles5D,
                 b: BlockVector,
                 b_eq: BlockVector,
                 unit_b1: BlockVector,
                 curl_unit_b2: BlockVector,
                 u_space: str,
                 solver: dict = options(default=True)['solver'],
                 coupling_params: dict,
                 epsilon: float = 1.):

        super().__init__(u)

        # assert parameters and expose some quantities to self
        assert isinstance(particles, (Particles5D))

        assert u_space in {'Hcurl', 'Hdiv', 'H1vec'}

        if u_space == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[u_space])

        self._epsilon = epsilon
        self._particles = particles
        self._b = b
        self._b_eq = b_eq
        self._unit_b1 = unit_b1
        self._curl_norm_b = curl_unit_b2

        self._info = solver['info']
        self._rank = self.derham.comm.Get_rank()

        self._scale_mat = coupling_params['Ah'] / \
            coupling_params['Ab'] / self._epsilon

        self._accumulator = Accumulator(particles,
                                        u_space,
                                        accum_kernels_gc.cc_lin_mhd_5d_D,
                                        self.derham,
                                        self.domain.args_domain,
                                        add_vector=False,
                                        symmetry='asym')

        # if self._particles.control_variate:

        #     # control variate method is only valid with Maxwellian distributions
        #     assert isinstance(self._particles.f0, Maxwellian)
        #     assert params['u_space'] == 'Hdiv'

        #     # evaluate and save f0.n / |det(DF)| at quadrature points
        #     quad_pts = [quad_grid[nquad].points.flatten()
        #                 for quad_grid, nquad in zip(self.derham.Vh_fem['0']._quad_grids, self.derham.Vh_fem['0'].nquads)]

        #     self._n0_at_quad = self.domain.push(
        #         self._particles.f0.n, *quad_pts, kind='3', squeeze_out=False)

        #     # prepare field evaluation
        #     quad_pts_array = self.domain.prepare_eval_pts(*quad_pts)[:3]

        #     u0_parallel = self._particles.f0.u(*quad_pts_array)[0]

        #     det_df_at_quad = self.domain.jacobian_det(*quad_pts, squeeze_out=False)

        #     # evaluate unit_b1 / |det(DF)| at quadrature points
        #     self._unit_b1_at_quad = WeightedMassOperator.eval_quad(self.derham.Vh_fem['1'], self._unit_b1)
        #     self._unit_b1_at_quad /= det_df_at_quad

        #     # evaluate unit_b1 (1form) dot epsilon * f0.u * curl_norm_b (2form) / |det(DF)| at quadrature points
        #     curl_norm_b_at_quad = WeightedMassOperator.eval_quad(self.derham.Vh_fem['2'], self._curl_norm_b)

        #     self._unit_b1_dot_curl_norm_b_at_quad = np.sum(p * q for p, q in zip(self._unit_b1_at_quad, curl_norm_b_at_quad))

        #     self._unit_b1_dot_curl_norm_b_at_quad /= det_df_at_quad
        #     self._unit_b1_dot_curl_norm_b_at_quad *= self._epsilon
        #     self._unit_b1_dot_curl_norm_b_at_quad *= u0_parallel

        #     # memory allocation for magnetic field at quadrature points
        #     self._b_quad1 = np.zeros_like(self._n0_at_quad)
        #     self._b_quad2 = np.zeros_like(self._n0_at_quad)
        #     self._b_quad3 = np.zeros_like(self._n0_at_quad)

        #     # memory allocation for parallel magnetic field at quadrature points
        #     self._B_para = np.zeros_like(self._n0_at_quad)

        #     # memory allocation for control_const at quadrature points
        #     self._control_const = np.zeros_like(self._n0_at_quad)

        #     # memory allocation for self._b_quad x self._nh0_at_quad * self._coupling_const
        #     self._mat12 = np.zeros_like(self._n0_at_quad)
        #     self._mat13 = np.zeros_like(self._n0_at_quad)
        #     self._mat23 = np.zeros_like(self._n0_at_quad)

        #     self._mat21 = np.zeros_like(self._n0_at_quad)
        #     self._mat31 = np.zeros_like(self._n0_at_quad)
        #     self._mat32 = np.zeros_like(self._n0_at_quad)

        u_id = self.derham.space_to_form[u_space]
        self._M = getattr(self.mass_ops, 'M' + u_id + 'n')

        self._E0T = self.derham.extraction_ops['0'].transpose()
        self._EuT = self.derham.extraction_ops[u_id].transpose()
        self._E1T = self.derham.extraction_ops['1'].transpose()
        self._E2T = self.derham.extraction_ops['2'].transpose()

        self._PB = getattr(self.basis_ops, 'PB')
        self._unit_b1 = self._E1T.dot(self._unit_b1)

        # preconditioner
        if solver['type'][1] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            self._pc = pc_class(self._M)

        # linear solver
        self._solver = inverse(self._M,
                               solver['type'][0],
                               pc=self._pc,
                               x0=self.feec_vars[0],
                               tol=solver['tol'],
                               maxiter=solver['maxiter'],
                               verbose=solver['verbose'])

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()
        self._rhs_v = u.space.zeros()
        self._u_new = u.space.zeros()

    def __call__(self, dt):
        # pointer to old coefficients
        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b_eq.copy(out=self._b_full1)

        if self._b is not None:
            b_full += self._b

        Eb_full = self._E2T.dot(b_full, out=self._b_full2)
        Eb_full.update_ghost_regions()

        # perform accumulation (either with or without control variate)
        # if self._particles.control_variate:

        #     # evaluate magnetic field at quadrature points (in-place)
        #     WeightedMassOperator.eval_quad(self.derham.Vh_fem['2'], self._b_full2,
        #                                    out=[self._b_quad1, self._b_quad2, self._b_quad3])

        #     # evaluate B_parallel
        #     self._B_para = np.sum(p * q for p, q in zip(self._unit_b1_at_quad, [self._b_quad1, self._b_quad2, self._b_quad3]))

        #     # evaluate coupling_const 1 - B_parallel / B^star_parallel
        #     self._control_const = 1 - (self._B_para / (self._B_para + self._unit_b1_dot_curl_norm_b_at_quad))

        #     # assemble (B x)
        #     self._mat12[:, :, :] = self._scale_mat * \
        #         self._b_quad3 * self._n0_at_quad * self._control_const
        #     self._mat13[:, :, :] = -self._scale_mat * \
        #         self._b_quad2 * self._n0_at_quad * self._control_const
        #     self._mat23[:, :, :] = self._scale_mat * \
        #         self._b_quad1 * self._n0_at_quad * self._control_const

        #     self._mat21[:, :, :] = -self._mat12
        #     self._mat31[:, :, :] = -self._mat13
        #     self._mat32[:, :, :] = -self._mat23

        #     self._accumulator.accumulate(self._particles, self._epsilon,
        #                                  Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
        #                                  self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
        #                                  self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
        #                                  self._space_key_int, self._scale_mat, 0.1,
        #                                  control_mat=[[None, self._mat12, self._mat13],
        #                                               [self._mat21, None, self._mat23],
        #                                               [self._mat31, self._mat32, None]])
        # else:
        #     self._accumulator.accumulate(self._particles, self._epsilon,
        #                                  Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
        #                                  self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
        #                                  self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
        #                                  self._space_key_int, self._scale_mat, 0.)

        self._accumulator(self._epsilon,
                          Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
                          self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                          self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                          self._space_key_int, self._scale_mat, 0.)

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


class ImplicitDiffusion(Propagator):
    r"""
    Weak, implicit discretization of the diffusion (or heat) equation (can be used as a Poisson solver too).

    Find :math:`\phi \in H^1` such that

    .. math::

        \int_\Omega \psi\, n_0(\mathbf x)\frac{\partial \phi}{\partial t}\,\textrm d \mathbf x + \int_\Omega \nabla \psi^\top D_0(\mathbf x) \nabla \phi \,\textrm d \mathbf x = \sum_i \int_\Omega \psi\, \rho_i(\mathbf x)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`n_0, \rho_i:\Omega \to \mathbb R` are real-valued functions and 
    :math:`D_0:\Omega \to \mathbb R^{3\times 3}`
    is a positive diffusion matrix. 
    Boundary terms from integration by parts are assumed to vanish.
    The equation is discretized as

    .. math::

        \left( \frac{\sigma_1}{\Delta t} \mathbb M^0_{n_0} + \mathbb G^\top \mathbb M^1_{D_0} \mathbb G \right)\, \boldsymbol\phi^{n+1} = \frac{\sigma_2}{\Delta t} \mathbb M^0_{n_0} \boldsymbol\phi^{n} + \frac{\sigma_3}{\Delta t} \sum_i(\Lambda^0, \rho_i  )_{L^2}\,,

    where :math:`M^0_{n_0}` and :math:`M^1_{D_0}` are :class:`WeightedMassOperators <struphy.feec.mass.WeightedMassOperators>`
    and :math:`\sigma_1, \sigma_2, \sigma_3 \in \mathbb R` are artificial parameters that can be tuned to
    change the model (see Notes).

    Notes
    -----

    * :math:`\sigma_1=\sigma_2=0` and :math:`\sigma_3 = \Delta t`: **Poisson solver** with a given charge density :math:`\sum_i\rho_i`. 
    * :math:`\sigma_2=0` and :math:`\sigma_1 = \sigma_3 = \Delta t` : Poisson with **adiabatic electrons**.
    * :math:`\sigma_1=\sigma_2=1` and :math:`\sigma_3 = 0`: **Implicit heat equation**. 

    Parameters
    ----------
    phi : StencilVector
        FE coefficients of the solution as a discrete 0-form.

    sigma_1, sigma_2, sigma_3 : float | int
        Equation parameters.

    divide_by_dt : bool
        Whether to divide the sigmas by dt during __call__.

    stab_mat : str
        Name of the matrix :math:`M^0_{n_0}`.

    diffusion_mat : str
        Name of the matrix :math:`M^1_{D_0}`.

    rho : StencilVector or tuple or list 
        (List of) right-hand side FE coefficients of a 0-form (optional, can be set with a setter later).
        Can be either a) StencilVector or b) 2-tuple, or a list of those.
        In case b) the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.

    x0 : StencilVector
        Initial guess for the iterative solver (optional, can be set with a setter later).

    solver : dict
        Parameters for the iterative solver (see ``__init__`` for details).
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['model'] = {'sigma_1': 1.,
                        'sigma_2': 0.,
                        'sigma_3': 1.,
                        'stab_mat': ['M0', 'M0ad'],
                        'diffusion_mat': ['M1', 'M1perp']}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': False}
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 phi: StencilVector,
                 *,
                 sigma_1: float = options()['model']['sigma_1'],
                 sigma_2: float = options()['model']['sigma_2'],
                 sigma_3: float = options()['model']['sigma_3'],
                 divide_by_dt: bool = False,
                 stab_mat: str = options(default=True)['model']['stab_mat'],
                 diffusion_mat: str = options(default=True)[
                     'model']['diffusion_mat'],
                 rho: StencilVector | tuple | list = None,
                 x0: StencilVector = None,
                 solver: dict = options(default=True)['solver']):

        assert phi.space == self.derham.Vh['0']

        super().__init__(phi)

        # always stabilize
        if np.abs(sigma_1) < 1e-14:
            sigma_1 = 1e-14
            print(f'Stabilizing Poisson solve with {sigma_1 = }')

        # model parameters
        self._sigma_1 = sigma_1
        self._sigma_2 = sigma_2
        self._sigma_3 = sigma_3
        self._divide_by_dt = divide_by_dt

        # collect rhs
        if rho is None:
            self._rho = [phi.space.zeros()]
        else:
            if isinstance(rho, list):
                for r in rho:
                    if isinstance(r, tuple):
                        assert isinstance(r[0], AccumulatorVector)
                        assert isinstance(r[1], Particles)
                        # assert r.space_id == 'H1'
                    else:
                        assert r.space == phi.space
            elif isinstance(rho, tuple):
                assert isinstance(rho[0], AccumulatorVector)
                assert isinstance(rho[1], Particles)
                # assert rho[0].space_id == 'H1'
                rho = [rho]
            else:
                assert rho.space == phi.space
                rho = [rho]
            self._rho = rho

        # initial guess and solver params
        self._x0 = x0
        self._info = solver['info']
        stab_mat = getattr(self.mass_ops, stab_mat)
        diffusion_mat = getattr(self.mass_ops, diffusion_mat)

        # Set lhs matrices (without dt)
        self._stab_mat = stab_mat
        self._diffusion_op = self.derham.grad.T @ diffusion_mat @ self.derham.grad

        # preconditioner and solver for Ax=b
        if solver['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver['type'][1])
            pc = pc_class(stab_mat)

        # solver just with A_2, but will be set during call with dt
        self._solver = inverse(self._diffusion_op,
                               solver['type'][0],
                               pc=pc,
                               x0=self.x0,
                               tol=solver['tol'],
                               maxiter=solver['maxiter'],
                               verbose=solver['verbose'],
                               recycle=solver['recycle'])

        # allocate memory for solution
        self._tmp = phi.space.zeros()
        self._rhs = phi.space.zeros()
        self._rhs2 = phi.space.zeros()

    @property
    def rho(self):
        """
        (List of) right-hand side FE coefficients of a 0-form.
        The list entries can be either a) StencilVectors or b) 2-tuples;
        in the latter case, the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.
        """
        return self._rho

    @rho.setter
    def rho(self, value):
        """ In-place setter for StencilVector/PolarVector.
        If rho is a list, len(value) msut be len(rho) and value can contain None.
        """
        if isinstance(value, list):
            assert len(value) == len(self.rho)
            for i, (val, r) in enumerate(zip(value, self.rho)):
                if val is None:
                    continue
                elif isinstance(val, tuple):
                    assert isinstance(val[0], AccumulatorVector)
                    assert isinstance(val[1], Particles)
                    assert isinstance(r, tuple)
                    self._rho[i] = val
                else:
                    assert val.space == r.space
                    r[:] = val[:]
        elif isinstance(ValueError, tuple):
            assert isinstance(value[0], AccumulatorVector)
            assert isinstance(value[1], Particles)
            assert len(self.rho) == 1
            # assert rho[0].space_id == 'H1'
            self._rho[0] = value
        else:
            assert value.space == self.derham.Vh['0']
            assert len(self.rho) == 1
            self._rho[0][:] = value[:]

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
        assert value.space == self.derham.Vh['0']
        assert value.space.symbolic_space == 'H1', f'Right-hand side must be in H1, but is in {value.space.symbolic_space}.'

        if self._x0 is None:
            self._x0 = value
        else:
            self._x0[:] = value[:]

    def __call__(self, dt):

        # set parameters
        if self._divide_by_dt:
            sig_1 = self._sigma_1 / dt
            sig_2 = self._sigma_2 / dt
            sig_3 = self._sigma_3 / dt
        else:
            sig_1 = self._sigma_1
            sig_2 = self._sigma_2
            sig_3 = self._sigma_3

        # compute rhs
        phin = self.feec_vars[0]
        rhs = self._stab_mat.dot(phin, out=self._rhs)
        rhs *= sig_2

        self._rhs2 *= 0.
        for rho in self._rho:
            if isinstance(rho, tuple):
                rho[0]()  # accumulate
                self._rhs2 += sig_3 * rho[0].vectors[0]
            else:
                self._rhs2 += sig_3 * rho

        rhs += self._rhs2

        # compute lhs
        self._solver.linop = sig_1 * self._stab_mat + self._diffusion_op

        # solve
        out = self._solver.solve(rhs, out=self._tmp)
        info = self._solver._info

        if self._info:
            print(info)

        self.feec_vars_update(out)


class VariationalMomentumAdvection(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf u \in (H^1)^3` such that

    .. math::

        \int_{\Omega} \partial_t ( \rho  \mathbf{u}) \cdot \mathbf{v} \,\textrm d \mathbf x - 
        \int_{\Omega} \rho \mathbf{u} \cdot  [\mathbf{u}, \mathbf{v}] \, \textrm d \mathbf x = 0 \,.

    On the logical domain:

    .. math::

        \int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \,\textrm d \boldsymbol \eta - 
        \int_{\hat{\Omega}} \hat{\rho}^3 \hat{\mathbf{u}} \cdot G [\hat{\mathbf{u}}, \hat{\mathbf{v}}] \, \textrm d \boldsymbol \eta = 0 \,,

    which is discretized as

    .. math::

        \mathbb M^v[\hat{\rho}_h^n] \frac{\mathbf u^{n+1}- \mathbf u^n}{\Delta t} - \left(\sum_{\mu} (\hat{\Pi}^{0}[\hat{\mathbf u}_h^{n+1/2} \cdot \vec{\boldsymbol \Lambda}^1] \mathbb G P_{\mu} - \hat{\Pi}^0[\hat{\mathbf A}^1_{\mu,h} \cdot \vec{\boldsymbol \Lambda}^v])^\top P_i \right) \mathbb M^v[\hat{\rho}_h^n] \mathbf u^{n} = 0 ~ .

    where :math:`P_\mu` stand for the :class:`~struphy.feec.basis_projection_ops.CoordinateProjector` and the weights
    in the the two :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{n+1/2} = (\mathbf{u}^{n+1/2})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \,, \qquad \hat{\mathbf A}^1_{\mu,h} = \nabla P_\mu((\mathbf u^{n+1/2})^\top \vec{\boldsymbol \Lambda}^v)] \in V_h^1\,, \qquad \hat{\rho}_h^{n} = (\rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['lin_solver'] = {'tol': 1e-12,
                             'maxiter': 500,
                             'type': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                      ('cg', None)],
                             'verbose': False}
        dct['nonlin_solver'] = {'tol': 1e-8,
                                'maxiter': 100,
                                'type': ['Newton', 'Picard'],
                                'info': False}
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(self,
                 u: BlockVector,
                 *,
                 mass_ops: WeightedMassOperator,
                 lin_solver: dict = options(default=True)['lin_solver'],
                 nonlin_solver: dict = options(default=True)['nonlin_solver']):

        super().__init__(u)

        assert mass_ops is not None

        self._mass_ops = mass_ops
        self._lin_solver = lin_solver
        self._nonlin_solver = nonlin_solver

        if self.derham.comm is not None:
            rank = self.derham.comm.Get_rank()
        else:
            rank = 0

        self._info = self._nonlin_solver['info'] and (rank == 0)

        self.WMM = mass_ops

        self._initialize_mass()

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_diff = u.space.zeros()
        self._tmp__pc_diff = u.space.zeros()
        self._tmp_update = u.space.zeros()
        self._tmp_weak_diff = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_advection = u.space.zeros()

        self.brack = BracketOperator(self.derham, self._tmp_mn)
        self._dt2_brack = 2.*self.brack
        self.derivative = self.WMM + self._dt2_brack
        self.inv_derivative = inverse(self.pc@self.derivative,
                                      'gmres',
                                      tol=self._lin_solver['tol'],
                                      maxiter=self._lin_solver['maxiter'],
                                      verbose=self._lin_solver['verbose'],
                                      recycle=True)

    def __call__(self, dt):
        if self._nonlin_solver['type'] == 'Newton':
            self.__call_newton(dt)
        elif self._nonlin_solver['type'] == 'Picard':
            self.__call_picard(dt)

    def __call_newton(self, dt):

        # Initialize variable for Newton iteration
        un = self.feec_vars[0]
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)
        tol = self._nonlin_solver['tol']
        err = tol+1
        self.pc.update_mass_operator(self._Mrho)
        # Jacobian matrix for Newton solve
        self._dt2_brack._scalar = dt/2
        if self._info:
            print()
            print("Newton iteration in VariationalMomentumAdvection")

        for it in range(self._nonlin_solver['maxiter']):

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

            if self._info:
                print("iteration : ", it, " error : ", err)
            if err < tol**2 or np.isnan(err):
                break

            # Newton step
            pc_diff = self.pc.dot(diff, out=self._tmp__pc_diff)
            update = self.inv_derivative.dot(pc_diff, out=self._tmp_update)
            if self._info:
                print("information on the linear solver : ",
                      self.inv_derivative._info)
            un1 -= update
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver['maxiter']-1 or np.isnan(err):
            print(
                f'!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err = } \n {tol**2 = }')

        self.feec_vars_update(un1)

    def __call_picard(self, dt):

        # Initialize variable for Picard iteration
        un = self.feec_vars[0]
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)
        tol = self._nonlin_solver['tol']
        err = tol+1
        self.pc.update_mass_operator(self._Mrho)
        # Jacobian matrix for Newton solve

        for it in range(self._nonlin_solver['maxiter']):

            # Picard iteration
            if err < tol**2 or np.isnan(err):
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

        if it == self._nonlin_solver['maxiter']-1 or np.isnan(err):
            print(
                f'!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err = } \n {tol**2 = }')

        self.feec_vars_update(un1)

    def _initialize_mass(self):
        """Initialization of the mass matrix solver"""
        # weighted mass matrix to go from m to u
        self._Mrho = self.WMM

        # Inverse weighted mass matrix
        if self._lin_solver['type'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._lin_solver['type'][1])
            self.pc = pc_class(self._Mrho)

        self._Mrhoinv = inverse(self._Mrho,
                                self._lin_solver['type'][0],
                                pc=self.pc,
                                tol=self._lin_solver['tol'],
                                maxiter=self._lin_solver['maxiter'],
                                verbose=self._lin_solver['verbose'],
                                recycle=True)

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv)
        self._inv_Mv = inverse(self.mass_ops.Mv,
                               'pcg',
                               pc=self.pc_Mv,
                               tol=1e-16,
                               maxiter=1000,
                               verbose=False)

    def _get_error_newton(self, mn_diff):
        weak_un_diff = self._inv_Mv.dot(
            self.derham.boundary_ops['v'].dot(mn_diff), out=self._tmp_weak_diff)
        err_u = weak_un_diff.dot(mn_diff)
        return err_u


class VariationalDensityEvolve(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\rho \in L^2` and  :math:`\mathbf u \in (H^1)^3` such that

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x + \int_\Omega \left(\frac{|\mathbf u|^2}{2} - \frac{\partial(\rho \mathcal U(\rho))}{\partial \rho}\right) \nabla \cdot (\rho \mathbf v) \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,.

    On the logical domain:

    .. math::

        \begin{align}
        &\partial_t \hat{\rho}^3 + \nabla \cdot ( \hat{\rho}^3 \hat{\mathbf{u}} ) = 0 ~ ,
        \\[4mm]
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta  
        + \int_{\hat{\Omega}} \left( \frac{| DF \hat{\mathbf{u}} |^2}{2} - \frac{\partial (\hat{\rho}^3 \mathcal U)}{\partial \hat{\rho}^3} \right) \nabla \cdot (\hat{\rho}^3 \hat{\mathbf{v}}) \, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        \end{align}

    where :math:`\mathcal U` depends on the chosen model. It is discretized as

    .. math::

        \begin{align}
        &\frac{\mathbb M^v[\hat{\rho}_h^{n+1}] \mathbf u^{n+1}- \mathbb M^v[\hat{\rho}_h^n] \mathbf u^n}{\Delta t} 
        + (\mathbb D \hat{\Pi}^{2}[\hat{\tilde{\rho}_h^{n+1}} \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3\left(\frac{DF \hat{\mathbf{u}}_h^{n+1} \cdot DF \hat{\mathbf{u}}_h^{n}}{2} 
        - \frac{\hat{\rho}_h^{n+1}\mathcal U(\hat{\rho}_h^{n+1})-\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n})}{\hat{\rho}_h^{n+1}-\hat{\rho}_h^n} \right) = 0 ~ ,
        \\[2mm]
        &\frac{\boldsymbol \rho^{n+1}- \boldsymbol \rho^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{\tilde{\rho}_h^{n+1}} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        &\frac{\tilde{\boldsymbol \rho}^{n+1}- \boldsymbol\rho^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{\tilde{\rho}_h^{n+1}} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n} = 0 ~ \text{if parameter 'implicit_transport'},
        \\[2mm]
        &\tilde{\boldsymbol \rho}^{n+1} = \boldsymbol\rho^n ~ \text{else},
        \end{align}

    where :math:`\hat{l}^3(f)` denotes the vector representing the linear form :math:`v_h \mapsto \int_{\hat{\Omega}} f(\boldsymbol \eta) v_h(\boldsymbol \eta) d \boldsymbol \eta`, that is the vector with components

    .. math::
        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    and the weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{k} = (\mathbf{u}^{k})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \, \text{for k in} \{n, n+1/2, n+1\}, \qquad \hat{\rho}_h^{k} = (\rho^{k})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \, \text{for k in} \{n, n+1/2, n+1\} .
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['lin_solver'] = {'tol': 1e-12,
                             'maxiter': 500,
                             'type': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                      ('cg', None)],
                             'verbose': False,
                             'recycle': True}
        dct['nonlin_solver'] = {'tol': 1e-8,
                                'maxiter': 100,
                                'type': ['Newton', 'Picard'],
                                'info': False,
                                'implicit_transport': False}
        dct['physics'] = {'gamma': 5/3}

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 rho: StencilVector,
                 u: BlockVector,
                 *,
                 model: str = 'barotropic',
                 gamma: float = options()['physics']['gamma'],
                 s: StencilVector = None,
                 mass_ops: WeightedMassOperator,
                 lin_solver: dict = options(default=True)['lin_solver'],
                 nonlin_solver: dict = options(default=True)['nonlin_solver']):

        super().__init__(rho, u)

        assert model in ['pressureless', 'barotropic', 'full']
        if model == 'full':
            assert s is not None
        assert mass_ops is not None

        self._model = model
        self._gamma = gamma
        self._s = s
        self._mass_ops = mass_ops
        self._lin_solver = lin_solver
        self._nonlin_solver = nonlin_solver
        self._implicit_transport = self._nonlin_solver['implicit_transport']

        if self.derham.comm is not None:
            rank = self.derham.comm.Get_rank()
        else:
            rank = 0
        print(self._nonlin_solver)
        self._info = self._nonlin_solver['info'] and (rank == 0)

        self.WMM = mass_ops

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
        if self._nonlin_solver['type'] == 'Newton':
            self.__call_newton(dt)
        elif self._nonlin_solver['type'] == 'Picard':
            self.__call_picard(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""

        if self._info:
            print()
            print("Newton iteration in VariationalDensityEvolve")

        # Initial variables
        rhon = self.feec_vars[0]
        un = self.feec_vars[1]
        self.rhof.vector = rhon
        self.rhof1.vector = rhon
        self._update_weighted_MM()
        mn = self._Mrho.dot(un, out=self._tmp_mn)

        # Compute implicit approximation of rho^{n+1}
        self.uf.vector = un
        if self._implicit_transport:
            self._update_Piu()

            if self._info:
                print("Compute the implicit approximation")
            # Check it's usefull to invert the transport operator to avoid calling on Id and causing bugs
            # Explicit call
            self._dt_divPiu._scalar = -dt
            rhon1 = self._transop.dot(rhon, out=self._tmp_rhon1)
            # Get diff
            rhon_diff = rhon1.copy(out=self._tmp_rhon_diff)
            rhon_diff -= rhon
            weak_rhon_diff = self.mass_ops.M3.dot(
                rhon_diff, out=self._tmp_rhon_weak_diff)
            err_rho = weak_rhon_diff.dot(rhon_diff)

            if err_rho > self._lin_solver['tol']:
                # Implicit call if needed
                self._dt_divPiu._scalar = dt
                rhon1 = self._inv_transop.dot(rhon, out=self._tmp_rhon1)
            if self._info:
                print("Linear solver info : ", self._inv_transop._info)

        else:
            # No implicit
            rhon1 = rhon.copy(out=self._tmp_rhon1)

        # Initialize variable for Newton iteration
        if self._model == 'full':
            s = self._s
            self.sf.vector = s

        self.rhof1.vector = rhon1
        self._update_Pirho()

        rhon1 = rhon.copy(out=self._tmp_rhon1)
        self.rhof1.vector = rhon1
        un1 = un.copy(out=self._tmp_un1)
        mn1 = mn.copy(out=self._tmp_mn1)
        tol = self._nonlin_solver['tol']
        err = tol+1

        for it in range(self._nonlin_solver['maxiter']):

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

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or np.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self.pc.update_mass_operator(self._Mrho)
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = rhon_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                print("information on the linear solver : ",
                      self._inv_Jacobian._solver._info)
            un1 -= incr[0]
            rhon1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            self.rhof1.vector = rhon1
            self._update_weighted_MM()
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver['maxiter'] - 1 or np.isnan(err):
            print(
                f'!!!Warning: Maximum iteration in VariationalDensityEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(rhon1, un1)

    def __call_picard(self, dt):
        """Solve the non linear system for updating the variables using Picard iteration method"""

        # Initialize variable for Picard iteration
        if self._model == 'full':
            s = self._s
            self.sf.vector = s
        rhon = self.feec_vars[0]
        rhon1 = rhon.copy(out=self._tmp_rhon1)
        self.rhof.vector = rhon
        self.rhof1.vector = rhon1
        self._update_weighted_MM()
        self._update_Pirho()
        un = self.feec_vars[1]
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        un1 = un.copy(out=self._tmp_un1)
        un2 = un1.copy(out=self._tmp_un2)
        mn1 = mn.copy(out=self._tmp_mn1)
        tol = self._nonlin_solver['tol']
        err = tol+1
        for it in range(self._nonlin_solver['maxiter']):

            # Picard iteration
            if err < tol**2 or np.isnan(err):
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
            un1 = self._Mrhoinv.dot(mn1, out=self._tmp_un1)

            # get the error
            un_diff = un1.copy(out=self._tmp_un_diff)
            un_diff -= un2
            un2 = un1.copy(out=self._tmp_un2)

            err = self._get_error_picard(un_diff, rhon_diff)

        if it == self._nonlin_solver['maxiter']-1 or np.isnan(err):
            print(
                f'!!!Warning: Maximum iteration in VariationalDensityEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(rhon1, un1)

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
            transposed=False, use_cache=True,
            V_extraction_op=self.derham.extraction_ops['v'],
            V_boundary_op=self.derham.boundary_ops['v'],
            P_boundary_op=IdentityOperator(self.derham.Vh_pol['2']))

        self.Piu = BasisProjectionOperator(
            P2, V3h, [[None], [None], [None]],
            transposed=False, use_cache=True,
            P_boundary_op=IdentityOperator(self.derham.Vh_pol['2']))

        self.PirhoT = self.Pirho.T

        # divergence
        self.div = self.derham.div_bcfree

        # Initialize the transport operator and transposed
        self.divPirho = self.div @ self.Pirho
        self.divPirhoT = self.PirhoT @ self.div.T
        self.divPiu = self.div@self.Piu

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
        self._uf_0_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_1])
        self._rhof_1_values = np.zeros(grid_shape, dtype=float)
        self._uf_1_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_2])
        self._rhof_2_values = np.zeros(grid_shape, dtype=float)
        self._uf_2_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]

        # weighted mass matrix to go from m to u
        self._Mrho = self.WMM

        # Inverse weighted mass matrix
        if self._lin_solver['type'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._lin_solver['type'][1])
            self.pc = pc_class(self._Mrho)

        self._Mrhoinv = inverse(self._Mrho,
                                self._lin_solver['type'][0],
                                pc=self.pc,
                                tol=self._lin_solver['tol'],
                                maxiter=self._lin_solver['maxiter'],
                                verbose=self._lin_solver['verbose'],
                                recycle=True)

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv)
        self._inv_Mv = inverse(self.mass_ops.Mv,
                               'pcg',
                               pc=self.pc_Mv,
                               tol=1e-16,
                               maxiter=1000,
                               verbose=False,
                               recycle=True)

        integration_grid = [grid_1d.flatten()
                            for grid_1d in self.derham.quad_grid_pts['0']]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = self.derham.prepare_eval_tp_fixed(
            integration_grid)

        metric = self.domain.metric(*integration_grid)
        self._mass_metric_term = deepcopy(metric)

        # tmps
        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid])
        self._rhof_values = np.zeros(grid_shape, dtype=float)

        self._full_term_mass = deepcopy(metric)

        # prepare for integration of linear form

        metric = self.domain.metric(
            *integration_grid)*self.domain.jacobian_det(*integration_grid)
        self._proj_u2_metric_term = deepcopy(metric)

        # implicit transport operator
        self._I3 = IdentityOperator(self.derham.Vh_pol['3'])
        self._dt_divPiu = 2 * self.divPiu

        self._transop = self._I3 + self._dt_divPiu
        self._inv_transop = inverse(self._transop,
                                    'gmres',
                                    tol=self._lin_solver['tol'],
                                    maxiter=self._lin_solver['maxiter'],
                                    verbose=False,
                                    recycle=True)

        # Other mass matrices for newton solve
        self._M_un = WeightedMassOperator(
            self.derham.Vh_fem['v'],
            self.derham.Vh_fem['3'],
            V_extraction_op=self.derham.extraction_ops['v'],
            W_extraction_op=self.derham.extraction_ops['3'],
            V_boundary_op=self.derham.boundary_ops['v'],
            W_boundary_op=self.derham.boundary_ops['3'])

        self._M_un1 = WeightedMassOperator(
            self.derham.Vh_fem['3'],
            self.derham.Vh_fem['v'],
            V_extraction_op=self.derham.extraction_ops['3'],
            W_extraction_op=self.derham.extraction_ops['v'],
            V_boundary_op=self.derham.boundary_ops['3'],
            W_boundary_op=self.derham.boundary_ops['v'])

        self._M_drho = WeightedMassOperator(
            self.derham.Vh_fem['3'],
            self.derham.Vh_fem['3'],
            V_extraction_op=self.derham.extraction_ops['3'],
            W_extraction_op=self.derham.extraction_ops['3'],
            V_boundary_op=self.derham.boundary_ops['3'],
            W_boundary_op=self.derham.boundary_ops['3'])

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid])

        self._Guf_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]

        self._Guf1_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]

        metric = self.domain.metric(*integration_grid)
        self._mass_u_metric_term = deepcopy(metric)

        Jacs = BlockVectorSpace(
            self.derham.Vh_pol['v'], self.derham.Vh_pol['3'])

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        # local version to avoid creating new version of LinearOperator every time
        self._dt_pc_divPirhoT = 2 * (self.divPirhoT)
        self._dt2_pc_divPirhoT = 2 * (self.divPirhoT)
        self._dt2_divPirho = 2 * self.divPirho

        self._Jacobian[0, 0] = self._Mrho + \
            self._dt2_pc_divPirhoT@self._M_un
        self._Jacobian[0, 1] = self._M_un1 + \
            self._dt_pc_divPirhoT@self._M_drho
        self._Jacobian[1, 0] = self._dt2_divPirho
        self._Jacobian[1, 1] = self._I3

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(self._Jacobian, 'pbicgstab',
                                             pc=self.pc,
                                             tol=self._lin_solver['tol'],
                                             maxiter=self._lin_solver['maxiter'],
                                             verbose=self._lin_solver['verbose'],
                                             recycle=True)

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],
        #                          recycle=True)

        # L2-projector for V3
        self._get_L2dofs_V3 = L2Projector('L2', self.mass_ops).get_dofs

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid])

        # tmps
        self._eval_dl_drho = np.zeros(grid_shape, dtype=float)

        self._uf_values = [np.zeros(grid_shape, dtype=float) for i in range(3)]
        self._uf1_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in integration_grid])

        self._tmp_int_grid = np.zeros(grid_shape, dtype=float)
        self._tmp_int_grid2 = np.zeros(grid_shape, dtype=float)
        self._rhof_values = np.zeros(grid_shape, dtype=float)
        self._rhof1_values = np.zeros(grid_shape, dtype=float)

        if self._model == 'full':
            self._sf_values = np.zeros(grid_shape, dtype=float)
            self._delta_rhof_values = np.zeros(grid_shape, dtype=float)
            self._rhof_mid_values = np.zeros(grid_shape, dtype=float)
            self._eta_values = np.zeros(grid_shape, dtype=float)
            self._e_rho1_s_values = np.zeros(grid_shape, dtype=float)
            self._e_rho_s_values = np.zeros(grid_shape, dtype=float)
            self._de_rhom_s_values = np.zeros(grid_shape, dtype=float)
            self._d2e_rho1_s_values = np.zeros(grid_shape, dtype=float)
            self._DG_values = np.zeros(grid_shape, dtype=float)

            gam = self._gamma
            metric = np.power(self.domain.jacobian_det(
                *integration_grid), 2-gam)
            self._proj_rho2_metric_term = deepcopy(metric)

            metric = np.power(self.domain.jacobian_det(
                *integration_grid), 1-gam)
            self._proj_drho_metric_term = deepcopy(metric)

    def __ener(self, rho, s, out=None):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = np.power(rho, gam)*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def __dener_drho(self, rho, s, out=None):
        """Derivative with respect to rho of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        dE(rho, s)/drho = (gamma*rho^{gamma-1} - s*rho^{gamma-2})*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = (gam * np.power(rho, gam-1) -
                   s * np.power(rho, gam-2))*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)

            np.power(rho, gam-1, out=self._tmp_int_grid)
            self._tmp_int_grid *= gam

            np.power(rho, gam-2, out=self._tmp_int_grid2)
            self._tmp_int_grid2 *= s

            self._tmp_int_grid -= self._tmp_int_grid2
            out *= self._tmp_int_grid
        return out

    def __d2ener_drho2(self, rho, s, out=None):
        """Second derivative with respect to (rho, rho) of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        d^2E(rho, s)/drho^2 = (gamma*(gamma-1) rho^{gamma-2}- 2*s*(gamma-1)*rho^{gamma-3}+ s^2*rho^{gamma-4})*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = (gam * (gam-1) * np.power(rho, gam-2)
                   - s * 2 * (gam-1) * np.power(rho, gam-3)
                   + s**2 * np.power(rho, gam-4))*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)

            np.power(rho, gam-2, out=self._tmp_int_grid)
            self._tmp_int_grid *= gam*(gam-1)

            np.power(rho, gam-3, out=self._tmp_int_grid2)
            self._tmp_int_grid2 *= s
            self._tmp_int_grid2 *= 2*(gam-1)
            self._tmp_int_grid -= self._tmp_int_grid2

            np.power(rho, gam-4, out=self._tmp_int_grid2)
            self._tmp_int_grid2 *= s
            self._tmp_int_grid2 *= s
            self._tmp_int_grid += self._tmp_int_grid2
            out *= self._tmp_int_grid
        return out

    def __eta(self, delta_x, out=None):
        if out is None:
            out = 1.-np.exp(-(delta_x/1e-5)**2)
        else:
            out *= 0.
            out += delta_x
            out /= 1e-5
            out **= 2
            out *= -1
            np.exp(out, out=out)
            out *= -1
            out += 1.
        return out

    def _update_Pirho(self,):
        """Update the weights of the `BasisProjectionOperator` Pirho"""

        rhof0_values = self.rhof1.eval_tp_fixed_loc(
            self.hist_grid_0_spans, self.hist_grid_0_bd, out=self._rhof_0_values)
        rhof1_values = self.rhof1.eval_tp_fixed_loc(
            self.hist_grid_1_spans, self.hist_grid_1_bd, out=self._rhof_1_values)
        rhof2_values = self.rhof1.eval_tp_fixed_loc(
            self.hist_grid_2_spans, self.hist_grid_2_bd, out=self._rhof_2_values)

        self.Pirho.update_weights([[rhof0_values, None, None],
                                   [None, rhof1_values, None],
                                   [None, None, rhof2_values]])

        self.PirhoT.update_weights([[rhof0_values, None, None],
                                    [None, rhof1_values, None],
                                    [None, None, rhof2_values]])

    def _update_Piu(self,):
        """Update the weights of the `BasisProjectionOperator` appearing in the equations"""

        uf0_values = self.uf.eval_tp_fixed_loc(
            self.hist_grid_0_spans, [self.hist_grid_0_bn]*3, out=self._uf_0_values)
        uf1_values = self.uf.eval_tp_fixed_loc(
            self.hist_grid_1_spans, [self.hist_grid_1_bn]*3, out=self._uf_1_values)
        uf2_values = self.uf.eval_tp_fixed_loc(
            self.hist_grid_2_spans, [self.hist_grid_2_bn]*3, out=self._uf_2_values)

        self.Piu.update_weights(
            [[uf0_values[0]], [uf1_values[1]], [uf2_values[2]]])

    def _update_weighted_MM(self,):
        """update the weighted mass matrix operator"""

        rhof_values = self.rhof1.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_bd, out=self._rhof_values)
        for i in range(3):
            for j in range(3):
                self._full_term_mass[i, j] = rhof_values * \
                    self._mass_metric_term[i, j]

        self._Mrho.assemble([[self._full_term_mass[0, 0], self._full_term_mass[0, 1], self._full_term_mass[0, 2]],
                             [self._full_term_mass[1, 0], self._full_term_mass[1,
                                                                               1], self._full_term_mass[1, 2]],
                             [self._full_term_mass[2, 0], self._full_term_mass[2, 1], self._full_term_mass[2, 2]]],
                            verbose=False)

    def _update_linear_form_u2(self,):
        """Update the linearform representing integration in V3 against kynetic energy"""

        uf_values = self.uf.eval_tp_fixed_loc(self.integration_grid_spans, [
                                              self.integration_grid_bn]*3, out=self._uf_values)
        uf1_values = self.uf1.eval_tp_fixed_loc(self.integration_grid_spans, [
                                                self.integration_grid_bn]*3, out=self._uf1_values)

        self._eval_dl_drho *= 0.
        for i in range(3):
            for j in range(3):
                self._tmp_int_grid *= 0
                self._tmp_int_grid += uf_values[i]
                self._tmp_int_grid *= self._proj_u2_metric_term[i, j]
                self._tmp_int_grid *= uf1_values[j]
                self._eval_dl_drho += self._tmp_int_grid

        self._eval_dl_drho *= 0.5

        if self._model == 'barotropic':

            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._rhof_values)
            rhof1_values = self.rhof1.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._rhof1_values)

            # self._eval_dl_drho -= (rhof_values + rhof1_values)/2
            rhof_values /= 2
            rhof1_values /= 2

            self._eval_dl_drho -= rhof_values
            self._eval_dl_drho -= rhof1_values

        if self._model == 'full':
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._rhof_values)
            rhof1_values = self.rhof1.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._rhof1_values)

            sf_values = self.sf.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._sf_values)

            # delta_rho_values = rhof1_values-rhof_values
            self._delta_rhof_values *= 0.
            self._delta_rhof_values += rhof1_values
            self._delta_rhof_values -= rhof_values
            delta_rho_values = self._delta_rhof_values

            # rho_mid_values = (rhof1_values+rhof_values)/2
            self._rhof_mid_values *= 0
            self._rhof_mid_values += rhof1_values
            self._rhof_mid_values += rhof_values
            self._rhof_mid_values /= 2
            rho_mid_values = self._rhof_mid_values

            eta = self.__eta(delta_rho_values, out=self._eta_values)

            e_rho1_s = self.__ener(
                rhof1_values, sf_values, out=self._e_rho1_s_values)
            e_rho_s = self.__ener(rhof_values, sf_values,
                                  out=self._e_rho_s_values)

            de_rhom_s = self.__dener_drho(
                rho_mid_values, sf_values, out=self._de_rhom_s_values)

            # eta*delta_rho_values*(e_rho1_s-e_rho_s)*delta_rho_values/(delta_rho_values**2+1e-40)
            self._DG_values *= 0.
            self._DG_values += e_rho1_s
            self._DG_values -= e_rho_s
            self._DG_values *= delta_rho_values
            delta_rho_values **= 2
            delta_rho_values += 1e-40
            self._DG_values /= delta_rho_values
            self._DG_values *= eta

            # (1-eta)*de_rhom_s
            eta -= 1.
            eta *= -1.
            de_rhom_s *= eta

            # metric_term * (DG_values + de_rhom_s)
            self._tmp_int_grid *= 0.
            self._tmp_int_grid += self._DG_values
            self._tmp_int_grid += de_rhom_s
            self._tmp_int_grid *= self._proj_rho2_metric_term

            # self._eval_dl_drho -= self._proj_rho2_metric_term * (self._DG_values + de_rhom_s)
            self._eval_dl_drho -= self._tmp_int_grid

        self._get_L2dofs_V3(self._eval_dl_drho, dofs=self._linear_form_dl_drho)

    def _get_jacobian(self, dt):
        uf_values = self.uf.eval_tp_fixed_loc(self.integration_grid_spans, [
                                              self.integration_grid_bn]*3, out=self._uf_values)
        uf1_values = self.uf1.eval_tp_fixed_loc(self.integration_grid_spans, [
                                                self.integration_grid_bn]*3, out=self._uf1_values)

        # Guf = metric @ uf
        for i in range(3):
            self._Guf_values[i] *= 0.
            self._Guf1_values[i] *= 0.
            for j in range(3):
                self._tmp_int_grid *= 0.
                self._tmp_int_grid += self._mass_u_metric_term[i, j]
                self._tmp_int_grid *= uf_values[j]
                self._Guf_values[i] += self._tmp_int_grid

                self._tmp_int_grid *= 0.
                self._tmp_int_grid += self._mass_u_metric_term[i, j]
                self._tmp_int_grid *= uf1_values[j]
                self._Guf1_values[i] += self._tmp_int_grid

        if self._model == 'barotropic':
            self._M_drho = -self.mass_ops.M3/2.

        if self._model == 'full':
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._rhof_values)
            rhof1_values = self.rhof1.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._rhof1_values)
            sf_values = self.sf.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._sf_values)

            # delta_rho_values = rhof1_values-rhof_values
            self._delta_rhof_values *= 0.
            self._delta_rhof_values += rhof1_values
            self._delta_rhof_values -= rhof_values
            delta_rho_values = self._delta_rhof_values

            eta = self.__eta(delta_rho_values)

            e_rho1_s = self.__ener(
                rhof1_values, sf_values, out=self._e_rho1_s_values)
            e_rho_s = self.__ener(rhof_values, sf_values,
                                  out=self._e_rho_s_values)

            de_rho1_s = self.__dener_drho(
                rhof1_values, sf_values, out=self._de_rhom_s_values)

            d2e_rho1_s = self.__d2ener_drho2(
                rhof1_values, sf_values, out=self._d2e_rho1_s_values)

            # eta*(de_rho1_s*delta_rho_values-e_rho1_s+e_rho_s)/(delta_rho_values**2+1e-40)
            self._DG_values *= 0.
            self._DG_values += de_rho1_s
            self._DG_values *= delta_rho_values
            self._DG_values -= e_rho1_s
            self._DG_values += e_rho_s
            delta_rho_values **= 2
            delta_rho_values += 1e-40
            self._DG_values /= delta_rho_values
            self._DG_values *= eta

            # (1-eta)*d2e_rho1_s
            eta -= 1.
            eta *= -1.
            d2e_rho1_s *= eta

            # -metric_term * (DG_values + d2e_rho1_s)
            self._tmp_int_grid *= 0.
            self._tmp_int_grid -= self._DG_values
            self._tmp_int_grid -= d2e_rho1_s
            self._tmp_int_grid *= self._proj_drho_metric_term

            self._M_drho.assemble([[self._tmp_int_grid]], verbose=False)

        self._M_un.assemble(
            [[self._Guf_values[0],   self._Guf_values[1],   self._Guf_values[2]]], verbose=False)
        self._M_un1.assemble(
            [[self._Guf1_values[0]], [self._Guf1_values[1]], [self._Guf1_values[2]]], verbose=False)

        # This way we can update only the scalar multiplying the operator and avoid creating multiple operators
        self._dt_pc_divPirhoT._scalar = dt
        self._dt2_pc_divPirhoT._scalar = dt/2
        self._dt2_divPirho._scalar = dt/2

    def _get_error_newton(self, mn_diff, rhon_diff):
        """Error for the newton method : max(|f(0)|,|f(1)|) where f is the function we're trying to nullify"""
        weak_un_diff = self._inv_Mv.dot(
            self.derham.boundary_ops['v'].dot(mn_diff), out=self._tmp_un_weak_diff)
        weak_rhon_diff = self.mass_ops.M3.dot(
            rhon_diff, out=self._tmp_rhon_weak_diff)
        err_rho = weak_rhon_diff.dot(rhon_diff)
        err_u = weak_un_diff.dot(mn_diff)
        return max(err_rho, err_u)

    def _get_error_picard(self, un_diff, rhon_diff):
        """Error for the picard method : difference between the two last iterations"""
        weak_un_diff = self.mass_ops.Mv.dot(
            un_diff, out=self._tmp_un_weak_diff)
        weak_rhon_diff = self.mass_ops.M3.dot(
            rhon_diff, out=self._tmp_rhon_weak_diff)
        err_rho = weak_rhon_diff.dot(rhon_diff)
        err_u = weak_un_diff.dot(un_diff)
        return max(err_rho, err_u)


class VariationalEntropyEvolve(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf u \in (H^1)^3` and :math:`s \in L^2` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \frac{\partial(\rho \mathcal U)}{\partial s} \nabla \cdot (s \mathbf v) \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = 0 \,.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta  
        - \int_{\hat{\Omega}} \left(\frac{\partial \hat{\rho}^3 \mathcal U}{\partial \hat{s}} \right) \nabla \cdot (\hat{s} \hat{\mathbf{v}}) \, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{s} + \nabla \cdot ( \hat{s} \hat{\mathbf{u}} ) = 0 ~ ,
        \end{align}

    where :math:`\mathcal U` depends on the chosen model. It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t} - 
        (\mathbb D \hat{\Pi}^{2}[\hat{\tilde{s}_h^{n+1}} \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3\left( \frac{\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n+1})-\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n})}{\hat{s}_h^{n+1}-\hat{s}_h^n} \right) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf s^{n+1}- \mathbf s^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{\tilde{s}_h^{n+1}} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        &\frac{\tilde{\boldsymbol s}^{n+1}- \boldsymbol s^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{\tilde{s}_h^{n+1}} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n} = 0 ~ \text{if parameter 'implicit_transport'},
        \\[2mm]
        &\tilde{\boldsymbol s}^{n+1} = \boldsymbol s^n ~ \text{else},
        \end{align}        

    where :math:`\hat{l}^3(f)` denotes the vector representing the linear form :math:`v_h \mapsto \int_{\hat{\Omega}} f(\boldsymbol \eta) v_h(\boldsymbol \eta) d \boldsymbol \eta`, that is the vector with components

    .. math::
        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    and the weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{k} = (\mathbf{u}^{k})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \, \text{for k in} \{n, n+1/2, n+1\}, \qquad \hat{s}_h^{k} = (s^{k})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \, \text{for k in} \{n, n+1/2, n+1\} \qquad \hat{\rho}_h^{n} = (\rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \.
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['lin_solver'] = {'tol': 1e-12,
                             'maxiter': 500,
                             'type': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                      ('cg', None)],
                             'verbose': False}
        dct['nonlin_solver'] = {'tol': 1e-8,
                                'maxiter': 100,
                                'type': ['Newton', 'Picard'],
                                'info': False,
                                'implicit_transport': False}
        dct['physics'] = {'gamma': 5/3}

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 s: StencilVector,
                 u: BlockVector,
                 *,
                 model: str = 'full',
                 gamma: float = options()['physics']['gamma'],
                 rho: StencilVector,
                 mass_ops: WeightedMassOperator,
                 lin_solver: dict = options(default=True)['lin_solver'],
                 nonlin_solver: dict = options(default=True)['nonlin_solver']):

        super().__init__(s, u)

        assert model in ['full']
        if model == 'full':
            assert rho is not None
        assert mass_ops is not None

        self._model = model
        self._gamma = gamma
        self._rho = rho
        self._mass_ops = mass_ops
        self._lin_solver = lin_solver
        self._nonlin_solver = nonlin_solver
        self._implicit_transport = nonlin_solver['implicit_transport']

        if self.derham.comm is not None:
            rank = self.derham.comm.Get_rank()
        else:
            rank = 0

        self._info = self._nonlin_solver['info'] and (rank == 0)

        self.WMM = mass_ops

        # Femfields for the projector
        self.rhof = self.derham.create_field("rhof", "L2")
        self.sf = self.derham.create_field("sf", "L2")
        self.sf1 = self.derham.create_field("sf1", "L2")
        self.uf = self.derham.create_field("uf", "H1vec")
        self.uf1 = self.derham.create_field("uf1", "H1vec")

        # Projector
        self._initialize_projectors_and_mass()

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
        if self._nonlin_solver['type'] == 'Newton':
            self.__call_newton(dt)
        elif self._nonlin_solver['type'] == 'Picard':
            self.__call_picard(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        if self._info:
            print()
            print("Newton iteration in VariationalEntropyEvolve")
        # Compute implicit approximation of s^{n+1}
        sn = self.feec_vars[0]
        un = self.feec_vars[1]
        self.uf.vector = un
        if self._implicit_transport:
            self._update_Piu()
            if self._info:
                print("Compute the implicit approximation")
            # Check it's usefull to invert the transport operator to avoid calling on Id and causing bugs
            # Explicit call
            self._dt_divPiu._scalar = -dt
            sn1 = self._transop.dot(sn, out=self._tmp_sn1)
            # Get diff
            sn_diff = sn1.copy(out=self._tmp_sn_diff)
            sn_diff -= sn
            weak_sn_diff = self.mass_ops.M3.dot(
                sn_diff, out=self._tmp_sn_weak_diff)
            err_s = weak_sn_diff.dot(sn_diff)

            if err_s > self._lin_solver['tol']:
                # Implicit call if needed
                self._dt_divPiu._scalar = dt
                sn1 = self._inv_transop.dot(sn, out=self._tmp_sn1)
            if self._info:
                print("Linear solver info : ", self._inv_transop._info)

        else:
            # No implicit
            sn1 = sn.copy(out=self._tmp_sn1)
        # Initialize variable for Newton iteration
        rho = self._rho
        self.rhof.vector = rho

        self.sf.vector = sn
        self.sf1.vector = sn1
        self._update_Pis()
        self.pc.update_mass_operator(self._Mrho)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        sn1 = sn.copy(out=self._tmp_sn1)
        self.sf1.vector = sn1
        un1 = un.copy(out=self._tmp_un1)
        mn1 = mn.copy(out=self._tmp_mn1)
        tol = self._nonlin_solver['tol']
        err = tol+1

        for it in range(self._nonlin_solver['maxiter']):

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

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or np.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = sn_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                print("information on the linear solver : ",
                      self._inv_Jacobian._solver._info)
            un1 -= incr[0]
            sn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            self.sf1.vector = sn1
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver['maxiter']-1 or np.isnan(err):
            print(
                f'!!!Warning: Maximum iteration in VariationalEntropyEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(sn1, un1)

    def __call_picard(self, dt):

        # Initialize variable for Picard iteration
        rho = self._rho
        self.rhof.vector = rho

        sn = self.feec_vars[0]
        sn1 = sn.copy(out=self._tmp_sn1)
        self.sf.vector = sn
        self.sf1.vector = sn1

        self._update_Pis()

        un = self.feec_vars[1]
        un1 = un.copy(out=self._tmp_un1)
        un2 = un1.copy(out=self._tmp_un2)
        self.uf.vector = un

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)

        self.pc.update_mass_operator(self._Mrho)

        tol = self._nonlin_solver['tol']
        err = tol+1
        for it in range(self._nonlin_solver['maxiter']):

            # Picard iteration
            if err < tol**2 or np.isnan(err):
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
            un1 = self._Mrhoinv.dot(mn1, out=self._tmp_un1)

            # get the error
            un_diff = un1.copy(out=self._tmp_un_diff)
            un_diff -= un2
            un2 = un1.copy(out=self._tmp_un2)

            err = self._get_error_picard(un_diff, sn_diff)

            if it == self._nonlin_solver['maxiter']-1 or np.isnan(err):
                print(
                    f'!!!Warning: Maximum iteration in VariationalEntropyEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(sn1, un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and `CoordinateProjector` needed to compute the bracket term"""

        from struphy.feec.projectors import L2Projector

        # Get the projector and the spaces
        P2 = self.derham.P['2']

        Xh = self.derham.Vh_fem['v']
        V3h = self.derham.Vh_fem['3']

        # Initialize the BasisProjectionOperators
        self.Pis = BasisProjectionOperator(
            P2, Xh, [[None, None, None],
                     [None, None, None],
                     [None, None, None]],
            transposed=False, use_cache=True,
            V_extraction_op=self.derham.extraction_ops['v'],
            V_boundary_op=self.derham.boundary_ops['v'],
            P_boundary_op=IdentityOperator(self.derham.Vh_pol['2']))

        self.Piu = BasisProjectionOperator(
            P2, V3h, [[None], [None], [None]],
            transposed=False, use_cache=True,
            P_boundary_op=IdentityOperator(self.derham.Vh_pol['2']))

        self.PisT = self.Pis.T

        self.div = self.derham.div_bcfree

        # Initialize the transport operator and transposed
        self.divPis = self.div @ self.Pis
        self.divPisT = self.PisT @ self.div.T
        self.divPiu = self.div@self.Piu

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
        self._uf_0_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_1])
        self._sf_1_values = np.zeros(grid_shape, dtype=float)
        self._uf_1_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_2])
        self._sf_2_values = np.zeros(grid_shape, dtype=float)
        self._uf_2_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]

        # weighted mass matrix to go from m to u
        self._Mrho = self.WMM

        # Inverse weighted mass matrix
        if self._lin_solver['type'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._lin_solver['type'][1])
            self.pc = pc_class(self._Mrho)

        self._Mrhoinv = inverse(self._Mrho,
                                self._lin_solver['type'][0],
                                pc=self.pc,
                                tol=self._lin_solver['tol'],
                                maxiter=self._lin_solver['maxiter'],
                                verbose=self._lin_solver['verbose'],
                                recycle=True)

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv)
        self._inv_Mv = inverse(self.mass_ops.Mv,
                               'pcg',
                               pc=self.pc_Mv,
                               tol=1e-16,
                               maxiter=1000,
                               verbose=False)

        # implicit transport operator
        self._I3 = IdentityOperator(self.derham.Vh_pol['3'])
        self._dt_divPiu = 2 * self.divPiu

        self._transop = self._I3 + self._dt_divPiu
        self._inv_transop = inverse(self._transop,
                                    'gmres',
                                    tol=self._lin_solver['tol'],
                                    maxiter=self._lin_solver['maxiter'],
                                    verbose=False,
                                    recycle=True)

        # For Newton solve
        self._M_ds = WeightedMassOperator(
            self.derham.Vh_fem['3'], self.derham.Vh_fem['3'],
            V_extraction_op=self.derham.extraction_ops['3'],
            W_extraction_op=self.derham.extraction_ops['3'],
            V_boundary_op=self.derham.boundary_ops['3'],
            W_boundary_op=self.derham.boundary_ops['3'])

        Jacs = BlockVectorSpace(
            self.derham.Vh_pol['v'], self.derham.Vh_pol['3'])

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        self._I3 = IdentityOperator(self.derham.Vh_pol['3'])

        # local version to avoid creating new version of LinearOperator every time
        self._dt_pc_divPisT = 2 * (self.divPisT)
        self._dt2_divPis = 2 * self.divPis

        self._Jacobian[0, 0] = self._Mrho
        self._Jacobian[0, 1] = self._dt_pc_divPisT@self._M_ds
        self._Jacobian[1, 0] = self._dt2_divPis
        self._Jacobian[1, 1] = self._I3

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(self._Jacobian, 'pcg',
                                             pc=self.pc,
                                             tol=self._lin_solver['tol'],
                                             maxiter=self._lin_solver['maxiter'],
                                             verbose=self._lin_solver['verbose'],
                                             recycle=True)

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],
        #                          recycle=True)

        # prepare for integration of linear form
        # L2-projector for V3
        self._get_L2dofs_V3 = L2Projector('L2', self.mass_ops).get_dofs

        integration_grid = [grid_1d.flatten()
                            for grid_1d in self.derham.quad_grid_pts['3']]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = self.derham.prepare_eval_tp_fixed(
            integration_grid)

        if self._model == 'full':
            grid_shape = tuple([len(loc_grid)
                                for loc_grid in integration_grid])
            self._sf_values = np.zeros(grid_shape, dtype=float)
            self._sf1_values = np.zeros(grid_shape, dtype=float)
            self._rhof_values = np.zeros(grid_shape, dtype=float)
            self._tmp_int_grid = np.zeros(grid_shape, dtype=float)
            self._delta_sf_values = np.zeros(grid_shape, dtype=float)
            self._sf_mid_values = np.zeros(grid_shape, dtype=float)
            self._eta_values = np.zeros(grid_shape, dtype=float)
            self._e_rho_s1_values = np.zeros(grid_shape, dtype=float)
            self._e_rho_s_values = np.zeros(grid_shape, dtype=float)
            self._de_rho_sm_values = np.zeros(grid_shape, dtype=float)
            self._d2e_rho_s1_values = np.zeros(grid_shape, dtype=float)

            gam = self._gamma
            metric = np.power(self.domain.jacobian_det(
                *integration_grid), 2-gam)
            self._proj_rho2_metric_term = deepcopy(metric)

            metric = np.power(self.domain.jacobian_det(
                *integration_grid), 1-gam)
            self._proj_ds_metric_term = deepcopy(metric)

    def __ener(self, rho, s, out=None):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = np.power(rho, gam)*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def __dener_ds(self, rho, s, out=None):
        """Derivative with respect to s of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        dE(rho, s)/ds = (rho^{gamma-1})*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = np.power(rho, gam-1)*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam-1, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def __d2ener_ds2(self, rho, s, out=None):
        """Second derivative with respect to (s, s) of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        d^2E(rho, s)/ds^2 = (rho^{gamma-2})*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = np.power(rho, gam-2)*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam-2, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def __eta(self, delta_x, out=None):
        if out is None:
            out = 1.-np.exp(-(delta_x/1e-5)**2)
        else:
            out *= 0.
            out += delta_x
            out /= 1e-5
            out **= 2
            out *= -1
            np.exp(out, out=out)
            out *= -1
            out += 1.
        return out

    def _update_Pis(self,):
        """Update the weights of the `BasisProjectionOperator`"""

        sf0_values = self.sf1.eval_tp_fixed_loc(
            self.hist_grid_0_spans, self.hist_grid_0_bd, out=self._sf_0_values)
        sf1_values = self.sf1.eval_tp_fixed_loc(
            self.hist_grid_1_spans, self.hist_grid_1_bd, out=self._sf_1_values)
        sf2_values = self.sf1.eval_tp_fixed_loc(
            self.hist_grid_2_spans, self.hist_grid_2_bd, out=self._sf_2_values)

        self.Pis.update_weights([[sf0_values, None, None],
                                 [None, sf1_values, None],
                                 [None, None, sf2_values]])

        self.PisT.update_weights([[sf0_values, None, None],
                                  [None, sf1_values, None],
                                  [None, None, sf2_values]])

    def _update_Piu(self,):
        """Update the weights of the `BasisProjectionOperator`"""

        uf0_values = self.uf.eval_tp_fixed_loc(
            self.hist_grid_0_spans, [self.hist_grid_0_bn]*3, out=self._uf_0_values)
        uf1_values = self.uf.eval_tp_fixed_loc(
            self.hist_grid_1_spans, [self.hist_grid_1_bn]*3, out=self._uf_1_values)
        uf2_values = self.uf.eval_tp_fixed_loc(
            self.hist_grid_2_spans, [self.hist_grid_2_bn]*3, out=self._uf_2_values)

        # print()
        # print()
        # print("ufvalues")
        # print(np.max(uf0_values))
        # print(np.max(uf1_values))
        # print(np.max(uf2_values))
        # print()
        # print()

        self.Piu.update_weights(
            [[uf0_values[0]], [uf1_values[1]], [uf2_values[2]]])

    def _update_linear_form_u2(self,):
        """Update the linearform representing integration in V3 against kynetic energy"""

        if self._model == 'full':
            sf_values = self.sf.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._sf_values)
            sf1_values = self.sf1.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._sf1_values)

            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._rhof_values)

            # delta_s_values = s1_values-sf_values
            self._delta_sf_values *= 0.
            self._delta_sf_values += sf1_values
            self._delta_sf_values -= sf_values
            delta_s_values = self._delta_sf_values

            # rho_mid_values = (rhof1_values+rhof_values)/2
            self._sf_mid_values *= 0.
            self._sf_mid_values += sf1_values
            self._sf_mid_values += sf_values
            self._sf_mid_values /= 2.
            s_mid_values = self._sf_mid_values

            eta = self.__eta(delta_s_values, out=self._eta_values)

            e_rho_s1 = self.__ener(
                rhof_values, sf1_values, out=self._e_rho_s1_values)
            e_rho_s = self.__ener(rhof_values, sf_values,
                                  out=self._e_rho_s_values)

            de_rho_sm = self.__dener_ds(
                rhof_values, s_mid_values, out=self._de_rho_sm_values)

            # metric_term * (eta*delta_s_values*(e_rho_s1-e_rho_s) / (delta_s_values**2+1e-40)+(1-eta)*de_rho_sm)

            # eta*delta_s_values*(e_rho_s1-e_rho_s) /(delta_s_values**2+1e-40)
            self._tmp_int_grid *= 0.
            self._tmp_int_grid += e_rho_s1
            self._tmp_int_grid -= e_rho_s
            self._tmp_int_grid *= delta_s_values
            self._tmp_int_grid *= eta

            # delta_s_values**2+1e-40
            delta_s_values **= 2
            delta_s_values += 1e-40
            self._tmp_int_grid /= delta_s_values

            # (1-eta)
            eta -= 1.
            eta *= -1.

            # (1-eta)*de_rho_sm
            de_rho_sm *= eta

            self._tmp_int_grid += de_rho_sm
            self._tmp_int_grid *= self._proj_rho2_metric_term
            self._tmp_int_grid *= -1.

        self._get_L2dofs_V3(self._tmp_int_grid, dofs=self._linear_form_dl_ds)

    def _get_jacobian(self, dt):

        if self._model == 'full':
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._rhof_values)
            sf_values = self.sf.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._sf_values)
            sf1_values = self.sf1.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._sf1_values)

            # delta_s_values = s1_values-sf_values
            self._delta_sf_values *= 0.
            self._delta_sf_values += sf1_values
            self._delta_sf_values -= sf_values
            delta_s_values = self._delta_sf_values

            eta = self.__eta(delta_s_values, out=self._eta_values)

            e_rho_s1 = self.__ener(
                rhof_values, sf1_values, out=self._e_rho_s1_values)
            e_rho_s = self.__ener(rhof_values, sf_values,
                                  out=self._e_rho_s_values)

            de_rho_s1 = self.__dener_ds(
                rhof_values, sf1_values, out=self._de_rho_sm_values)

            d2e_rho_s1 = self.__d2ener_ds2(
                rhof_values, sf1_values, out=self._d2e_rho_s1_values)

            # de_rho_s1*delta_s_values-e_rho_s1+e_rho_s
            self._tmp_int_grid *= 0.
            self._tmp_int_grid += de_rho_s1
            self._tmp_int_grid *= delta_s_values
            self._tmp_int_grid -= e_rho_s1
            self._tmp_int_grid += e_rho_s

            # (delta_s_values**2+1e-40)
            delta_s_values **= 2
            delta_s_values += 1e-40

            # eta*(de_rho_s1*delta_s_values-e_rho_s1+e_rho_s)/(delta_s_values**2+1e-40)
            self._tmp_int_grid /= delta_s_values
            self._tmp_int_grid *= eta

            # (1-eta)*d2e_rho_s1
            eta -= 1.
            eta *= -1.
            d2e_rho_s1 *= eta

            # -metric *(eta*(de_rho_s1*delta_s_values-e_rho_s1+e_rho_s)/(delta_s_values**2+1e-40) + (1-eta)*d2e_rho_s1)
            self._tmp_int_grid += d2e_rho_s1
            self._tmp_int_grid *= self._proj_ds_metric_term
            self._tmp_int_grid *= -1.

            self._M_ds.assemble([[self._tmp_int_grid]], verbose=False)

        # This way we can update only the scalar multiplying the operator and avoid creating multiple operators
        self._dt_pc_divPisT._scalar = dt
        self._dt2_divPis._scalar = dt/2

    def _get_error_newton(self, mn_diff, sn_diff):
        weak_un_diff = self._inv_Mv.dot(
            self.derham.boundary_ops['v'].dot(mn_diff), out=self._tmp_un_weak_diff)
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
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf u \in (H^1)^3` and :math:`s \in L^2` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times (\mathbf B \times \mathbf v) \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \cdot ( \mathbf B \times \mathbf u ) = 0 \,.

    On the logical domain:

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
        - (\mathbb C \hat{\Pi}^{1}[\hat{\tilde{B}_h^{n+1}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \mathbb M^2 B^{n+\frac{1}{2}} \big) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf b^{n+1}- \mathbf b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{\tilde{B}_h^{n+1}} \cdot \vec{\boldsymbol \Lambda}^v]] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        &\frac{\tilde{\boldsymbol b}^{n+1}-\boldsymbol b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{\tilde{B}_h^{n+1}} \cdot \vec{\boldsymbol \Lambda}^v] \mathbf u^{n} = 0 ~ \text{if parameter 'implicit_transport'},
        \\[2mm]
        &\tilde{\boldsymbol b}^{n+1} = \boldsymbol b^n ~ \text{else},
        \end{align}

    where weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{B}}_h^{n+1/2} = (\mathbf{b}^{n+1/2})^\top \vec{\boldsymbol \Lambda}^2 \in V_h^2 \, \qquad \hat{\rho}_h^{n} = (\boldsymbol \rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.

    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['lin_solver'] = {'tol': 1e-12,
                             'maxiter': 500,
                             'non_linear_maxiter': 100,
                             'type': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                      ('cg', None)],
                             'verbose': False}
        dct['nonlin_solver'] = {'tol': 1e-8,
                                'maxiter': 100,
                                'type': ['Newton', 'Picard'],
                                'info': False,
                                'implicit_transport': False}

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 b: BlockVector,
                 u: BlockVector,
                 *,
                 mass_ops: WeightedMassOperator,
                 lin_solver: dict = options(default=True)['lin_solver'],
                 nonlin_solver: dict = options(default=True)['nonlin_solver']):

        super().__init__(b, u)

        self._mass_ops = mass_ops
        self._lin_solver = lin_solver
        self._nonlin_solver = nonlin_solver
        self._implicit_transport = nonlin_solver['implicit_transport']

        if self.derham.comm is not None:
            rank = self.derham.comm.Get_rank()
        else:
            rank = 0

        self._info = self._nonlin_solver['info'] and (rank == 0)

        self.WMM = mass_ops

        # Femfields for the projector
        self.bf = self.derham.create_field("bf", "Hdiv")

        self.uf = self.derham.create_field("uf", "H1vec")
        self.uf1 = self.derham.create_field("uf1", "H1vec")

        # Projector
        self._initialize_projectors_and_mass()

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
        if self._nonlin_solver['type'] == 'Newton':
            self.__call_newton(dt)
        elif self._nonlin_solver['type'] == 'Picard':
            self.__call_picard(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        if self._info:
            print()
            print("Newton iteration in VariationalMagFieldEvolve")
        # Compute implicit approximation of s^{n+1}
        bn = self.feec_vars[0]
        un = self.feec_vars[1]
        self.uf.vector = un
        if self._implicit_transport:
            self._update_Piu()
            if self._info:
                print("Compute the implicit approximation")
            # Check it's usefull to invert the transport operator to avoid calling on Id and causing bugs
            # Explicit call
            self._dt_curlPiu._scalar = -dt
            bn1 = self._transop.dot(bn, out=self._tmp_bn1)
            # Get diff
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            weak_bn_diff = self.mass_ops.M2.dot(
                bn_diff, out=self._tmp_bn_weak_diff)
            err_b = weak_bn_diff.dot(bn_diff)

            if err_b > self._lin_solver['tol']:
                # Implicit call if needed
                self._dt_curlPiu._scalar = dt
                bn1 = self._inv_transop.dot(bn, out=self._tmp_bn1)
            if self._info:
                print("Linear solver info : ", self._inv_transop._info)
        else:
            # No implicit
            bn1 = bn.copy(out=self._tmp_bn1)
        # Initialize variable for Newton iteration

        self.bf.vector = bn1
        self._update_Pib()
        self.pc.update_mass_operator(self._Mrho)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        bn1 = bn.copy(out=self._tmp_bn1)
        self.bf.vector = bn1
        un1 = un.copy(out=self._tmp_un1)
        mn1 = mn.copy(out=self._tmp_mn1)
        tol = self._nonlin_solver['tol']
        err = tol+1

        for it in range(self._nonlin_solver['maxiter']):

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

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or np.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = bn_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                print("information on the linear solver : ",
                      self._inv_Jacobian._solver._info)
            un1 -= incr[0]
            bn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver['maxiter']-1 or np.isnan(err):
            print(
                f'!!!Warning: Maximum iteration in VariationalMagFieldEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(bn1, un1)

    def __call_picard(self, dt):

        # Initialize variable for Picard iteration

        bn = self.feec_vars[0]
        bn1 = bn.copy(out=self._tmp_bn1)
        self.bf.vector = bn
        self._update_Pib()

        un = self.feec_vars[1]
        un1 = un.copy(out=self._tmp_un1)
        un2 = un1.copy(out=self._tmp_un2)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)

        self.pc.update_mass_operator(self._Mrho)

        tol = self._nonlin_solver['tol']
        err = tol+1
        for it in range(self._nonlin_solver['maxiter']):

            # Picard iteration
            if err < tol**2 or np.isnan(err):
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
            un1 = self._Mrhoinv.dot(mn1, out=self._tmp_un1)

            # get the error
            un_diff = un1.copy(out=self._tmp_un_diff)
            un_diff -= un2
            un2 = un1.copy(out=self._tmp_un2)

            err = self._get_error_picard(un_diff, bn_diff)

            if it == self._nonlin_solver['maxiter']-1 or np.isnan(err):
                print(
                    f'!!!Warning: Maximum iteration in VariationalMagFieldEvolve reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(bn1, un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        # Get the projector and the spaces
        P1 = self.derham.P['1']

        Xh = self.derham.Vh_fem['v']
        V2h = self.derham.Vh_fem['2']

        # Initialize the BasisProjectionOperators
        self.Pib = BasisProjectionOperator(
            P1, Xh, [[None, None, None],
                     [None, None, None],
                     [None, None, None]],
            transposed=False, use_cache=True,
            V_extraction_op=self.derham.extraction_ops['v'],
            V_boundary_op=self.derham.boundary_ops['v'],
            P_boundary_op=IdentityOperator(self.derham.Vh_pol['1']))

        self.Piu = BasisProjectionOperator(
            P1, V2h, [[None, None, None],
                      [None, None, None],
                      [None, None, None]],
            transposed=False, use_cache=True,
            P_boundary_op=IdentityOperator(self.derham.Vh_pol['1']))

        self.PibT = self.Pib.T

        # gradient of the component of the vector field
        self.curl = self.derham.curl_bcfree

        # Initialize the transport operator and transposed
        self.curlPib = self.curl @ self.Pib
        self.curlPibT = self.PibT @ self.curl.T

        self.curlPiu = self.curl @ self.Piu

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
        self._uf0_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        self.hist_grid_0_b = [[self.hist_grid_0_bn[0], self.hist_grid_0_bd[1], self.hist_grid_0_bd[2]],
                              [self.hist_grid_0_bd[0], self.hist_grid_0_bn[1],
                                  self.hist_grid_0_bd[2]],
                              [self.hist_grid_0_bd[0], self.hist_grid_0_bd[1], self.hist_grid_0_bn[2]]]
        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_1])
        self._bf1_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        self._uf1_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        self.hist_grid_1_b = [[self.hist_grid_1_bn[0], self.hist_grid_1_bd[1], self.hist_grid_1_bd[2]],
                              [self.hist_grid_1_bd[0], self.hist_grid_1_bn[1],
                                  self.hist_grid_1_bd[2]],
                              [self.hist_grid_1_bd[0], self.hist_grid_1_bd[1], self.hist_grid_1_bn[2]]]

        grid_shape = tuple([len(loc_grid)
                           for loc_grid in hist_grid_2])
        self._bf2_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        self._uf2_values = [np.zeros(grid_shape, dtype=float)
                            for i in range(3)]
        self.hist_grid_2_b = [[self.hist_grid_2_bn[0], self.hist_grid_2_bd[1], self.hist_grid_2_bd[2]],
                              [self.hist_grid_2_bd[0], self.hist_grid_2_bn[1],
                                  self.hist_grid_2_bd[2]],
                              [self.hist_grid_2_bd[0], self.hist_grid_2_bd[1], self.hist_grid_2_bn[2]]]

        # weighted mass matrix to go from m to u
        self._Mrho = self.WMM

        # Inverse weighted mass matrix
        if self._lin_solver['type'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._lin_solver['type'][1])
            self.pc = pc_class(self._Mrho)

        self._Mrhoinv = inverse(self._Mrho,
                                self._lin_solver['type'][0],
                                pc=self.pc,
                                tol=self._lin_solver['tol'],
                                maxiter=self._lin_solver['maxiter'],
                                verbose=self._lin_solver['verbose'],
                                recycle=True)

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv)
        self._inv_Mv = inverse(self.mass_ops.Mv,
                               'pcg',
                               pc=self.pc_Mv,
                               tol=1e-16,
                               maxiter=1000,
                               verbose=False)

        self._I2 = IdentityOperator(self.derham.Vh_pol['2'])
        self._dt_curlPiu = 2 * self.curlPiu

        self._transop = self._I2 + self._dt_curlPiu
        self._inv_transop = inverse(self._transop,
                                    'gmres',
                                    tol=self._lin_solver['tol'],
                                    maxiter=self._lin_solver['maxiter'],
                                    verbose=False,
                                    recycle=True)

        Jacs = BlockVectorSpace(
            self.derham.Vh_pol['v'], self.derham.Vh_pol['2'])

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        # local version to avoid creating new version of LinearOperator every time
        self._mdt2_pc_curlPibT_M = 2 * \
            (self.curlPibT@self.mass_ops.M2)
        self._dt2_curlPib = 2 * self.curlPib

        self._Jacobian[0, 0] = self._Mrho
        self._Jacobian[0, 1] = self._mdt2_pc_curlPibT_M
        self._Jacobian[1, 0] = self._dt2_curlPib
        self._Jacobian[1, 1] = self._I2

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(self._Jacobian, 'pcg',
                                             pc=self.pc,
                                             tol=self._lin_solver['tol'],
                                             maxiter=self._lin_solver['maxiter'],
                                             verbose=self._lin_solver['verbose'],
                                             recycle=True)

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],
        #                          recycle=True)

    def _update_Pib(self,):
        """Update the weights of the `BasisProjectionOperator`"""

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

    def _update_Piu(self,):
        """Update the weights of the `BasisProjectionOperator`"""

        uf0_values = self.uf.eval_tp_fixed_loc(
            self.hist_grid_0_spans, [self.hist_grid_0_bn]*3, out=self._uf0_values)
        uf1_values = self.uf.eval_tp_fixed_loc(
            self.hist_grid_1_spans, [self.hist_grid_1_bn]*3, out=self._uf1_values)
        uf2_values = self.uf.eval_tp_fixed_loc(
            self.hist_grid_2_spans, [self.hist_grid_2_bn]*3, out=self._uf2_values)

        self.Piu.update_weights([[None, uf0_values[2], -uf0_values[1]],
                                 [-uf1_values[2], None, uf1_values[0]],
                                 [uf2_values[1], -uf2_values[0], None]])

    def _update_linear_form_u2(self,):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        wb = self.mass_ops.M2.dot(self._tmp_bn12, out=self._linear_form_dl_db)
        wb *= -1

    def _get_error_newton(self, mn_diff, bn_diff):
        weak_un_diff = self._inv_Mv.dot(
            self.derham.boundary_ops['v'].dot(mn_diff), out=self._tmp_un_weak_diff)
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

    def _get_jacobian(self, dt):
        self._mdt2_pc_curlPibT_M._scalar = -dt/2
        self._dt2_curlPib._scalar = dt/2


class VariationalViscosity(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\rho \in L^2` and  :math:`\mathbf u \in (H^1)^3` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x + \int_\Omega (\mu + \mu_a(\mathbf x)) \nabla \mathbf u : \nabla \mathbf v \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3 \,,
        \\[4mm]
        &\int_\Omega \frac{\partial \mathcal U}{\partial s} \partial_t s \, q \,\textrm d \mathbf x - \mu \int_\Omega |\nabla \mathbf u|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\,.

    With :math:`\mu_a(\mathbf x) = \mu_a |\nabla \mathbf u(\mathbf x)|`

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta  
        + \mu \int_{\hat{\Omega}} \nabla (DF \hat{\mathbf{u}}) : \nabla (DF \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\hat{\rho} \hat{e}(\hat{\rho}, \hat{s})) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta -  \int_{\hat{\Omega}} (\mu + \mu_a(\boldsymbol \eta)) \nabla (DF \hat{\mathbf{u}}) : \nabla (DF \hat{\mathbf{u}}) \hat{w} \, \textrm d \boldsymbol \eta = 0 ~ .
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        +  \sum_\nu (\mathbb G \mathcal{X}^v_\nu)^T (\mu \mathbb M_0 + \mu_a \mathbb M_0[|\nabla u|] \mathbb G \mathcal{X}^v_\nu \mathbf u^{n+1} = 0 ~ ,
        \\[2mm]
        &\frac{P^{3}(\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n}))- P^{3}(\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n+1}))}{\Delta t} - \mu P^3(\sum_\nu DF \mathcal{X}^v_\nu \frac{ \mathbf u^{n+1}+\mathbf u^n}{2} \cdot DF \mathcal{X}^v_\nu \mathbf u^{n+1}) = 0 ~ ,
        \end{align}

    where $P^3$ denotes the $L^2$ projection in the last space of the de Rham sequence and the weights in :math:`\mathbb M_0[|\nabla u|]` are given by

    .. math::
        P^0(g \sqrt{\sum_\nu |(\mathbb G \mathcal{X}^v_\nu \mathbb u)^\top \vec{\boldsymbol \Lambda}^0 |^2]})^\top \vec{\boldsymbol \Lambda}^0 ~.

    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['lin_solver'] = {'tol': 1e-12,
                             'maxiter': 500,
                             'type': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                      ('cg', None)],
                             'verbose': False, }
        dct['nonlin_solver'] = {'tol': 1e-8,
                                'maxiter': 100,
                                'type': ['Newton'],
                                'info': False}
        dct['physics'] = {'gamma': 1.66666666667,
                          'mu': 0., 'mu_a': 0.}

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 s: StencilVector,
                 u: BlockVector,
                 *,
                 model: str = 'barotropic',
                 gamma: float = options()['physics']['gamma'],
                 rho: StencilVector,
                 mu: float = options()['physics']['mu'],
                 mu_a: float = options()['physics']['mu_a'],
                 mass_ops: WeightedMassOperator,
                 lin_solver: dict = options(default=True)['lin_solver'],
                 nonlin_solver: dict = options(default=True)['nonlin_solver']):

        super().__init__(s, u)

        assert model == 'full'

        self._model = model
        self._gamma = gamma
        self._mass_ops = mass_ops
        self._lin_solver = lin_solver
        self._nonlin_solver = nonlin_solver
        self._mu_a = mu_a
        self._mu = mu
        self._rho = rho

        if self.derham.comm is not None:
            rank = self.derham.comm.Get_rank()
        else:
            rank = 0

        self._info = self._nonlin_solver['info'] and (rank == 0)

        self._Mrho = mass_ops

        # Femfields for the projector
        self.rhof = self.derham.create_field("rhof", "L2")
        self.sf = self.derham.create_field("sf", "L2")
        self.sf1 = self.derham.create_field("sf1", "L2")
        self.uf = self.derham.create_field("uf", "H1vec")
        self.uf1 = self.derham.create_field("uf1", "H1vec")
        self.gu0f = self.derham.create_field("gu0", "Hcurl")
        self.gu1f = self.derham.create_field("gu1", "Hcurl")
        self.gu2f = self.derham.create_field("gu2", "Hcurl")
        self.gu120f = self.derham.create_field("gu120", "Hcurl")
        self.gu121f = self.derham.create_field("gu121", "Hcurl")
        self.gu122f = self.derham.create_field("gu122", "Hcurl")

        # Projector
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn_incr = s.space.zeros()
        self._tmp_sn_weak_diff = s.space.zeros()
        self._tmp_gu0 = self.derham.Vh['1'].zeros()
        self._tmp_gu1 = self.derham.Vh['1'].zeros()
        self._tmp_gu2 = self.derham.Vh['1'].zeros()
        self._tmp_gu120 = self.derham.Vh['1'].zeros()
        self._tmp_gu121 = self.derham.Vh['1'].zeros()
        self._tmp_gu122 = self.derham.Vh['1'].zeros()
        self._linear_form_tot_e = s.space.zeros()
        self._linear_form_e_sn1 = s.space.zeros()
        self.tot_rhs = s.space.zeros()

    def __call__(self, dt):
        if self._nonlin_solver['type'] == 'Newton':
            self.__call_newton(dt)
        else:
            raise ValueError(
                'wrong value for solver type in VariationalViscosity')

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # Compute dissipation implicitely
        self.pc.update_mass_operator(self._Mrho)
        sn = self.feec_vars[0]
        un = self.feec_vars[1]
        if self._mu < 1.e-15 and self._mu_a < 1.e-15:
            self.feec_vars_update(sn, un)
            return

        if self._info:
            print()
            print("Computing the dissipation in VariationalViscosity")

        # Update artificial viscosity weighted mass matrix
        gu0 = self.grad_0.dot(un, out=self._tmp_gu0)
        gu1 = self.grad_1.dot(un, out=self._tmp_gu1)
        gu2 = self.grad_2.dot(un, out=self._tmp_gu2)

        self.gu0f.vector = gu0
        self.gu1f.vector = gu1
        self.gu2f.vector = gu2

        gu0_v = self.gu0f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_gradient, out=self._guf0_values)
        gu1_v = self.gu1f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_gradient, out=self._guf1_values)
        gu2_v = self.gu2f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_gradient, out=self._guf2_values)

        gu_sq_v = self._gu_init_values
        gu_sq_v *= 0.
        for i in range(3):
            gu0_v[i] **= 2
            gu1_v[i] **= 2
            gu2_v[i] **= 2
            gu_sq_v += gu0_v[i]
            gu_sq_v += gu1_v[i]
            gu_sq_v += gu2_v[i]

        np.sqrt(gu_sq_v, out=gu_sq_v)

        gu_sq_v *= dt*self._mu_a  # /2

        self.M1_du.assemble([[gu_sq_v*self._mass_M1_metric[0, 0], gu_sq_v*self._mass_M1_metric[0, 1], gu_sq_v*self._mass_M1_metric[0, 2]],
                             [gu_sq_v*self._mass_M1_metric[1, 0], gu_sq_v *
                                 self._mass_M1_metric[1, 1], gu_sq_v*self._mass_M1_metric[1, 2]],
                             [gu_sq_v*self._mass_M1_metric[2, 0], gu_sq_v*self._mass_M1_metric[1, 2], gu_sq_v*self._mass_M1_metric[2, 2]]],
                            verbose=False)

        # gu_sq_v *= 2.
        gu_sq_v += dt*self._mu

        self._scaled_stiffness._scalar = dt*self._mu  # /2.
        # self.evol_op._multiplicants[1]._addends[0]._scalar = - dt*self._mu/2.
        un1 = self.evol_op.dot(un, out=self._tmp_un1)
        if self._info:
            print("information on the linear solver : ", self.inv_lop._info)

        # Energy balance term
        # 1) Pointwize energy change
        un12 = un.copy(out=self._tmp_un12)
        un12 += un1
        un12 /= 2.
        gu0 = self.grad_0.dot(un1, out=self._tmp_gu0)
        gu1 = self.grad_1.dot(un1, out=self._tmp_gu1)
        gu2 = self.grad_2.dot(un1, out=self._tmp_gu2)

        gu012 = self.grad_0.dot(un12, out=self._tmp_gu120)
        gu112 = self.grad_1.dot(un12, out=self._tmp_gu121)
        gu212 = self.grad_2.dot(un12, out=self._tmp_gu122)

        self.gu0f.vector = gu0
        self.gu1f.vector = gu1
        self.gu2f.vector = gu2

        self.gu120f.vector = gu012
        self.gu121f.vector = gu112
        self.gu122f.vector = gu212

        gu0_v = self.gu0f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_gradient, out=self._guf0_values)
        gu1_v = self.gu1f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_gradient, out=self._guf1_values)
        gu2_v = self.gu2f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_gradient, out=self._guf2_values)

        gu120_v = self.gu120f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_gradient, out=self._guf120_values)
        gu121_v = self.gu121f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_gradient, out=self._guf121_values)
        gu122_v = self.gu122f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_gradient, out=self._guf122_values)

        gu_sq_v = self._gu_sq_values
        gu_sq_v *= 0.
        for i in range(3):
            for j in range(3):
                gu_sq_v += gu0_v[i]*self._mass_M1_metric[i, j]*gu120_v[j]
                gu_sq_v += gu1_v[i]*self._mass_M1_metric[i, j]*gu121_v[j]
                gu_sq_v += gu2_v[i]*self._mass_M1_metric[i, j]*gu122_v[j]

        gu_sq_v *= self._gu_init_values
        # 2) Initial energy and linear form
        rho = self._rho
        self.rhof.vector = rho
        self.sf.vector = sn

        sf_values = self.sf.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_bd, out=self._sf_values)

        rhof_values = self.rhof.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_bd, out=self._rhof_values)

        e_rho_s = self.__ener(rhof_values, sf_values,
                              out=self._e_rho_s_values)

        e_rho_s *= self._energy_metric

        gu_sq_v += e_rho_s

        self._get_L2dofs_V3(gu_sq_v, dofs=self._linear_form_tot_e)

        # 3) Newton iteration
        sn1 = sn.copy(out=self._tmp_sn1)

        tol = self._nonlin_solver['tol']
        err = tol+1

        for it in range(self._nonlin_solver['maxiter']):

            self.sf1.vector = sn1

            sf1_values = self.sf1.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._sf1_values)

            e_rho_s1 = self.__ener(
                rhof_values, sf1_values, out=self._e_rho_s1_values)

            e_rho_s1 *= self._energy_metric

            self._get_L2dofs_V3(e_rho_s1, dofs=self._linear_form_e_sn1)

            self.tot_rhs *= 0.
            self.tot_rhs -= self._linear_form_e_sn1
            self.tot_rhs += self._linear_form_tot_e

            err = self._get_error_newton(self.tot_rhs)

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or np.isnan(err):
                break

            deds = self.__dener_ds(
                rhof_values, sf1_values, out=self._de_rho_s1_values)
            deds *= self._mass_metric_term

            self.M_de_ds.assemble([[deds]], verbose=False)
            self.pc_jac.update_mass_operator(self.M_de_ds)

            incr = self.inv_jac.dot(self.tot_rhs, out=self._tmp_sn_incr)

            if self._info:
                print("information on the linear solver : ", self.inv_jac._info)

            sn1 += incr

        if it == self._nonlin_solver['maxiter']-1 or np.isnan(err):
            print(
                f'!!!Warning: Maximum iteration in VariationalViscosity reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(sn1, un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        from struphy.feec.projectors import L2Projector

        Xv = getattr(self.basis_ops, 'Xv')
        Pcoord0 = CoordinateProjector(
            0, self.derham.Vh_pol['v'], self.derham.Vh_pol['0'])
        Pcoord1 = CoordinateProjector(
            1, self.derham.Vh_pol['v'], self.derham.Vh_pol['0'])
        Pcoord2 = CoordinateProjector(
            2, self.derham.Vh_pol['v'], self.derham.Vh_pol['0'])

        g = self.mass_ops.sqrt_g

        M1 = self.mass_ops.M1

        self.M1_du = WeightedMassOperator(
            self.derham.Vh_fem['1'],
            self.derham.Vh_fem['1'],
            V_extraction_op=self.derham.extraction_ops['1'],
            W_extraction_op=self.derham.extraction_ops['1'])

        self.pc_M3 = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.M3)
        self._inv_M3 = inverse(self.mass_ops.M3,
                               'pcg',
                               pc=self.pc_M3,
                               tol=1e-16,
                               maxiter=1000,
                               verbose=False)

        self.M_de_ds = WeightedMassOperator(
            self.derham.Vh_fem['3'],
            self.derham.Vh_fem['3'],
            V_extraction_op=self.derham.extraction_ops['3'],
            W_extraction_op=self.derham.extraction_ops['3'])

        if self._lin_solver['type'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._lin_solver['type'][1])
            self.pc_jac = pc_class(self.M_de_ds)

        self.inv_jac = inverse(self.M_de_ds,
                               'pcg',
                               pc=self.pc_jac,
                               tol=self._lin_solver['tol'],
                               maxiter=self._lin_solver['maxiter'],
                               verbose=False,
                               recycle=True)

        grad = self.derham.grad
        self.scalar_stiffness = grad.T@M1@grad
        self.log_stiffness = Pcoord0.T@self.scalar_stiffness@Pcoord0 \
            + Pcoord1.T@self.scalar_stiffness@Pcoord1 \
            + Pcoord2.T@self.scalar_stiffness@Pcoord2

        self.phy_stiffness = Xv.T@self.log_stiffness@Xv

        self._scaled_stiffness = .00001 * self.phy_stiffness

        self.du_stiffness = grad.T@self.M1_du@grad
        self.du_log_stiffness = Pcoord0.T@self.du_stiffness@Pcoord0 \
            + Pcoord1.T@self.du_stiffness@Pcoord1 \
            + Pcoord2.T@self.du_stiffness@Pcoord2

        self.du_phy_stiffness = Xv.T@self.du_log_stiffness@Xv

        self._scaled_stiffness = .00001 * self.phy_stiffness

        self.r_op = self._Mrho  # - self._scaled_stiffness - self.du_phy_stiffness
        self.l_op = self._Mrho + self._scaled_stiffness + self.du_phy_stiffness

        self.grad_0 = grad@Pcoord0@Xv
        self.grad_1 = grad@Pcoord1@Xv
        self.grad_2 = grad@Pcoord2@Xv

        if self._lin_solver['type'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._lin_solver['type'][1])
            self.pc = pc_class(self._Mrho)

        self.inv_lop = inverse(self.l_op,
                               'pcg',
                               pc=self.pc,
                               tol=self._lin_solver['tol'],
                               maxiter=self._lin_solver['maxiter'],
                               verbose=False,
                               recycle=True)

        self.evol_op = self.inv_lop @ self.r_op
        # self.evol_op = IdentityOperator(self.derham.Vh_pol['v'])
        integration_grid = [grid_1d.flatten()
                            for grid_1d in self.derham.quad_grid_pts['3']]
        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = self.derham.prepare_eval_tp_fixed(
            integration_grid)

        self.integration_grid_gradient = [[self.integration_grid_bd[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
                                          [self.integration_grid_bn[0], self.integration_grid_bd[1],
                                           self.integration_grid_bn[2]],
                                          [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bd[2]]]

        grid_shape = tuple([len(loc_grid)
                            for loc_grid in integration_grid])

        self._guf0_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]
        self._guf1_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]
        self._guf2_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]

        self._guf120_values = [
            np.zeros(grid_shape, dtype=float)for i in range(3)]
        self._guf121_values = [
            np.zeros(grid_shape, dtype=float)for i in range(3)]
        self._guf122_values = [
            np.zeros(grid_shape, dtype=float)for i in range(3)]

        self._gu_sq_values = np.zeros(grid_shape, dtype=float)
        self._gu_init_values = np.zeros(grid_shape, dtype=float)

        self._sf_values = np.zeros(grid_shape, dtype=float)
        self._sf1_values = np.zeros(grid_shape, dtype=float)
        self._rhof_values = np.zeros(grid_shape, dtype=float)

        self._e_rho_s1_values = np.zeros(grid_shape, dtype=float)
        self._e_rho_s_values = np.zeros(grid_shape, dtype=float)

        self._de_rho_s1_values = np.zeros(grid_shape, dtype=float)

        self._tmp_int_grid = np.zeros(grid_shape, dtype=float)

        gam = self._gamma

        metric = np.power(self.domain.jacobian_det(
            *integration_grid), -gam)
        self._mass_metric_term = deepcopy(metric)

        metric = np.power(self.domain.jacobian_det(
            *integration_grid), 1-gam)
        self._energy_metric = deepcopy(metric)

        metric = np.power(self.domain.jacobian_det(
            *integration_grid), 1)
        self._sq_term_metric = deepcopy(metric)

        metric = self.domain.metric_inv(
            *integration_grid)*self.domain.jacobian_det(*integration_grid)
        self._mass_M1_metric = deepcopy(metric)

        self._get_L2dofs_V3 = L2Projector('L2', self.mass_ops).get_dofs

    def __ener(self, rho, s, out=None):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = np.power(rho, gam)*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def __dener_ds(self, rho, s, out=None):
        """Derivative with respect to s of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        dE(rho, s)/ds = (rho^{gamma-1})*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = np.power(rho, gam-1)*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam-1, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def _get_error_newton(self, sn_diff):
        weak_sn_diff = self._inv_M3.dot(sn_diff, out=self._tmp_sn_weak_diff)
        err_s = weak_sn_diff.dot(sn_diff)
        return err_s


class VariationalResistivity(Propagator):
    r''':ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`s \in L^2` and  :math:`\mathbf B \in H(\textrm{div})` such that

    .. math::

        &\int_\Omega \partial_t \mathbf B \cdot \mathbf v \,\textrm d \mathbf x + \int_\Omega (\eta + \eta_a(\mathbf x)) \nabla \times \mathbf B \cdot \nabla \times \mathbf v \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in H(\textrm{div}) \,,
        \\[4mm]
        &\int_\Omega \frac{\partial \mathcal U}{\partial s} \partial_t s \, q\,\textrm d \mathbf x - \int_\Omega (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\,.

    With :math:`\eta_a(\mathbf x) = \eta_a |\nabla \times \mathbf B(\mathbf x)|`

    On the logical domain:

    .. math::

        \begin{align}
        &\partial_t \hat{\boldsymbol B} - \eta \Delta \hat{\boldsymbol B} = 0 ~ ,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\hat{\rho} \mathcal U(\hat{\rho}, \hat{s})) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta - \int_{\hat{\Omega}} (\eta + \eta_a(\mathbf \eta)) |DF^{-T}\tilde{\nabla} \times \hat{\boldsymbol B}|^2  \hat{w} \, \textrm d \boldsymbol \eta = 0 ~ .
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\frac{\mathbf B^{n+1}-\mathbf B^n}{\Delta t} 
        + \, \mathbb C \mathbb M_1^{-1} (\eta M_1 + \eta_a M_1[|\nabla \times \mathbf B|]) M_1^{-1} \mathbb C^T \mathbb M_2  \mathbf B^{n+1} = 0 ~ ,
        \\[2mm]
        &\frac{P^{3}(\rho e(s^{n+1})- P^{3}(\rho e(s^{n}))}{\Delta t} - P^3((\eta + \eta_a(\mathbf x)) DF^{-T} \tilde{\mathbb C} \frac{ \mathbf B^{n+1}+\mathbf B^n}{2} \cdot DF^{-T} \tilde{\mathbb C} \mathbf B^{n+1}) = 0 ~ ,
        \end{align}

    where $P^3$ denotes the $L^2$ projection in the last space of the de Rham sequence.

    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['lin_solver'] = {'tol': 1e-12,
                             'maxiter': 500,
                             'type': [('pcg', 'MassMatrixDiagonalPreconditioner'),
                                      ('cg', None)],
                             'verbose': False}
        dct['nonlin_solver'] = {'tol': 1e-8,
                                'maxiter': 100,
                                'type': ['Newton'],
                                'info': False}
        dct['physics'] = {'eta': 0.,
                          'eta_a': 0.,
                          'gamma': 5/3}
        dct['linearize_current'] = False

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 s: StencilVector,
                 b: BlockVector,
                 *,
                 model: str = 'full',
                 gamma: float = options()['physics']['gamma'],
                 rho: StencilVector,
                 eta: float = options()['physics']['eta'],
                 eta_a: float = options()['physics']['eta_a'],
                 lin_solver: dict = options(default=True)['lin_solver'],
                 nonlin_solver: dict = options(default=True)['nonlin_solver'],
                 linearize_current: dict = options(default=True)['linearize_current']):

        super().__init__(s, b)

        assert model in ['full']

        self._model = model
        self._gamma = gamma
        self._eta = eta
        self._eta_a = eta_a
        self._lin_solver = lin_solver
        self._nonlin_solver = nonlin_solver
        self._rho = rho
        self._linearize_current = linearize_current

        if self.derham.comm is not None:
            rank = self.derham.comm.Get_rank()
        else:
            rank = 0

        self._info = self._nonlin_solver['info'] and (rank == 0)

        # Femfields for the projector
        self.rhof = self.derham.create_field("rhof", "L2")
        self.sf = self.derham.create_field("sf", "L2")
        self.sf1 = self.derham.create_field("sf1", "L2")
        self.bf = self.derham.create_field("Bf", "Hdiv")
        self.bf1 = self.derham.create_field("Bf1", "Hdiv")
        self.cbf1 = self.derham.create_field("cBf", "Hcurl")
        self.cbf12 = self.derham.create_field("cBf", "Hcurl")

        # Projector
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn_incr = s.space.zeros()
        self._tmp_sn_weak_diff = s.space.zeros()
        self._tmp_cb12 = self.derham.Vh['1'].zeros()
        self._tmp_cb1 = self.derham.Vh['1'].zeros()
        self._linear_form_tot_e = s.space.zeros()
        self._linear_form_e_sn1 = s.space.zeros()
        self.tot_rhs = s.space.zeros()

    def __call__(self, dt):
        if self._nonlin_solver['type'] == 'Newton':
            self.__call_newton(dt)
        else:
            raise ValueError(
                'wrong value for solver type in VariationalResistivity')

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # Compute dissipation implicitely
        sn = self.feec_vars[0]
        bn = self.feec_vars[1]
        if self._eta < 1.e-15 and self._eta_a < 1.e-15:
            self.feec_vars_update(sn, bn)
            return

        if self._info:
            print()
            print("Computing the dissipation in VariationalResistivity")

        # Update weighted mass matrix for artificial resistivity
        cb = self.Tcurl.dot(bn, out=self._tmp_cb1)
        self.cbf1.vector = cb
        cb_v = self.cbf1.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_curl, out=self._cb1_values)

        cb_sq_v = self._cb_sq_values_init
        cb_sq_v *= 0.
        for i in range(3):
            for j in range(3):
                cb_sq_v += cb_v[i]*self._sq_term_metric_no_jac[i, j]*cb_v[j]

        np.sqrt(cb_sq_v, out=cb_sq_v)

        cb_sq_v *= dt*self._eta_a

        self.M1_cb.assemble([[cb_sq_v*self._sq_term_metric[0, 0], cb_sq_v*self._sq_term_metric[0, 1], cb_sq_v*self._sq_term_metric[0, 2]],
                             [cb_sq_v*self._sq_term_metric[1, 0], cb_sq_v *
                                 self._sq_term_metric[1, 1], cb_sq_v*self._sq_term_metric[1, 2]],
                             [cb_sq_v*self._sq_term_metric[2, 0], cb_sq_v*self._sq_term_metric[2, 1], cb_sq_v*self._sq_term_metric[2, 2]]],
                            verbose=False)

        cb_sq_v += dt*self._eta

        self._scaled_stiffness._scalar = dt*self._eta
        # self.evol_op._multiplicants[1]._addends[0]._scalar = -dt*self._eta/2.
        if self._linearize_current:
            bn1 = self.evol_op.dot(bn + dt*self._eta*self.derham.curl.dot(
                self.Tcurl.dot(self.projected_mhd_equil.b2)), out=self._tmp_bn1)
        else:
            bn1 = self.evol_op.dot(bn, out=self._tmp_bn1)
        if self._info:
            print("information on the linear solver : ", self.inv_lop._info)

        # Energy balance term
        # 1) Pointwize energy change
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 /= 2.
        if self._linearize_current:
            cb1 = self.Tcurl.dot(
                bn1-self.projected_mhd_equil.b2, out=self._tmp_cb1)
        else:
            cb1 = self.Tcurl.dot(bn1, out=self._tmp_cb1)

        cb12 = self.Tcurl.dot(bn12, out=self._tmp_cb12)

        self.cbf12.vector = cb12
        self.cbf1.vector = cb1

        cb12_v = self.cbf12.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_curl, out=self._cb12_values)
        cb1_v = self.cbf1.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_curl, out=self._cb1_values)

        cb_sq_v = self._cb_sq_values
        cb_sq_v *= 0.
        for i in range(3):
            for j in range(3):
                cb_sq_v += cb12_v[i]*self._sq_term_metric[i, j]*cb1_v[j]

        cb_sq_v *= self._cb_sq_values_init
        # 2) Initial energy and linear form
        rho = self._rho
        self.rhof.vector = rho
        self.sf.vector = sn

        sf_values = self.sf.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_bd, out=self._sf_values)

        rhof_values = self.rhof.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_bd, out=self._rhof_values)

        e_rho_s = self.__ener(rhof_values, sf_values,
                              out=self._e_rho_s_values)

        e_rho_s *= self._energy_metric

        cb_sq_v += e_rho_s

        self._get_L2dofs_V3(cb_sq_v, dofs=self._linear_form_tot_e)

        # 3) Newton iteration
        sn1 = sn.copy(out=self._tmp_sn1)

        tol = self._nonlin_solver['tol']
        err = tol+1

        for it in range(self._nonlin_solver['maxiter']):

            self.sf1.vector = sn1

            sf1_values = self.sf1.eval_tp_fixed_loc(
                self.integration_grid_spans, self.integration_grid_bd, out=self._sf1_values)

            e_rho_s1 = self.__ener(
                rhof_values, sf1_values, out=self._e_rho_s1_values)

            e_rho_s1 *= self._energy_metric

            self._get_L2dofs_V3(e_rho_s1, dofs=self._linear_form_e_sn1)

            self.tot_rhs *= 0.
            self.tot_rhs -= self._linear_form_e_sn1
            self.tot_rhs += self._linear_form_tot_e

            err = self._get_error_newton(self.tot_rhs)

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or np.isnan(err):
                break

            deds = self.__dener_ds(
                rhof_values, sf1_values, out=self._de_rho_s1_values)
            deds *= self._mass_metric_term

            self.M_de_ds.assemble([[deds]], verbose=False)
            self.pc_jac.update_mass_operator(self.M_de_ds)

            incr = self.inv_jac.dot(self.tot_rhs, out=self._tmp_sn_incr)

            if self._info:
                print("information on the linear solver : ", self.inv_jac._info)

            sn1 += incr

        if it == self._nonlin_solver['maxiter']-1 or np.isnan(err):
            print(
                f'!!!Warning: Maximum iteration in VariationalResistivity reached - not converged:\n {err = } \n {tol**2 = }')

        self.feec_vars_update(sn1, bn1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        from struphy.feec.projectors import L2Projector

        pc_M1 = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.M1)
        inv_M1 = inverse(self.mass_ops.M1,
                         'pcg',
                         pc=pc_M1,
                         tol=1e-16,
                         maxiter=1000,
                         verbose=False)

        pc_M3 = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.M3)
        self._inv_M3 = inverse(self.mass_ops.M3,
                               'pcg',
                               pc=pc_M3,
                               tol=1e-16,
                               maxiter=1000,
                               verbose=False)

        M2 = self.mass_ops.M2

        self.M_de_ds = WeightedMassOperator(
            self.derham.Vh_fem['3'],
            self.derham.Vh_fem['3'],
            V_extraction_op=self.derham.extraction_ops['3'],
            W_extraction_op=self.derham.extraction_ops['3'])

        g = self.mass_ops.sqrt_g

        self.M1_cb = WeightedMassOperator(
            self.derham.Vh_fem['1'],
            self.derham.Vh_fem['1'],
            V_extraction_op=self.derham.extraction_ops['1'],
            W_extraction_op=self.derham.extraction_ops['1'],
            weights_info=[[g, None, None],
                          [None, g, None],
                          [None, None, g]])

        if self._lin_solver['type'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._lin_solver['type'][1])
            self.pc_jac = pc_class(self.M_de_ds)

        self.inv_jac = inverse(self.M_de_ds,
                               'pcg',
                               pc=self.pc_jac,
                               tol=self._lin_solver['tol'],
                               maxiter=self._lin_solver['maxiter'],
                               verbose=False,
                               recycle=True)

        curl = self.derham.curl
        self.Tcurl = inv_M1@curl.T@M2

        self.phy_stiffness = M2@curl@inv_M1@curl.T@M2
        self.phy_cb_stiffness = self.Tcurl.T@self.M1_cb@self.Tcurl

        self._scaled_stiffness = .00001 * self.phy_stiffness

        self.r_op = M2  # - self._scaled_stiffness
        self.l_op = M2 + self._scaled_stiffness + self.phy_cb_stiffness

        if self._lin_solver['type'][1] is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner, self._lin_solver['type'][1])
            self.pc = pc_class(M2)

        self.inv_lop = inverse(self.l_op,
                               'pcg',
                               pc=self.pc,
                               tol=self._lin_solver['tol'],
                               maxiter=self._lin_solver['maxiter'],
                               verbose=False,
                               recycle=True)

        self.evol_op = self.inv_lop@self.r_op
        # self.evol_op = IdentityOperator(self.derham.Vh_pol['v'])
        integration_grid = [grid_1d.flatten()
                            for grid_1d in self.derham.quad_grid_pts['3']]
        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = self.derham.prepare_eval_tp_fixed(
            integration_grid)

        self.integration_grid_curl = [[self.integration_grid_bd[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
                                      [self.integration_grid_bn[0], self.integration_grid_bd[1],
                                       self.integration_grid_bn[2]],
                                      [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bd[2]]]

        grid_shape = tuple([len(loc_grid)
                            for loc_grid in integration_grid])

        self._cb12_values = [np.zeros(grid_shape, dtype=float)
                             for i in range(3)]
        self._cb1_values = [np.zeros(grid_shape, dtype=float)for i in range(3)]

        self._cb_sq_values = np.zeros(grid_shape, dtype=float)
        self._cb_sq_values_init = np.zeros(grid_shape, dtype=float)

        self._sf_values = np.zeros(grid_shape, dtype=float)
        self._sf1_values = np.zeros(grid_shape, dtype=float)
        self._rhof_values = np.zeros(grid_shape, dtype=float)

        self._e_rho_s1_values = np.zeros(grid_shape, dtype=float)
        self._e_rho_s_values = np.zeros(grid_shape, dtype=float)

        self._de_rho_s1_values = np.zeros(grid_shape, dtype=float)

        self._tmp_int_grid = np.zeros(grid_shape, dtype=float)

        gam = self._gamma

        metric = np.power(self.domain.jacobian_det(
            *integration_grid), -gam)
        self._mass_metric_term = deepcopy(metric)

        metric = np.power(self.domain.jacobian_det(
            *integration_grid), 1-gam)
        self._energy_metric = deepcopy(metric)

        metric = self.domain.metric_inv(
            *integration_grid)*self.domain.jacobian_det(*integration_grid)
        self._sq_term_metric = deepcopy(metric)

        metric = self.domain.metric_inv(
            *integration_grid)
        self._sq_term_metric_no_jac = deepcopy(metric)

        self._get_L2dofs_V3 = L2Projector('L2', self.mass_ops).get_dofs

    def __ener(self, rho, s, out=None):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = np.power(rho, gam)*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def __dener_ds(self, rho, s, out=None):
        """Derivative with respect to s of the thermodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        dE(rho, s)/ds = (rho^{gamma-1})*exp(s/rho)"""
        gam = self._gamma
        if out is None:
            out = np.power(rho, gam-1)*np.exp(s/rho)
        else:
            out *= 0.
            out += s
            out /= rho
            np.exp(out, out=out)
            np.power(rho, gam-1, out=self._tmp_int_grid)
            out *= self._tmp_int_grid
        return out

    def _get_error_newton(self, sn_diff):
        weak_sn_diff = self._inv_M3.dot(sn_diff, out=self._tmp_sn_weak_diff)
        err_s = weak_sn_diff.dot(sn_diff)
        return err_s


class TimeDependentSource(Propagator):
    r'''Propagates a source term :math:`S(t) \in V_h^n` of the form

    .. math::

        S(t) = \sum_{ijk} c_{ijk} \Lambda^n_{ijk} * h(\omega t)\,,

    where :math:`h(\omega t)` is one of the functions in Notes.

    Notes
    -----

    * :math:`h(\omega t) = \cos(\omega t)` (default)
    * :math:`h(\omega t) = \sin(\omega t)` 
    '''

    @staticmethod
    def options(default=False):
        dct = {}
        dct['omega'] = 1.
        dct['hfun'] = ['cos', 'sin']
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(self,
                 c: StencilVector,
                 *,
                 omega: float = options()['omega'],
                 hfun: str = options(default=True)['hfun']):

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


class AdiabaticPhi(Propagator):
    r"""
    Electrostatic potential for adiabatic electrons, computed from

    .. math::

        n_e = n_{e0}\,\exp \left( \frac{e \phi}{k_B T_e} \right) \approx n_{e0} \left( 1 + \frac{e \phi}{k_B T_{e0}} \right)\,,

    where :math:`n_{e0}` and :math:`T_{e0}` denote electron equilibrium density and temperature, respectively.

    This is solved in weak form: find :math:`\phi \in H^1` such that

    .. math::

        \int_\Omega \psi\, \frac{n_{e0}(\mathbf x)}{T_{e0}(\mathbf x)}\phi \,\textrm d \mathbf x  = \int_\Omega \psi\, (n_{e}(\mathbf x) - n_{e0}(\mathbf x))\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,.

    The equation is discretized as

    .. math::

        \sigma_1 \mathbb M^0_{n/T} \boldsymbol \phi = (\Lambda^0, n_{e} - n_{e0} )_{L^2}\,,

    where :math:`M^0_{n/T}` is a :class:`~struphy.feec.mass.WeightedMassOperator` and :math:`\sigma_1`
    is a normalization parameter. 

    Parameters
    ----------
    phi : StencilVector
        FE coefficients of the solution as a discrete 0-form.

    A_mat : WeightedMassOperator
        The matrix to invert.

    rho : StencilVector or tuple
        Right-hand side FE coefficients of a 0-form (optional, can be set with a setter later).
        Can be either a) StencilVector or b) 2-tuple.
        In case b) the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.

    sigma_1 : float
        Normalization parameter.

    x0 : StencilVector
        Initial guess for the iterative solver (optional, can be set with a setter later).

    **params : dict
        Parameters for the iterative solver (see ``__init__`` for details).
    """

    def __init__(self,
                 phi: StencilVector,
                 *,
                 A_mat: WeightedMassOperator = 'M0',
                 rho: StencilVector | tuple = None,
                 sigma_1: float = 1.,
                 x0: StencilVector = None,
                 **params):

        assert phi.space == self.derham.Vh['0']

        super().__init__(phi)

        # solver parameters
        params_default = {'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'recycle': False}

        params = set_defaults(params, params_default)

        # collect rhs
        if rho is None:
            rho = phi.space.zeros()
        else:
            if isinstance(rho, tuple):
                assert isinstance(rho[0], AccumulatorVector)
                assert isinstance(rho[1], Particles)
                # assert rho[0].space_id == 'H1'
            else:
                assert rho.space == phi.space
        self._rho = rho

        # initial guess and solver params
        self._x0 = x0
        self._params = params
        A_mat = getattr(self.mass_ops, A_mat)

        # Set lhs matrices
        self._A = sigma_1 * A_mat

        # preconditioner and solver for Ax=b
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(A_mat)

        # solver just with A_2, but will be set during call with dt
        self._solver = inverse(self._A,
                               params['type'][0],
                               pc=pc,
                               x0=self.x0,
                               tol=params['tol'],
                               maxiter=params['maxiter'],
                               verbose=params['verbose'],
                               recycle=params['recycle'])

        # allocate memory for solution
        self._tmp = phi.space.zeros()
        self._tmp2 = phi.space.zeros()
        self._rhs = phi.space.zeros()
        self._rhs2 = phi.space.zeros()

    @property
    def rho(self):
        """
        Right-hand side FE coefficients of a 0-form.
        Can be either a) StencilVector or b) 2-tuple.
        In the latter case, the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.
        """
        return self._rho

    @rho.setter
    def rho(self, value):
        """ In-place setter for StencilVector/PolarVector.
        """
        if isinstance(value, tuple):
            assert isinstance(value[0], AccumulatorVector)
            assert isinstance(value[1], Particles)
            self._rho = value
        else:
            assert value.space == self.derham.Vh['0']
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
        assert value.space == self.derham.Vh['0']
        assert value.space.symbolic_space == 'H1', f'Right-hand side must be in H1, but is in {value.space.symbolic_space}.'

        if self._x0 is None:
            self._x0 = value
        else:
            self._x0[:] = value[:]

    def __call__(self, dt):

        self._rhs *= 0.
        if isinstance(self._rho, tuple):
            self._rho[0]()  # accumulate
            self._rhs += self._rho[0].vectors[0]
        else:
            self._rhs += self._rho

        # solve
        out = self._solver.solve(self._rhs, out=self._tmp)
        info = self._solver._info

        if self._lin_solver['info']:
            print(info)

        dphi = self.feec_vars_update(out)

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False,
                         'recycle': True}
        return dct
