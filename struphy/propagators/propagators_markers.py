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


class StepStaticEfield( Propagator ):
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

        mass_ops : struphy.psydac_api.mass.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass.

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


class StepStaticBfield( Propagator ):
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

        mass_ops : struphy.psydac_api.mass.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass.

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
    

class StepPushEtaFullPC( Propagator ):
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
            
        bc : list[str]
            Kinetic boundary conditions in each direction.
    '''
    def __init__(self, u, particles, derham, domain, u_space, bc):

        assert isinstance(u, BlockVector)

        self._derham = derham
        self._domain = domain
        self._u = u
        self._particles = particles
        self._u_space = u_space
        self._bc = bc

        # call Pusher class
        if self._u_space == 'Hcurl':
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_eta_rk4_Hcurl_full', n_stages=4)

        elif self._u_space == 'Hdiv':
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_eta_rk4_Hdiv_full', n_stages=4)

        else:
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_eta_rk4_H1vec_full', n_stages=4)

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
                     bc=self._bc, mpi_sort='each')


class StepPushEtaPC( Propagator ):
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
            
        bc : list[str]
            Kinetic boundary conditions in each direction.
    '''
    def __init__(self, u, particles, derham, domain, u_space, bc):

        assert isinstance(u, BlockVector)

        self._derham = derham
        self._domain = domain
        self._u = u
        self._particles = particles
        self._u_space = u_space
        self._bc = bc

        # call Pusher class
        if self._u_space == 'Hcurl':
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_eta_rk4_Hcurl', n_stages=4)

        elif self._u_space == 'Hdiv':
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_eta_rk4_Hdiv', n_stages=4)

        else:
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_eta_rk4_H1vec', n_stages=4)

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
                     self._u[0]._data, self._u[1]._data,
                     bc=self._bc, mpi_sort='each')


class StepPushVxB( Propagator ):
    r"""Solves

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector. Available algorithms: 
        * analytic
        * implicit
    
    Parameters
    ----------
        particles : struphy.pic.particles.Particles6D
            Holdes the markers to push.
            
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        algo : str
            The used algorithm.
            
        b : psydac.linalg.block.BlockVector
            FE coefficients of a dynamical magnetic field (2-form).
            
        b_static : psydac.linalg.block.BlockVector (optional)
            FE coefficients of a static (background) magnetic field (2-form).
    """
    
    def __init__(self, particles, derham, algo, b, b_static=None):
        
        self._particles = particles
        
        # load pusher
        from struphy.pic.pusher import Pusher
        
        kernel_name = 'push_vxb_' + algo
        
        self._pusher = Pusher(derham, particles.domain, kernel_name)
        
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
        

class StepPushEta( Propagator ):
    r"""Solves  

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant. Available algorithms: 
        * forward_euler (1st order)
        * heun2 (2nd order)
        * rk2 (2nd order)
        * heun3 (3rd order)
        * rk4 (4th order)
    
    Parameters
    ----------
        particles : struphy.pic.particles.Particles6D
            Holdes the markers to push.
            
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        algo : str
            The used algorithm.
            
        bc : list[str]
            Kinetic boundary conditions in each direction.
    """
    
    def __init__(self, particles, derham, algo, bc):
        
        self._particles = particles
        self._bc = bc
        
        # load butcher tableau and pusher
        from struphy.pic.pusher import Pusher
        from struphy.pic.pusher import ButcherTableau
        
        if algo == 'forward_euler':
            a = []
            b = [1.]
            c = [0.]
        elif algo == 'heun2':
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif algo == 'rk2':
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif algo == 'heun3':
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif algo == 'rk4':
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        else:
            raise NotImplementedError('Chosen algorithm is not implemented.')
        
        self._butcher = ButcherTableau(a, b, c)
        self._pusher = Pusher(derham, particles.domain, 'push_eta_stage', self._butcher.n_stages)
        
    @property
    def variables(self):
        return self._particles

    def __call__(self, dt):
        self._pusher(self._particles, dt, 
                     self._butcher.a, self._butcher.b, self._butcher.c, 
                     bc=self._bc, mpi_sort='last')