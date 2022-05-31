from http.client import PRECONDITION_FAILED
from propagators import Propagator

from struphy.psydac_linear_operators.preconditioner import MassMatrixPreConditioner as MassPre
from struphy.linear_algebra.schur_solver            import Schur_solver
from struphy.pic.lin_Vlasov_Maxwell                 import pusher_weights
from struphy.pic.lin_Vlasov_Maxwell                 import accumulation

import numpy as np


class StepEWLinVlasovMaxwell( Propagator ):
    """
    Split step of the linearized Vlasov Maxwell system where E-field and weights are updated using psydac functions
    
    Parameters
    ----------
        particles : array
            contains particles, positions [0:3], velocities [3:6], and weights [6]

        Np_loc : double
            number of particles

        dts : list
            Time steps, one for each split step.
        
        DOMAIN : obj
            Domain object from geometry/domain_3d.

        SPACES : obj
            FEEC self.SPACES.

        Np_loc : int
            Number of particles per rank.

        TODO
    """

    def __init__(self, particles, efield, Np_loc, DERHAM, params_f_0, solver_params):

        self._particles     = particles
        self._e             = efield
        self._DERHAM        = DERHAM
        self._Np_loc        = Np_loc
        self._ACCUM         = accumulation.Accumulation()
        self._MPI_COMM      = DERHAM.comm
        self._params_f_0    = params_f_0
        self._solver_params = solver_params

        precon = solver_params['pc']

        self._v_shift = np.array([params_f_0['v0_x'],
                                  params_f_0['v0_y'],
                                  params_f_0['v0_z']])

        self._v_th    = np.array([params_f_0['vth_x'],
                                  params_f_0['vth_y'],
                                  params_f_0['vth_z']])

        self._n0      =           params_f_0['nh0']

        # Preconditioner
        if precon == None:
            self._precon = None
        elif precon == 'fft':
            self._precon = MassPre(self._DERHAM.V1)
        else:
            raise ValueError(f'Preconditioner "{precon}" not implemented.')

        # accumulate and assemble accumulation matrix and vector
        self._ACCUM.accumulate_e_W_step(self._particles, self._MPI_COMM, self._Np_loc, self.v_shift, self.v_th, self.n0)
        self._Accum_mat, self._Accum_vec = self._ACCUM.assemble_step_e_W(self._Np_loc)

        # Define block matrix (without time step size dt in the diangonals)
        _A  = self._DERHAM.M1
        _BC = self._Accum_mat

        # Instantiate Schur solver
        self._schur_solver = Schur_solver(_A, _BC, pc=self._precon, tol=self._params['tol'], maxiter=self._params['maxiter'], verbose=self._params['verbose'])
    
    @property
    def variables(self):
        return [self._e, self._particles]
    
    @property
    def v_shift( self ):
        """Shift in the Maxwellian velocity distribution"""
        return self._v_shift
    
    @property
    def v_th( self ):
        """Thermal velocity in the Maxwellian velocity distribution"""
        return self._v_th
    
    @property
    def n0( self ):
        """Spatial distribution factor (const) in the Maxwellian velocity distribution"""
        return self._n0


    def push(self, dt, print_info=False):
        """
        E_W substep (subsystem 3) in the linearized Vlasov-Maxwell system

        Parameters:
        -----------
            particles : np.array
                Shape (7, Np), where the rows hold the positions [:3] and velocities [3:6] and weights [6]

            efield : np.array
                Shape (Nel[0]*Nel[1]*Nel[2]*3, ), contains efield coefficients

            print_info : boolean
                Print the maximal absolute difference between input and output
        """

        # store initial values
        old_e         = self._e.copy()
        old_particles = self._particles.copy()

        # accumulate and assemble accumulation matrix and vector
        self._ACCUM.accumulate_e_W_step(self._particles, self._MPI_COMM, self._Np_loc, self.v_shift, self.v_th, self.n0)
        self._Accum_mat, self._Accum_vec = self._ACCUM.assemble_step_e_W(self._Np_loc)

        # Define block matrix (without time step size dt in the diangonals)
        _A  = self._DERHAM.M1
        _BC = self._Accum_mat

        efield_new, = self._schur_solver(self._e, self._Accum_vec, dt)
        
        # In-place update of the electric field coefficients
        self.in_place_update(self._e, efield_new)
        
        pusher_weights.push_weights(self._particles,
                                    self._Np_loc,
                                    self._DERHAM.V0.degree,
                                    self._DERHAM.V0.knots[0], self._DERHAM.V0.knots[1], self._DERHAM.V0.knots[2],
                                    self._DERHAM.indN[0], self._DERHAM.indN[1], self._DERHAM.indN[2],
                                    self._DERHAM.indD[0], self._DERHAM.indD[1], self._DERHAM.indD[2],
                                    self._DERHAM.NbaseN, self._DERHAM.NbaseD,
                                    self._e + old_e,
                                    self.__dts[0],
                                    self.v_shift, self.v_th, self.n0
                                    )
        
        # print info
        if print_info:
            print('Maxdiff    e1        for step_e_W:', np.max(np.abs(self._e - old_e)))
            print('Maxdiff    weights   for step_e_W:', np.max(np.abs(self._particles[6,:] - old_particles[6,:])))
            print()

        # Delete Schur solver since it has to be 
        del(self._schur_solver)



class StepXVLinVlasovMaxwell( Propagator ):
    """
    
    """
