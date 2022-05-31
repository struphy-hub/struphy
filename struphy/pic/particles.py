import numpy as np
import h5py
import scipy.special as sp

from struphy.kinetic_equil.analytical import gaussian, moments
from struphy.pic import sampling, sobol_seq
from struphy.kinetic_init.kinetic_init import KineticInit6D


class Particles6D:
    """
    A class for initializing particles in models that use the full 6D phase space.

    Parameters:
    -----------
        name : str
            name of the particle species

        DOMAIN: Domain obj
            From struphy/geometry/domain_3d.Domain.

        params_markers : dict
            parameters under key-word markers in the parameters file

        mpi_comm : MPI
            MPI communicator
    """

    def __init__(self, name, DOMAIN, params_markers, mpi_comm):

        self._name = name
        self._DOMAIN = DOMAIN
        self._Np = params_markers['Np']
        self.mpi_comm = mpi_comm

        self._Np_loc = int(self._Np / mpi_comm.Get_size())
        self._params = params_markers

        mpi_size = self.mpi_comm.Get_size()
        mpi_rank = self.mpi_comm.Get_rank()

        # initialize particle arrays
        # positions, velocities, weights, and process number
        self._particles_loc = np.empty((8, self._Np_loc), dtype=float)
        # initial sampling density: hat_s_ini(eta_0, v_0) for each process
        self.s0_loc = np.empty(self._Np_loc, dtype=float)
        self.f0_loc = np.empty(self._Np_loc, dtype=float)

        if mpi_rank == 0:
            self.particles_recv = np.empty((8, self._Np_loc), dtype=float)
            self.w0_recv = np.empty(self._Np_loc, dtype=float)
            self.s0_recv = np.empty(self._Np_loc, dtype=float)
        else:
            self.particles_recv = None
            self.w0_recv = None
            self.s0_recv = None

        # Step 1: create numbers in (0, 1)
        # pseudo-random
        if self._params['loading'] == 'pseudo_random':

            np.random.seed(self._params['seed'])

            for i in range(mpi_size):
                temp = np.random.rand(self._Np_loc, 6)

                if i == mpi_rank:
                    self._particles_loc[:6] = temp.T
                    break

            del temp
        # plain sobol numbers (skip first 1000 numbers)
        elif self._params['loading'] == 'sobol_standard':
            self._particles_loc[:6] = sobol_seq.i4_sobol_generate(
                6, self._Np_loc, 1000 + self._Np_loc*mpi_rank).T
        # symmetric sobol numbers in all 6 dimensions (skip first 1000 numbers)
        elif self._params['loading'] == 'sobol_antithetic':
            sampling.set_particles_symmetric(sobol_seq.i4_sobol_generate(6,
                                             self._Np_loc//64, 1000 + self._Np_loc//64*mpi_rank), self._particles_loc, self._Np_loc)
        # load numbers from an external files
        elif self._params['loading'] == 'external':

            if mpi_rank == 0:
                file = h5py.File(self._params['dir_particles'], 'r')

                self._particles_loc[:,
                                    :] = file['particles'][0, :6, :self._Np_loc]

                for i in range(1, mpi_size):
                    self.mpi_comm.Send(
                        file['particles'][0, :6, i*self._Np_loc:(i + 1)*self._Np_loc], dest=i, tag=11)
            else:
                self.mpi_comm.Recv(self._particles_loc, source=0, tag=11)
        else:
            raise ValueError(
                'Specified particle loading method does not exist!')

        # Step 2: inversion of Gaussian in velocity space (sh3_ini from above)
        if self._params['loading'] != 'external':
            self._particles_loc[3] = sp.erfinv(
                2*self._particles_loc[3] - 1)*self._params['loading']['vth_x'] + self._params['loading']['v0_x']
            self._particles_loc[4] = sp.erfinv(
                2*self._particles_loc[4] - 1)*self._params['loading']['vth_y'] + self._params['loading']['v0_y']
            self._particles_loc[5] = sp.erfinv(
                2*self._particles_loc[5] - 1)*self._params['loading']['vth_z'] + self._params['loading']['v0_z']

        # check if all particle positions are in [0, 1]^3
        if np.any(self._particles_loc[:3] >= 1.0) or np.any(self._particles_loc[:3] <= 0.0):
            raise ValueError(
                'There are particles outside of the logical domain - aborting...')

        # Step 3: initial sampling density
        self.s0_loc[:] = self.sh0_ini(self._particles_loc[0], self._particles_loc[1], self._particles_loc[2],
                                      self._particles_loc[3], self._particles_loc[4], self._particles_loc[5])

    @property
    def name(self):
        '''Name of the kinetic species in DATA container.'''
        return self._name

    @property
    def Np(self):
        '''Total number of particles.'''
        return self._Np

    @property
    def Np_loc(self):
        '''Number of particles on each process.'''
        return self._Np_loc

    @property
    def particles_loc(self):
        """Numpy array holding the particle information: positions, velocities, weights, MPI process"""
        return self._particles_loc

    @particles_loc.setter
    def particles_loc(self, particle_array):
        self._particles_loc = particle_array

    def set_initial_conditions(self, init, params):
        """
        Sets the initial conditions for the weights in self.particles_loc.

        Parameters
        ----------
            Kinetic_EQ : obj

        """

        assert hasattr(
            self, 'EQ_Kinetic'), 'Kinetic equilibrium has not been set yet!'

        self.w0_loc = np.empty(self._Np_loc, dtype=float)

        KINETIC_INIT = KineticInit6D(self._DOMAIN, init, params)

        f0_loc = np.empty(self._Np_loc, dtype=float)
        f0_loc[:] = KINETIC_INIT.f0(self.particles_loc[0], self.particles_loc[1], self.particles_loc[2],
                                    self.particles_loc[3], self.particles_loc[4], self.particles_loc[5])
        self.w0_loc[:] = f0_loc / self.s0_loc

    def set_kinetic_equil(self, eq_kinetic):
        """
        Sets the kinetic equilibrium needed by set_initial_conditions
        """
        self._EQ_Kinetic = eq_kinetic

    @property
    def EQ_Kinetic(self):
        """Kinetic equilibrium"""
        return self._EQ_Kinetic

    # density for modes_k
    def nh_tot(self, x, y, z):
        '''Total normalized EP density nh/nh_eq = 1 + sum_i amp_i sin(k_i*x).'''

        value = 1.*x

        if self.init_type == 'modes_k':
            for i in range(len(self.amp)):
                value += self.amp[i]*np.sin(self.modes_k[0][i] *
                                            x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return value

    # sampling distribution s3 (3-form) on logical domain, always a Gaussian (no importance sampling)
    def sh3_ini(self, eta1, eta2, eta3, vx, vy, vz):
        '''Gaussian velocity distribution for sampling markers. 
        Parameters are such that Gaussian is close to kinetic equilibrium.
        MUST be normalized to 1 in the logical domain.'''

        MOMENTS = moments.Kinetic_homogen_slab({'vth_x': self._params['loading']['vth_x'],
                                                'vth_y': self._params['loading']['vth_y'],
                                                'vth_z': self._params['loading']['vth_z'],
                                                'v0_x': self._params['loading']['v0_z'],
                                                'v0_y': self._params['loading']['v0_y'],
                                                'v0_z': self._params['loading']['v0_z'],
                                                'nh0': None,
                                                })
        EQ = gaussian.Gaussian_3d(MOMENTS)

        return EQ.velocity_distribution(eta1, eta2, eta3, vx, vy, vz)

    # sampling distribution s0 (0-form) to compute weights
    def sh0_ini(self, eta1, eta2, eta3, vx, vy, vz):
        '''Sampling distribution as 0-form.'''
        fun = self.sh3_ini(eta1, eta2, eta3, vx, vy, vz)
        return self._DOMAIN.transformation(fun, eta1, eta2, eta3, '3_to_0', flat_eval=True)

    # when using control variate
    def update_weights(self):
        assert hasattr(
            self, 'EQ_KINETIC'), 'Kinetic equilibrium has not been set yet!'

        self.particles_loc[6] = self.w0_loc - 1./self.s0_loc * self.EQ_Kinetic.fh0_eq(self.particles_loc[0],
                                                                                      self.particles_loc[1],
                                                                                      self.particles_loc[2],
                                                                                      self.particles_loc[3],
                                                                                      self.particles_loc[4],
                                                                                      self.particles_loc[5])


class Particles5D:
    """
    A class for intializing particles of a driftkinetic or gyrokinetic model

    Parameters:
    -----------
        name : str
            name of the particle species

        DOMAIN: Domain obj
            From struphy/geometry/domain_3d.Domain.

        params_markers : dict
            parameters under key-word markers in the parameters file

        mpi_comm : MPI
            MPI communicator
    """

    def __init__(self, name, DOMAIN, params_markers, mpi_comm):

        self._name = name
        self._DOMAIN = DOMAIN
        self.Np = params_markers['Np']
        self.mpi_comm = mpi_comm

        self.Np_loc = int(self.Np / mpi_comm.Get_size())
        self._params = params_markers

        mpi_size = self.mpi_comm.Get_size()
        mpi_rank = self.mpi_comm.Get_rank()
