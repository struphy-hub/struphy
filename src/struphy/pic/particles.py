from abc import ABCMeta, abstractmethod

import numpy as np
import h5py
import scipy.special as sp

from struphy.pic import sampling, sobol_seq
from struphy.pic.pusher_utilities import reflect
from struphy.kinetic_background import analytical
from struphy.fields_background.mhd_equil.equils import set_defaults


class Particles(metaclass=ABCMeta):
    """
    Base class for a particle based kinetic species.

    Loading and compute initial particles and save the values at the corresponding column of markers array.
    | 0 | 1 | 2 | 3 | ... | 3+(vdim-1) | 3+vdim | 4+vdim | 5+vdim |
    |    eta    |      velocities      | weight |   s0   |   w0   |

    Parameters
    ----------
    name : str
        Name of particle species.

    **params : dict
        Marker parameters.
    """

    def __init__(self, name: str, **params):

        params_default = {'type': 'full_f',
                          'ppc': None,
                          'Np': 4,
                          'eps': .25,
                          'bc_type': ['periodic', 'periodic', 'periodic'],
                          'loading': {'type': 'pseudo:random', 'seed': 1234, 'dir_particles': None, 'moments': [0., 0., 0., 1., 1., 1.]},
                          'comm': None,
                          'domain': None,
                          'domain_array': None
                          }

        self._params = set_defaults(params, params_default)

        self._name = name
        self._domain = params['domain']
        self._bc =     params['bc_type']
        self._domain_decomp = params['domain_array']

        assert params['comm'] is not None
        self._mpi_comm = params['comm']
        self._mpi_size = params['comm'].Get_size()
        self._mpi_rank = params['comm'].Get_rank()

        # create marker array
        self.create_marker_array()

        # draw markers
        self.draw_markers()

        # number of holes and markers on process
        self._holes = self._markers[:, 0] == -1.
        self._n_holes_loc = np.count_nonzero(self._holes)
        self._n_mks_loc = self._markers.shape[0] - self._n_holes_loc

        # check if all particle positions are inside the unit cube [0, 1]^3
        n_mks_load_loc = self._n_mks_load[self._mpi_rank]

        assert np.all(~self._holes[:n_mks_load_loc]) and np.all(
            self._holes[n_mks_load_loc:])

        # Assume full-f if type is not in parameters
        if 'type' in params.keys():
            if params['type'] == 'control_variate':
                self._use_control_variate = True
            else:
                self._use_control_variate = False
        else:
            self._use_control_variate = False

    @abstractmethod
    def svol(self, eta1, eta2, eta3, *v):
        """ Sampling density function as volume form.
        """
        pass

    @abstractmethod
    def s0(self, eta1, eta2, eta3, *v, remove_holes=True):
        """ Sampling density function as 0 form.
        """
        pass

    @property
    @abstractmethod
    def n_cols(self):
        """Number of the columns at each markers.
        """
        pass

    @property
    @abstractmethod
    def vdim(self):
        """Dimension of the velocity space.
        """
        pass

    @property
    def kinds(self):
        """ Name of the class
        """
        return self.__class__.__name__

    @property
    def name(self):
        """ Name of the kinetic species in DATA container.
        """
        return self._name

    @property
    def params(self):
        """ Parameters for markers.
        """
        return self._params

    @property
    def domain_decomp(self):
        """ Array containing domain decomposition information.
        """
        return self._domain_decomp

    @property
    def comm(self):
        """ MPI communicator.
        """
        return self._mpi_comm

    @property
    def mpi_size(self):
        """ Number of MPI processes.
        """
        return self._mpi_size

    @property
    def mpi_rank(self):
        """ Rank of current process.
        """
        return self._mpi_rank

    @property
    def n_mks(self):
        """ Total number of markers.
        """
        return self._n_mks

    @property
    def n_mks_loc(self):
        """ Number of markers on process (without holes).
        """
        return self._n_mks_loc

    @property
    def n_mks_load(self):
        """ Array of number of markers on each process at loading stage
        """
        return self._n_mks_load

    @property
    def markers(self):
        """ Array holding the marker information, including holes. The i-th row holds the i-th marker info.
        """
        return self._markers

    @property
    def holes(self):
        """ Array of booleans stating if an entry in the markers array is a hole or not. 
        """
        return self._holes

    @property
    def n_holes_loc(self):
        """ Number of holes on process (= marker.shape[0] - n_mks_loc).
        """
        return self._n_holes_loc

    @property
    def markers_wo_holes(self):
        """ Array holding the marker information, excluding holes. The i-th row holds the i-th marker info.
        """
        return self._markers[~self._holes]

    @property
    def domain(self):
        """ struphy.geometry.domains
        """
        return self._domain
    
    @property
    def bc(self):
        """ Kinetic boundary conditions in each direction.
        """
        return self._bc

    def create_marker_array(self):
        """Create marker array. (self.markers)
        """

        # number of cells on current process
        n_cells_loc = np.prod(
            self._domain_decomp[self._mpi_rank, 2::3], dtype=int)

        # total number of cells
        n_cells = np.sum(
            np.prod(self._domain_decomp[:, 2::3], axis=1, dtype=int))

        # number of markers to load on each process (depending on relative domain size)
        if self.params['ppc'] is not None:
            assert isinstance(self.params['ppc'], int)
            ppc = self.params['ppc']
            Np = ppc*n_cells
        else:
            Np = self.params['Np']
            assert isinstance(Np, int)
            ppc = Np/n_cells

        Np = int(Np)
        assert Np >= self._mpi_size

        # array of number of markers on each process at loading stage
        self._n_mks_load = np.zeros(self._mpi_size, dtype=int)
        self._mpi_comm.Allgather(np.array([int(ppc*n_cells_loc)]), self._n_mks_load)

        # add deviation from Np to rank 0
        self._n_mks_load[0] += Np - np.sum(self._n_mks_load)

        # check if all markers are there
        assert np.sum(self._n_mks_load) == Np
        self._n_mks = Np

        # number of markers on the local process at loading stage
        n_mks_load_loc = self._n_mks_load[self._mpi_rank]

        # create markers array (3 x positions, vdim x velocities, weight, s0, w0, ..., ID) with eps send/receive buffer
        markers_size = round(n_mks_load_loc*(1 + 1/np.sqrt(n_mks_load_loc) + self.params['eps']))
        self._markers = np.zeros((markers_size, self.n_cols), dtype=float)

    def draw_markers(self):
        """ Draw markers. 
        """

        # number of markers on the local process at loading stage
        n_mks_load_loc = self.n_mks_load[self._mpi_rank]

        # cumulative sum of number of markers on each process at loading stage. 
        n_mks_load_cum_sum = np.cumsum(self.n_mks_load)

        # load markers from external .hdf5 file
        if self._params['loading']['type'] == 'external':

            if self._mpi_rank == 0:
                file = h5py.File(self._params['loading']['dir_markers'], 'r')

                self._markers[:n_mks_load_cum_sum[0], :
                              ] = file['markers'][:n_mks_load_cum_sum[0], :]

                for i in range(1, self._mpi_size):
                    self._mpi_comm.Send(
                        file['markers'][n_mks_load_cum_sum[i - 1]:n_mks_load_cum_sum[i], :], dest=i, tag=123)

                file.close()
            else:
                recvbuf = np.zeros(
                    (n_mks_load_loc, self._markers.shape[1]), dtype=float)
                self._mpi_comm.Recv(recvbuf, source=0, tag=123)
                self._markers[:n_mks_load_loc, :] = recvbuf

        # load fresh markers
        else:

            # 1. standard random number generator (pseudo-random)
            if self._params['loading']['type'] == 'pseudo_random':

                np.random.seed(self._params['loading']['seed'])

                for i in range(self._mpi_size):
                    temp = np.random.rand(self.n_mks_load[i], 3+self.vdim)

                    if i == self._mpi_rank:
                        self._markers[:n_mks_load_loc, :3+self.vdim] = temp
                        break

                del temp

            # 2. plain sobol numbers with skip of first 1000 numbers
            elif self._params['loading']['type'] == 'sobol_standard':

                self._markers[:n_mks_load_loc, :6] = sobol_seq.i4_sobol_generate(
                    3+self.vdim, n_mks_load_loc, 1000 + (n_mks_load_cum_sum - self.n_mks_load)[self._mpi_rank])

            # 3. symmetric sobol numbers in all 6 dimensions with skip of first 1000 numbers
            elif self._params['loading']['type'] == 'sobol_antithetic':

                temp_markers = sobol_seq.i4_sobol_generate(
                    3+self.vdim, n_mks_load_loc//64, 1000 + (n_mks_load_cum_sum - self.n_mks_load)[self._mpi_rank]//64)

                sampling.set_particles_symmetric_3d_3v(
                    temp_markers, self._markers)

            # 4. Wrong specification
            else:
                raise ValueError(
                    'Specified particle loading method does not exist!')

            # inversion of Gaussian in velocity space
            for i in range(self.vdim):
                self._markers[:n_mks_load_loc, 3 + i] = sp.erfinv(
                    2*self._markers[:n_mks_load_loc, 3 + i] - 1)*self._params['loading']['moments'][self.vdim + i] + self._params['loading']['moments'][i]

        # fill holes in markers array with -1
        self._markers[n_mks_load_loc:] = -1.

        # set markers ID in last column
        self._markers[:n_mks_load_loc, -1] = (n_mks_load_cum_sum - self.n_mks_load)[
            self._mpi_rank] + np.arange(n_mks_load_loc, dtype=float)

        # set specific initial condition for some particles
        if 'initial' in self.params['loading']:
            specific_markers = self.params['loading']['initial']

            counter = 0
            for i in range(len(specific_markers)):
                if i == int(self._markers[counter, -1]):

                    for j in range(3+self.vdim):
                        if specific_markers[i][j] is not None:
                            self._markers[counter, j] = specific_markers[i][j]

                    counter += 1

    def mpi_sort_markers(self, do_test=False):
        """ 
        Sorts markers according to domain decomposition.

        Parameters
        ----------
        do_test : bool
            Check if all markers are on the right process after sorting.
        """

        self.comm.Barrier()

        # before sorting, apply kinetic bc
        self.apply_kinetic_bc()

        # create new markers_to_be_sent array and make corresponding holes in markers array
        markers_to_be_sent, hole_inds_after_send = sendrecv_determine_mtbs(
            self._markers, self._holes, self.domain_decomp, self.mpi_rank)

        # determine where to send markers_to_be_sent
        send_info, send_list = sendrecv_get_destinations(
            markers_to_be_sent, self.domain_decomp, self.mpi_size)

        # transpose send_info
        recv_info = sendrecv_all_to_all(send_info, self.comm)

        # send and receive markers
        sendrecv_markers(send_list, recv_info, hole_inds_after_send,
                         self._markers, self.comm)

        # new holes and new number of holes and markers on process
        self._holes = self._markers[:, 0] == -1.
        self._n_holes_loc = np.count_nonzero(self._holes)
        self._n_mks_loc = self._markers.shape[0] - self._n_holes_loc

        # check if all markers are on the right process after sorting
        if do_test:
            all_on_right_proc = np.all(np.logical_and(
                self.markers[~self._holes,
                             :3] > self.domain_decomp[self.mpi_rank, 0::3],
                self.markers[~self._holes, :3] < self.domain_decomp[self.mpi_rank, 1::3]))

            assert all_on_right_proc

        self.comm.Barrier()

    def initialize_weights(self, fun_params, bckgr_params=None):
        """
        Computes w0 = f0(t=0, eta(t=0), v(t=0)) / s0(t=0, eta(t=0), v(t=0)) from the initial
        distribution function and sets the corresponding columns for w0, s0 and weights in markers array.
        For the control variate method, the background is subtracted.

        Parameters
        ----------
        fun_params : dict
            Dictionary of the form {type : class_name, class_name : params_dict} defining the initial condition.

        bckgr_params : dict (optional)
            Dictionary of the form {type : class_name, class_name : params_dict} defining the background.
        """
        if self._use_control_variate:
            assert bckgr_params is not None, 'When control variate is used, background parameters must be given!'

        # compute s0 and save at vdim+4
        self._markers[~self._holes, self.vdim+4] = self.s0(*self.markers_wo_holes[:, :self.vdim+3].T)

        # load distribution function (with given parameters or default parameters)
        fun_name = fun_params['type']

        if fun_name in fun_params:
            f_init = getattr(analytical, fun_name)(**fun_params[fun_name])
        else:
            f_init = getattr(analytical, fun_name)()

        # compute w0 and save at vdim+5
        self._markers[~self._holes, self.vdim+5] = f_init(
            *self.markers_wo_holes[:, :self.vdim+3].T)/self.markers_wo_holes[:, self.vdim+4]

        # compute weights and save at vdim+3
        if self._use_control_variate:
            fun_name = bckgr_params['type']

            if fun_name in bckgr_params:
                f_bckgr = getattr(analytical, fun_name)(
                    **bckgr_params[fun_name])
            else:
                f_bckgr = getattr(analytical, fun_name)()

            self._markers[~self._holes, self.vdim+3] = self.markers_wo_holes[:, self.vdim+5] - \
                f_bckgr(*self.markers_wo_holes[:, :self.vdim+3].T) / \
                self.markers_wo_holes[:, self.vdim+4]
        else:
            self._markers[~self._holes, self.vdim+3] = self.markers_wo_holes[:, self.vdim+5]

    def update_weights(self, f0):
        """
        Updates the marker weights according to w0 - control*f0(eta, v)/s0, where control=True or control=False.

        Parameters
        ----------
        f0 : callable
            The distribution function used as a control variate. Is called as f0(eta1, eta2, eta3, vx, vy, vz).
        """

        if self._use_control_variate:
            self._markers[~self._holes, self.vdim+3] = self.markers_wo_holes[:, self.vdim+5] - \
                f0(*self.markers_wo_holes[:, :self.vdim+3].T)/self.markers_wo_holes[:, self.vdim+4]

    def binning(self, components, bin_edges, domain=None):
        """
        Computes the distribution function via marker binning in logical space using numpy's histogramdd.

        Parameters
        ----------
        components : list[bool]
            List of length 6 giving the directions in phase space in which to bin.

        bin_edges : list[array]
            List of bin edges (resolution) having the length of True entries in components.

        domain : struphy.geometry.domains
            Mapping info for evaluating metric coefficients.

        Returns
        -------
        f_sclice : array-like
            The reconstructed distribution function.
        """

        assert np.count_nonzero(components) == len(bin_edges)

        # volume of a bin
        bin_vol = 1.

        for bin_edges_i in bin_edges:
            bin_vol *= bin_edges_i[1] - bin_edges_i[0]

        # extend components list to number of columns of markers array
        slicing = components + [False] * (self._markers.shape[1] - 6)

        # binning with weight transformation
        if domain is not None:
            f_slice = np.histogramdd(self.markers_wo_holes[:, slicing],
                                     bins=bin_edges,
                                     weights=self.markers_wo_holes[:, 6]
                                     / domain.jacobian_det(self.markers))[0]

        # binning without weight transformation
        else:
            f_slice = np.histogramdd(self.markers_wo_holes[:, slicing],
                                     bins=bin_edges,
                                     weights=self.markers_wo_holes[:, 6])[0]

        return f_slice/(self._n_mks*bin_vol)

    def show_distribution_function(self, components, bin_edges, domain=None):
        """
        1D and 2D plots of slices of the distribution function via marker binning.

        Parameters
        ----------
        components : list[bool]
            List of length 6 giving the directions in phase space in which to bin.

        bin_edges : list[array]
            List of bin edges (resolution) having the length of True entries in components.

        domain : struphy.geometry.domains
            Mapping info for evaluating metric coefficients.
        """

        import matplotlib.pyplot as plt

        n_dim = np.count_nonzero(components)

        assert n_dim == 1 or n_dim == 2

        f_slice = self.binning(components, bin_edges, domain)

        bin_centers = [bi[:-1] + (bi[1] - bi[0])/2 for bi in bin_edges]

        #labels = {0 : '$\eta_1$', 1 : '$\eta_2$', 2 : '$\eta_3$', 3 : '$v_x$', 4 : '$v_y$', 5 : '$v_z$'}

        indices = np.nonzero(components)[0]

        if n_dim == 1:
            plt.plot(bin_centers[0], f_slice)
            # plt.xlabel(labels[indices[0]])
        else:
            plt.contourf(bin_centers[0], bin_centers[1], f_slice, levels=20)
            plt.colorbar()
            plt.axis('square')
            # plt.xlabel(labels[indices[0]])
            # plt.ylabel(labels[indices[1]])

        plt.show()

    def apply_kinetic_bc(self):
        """
        Apply boundary conditions to markers that are outside of the logical unit cube.

        Parameters
        ----------
        """

        self.comm.Barrier()

        for axis, bc in enumerate(self.bc):

            # sorting out particles outside of the logical unit cube
            is_outside_cube = np.logical_or(self.markers[:,axis] > 1.,
                                            self.markers[:,axis] < 0.)
            
            # exclude holes
            is_outside_cube[self.holes] = False

            # indices or particles that are outside of the logical unit cube
            outside_inds = np.nonzero(is_outside_cube)[0]

            # apply boundary conditions
            if bc == 'remove':
                self.markers[outside_inds, :-1] = -1.

            elif bc == 'periodic':
                self.markers[outside_inds, axis] = self.markers[outside_inds, axis]%1.

            elif bc == 'reflect':
                reflect(self.markers, *self.domain.args_map, outside_inds, axis)

            else:
                raise NotImplementedError('Given bc_type is not implemented!')

        self.comm.Barrier()


class Particles6D(Particles):
    """
    A class for initializing particles in models that use the full 6D phase space.

    | 0 | 1 | 2 | 3 | 4 | 5 |   6  | 7  | 8  |
    |    eta    |     v     |weight| s0 | w0 |

    Parameters
    ----------
    name : str
        Name of the particle species.

    **params : dict
        Parameters for markers.
    """

    def __init__(self, name, **params):

        # base class params
        base_params = {}

        list_base_params = ['type', 'ppc', 'Np', 'eps', 'bc_type', 'loading', 'comm', 'domain', 'domain_array']

        for key, val in params.items():
            if key in list_base_params:
                base_params[key] = val

        super().__init__(name, **base_params)

    @property
    def n_cols(self):
        """Number of the columns at each markers.
        """
        return 16
    
    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 3
    
    def svol(self, eta1, eta2, eta3, *v):
        """ 
        Sampling density function as volume form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """
        # load sampling density svol = s6 = s3 (normalized to 1 in logical space!)
        Maxwellian6DUniform = getattr(analytical, 'Maxwellian6DUniform')

        s3 = Maxwellian6DUniform(n=1.,
                                 ux=self._params['loading']['moments'][0],
                                 uy=self._params['loading']['moments'][1],
                                 uz=self._params['loading']['moments'][2],
                                 vthx=self._params['loading']['moments'][3],
                                 vthy=self._params['loading']['moments'][4],
                                 vthz=self._params['loading']['moments'][5])
        
        return s3(eta1, eta2, eta3, *v)

    def s0(self, eta1, eta2, eta3, *v, remove_holes=True):
        """ 
        Sampling density function as 0 form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        -------
        """
        return self.domain.transform(self.svol(eta1, eta2, eta3, *v), self.markers, kind='3_to_0', remove_outside=remove_holes)


class Particles5D(Particles):
    """
    A class for initializing particles in guiding-center, drift-kinetic or gyro-kinetic models that use the 5D phase space.

    | 0 | 1 | 2 |     3      |        4        |  5   | 6  | 7  |
    |    eta    | v_parallel | magnetic moment |weight| s0 | w0 |         

    Parameters
    ----------
    name : str
        Name of the particle species.

    **params : dict
        Parameters for markers.
    """

    def __init__(self, name, **params):

        # base class params
        base_params = {}

        list_base_params = ['type', 'ppc', 'Np', 'eps', 'bc_type', 'loading', 'comm', 'domain', 'domain_array']

        for key, val in params.items():
            if key in list_base_params:
                base_params[key] = val

        super().__init__(name, **base_params)

        # child class params
        child_params = {}

        list_child_params = ['A', 'Z', 'mhd_equil', 'units_basic']

        for key, val in params.items():
            if key in list_child_params:
                child_params[key] = val

        params_default = {'A': 1, 
                          'Z': 1,
                          'mhd_equil': None,
                          'units_basic': None
                          }

        child_params = set_defaults(child_params, params_default)

        self._mhd_equil = params['mhd_equil']

        # compute kappa
        ee = 1.602176634e-19 # elementary charge (C)
        mH = 1.67262192369e-27 # proton mass (kg)

        Ah = child_params['A']
        Zh = child_params['Z']
        
        omega_ch = (Zh*ee*child_params['units_basic']['B'])/(Ah*mH)
        self._kappa = omega_ch*child_params['units_basic']['t']

    @property
    def n_cols(self):
        """Number of the columns at each markers.
        """
        return 25
    
    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 2
    
    def svol(self, eta1, eta2, eta3, *v):
        """ 
        Sampling density function as volume-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """
        # load sampling density svol = s5 (normalized to 1 in logical space!)
        Maxwellian5DUniform = getattr(analytical, 'Maxwellian5DUniform')

        s5 = Maxwellian5DUniform(n=1.,
                                 u_parallel=self.params['loading']['moments'][0],
                                 u_perp=self.params['loading']['moments'][1],
                                 vth_parallel=self.params['loading']['moments'][2],
                                 vth_perp=self.params['loading']['moments'][3])
        
        return s5(eta1, eta2, eta3, *v)

    def s3(self, eta1, eta2, eta3, *v):
        """
        Sampling density function as 3-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The 3-form sampling density.
        -------
        """
        # call equilibrium arrays
        etas = (np.vstack((eta1, eta2, eta3)).T).copy()
        bv = self._mhd_equil.bv(etas)
        curlb = self._mhd_equil.jv(etas)/self._mhd_equil.absB0(etas)
        unit_b1 = self._mhd_equil.unit_b1(etas)

        # contra-variant components of B* = B + 1/kappa*v_parallel*curlb0
        bstar = bv + 1/self._kappa*v[0]*curlb
        
        # B*_parallel = b0 . B*
        jacobian_det = np.einsum('ij,ij->j',unit_b1, bstar)/v[1]

        return self.svol(eta1, eta2, eta3, *v)/np.abs(jacobian_det)

    def s0(self, eta1, eta2, eta3, *v, remove_holes=True):
        """ 
        Sampling density function as 0-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_parallel, v_perp : array_like
            Velocity evaluation points.

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        -------
        """
        return self.domain.transform(self.s3(eta1, eta2, eta3, *v), self.markers, kind='3_to_0', remove_outside=remove_holes)

    def save_magnetic_moment(self, derham):
        r"""
        Calculate magnetic moment of each particles :math:`\mu = \frac{m v_\perp^2}{2B}` and asign it into markers[:,4].
        """
        from struphy.pic.utilities_kernels import eval_magnetic_moment_5d

        T1, T2, T3 = derham.Vh_fem['0'].knots

        absB = derham.P['0'](self._mhd_equil.absB0)

        E0T = derham.E['0'].transpose()

        absB = E0T.dot(absB)

        eval_magnetic_moment_5d(self._markers,
                                np.array(derham.p), T1, T2, T3,
                                np.array(derham.Vh['0'].starts),
                                absB._data)

    def save_magnetic_energy(self, derham, PB):
        r"""
        Calculate magnetic field energy at each particles' position and asign it into markers[:,8].
        """
        from struphy.pic.utilities_kernels import eval_magnetic_energy

        T1, T2, T3 = derham.Vh_fem['0'].knots

        eval_magnetic_energy(self._markers,
                             np.array(derham.p), T1, T2, T3,
                             np.array(derham.Vh['0'].starts),
                             PB._data)


def sendrecv_determine_mtbs(markers, holes, domain_decomp, mpi_rank):
    """
    Determine which markers have to be sent from current process and put them in a new array. 
    Corresponding rows in markers array become holes and are therefore set to -1.
    This can be done purely with numpy functions (fast, vectorized).

    Parameters
    ----------
        markers : array[float]
            Local markers array of shape (n_mks_loc + n_holes_loc, :).

        holes : array[bool]
            Local array stating whether a row in the markers array is empty (i.e. a hole) or not.

        domain_decomp : array[float]
            2d array of shape (mpi_size, 9) defining the domain of each process.

        mpi_rank : int
            Rank of calling MPI process.

    Returns
    -------
        markers_to_be_sent : array[float]
            Markers of shape (n_send, :) to be sent.

        hole_inds_after_send : array[int]
            Indices of empty columns in markers after send.
    """

    # check which particles are in a certain interval (e.g. the process domain)
    is_on_proc_domain = np.logical_and(
        markers[:, :3] > domain_decomp[mpi_rank, 0::3],
        markers[:, :3] < domain_decomp[mpi_rank, 1::3])

    # to can_stay on the current process, all three columns must be True
    can_stay = np.all(is_on_proc_domain, axis=1)

    # holes can stay, too
    can_stay[holes] = True

    # True values can can_stay on the process, False must be sent, already empty rows (-1) cannot be sent
    send_inds = np.nonzero(~can_stay)[0]

    hole_inds_after_send = np.nonzero(np.logical_or(~can_stay, holes))[0]

    # New array for sending particles.
    # TODO: do not create new array, but just return send_inds?
    # Careful: just markers[send_ids] already creates a new array in memory
    markers_to_be_sent = markers[send_inds]

    # set new holes in markers array to -1
    markers[send_inds] = -1.

    return markers_to_be_sent, hole_inds_after_send


def sendrecv_get_destinations(markers_to_be_sent, domain_decomp, mpi_size):
    """
    Determine to which process particles have to be sent.

    Parameters
    ----------
        markers_to_be_sent : array[float]
            Markers of shape (n_send, :) to be sent.

        domain_decomp : array[float]
            2d array of shape (mpi_size, 9) defining the domain of each process.

        mpi_size : int
            Total number of MPI processes.

    Returns
    -------
        send_info : array[int]
            Amount of particles sent to i-th process.

        send_list : list[array]
            Particles sent to i-th process.
    """

    # One entry for each process
    send_info = np.zeros(mpi_size, dtype=int)
    send_list = []

    # TODO: do not loop over all processes, start with neighbours and work outwards (using while)
    for i in range(mpi_size):

        conds = np.logical_and(
            markers_to_be_sent[:, :3] > domain_decomp[i, 0::3],
            markers_to_be_sent[:, :3] < domain_decomp[i, 1::3])

        send_to_i = np.nonzero(np.all(conds, axis=1))[0]
        send_info[i] = send_to_i.size

        send_list += [markers_to_be_sent[send_to_i]]

    return send_info, send_list


def sendrecv_all_to_all(send_info, comm):
    """
    Distribute info on how many markers will be sent/received to/from each process via all-to-all.

    Parameters
    ----------
        send_info : array[int]
            Amount of markers to be sent to i-th process.

        comm : Intracomm
            MPI communicator from mpi4py.MPI.Intracomm.

    Returns
    -------
        recv_info : array[int]
            Amount of marticles to be received from i-th process.
    """

    recv_info = np.zeros(comm.Get_size(), dtype=int)

    comm.Alltoall(send_info, recv_info)

    return recv_info


def sendrecv_markers(send_list, recv_info, hole_inds_after_send, markers, comm):
    """
    Use non-blocking communication. In-place modification of markers

    Parameters
    ----------
        send_list : list[array]
            Markers to be sent to i-th process.

        recv_info : array[int]
            Amount of markers to be received from i-th process.

        hole_inds_after_send : array[int]
            Indices of empty rows in markers after send.

        markers : array[float]
            Local markers array of shape (n_mks_loc + n_holes_loc, :).

        comm : Intracomm
            MPI communicator from mpi4py.MPI.Intracomm.
    """

    # i-th entry holds the number (not the index) of the first hole to be filled by data from process i
    first_hole = np.cumsum(recv_info) - recv_info

    # Initialize send and receive commands
    reqs = []
    recvbufs = []
    for i, (data, N_recv) in enumerate(zip(send_list, list(recv_info))):
        if i == comm.Get_rank():
            reqs += [None]
            recvbufs += [None]
        else:
            comm.Isend(data, dest=i, tag=comm.Get_rank())

            recvbufs += [np.zeros((N_recv, markers.shape[1]), dtype=float)]
            reqs += [comm.Irecv(recvbufs[-1], source=i, tag=i)]

    # Wait for buffer, then put markers into holes
    test_reqs = [False] * (recv_info.size - 1)
    while len(test_reqs) > 0:
        # loop over all receive requests
        for i, req in enumerate(reqs):
            if req is None:
                continue
            else:
                # check if data has been received
                if req.Test():

                    markers[hole_inds_after_send[first_hole[i] +
                                                 np.arange(recv_info[i])]] = recvbufs[i]

                    test_reqs.pop()
                    reqs[i] = None
