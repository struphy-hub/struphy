from abc import ABCMeta, abstractmethod

import numpy as np
import h5py
import scipy.special as sp

from struphy.pic import sampling_kernels, sobol_seq
from struphy.pic.pushing.pusher_utilities_kernels import reflect
from struphy.pic.utilities_kernels import eval_magnetic_energy
from struphy.kinetic_background import maxwellians
from struphy.fields_background.mhd_equil.equils import set_defaults


class Particles(metaclass=ABCMeta):
    """
    Base class for a particle based kinetic species.

    Loading and compute initial particles and save the values at the corresponding column of markers array:

    ===== ============== ======================= ======= ====== ====== ========== === ===
    index  | 0 | 1 | 2 | | 3 | ... | 3+(vdim-1)|  3+vdim 4+vdim 5+vdim >=6+vdim   ... -1
    ===== ============== ======================= ======= ====== ====== ========== === ===
    value position (eta)    velocities           weight   s0     w0      other    ... ID
    ===== ============== ======================= ======= ====== ====== ========== === ===

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
                          'bc': {'type': ['periodic', 'periodic', 'periodic']},
                          'loading': {'type': 'pseudo:random', 'seed': 1234, 'dir_particles': None, 'moments': [0., 0., 0., 1., 1., 1.]},
                          'derham': None,
                          'domain': None,
                          'f0_params': None}

        self._params = set_defaults(params, params_default)

        self._name = name
        self._derham = params['derham']
        self._domain = params['domain']
        self._domain_decomp = params['derham'].domain_array
        f0_params = params['f0_params']

        assert params['derham'].comm is not None
        self._mpi_comm = params['derham'].comm
        self._mpi_size = params['derham'].comm.Get_size()
        self._mpi_rank = params['derham'].comm.Get_rank()

        # create marker array
        self.create_marker_array()

        # Assume full-f if type is not in parameters
        if params['type'] not in 'full_f':

            self._control_variate = True

            assert f0_params is not None, \
                'When control variate is used, background parameters must be given!'
            fun_name = f0_params['type']

            if fun_name in f0_params:
                self._f_backgr = getattr(maxwellians, fun_name)(
                    **f0_params[fun_name])
            else:
                self._f_backgr = getattr(maxwellians, fun_name)()
        else:

            self._control_variate = False
            self._f_backgr = None

    @abstractmethod
    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """ Jacobian determinant of the velocity coordinate transformation.
        """
        pass

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
    def f_init(self):
        assert hasattr(self, '_f_init'), AttributeError(
            'The method "initialize_weights" has not yet been called.')
        return self._f_init

    @property
    def f_backgr(self):
        assert hasattr(self, '_f_backgr'), AttributeError(
            'No background distribution available, maybe this is a full-f model?')
        return self._f_backgr

    @property
    def control_variate(self):
        '''Boolean for whether to use the `<https://struphy.pages.mpcdf.de/struphy/sections/discretization.html#control-variate-method>`_.'''
        return self._control_variate

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
        """ Numpy array holding the marker information, including holes. The i-th row holds the i-th marker info.

        ===== ============== ======================= ======= ====== ====== ========== === ===
        index  | 0 | 1 | 2 | | 3 | ... | 3+(vdim-1)|  3+vdim 4+vdim 5+vdim >=6+vdim   ... -1
        ===== ============== ======================= ======= ====== ====== ========== === ===
        value position (eta)    velocities           weight   s0     w0      other    ... ID
        ===== ============== ======================= ======= ====== ====== ========== === ===
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
        return self.markers[~self.holes]

    @property
    def derham(self):
        """ struphy.feec.psydac_derham.Derham
        """
        return self._derham

    @property
    def domain(self):
        """ struphy.geometry.domains
        """
        return self._domain

    @property
    def lost_markers(self):
        """ Array containing the last infos of removed markers
        """
        return self._lost_markers

    @property
    def n_lost_markers(self):
        """ Number of removed particles.
        """
        return self._n_lost_markers

    @property
    def index(self):
        """ Dict holding the column indices referring to specific marker parameters (coordinates).
        """
        out = {}
        out['pos'] = slice(0, 3) # positions
        out['vel'] = slice(3, 3 + self.vdim) # velocities
        out['coords'] = slice(0, 3 + self.vdim) # phasespace_coords
        out['weights'] = 3 + self.vdim # weights
        out['s0'] = 4 + self.vdim # sampling_density
        out['w0'] = 5 + self.vdim # weights0
        out['ids'] = -1 # marker_inds
        return out

    @property
    def positions(self):
        """ Array holding the marker positions in logical space, excluding holes. The i-th row holds the i-th marker info.
        """
        return self.markers[~self.holes, self.index['pos']]

    @positions.setter
    def positions(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, 3)
        self._markers[~self.holes, self.index['pos']] = new

    @property
    def velocities(self):
        """ Array holding the marker velocities in logical space, excluding holes. The i-th row holds the i-th marker info.
        """
        return self.markers[~self.holes, self.index['vel']]

    @velocities.setter
    def velocities(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, self.vdim)
        self._markers[~self.holes, self.index['vel']] = new

    @property
    def phasespace_coords(self):
        """ Array holding the marker velocities in logical space, excluding holes. The i-th row holds the i-th marker info.
        """
        return self.markers[~self.holes, self.index['coords']]

    @phasespace_coords.setter
    def phasespace_coords(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, 3 + self.vdim)
        self._markers[~self.holes, self.index['coords']] = new

    @property
    def weights(self):
        """ Array holding the current marker weights, excluding holes. The i-th row holds the i-th marker info.
        """
        return self.markers[~self.holes, self.index['weights']]

    @weights.setter
    def weights(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[~self.holes, self.index['weights']] = new

    @property
    def sampling_density(self):
        """ Array holding the current marker 0form sampling density s0, excluding holes. The i-th row holds the i-th marker info.
        """
        return self.markers[~self.holes, self.index['s0']]

    @sampling_density.setter
    def sampling_density(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[~self.holes, self.index['s0']] = new

    @property
    def weights0(self):
        """ Array holding the initial marker weights, excluding holes. The i-th row holds the i-th marker info.
        """
        return self.markers[~self.holes, self.index['w0']]

    @weights0.setter
    def weights0(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[~self.holes, self.index['w0']] = new

    @property
    def marker_ids(self):
        """ Array holding the marker id's on the current process.
        """
        return self.markers[~self.holes, self.index['ids']]

    @marker_ids.setter
    def marker_ids(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[~self.holes, self.index['ids']] = new

    @property
    def pforms(self):
        """ Tuple of size 2; each entry must be either "vol" or None, defining the p-form 
        (space and velocity, respectively) of f_init.
        """
        return self._pforms

    def create_marker_array(self):
        """ Create marker array (self.markers).
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
        self._mpi_comm.Allgather(np.array([int(ppc*n_cells_loc)]),
                                 self._n_mks_load)

        # add deviation from Np to rank 0
        self._n_mks_load[0] += Np - np.sum(self._n_mks_load)

        # check if all markers are there
        assert np.sum(self._n_mks_load) == Np
        self._n_mks = Np

        # number of markers on the local process at loading stage
        n_mks_load_loc = self._n_mks_load[self._mpi_rank]

        # create markers array (3 x positions, vdim x velocities, weight, s0, w0, ..., ID) with eps send/receive buffer
        markers_size = round(n_mks_load_loc *
                             (1 + 1/np.sqrt(n_mks_load_loc) + self.params['eps']))
        self._markers = np.zeros((markers_size, self.n_cols), dtype=float)

        # create array container (3 x positions, vdim x velocities, weight, s0, w0, ID) for removed markers
        self._n_lost_markers = 0
        self._lost_markers = np.zeros((int(markers_size*0.5), 10), dtype=float)

    def draw_markers(self):
        r""" 
        Drawing markers according to the volume density :math:`s^n_{\textnormal{in}}`.
        In Struphy, the initial marker distribution :math:`s^n_{\textnormal{in}}` is always of the form

        .. math::

            s^n_{\textnormal{in}}(\eta,v) = n^3(\eta)\, \mathcal M(v)\,,

        with :math:`\mathcal M(v)` a multi-variate Gaussian:

        .. math:: 

            \mathcal M(v) = \prod_{i=1}^{d_v} \frac{1}{\sqrt{2\pi}\,v_{\mathrm{th},i}}
                \exp\left[-\frac{(v_i-u_i)^2}{2 v_{\mathrm{th},i}^2}\right]\,,

        where :math:`d_v` stands for the dimension in velocity space, :math:`u_i` are velocity constant shifts
        and :math:`v_{\mathrm{th},i}` are constant thermal velocities (standard deviations).
        The function :math:`n^3:(0,1)^3 \to \mathbb R^+` is a normalized 3-form on the unit cube,

        .. math::

            \int_{(0,1)^3} n^3(\eta)\,\textnormal d \eta = 1\,.

        The following choices are available in Struphy:

        1. Uniform distribution on the unit cube: :math:`n^3(\eta) = 1`

        2. Uniform distribution on the disc: :math:`n^3(\eta) = 2\eta_1` (radial coordinate = volume element of square-to-disc mapping) 

        Velocities are sampled via inverse transform sampling.
        In case of Particles6D, velocities are sampled as a Maxwellian in each 3 directions,

        .. math::

            r_i = \int^{v_i}_{-\infty} \mathcal M(v^\prime_i) \textnormal{d} v^\prime_i = \frac{1}{2}\left[ 1 + \text{erf}\left(\frac{v_i - u_i}{\sqrt{2}v_{\mathrm{th},i}}\right)\right] \,,

        where :math:`r_i \in \mathcal R(0,1)` is a uniformly drawn random number in the unit interval. So then

        .. math::

            v_i = \text{erfinv}(2r_i - 1)\sqrt{2}v_{\mathrm{th},i} + u_i \,.

        In case of Particles5D, parallel velocity is sampled as a Maxwellian and perpendicular particle speed :math:`v_\perp = \sqrt{v_1^2 + v_2^2}` is sampled as a 2D Maxweillian in polar coordinates,

        .. math::

            \mathcal{M}(v_1, v_2) \, \textnormal{d} v_1 \textnormal{d} v_2 &=  \prod_{i=1}^{2} \frac{1}{\sqrt{2\pi}}\frac{1}{v_{\mathrm{th},i}}
                \exp\left[-\frac{(v_i-u_i)^2}{2 v_{\mathrm{th},i}^2}\right] \textnormal{d} v_i\,,
            \\
            &= \frac{1}{v_\mathrm{th}^2}v_\perp \exp\left[-\frac{(v_\perp-u)^2}{2 v_\mathrm{th}^2}\right] \textnormal{d} v_\perp\,,
            \\
            &= \mathcal{M}^{\textnormal{pol}}(v_\perp) \, \textnormal{d} v_\perp \,.

        Then,

        .. math::

            r = \int^{v_\perp}_0 \mathcal{M}^{\textnormal{pol}} \textnormal{d} v_\perp = 1 - \exp\left[-\frac{(v_\perp-u)^2}{2 v_\mathrm{th}^2}\right] \,.

        So then,

        .. math::

            v_\perp = \sqrt{- \ln(1-r)}\sqrt(2)v_\mathrm{th} + u \,.

        All needed parameters can be set in the parameter file, in the section ``kinetic/<species>/markers/loading``.
        """

        # number of markers on the local process at loading stage
        n_mks_load_loc = self.n_mks_load[self.mpi_rank]

        # fill holes in markers array with -1 (all holes are at end of array at loading stage)
        self._markers[n_mks_load_loc:] = -1.

        # number of holes and markers on process
        self._holes = self.markers[:, 0] == -1.
        self._n_holes_loc = np.count_nonzero(self._holes)
        self._n_mks_loc = self.markers.shape[0] - self._n_holes_loc

        # cumulative sum of number of markers on each process at loading stage.
        n_mks_load_cum_sum = np.cumsum(self.n_mks_load)

        if self.mpi_rank == 0:
            print('\nMARKERS:')
            for key, val in self.params.items():
                if 'loading' not in key and 'derham' not in key and 'domain' not in key:
                    print((key + ' :').ljust(25), val)

        # load markers from external .hdf5 file
        if self.params['loading']['type'] == 'external':

            if self.mpi_rank == 0:
                file = h5py.File(self.params['loading']['dir_markers'], 'r')
                print('Loading markers from file: '.ljust(25), file)

                self._markers[:n_mks_load_cum_sum[0], :
                              ] = file['markers'][:n_mks_load_cum_sum[0], :]

                for i in range(1, self._mpi_size):
                    self._mpi_comm.Send(
                        file['markers'][n_mks_load_cum_sum[i - 1]:n_mks_load_cum_sum[i], :], dest=i, tag=123)

                file.close()
            else:
                recvbuf = np.zeros(
                    (n_mks_load_loc, self.markers.shape[1]), dtype=float)
                self._mpi_comm.Recv(recvbuf, source=0, tag=123)
                self._markers[:n_mks_load_loc, :] = recvbuf

        # load fresh markers
        else:

            if self.mpi_rank == 0:
                print('\nLoading fresh markers:')
                for key, val in self.params['loading'].items():
                    print((key + ' :').ljust(25), val)

            # 1. standard random number generator (pseudo-random)
            if self.params['loading']['type'] == 'pseudo_random':

                _seed = self.params['loading']['seed']
                if _seed is not None:
                    np.random.seed(_seed)

                for i in range(self._mpi_size):
                    temp = np.random.rand(self.n_mks_load[i], 3 + self.vdim)

                    if i == self._mpi_rank:
                        self.phasespace_coords = temp
                        break

                del temp

            # 2. plain sobol numbers with skip of first 1000 numbers
            elif self.params['loading']['type'] == 'sobol_standard':

                self.phasespace_coords = sobol_seq.i4_sobol_generate(
                    3 + self.vdim, n_mks_load_loc, 1000 + (n_mks_load_cum_sum - self.n_mks_load)[self._mpi_rank])

            # 3. symmetric sobol numbers in all 6 dimensions with skip of first 1000 numbers
            elif self.params['loading']['type'] == 'sobol_antithetic':

                assert self.vdim == 3, NotImplementedError(
                    '"sobol_antithetic" requires vdim=3 at the moment.')

                temp_markers = sobol_seq.i4_sobol_generate(
                    3 + self.vdim, n_mks_load_loc//64, 1000 + (n_mks_load_cum_sum - self.n_mks_load)[self._mpi_rank]//64)

                sampling_kernels.set_particles_symmetric_3d_3v(
                    temp_markers, self.markers)

            # 4. Wrong specification
            else:
                raise ValueError(
                    'Specified particle loading method does not exist!')

            # inverse transform sampling in velocity space
            u_mean = np.array(self.params['loading']['moments'][:self.vdim])
            v_th = np.array(self.params['loading']['moments'][self.vdim:])

            # Particles6D: (1d Maxwellian, 1d Maxwellian, 1d Maxwellian)
            if self.vdim == 3:
                self.velocities = sp.erfinv(2*self.velocities - 1)*np.sqrt(2)*v_th + u_mean

            # Particles5D: (1d Maxwellian, 2d Maxwellian)
            elif self.vdim == 2:
                self.markers[:n_mks_load_loc, 3] = sp.erfinv(2*self.velocities[:,0] - 1)*np.sqrt(2)*v_th[0] + u_mean[0]
                self.markers[:n_mks_load_loc, 4] = np.sqrt(-1*np.log(1-self.velocities[:,1]))*np.sqrt(2)*v_th[1] + u_mean[1]

            else:
                raise NotImplementedError('Inverse transform sampling of given vdim is not implemented!')

            # inversion method for drawing uniformly on the disc
            _spatial = self.params['loading']['spatial']
            if _spatial == 'disc':
                self._markers[:n_mks_load_loc, 0] = np.sqrt(
                    self.markers[:n_mks_load_loc, 0])
            else:
                assert _spatial == 'uniform', f'Spatial drawing must be "uniform" or "disc", is {_spatial}.'

        # set markers ID in last column
        self.marker_ids = (n_mks_load_cum_sum - self.n_mks_load)[
            self._mpi_rank] + np.arange(n_mks_load_loc, dtype=float)

        # set specific initial condition for some particles
        if 'initial' in self.params['loading']:
            specific_markers = self.params['loading']['initial']

            counter = 0
            for i in range(len(specific_markers)):
                if i == int(self.markers[counter, -1]):

                    for j in range(3+self.vdim):
                        if specific_markers[i][j] is not None:
                            self._markers[counter, j] = specific_markers[i][j]

                    counter += 1

        # check if all particle positions are inside the unit cube [0, 1]^3
        n_mks_load_loc = self._n_mks_load[self._mpi_rank]

        assert np.all(~self._holes[:n_mks_load_loc]) and np.all(
            self._holes[n_mks_load_loc:])

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
        self._holes = self.markers[:, 0] == -1.
        self._n_holes_loc = np.count_nonzero(self.holes)
        self._n_mks_loc = self.markers.shape[0] - self._n_holes_loc

        # check if all markers are on the right process after sorting
        if do_test:
            all_on_right_proc = np.all(np.logical_and(
                self.positions > self.domain_decomp[self.mpi_rank, 0::3],
                self.positions < self.domain_decomp[self.mpi_rank, 1::3]))

            assert all_on_right_proc

        self.comm.Barrier()

    def initialize_weights(self, fun_params, bckgr_params=None):
        r"""
        Computes the initial weights

        .. math::

            w_{k0} := \frac{f^0(t, q_k(t)) }{s^0(t, q_k(t)) } = \frac{f^0(0, q_k(0)) }{s^0(0, q_k(0)) } = \frac{f^0_{\textnormal{in}}(q_{k0}) }{s^0_{\textnormal{in}}(q_{k0}) }

        from the initial distribution function :math:`f^0_{\textnormal{in}}` specified in the parmeter file
        and from the initial volume density :math:`s^n_{\textnormal{in}}` specified in :meth:`struphy.pic.particles.Particles.draw_markers`.
        Moreover, it sets the corresponding columns for "w0", "s0" and "weights" in the markers array.
        For the control variate method, the background is subtracted.

        Parameters
        ----------
        fun_params : dict
            Dictionary of the form {type : class_name, pforms : differential forms, class_name : params_dict} defining the initial condition.

        bckgr_params : dict (optional)
            Dictionary of the form {type : class_name, class_name : params_dict} defining the background.
        """

        assert self.domain is not None, 'A domain is needed to initialize weights.'

        # compute s0 and save at vdim + 4
        self.sampling_density = self.s0(*self.phasespace_coords.T)

        # load distribution function (with given parameters or default parameters)
        fun_name = fun_params['type']

        if fun_name in fun_params:
            self._f_init = getattr(maxwellians, fun_name)(
                **fun_params[fun_name])
        else:
            self._f_init = getattr(maxwellians, fun_name)()

        f_init = self.f_init(*self.phasespace_coords.T)

        # if diffential forms of f_init is not specified, consider it as 0-form
        if 'pforms' in fun_params:
            assert len(fun_params['pforms']) == 2
            self._pforms = fun_params['pforms']
        else:
            self._pforms = (None, None)

        # if f_init is vol-form, tramsform to 0-form
        if self.pforms[0] == 'vol':
            f_init /= self.domain.jacobian_det(self.markers_wo_holes)

        if self.pforms[1] == 'vol':
            f_init /= self.velocity_jacobian_det(*self.phasespace_coords.T)

        # compute w0 and save at vdim + 5
        self.weights0 = f_init / self.sampling_density

        # compute weights
        if self._control_variate:
            self.update_weights()
        else:
            self.weights = self.weights0

    def update_weights(self):
        """
        Applies the control variate method.

        Updates the time-dependent marker weights according to the algorithm in the 
        `Struphy documentation <https://struphy.pages.mpcdf.de/struphy/sections/discretization.html#control-variate-method>`_.
        """

        f_backgr = self.f_backgr(*self.phasespace_coords.T)

        # if f_init is vol-form, transform to 0-form
        if self.pforms[0] == 'vol':
            f_backgr /= self.domain.jacobian_det(self.markers_wo_holes)

        if self.pforms[1] == 'vol':
            f_backgr /= self.velocity_jacobian_det(
                *self.phasespace_coords.T)

        self.weights = self.weights0 - f_backgr/self.sampling_density

    def binning(self, components, bin_edges, pforms=['0', '0']):
        r"""
        Computes the distribution function via marker binning in logical space using numpy's histogramdd,
        following the algorithm outlined in the `Struphy documentation <https://struphy.pages.mpcdf.de/struphy/sections/discretization.html#particle-binning>`_.
        If pforms=['vol','vol'], approximations of the volume density :math:`f^n(t)` are computed (of :math:`f^0(t)` by default). 

        Parameters
        ----------
        components : list[bool]
            List of length n (dim. of phase space) giving the directions in phase space in which to bin.

        bin_edges : list[array]
            List of bin edges (resolution) having the length of True entries in components.

        pforms : list
            List differential forms ('0' or 'vol') of the distribution function in [domain-space, velocity-space] in which to bin.

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
        _n = len(components)
        slicing = components + [False] * (self.markers.shape[1] - _n)

        # compute weights of histogram:
        _weights = self.markers_wo_holes[:, _n]

        if pforms[0] == '0':
            _weights /= self.domain.jacobian_det(self.markers_wo_holes)

        if pforms[1] == '0':
            _weights /= self.velocity_jacobian_det(*self.phasespace_coords.T)

        f_slice = np.histogramdd(self.markers_wo_holes[:, slicing],
                                 bins=bin_edges,
                                 weights=_weights)[0]

        return f_slice/(self._n_mks*bin_vol)

    def show_distribution_function(self, components, bin_edges, pforms=['0', '0']):
        """
        1D and 2D plots of slices of the distribution function via marker binning.
        This routine is mainly for de-bugging.

        Parameters
        ----------
        components : list[bool]
            List of length 6 giving the directions in phase space in which to bin.

        bin_edges : list[array]
            List of bin edges (resolution) having the length of True entries in components.

        pforms : list
            List differential forms ('0' or 'vol') of the distribution function in [domain-space, velocity-space] in which to bin.
        """

        import matplotlib.pyplot as plt

        n_dim = np.count_nonzero(components)

        assert n_dim == 1 or n_dim == 2, f'Distribution function can only be shown in 1D or 2D slices, not {n_dim}.'

        f_slice = self.binning(components, bin_edges, pforms)

        bin_centers = [bi[:-1] + (bi[1] - bi[0])/2 for bi in bin_edges]

        labels = {0: '$\eta_1$', 1: '$\eta_2$',
                  2: '$\eta_3$', 3: '$v_1$', 4: '$v_2$', 5: '$v_3$'}
        indices = np.nonzero(components)[0]

        if n_dim == 1:
            plt.plot(bin_centers[0], f_slice)
            plt.xlabel(labels[indices[0]])
        else:
            plt.contourf(bin_centers[0], bin_centers[1], f_slice, levels=20)
            plt.colorbar()
            plt.axis('square')
            plt.xlabel(labels[indices[0]])
            plt.ylabel(labels[indices[1]])

        plt.show()

    def apply_kinetic_bc(self):
        """
        Apply boundary conditions to markers that are outside of the logical unit cube.

        Parameters
        ----------
        """

        for axis, bc in enumerate(self.params['bc']['type']):

            # sorting out particles outside of the logical unit cube
            is_outside_cube = np.logical_or(self.markers[:, axis] > 1.,
                                            self.markers[:, axis] < 0.)

            # exclude holes
            is_outside_cube[self.holes] = False

            # indices or particles that are outside of the logical unit cube
            outside_inds = np.nonzero(is_outside_cube)[0]

            # apply boundary conditions
            if bc == 'remove':

                if self.params['bc']['remove']['boundary_transfer']:
                    # boundary transfer
                    outside_inds = self.boundary_transfer(is_outside_cube)

                if self.params['bc']['remove']['save']:
                    # save the positions and velocities just before the pushing step
                    if self.vdim == 3:
                        self.lost_markers[self.n_lost_markers:self.n_lost_markers +
                                          len(outside_inds), 0:3] = self.markers[outside_inds, 9:12]
                        self.lost_markers[self.n_lost_markers:self.n_lost_markers +
                                          len(outside_inds), 3:9] = self.markers[outside_inds, 3:9]
                        self.lost_markers[self.n_lost_markers:self.n_lost_markers +
                                          len(outside_inds), -1] = self.markers[outside_inds, -1]

                    elif self.vdim == 2:
                        self.lost_markers[self.n_lost_markers:self.n_lost_markers +
                                          len(outside_inds), 0:4] = self.markers[outside_inds, 9:13]
                        self.lost_markers[self.n_lost_markers:self.n_lost_markers +
                                          len(outside_inds), 4:9] = self.markers[outside_inds, 4:9]
                        self.lost_markers[self.n_lost_markers:self.n_lost_markers +
                                          len(outside_inds), -1] = self.markers[outside_inds, -1]

                self._markers[outside_inds, :-1] = -1.

                self._n_lost_markers += len(outside_inds)

            elif bc == 'periodic':
                self.markers[outside_inds, axis] = \
                    self.markers[outside_inds, axis] % 1.

            elif bc == 'reflect':
                reflect(self.markers, *self.domain.args_map, outside_inds, axis)

            else:
                raise NotImplementedError('Given bc_type is not implemented!')

    def boundary_transfer(self, is_outside_cube):
        """
        Still draft. ONLY valid for the poloidal geometry (eta1: clamped r-direction, eta2: periodic theta-direction). 

        When particles reach to the inner boundary circle, transfer them to the opposite side of the circle.

        Parameters
        ----------
        """
        # sorting out particles which are inside of the inner hole
        smaller_than_rmin = self.markers[:, 0] < 0.

        # exclude holes
        smaller_than_rmin[self.holes] = False

        # indices or particles that are inside of the inner hole
        transfer_inds = np.nonzero(smaller_than_rmin)[0]

        self._markers[transfer_inds, 0] = 1e-8
        self._markers[transfer_inds, 1] = 1. - self.markers[transfer_inds, 1]
        self._markers[transfer_inds, 9] = -1.
        self._markers[transfer_inds, 10] = 0

        is_outside_cube[transfer_inds] = False
        outside_inds = np.nonzero(is_outside_cube)[0]

        return outside_inds


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

    # to stay on the current process, all three columns must be True
    can_stay = np.all(is_on_proc_domain, axis=1)

    # holes can stay, too
    can_stay[holes] = True

    # True values can stay on the process, False must be sent, already empty rows (-1) cannot be sent
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
