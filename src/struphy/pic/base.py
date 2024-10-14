from abc import ABCMeta, abstractmethod

import struphy
import os
import yaml
import numpy as np
import h5py
import scipy.special as sp
import copy

from struphy.pic import sampling_kernels, sobol_seq
from struphy.pic.pushing.pusher_utilities_kernels import reflect
from struphy.pic.pushing.pusher_args_kernels import MarkerArguments
from struphy.kinetic_background import maxwellians
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.io.output_handling import DataContainer

from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain
from struphy.fields_background.mhd_equil.base import MHDequilibrium
from struphy.fields_background.braginskii_equil.base import BraginskiiEquilibrium


class Particles(metaclass=ABCMeta):
    """
    Base class for particle species.

    The marker information is stored in a 2D numpy array, 
    see `Tutorial on PIC data structures <https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_06_data_structures.html#PIC-data-structures>`_.

    Parameters
    ----------
    name : str
        Name of particle species.

    derham : Derham
        Struphy Derham object. 

    domain : Domain
        Struphy domain object.

    mhd_equil : MHDequilibrium
        Struphy MHD equilibrium object

    braginskii_equil : BraginskiiEquilibrium
        Struphy Braginskii equilibrium object

    bckgr_params : dict
        Kinetic background parameters.

    pert_params : dict
        Kinetic perturbation parameters.

    marker_params : dict
        Marker parameters for loading.
    """

    def __init__(self,
                 name: str,
                 derham: Derham,
                 *,
                 domain: Domain = None,
                 mhd_equil: MHDequilibrium = None,
                 braginskii_equil: BraginskiiEquilibrium = None,
                 bckgr_params: dict = None,
                 pert_params: dict = None,
                 **marker_params):

        if domain is None:
            from struphy.geometry.domains import Cuboid
            domain = Cuboid()
        
        self._name = name
        self._derham = derham
        self._domain = domain
        self._mhd_equil = mhd_equil
        self._braginskii_equil = braginskii_equil
        
        self._mpi_comm = derham.comm
        self._mpi_size = derham.comm.Get_size()
        self._mpi_rank = derham.comm.Get_rank()

        if bckgr_params is None:
            bckgr_params = {'type': 'Maxwellian3D'}

        self._bckgr_params = bckgr_params
        self._pert_params = pert_params

        marker_params_default = {
            'type': 'full_f',
            'ppc': None,
            'Np': 4,
            'eps': .25,
            'bc': {'type': ['periodic', 'periodic', 'periodic']},
            'loading': {'type': 'pseudo_random',
                        'seed': 1234,
                        'dir_particles': None,
                        'moments': [0., 0., 0., 1., 1., 1.],
                        'spatial': 'uniform'},
        }

        self._marker_params = set_defaults(
            marker_params, marker_params_default)

        self._domain_decomp = derham.domain_array

        # background p-form description (default: None, which means 0-form)
        if 'pforms' in bckgr_params:
            assert len(bckgr_params['pforms']) == 2, \
                'Only two form degrees can be given!'
            self._pforms = bckgr_params['pforms']
        else:
            self._pforms = [None, None]

        # create marker array
        self.create_marker_array()

        # allocate arrays for sorting
        n_rows = self.markers.shape[0]
        self._is_outside_right = np.zeros(n_rows, dtype=bool)
        self._is_outside_left = np.zeros(n_rows, dtype=bool)
        self._is_outside = np.zeros(n_rows, dtype=bool)

        # Check if control variate
        self._control_variate = (
            self.marker_params['type'] == 'control_variate')

        # set background function
        bckgr_type = bckgr_params['type']

        if not isinstance(bckgr_type, list):
            bckgr_type = [bckgr_type]

        self._f0 = None
        for fi in bckgr_type:
            if fi[-2] == '_':
                fi_type = fi[:-2]
            else:
                fi_type = fi
            if fi in bckgr_params:
                maxw_params = bckgr_params[fi]
                pass_mhd_equil = mhd_equil
                pass_braginskii_equil = braginskii_equil
            else:
                maxw_params = None
                pass_mhd_equil = None
                pass_braginskii_equil = None

                print(
                    f'\n{fi} is not in bckgr_params; default background parameters are used.')

            if self._f0 is None:
                self._f0 = getattr(maxwellians, fi_type)(
                    maxw_params=maxw_params,
                    mhd_equil=pass_mhd_equil,
                    braginskii_equil=pass_braginskii_equil
                )
            else:
                self._f0 = self._f0 + getattr(maxwellians, fi_type)(
                    maxw_params=maxw_params,
                    mhd_equil=pass_mhd_equil,
                    braginskii_equil=pass_braginskii_equil
                )

        # set coordinates of the background distribution
        if self.f0.coords == 'constants_of_motion':
            self._f_coords_index = self.index['com']
            self._f_jacobian_coords_index = self.index['pos+energy']

        else:
            self._f_coords_index = self.index['coords']
            self._f_jacobian_coords_index = self.index['coords']

        # Marker arguments for kernels
        self._args_markers = MarkerArguments(self.markers,
                                             self.vdim,
                                             self.bufferindex)

        # Have at least 3 spare places in markers array
        assert self.args_markers.first_free_idx + 2 < self.n_cols - \
            1, f'{self.args_markers.first_free_idx + 2} is not smaller than {self.n_cols - 1 = }; not enough columns in marker array !!'

    @classmethod
    @abstractmethod
    def default_bckgr_params(cls):
        """ Dictionary holding the minimal information of the default background.

        Must contain at least a keyword 'type' with corresponding value a valid choice of background.
        """
        pass

    @abstractmethod
    def svol(self, eta1, eta2, eta3, *v):
        r""" Marker sampling distribution function :math:`s^\textrm{vol}` as a volume form, see :ref:`monte_carlo`.
        """
        pass

    @abstractmethod
    def s0(self, eta1, eta2, eta3, *v, remove_holes=True):
        r""" Marker sampling distribution function :math:`s^0` as 0-form, see :ref:`monte_carlo`.
        """
        pass

    @property
    @abstractmethod
    def n_cols(self):
        """Number of columns in the :attr:`~struphy.pic.base.Particles.markers` array.
        """
        pass

    @property
    @abstractmethod
    def vdim(self):
        """Dimension of the velocity space.
        """
        pass

    @property
    @abstractmethod
    def bufferindex(self):
        """Starting buffer marker index number
        """
        pass

    @property
    def kinds(self):
        """ Name of the class.
        """
        return self.__class__.__name__

    @property
    def name(self):
        """ Name of the kinetic species in DATA container.
        """
        return self._name

    @property
    def bckgr_params(self):
        """ Kinetic background parameters.
        """
        return self._bckgr_params

    @property
    def pert_params(self):
        """ Kinetic perturbation parameters.
        """
        return self._pert_params

    @property
    def marker_params(self):
        """ Parameters for markers.
        """
        return self._marker_params

    @property
    def f_init(self):
        assert hasattr(self, '_f_init'), AttributeError(
            'The method "initialize_weights" has not yet been called.')
        return self._f_init

    @property
    def f0(self):
        assert hasattr(self, '_f0'), AttributeError(
            'No background distribution available, maybe this is a full-f model?')
        return self._f0

    @property
    def control_variate(self):
        '''Boolean for whether to use the :ref:`control_var` during time stepping.'''
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
        """ 2D numpy array holding the marker information, including holes. 
        The i-th row holds the i-th marker info.

        ===== ============== ======================= ======= ====== ====== ========== === ===
        index  | 0 | 1 | 2 | | 3 | ... | 3+(vdim-1)|  3+vdim 4+vdim 5+vdim >=6+vdim   ... -1
        ===== ============== ======================= ======= ====== ====== ========== === ===
        value position (eta)    velocities           weight   s0     w0      other    ... ID
        ===== ============== ======================= ======= ====== ====== ========== === ===

        The column indices referring to different attributes can be obtained from
        :attr:`~struphy.pic.base.Particles.index`.
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
        """ :class:`~struphy.feec.psydac_derham.Derham`
        """
        return self._derham

    @property
    def domain(self):
        """ From :mod:`struphy.geometry.domains`.
        """
        return self._domain

    @property
    def mhd_equil(self):
        """ From :mod:`struphy.fields_background.mhd_equil.equils`.
        """
        return self._mhd_equil

    @property
    def braginskii_equil(self):
        """ From :mod:`struphy.fields_background.braginskii_equil.equils`.
        """
        return self._braginskii_equil

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
        out['pos'] = slice(0, 3)  # positions
        out['vel'] = slice(3, 3 + self.vdim)  # velocities
        out['coords'] = slice(0, 3 + self.vdim)  # phasespace_coords
        out['com'] = slice(8, 11)  # constants of motion
        out['pos+energy'] = list(range(0, 3)) + [8] # positions + energy
        out['weights'] = 3 + self.vdim  # weights
        out['s0'] = 4 + self.vdim  # sampling_density
        out['w0'] = 5 + self.vdim  # weights0
        out['ids'] = -1  # marker_inds
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
    def constants_of_motion(self):
        """ Array holding the constants of motion of marker, excluding holes. The i-th row holds the i-th marker info.
        """
        return self.markers[~self.holes, self.index['com']]

    @constants_of_motion.setter
    def constants_of_motion(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, 3)
        self._markers[~self.holes, self.index['com']] = new

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

    @property
    def spatial(self):
        """ Drawing particles uniformly on the unit cube('uniform') or on the disc('disc')
        """
        return self._spatial

    @property
    def f_coords_index(self):
        """Dict holding the column indices referring to coords of the distribution fuction.
        """
        return self._f_coords_index

    @property
    def f_jacobian_coords_index(self):
        """Dict holding the column indices referring to coords of the velocity jacobian determinant of the distribution fuction.
        """
        return self._f_jacobian_coords_index

    @property
    def f_coords(self):
        """ Coordinates of the distribution function.
        """
        return self.markers[~self.holes, self.f_coords_index]

    @f_coords.setter
    def f_coords(self, new):
        assert isinstance(new, np.ndarray)
        self.markers[~self.holes, self.f_coords_index] = new

    @property
    def args_markers(self):
        '''Collection of mandatory arguments for pusher kernels.
        '''
        return self._args_markers
    
    @property
    def f_jacobian_coords(self):
        """ Coordinates of the velocity jacobian determinant of the distribution fuction.
        """
        if isinstance(self.f_jacobian_coords_index, list):
            return self.markers[np.ix_(~self.holes, self.f_jacobian_coords_index)]
        else:
            return self.markers[~self.holes, self.f_jacobian_coords_index]

    @f_jacobian_coords.setter
    def f_jacobian_coords(self, new):
        assert isinstance(new, np.ndarray)
        if isinstance(self.f_jacobian_coords_index, list):
            self.markers[np.ix_(~self.holes, self.f_jacobian_coords_index)] = new
        else:
            self.markers[~self.holes, self.f_jacobian_coords_index] = new

    def create_marker_array(self):
        """ Create marker array :attr:`~struphy.pic.base.Particles.markers`.
        """

        # number of cells on current process
        n_cells_loc = np.prod(
            self._domain_decomp[self._mpi_rank, 2::3], dtype=int)

        # total number of cells
        n_cells = np.sum(
            np.prod(self._domain_decomp[:, 2::3], axis=1, dtype=int))

        # number of markers to load on each process (depending on relative domain size)
        if self.marker_params['ppc'] is not None:
            assert isinstance(self.marker_params['ppc'], int)
            ppc = self.marker_params['ppc']
            Np = ppc*n_cells
        else:
            Np = self.marker_params['Np']
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
        n_rows = round(n_mks_load_loc *
                       (1 + 1/np.sqrt(n_mks_load_loc) + self.marker_params['eps']))
        self._markers = np.zeros((n_rows, self.n_cols), dtype=float)

        # create array container (3 x positions, vdim x velocities, weight, s0, w0, ID) for removed markers
        self._n_lost_markers = 0
        self._lost_markers = np.zeros((int(n_rows*0.5), 10), dtype=float)

    def draw_markers(self):
        r""" 
        Drawing markers according to the volume density :math:`s^\textrm{vol}_{\textnormal{in}}`.
        In Struphy, the initial marker distribution :math:`s^\textrm{vol}_{\textnormal{in}}` is always of the form

        .. math::

            s^\textrm{vol}_{\textnormal{in}}(\eta,v) = n^3(\eta)\, \mathcal M(v)\,,

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

        In case of Particles5D, parallel velocity is sampled as a Maxwellian and perpendicular particle speed :math:`v_\perp = \sqrt{v_1^2 + v_2^2}` 
        is sampled as a 2D Maxwellian in polar coordinates,

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

            v_\perp = \sqrt{- \ln(1-r)}\sqrt{2}v_\mathrm{th} + u \,.

        All needed parameters can be set in the parameter file, see :ref:`params_yml`.
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
            for key, val in self.marker_params.items():
                if 'loading' not in key and 'derham' not in key and 'domain' not in key:
                    print((key + ' :').ljust(25), val)

        # load markers from external .hdf5 file
        if self.marker_params['loading']['type'] == 'external':

            if self.mpi_rank == 0:
                file = h5py.File(
                    self.marker_params['loading']['dir_external'], 'r')
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

        # load markers from restart .hdf5 file
        elif self.marker_params['loading']['type'] == 'restart':

            libpath = struphy.__path__[0]

            with open(os.path.join(libpath, 'state.yml')) as f:
                state = yaml.load(f, Loader=yaml.FullLoader)

            o_path = state['o_path']

            if self.marker_params['loading']['dir_particles_abs'] is None:
                data_path = os.path.join(
                    o_path, self.marker_params['loading']['dir_particles'])
            else:
                data_path = self.marker_params['loading']['dir_particles_abs']

            data = DataContainer(data_path, comm=self.comm)

            self.markers[:, :] = data.file['restart/' +
                                           self.marker_params['loading']['key']][-1, :, :]

        # load fresh markers
        else:

            if self.mpi_rank == 0:
                print('\nLoading fresh markers:')
                for key, val in self.marker_params['loading'].items():
                    print((key + ' :').ljust(25), val)

            # 1. standard random number generator (pseudo-random)
            if self.marker_params['loading']['type'] == 'pseudo_random':
                
                # Set seed
                _seed = self.marker_params['loading']['seed']
                if _seed is not None:
                    np.random.seed(_seed)
                # Draw pseudo_random markers
                break_outer_loop = False
                # The inner loop is over number of clones so that the set of markers
                # for the first domain in each clone has the same markers independent
                # of the number of clones
                for i in range(self._mpi_size):
                    for iclone in range(self.derham.Nclones):  
                        temp = np.random.rand(self.n_mks_load[i], 3 + self.vdim) 
                        if i == self._mpi_rank:
                            if self.derham.Nclones == 1:
                                self.phasespace_coords = temp
                                break_outer_loop = True
                                #print(iclone,self._mpi_rank)
                                break
                            else:
                                if iclone == self.derham.inter_comm.Get_rank():
                                    self.phasespace_coords = temp
                                    break_outer_loop = True

                                    #print('b',iclone,self._mpi_rank)
                                    break
                    # Check the flag variable to break the outer loop
                    if break_outer_loop:
                        break
                # print(f"{break_outer_loop = }")
                # exit()
                del temp

            # 2. plain sobol numbers with skip of first 1000 numbers
            elif self.marker_params['loading']['type'] == 'sobol_standard':

                self.phasespace_coords = sobol_seq.i4_sobol_generate(
                    3 + self.vdim, n_mks_load_loc, 1000 + (n_mks_load_cum_sum - self.n_mks_load)[self._mpi_rank])

            # 3. symmetric sobol numbers in all 6 dimensions with skip of first 1000 numbers
            elif self.marker_params['loading']['type'] == 'sobol_antithetic':

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
            u_mean = np.array(
                self.marker_params['loading']['moments'][:self.vdim])
            v_th = np.array(
                self.marker_params['loading']['moments'][self.vdim:])

            # Particles6D: (1d Maxwellian, 1d Maxwellian, 1d Maxwellian)
            if self.vdim == 3:
                self.velocities = sp.erfinv(
                    2*self.velocities - 1)*np.sqrt(2)*v_th + u_mean
            # Particles5D: (1d Maxwellian, polar Maxwellian as volume-form)
            elif self.vdim == 2:
                self.markers[:n_mks_load_loc, 3] = sp.erfinv(
                    2*self.velocities[:, 0] - 1)*np.sqrt(2)*v_th[0] + u_mean[0]

                self.markers[:n_mks_load_loc, 4] = np.sqrt(
                    -1*np.log(1-self.velocities[:, 1]))*np.sqrt(2)*v_th[1] + u_mean[1]
            elif self.vdim == 0:
                pass
            else:
                raise NotImplementedError(
                    'Inverse transform sampling of given vdim is not implemented!')

            # inversion method for drawing uniformly on the disc
            self._spatial = self.marker_params['loading']['spatial']
            if self._spatial == 'disc':
                self._markers[:n_mks_load_loc, 0] = np.sqrt(
                    self.markers[:n_mks_load_loc, 0])
            else:
                assert self._spatial == 'uniform', f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.'

            # set markers ID in last column
            self.marker_ids = (n_mks_load_cum_sum - self.n_mks_load)[
                self._mpi_rank] + np.arange(n_mks_load_loc, dtype=float)

            # set specific initial condition for some particles
            if 'initial' in self.marker_params['loading']:
                specific_markers = self.marker_params['loading']['initial']

                counter = 0
                for i in range(len(specific_markers)):
                    if i == int(self.markers[counter, -1]):

                        for j in range(3+self.vdim):
                            if specific_markers[i][j] is not None:
                                self._markers[counter,
                                              j] = specific_markers[i][j]

                        counter += 1

            # check if all particle positions are inside the unit cube [0, 1]^3
            n_mks_load_loc = self._n_mks_load[self._mpi_rank]

            assert np.all(~self._holes[:n_mks_load_loc]) and np.all(
                self._holes[n_mks_load_loc:])

    def mpi_sort_markers(self,
                         apply_bc: bool = True,
                         alpha: tuple | list | int | float = 1.,
                         do_test=False):
        """ 
        Sorts markers according to MPI domain decomposition.

        Markers are sent to the process corresponding to the alpha-weighted position
        alpha*markers[:, 0:3] + (1 - alpha)*markers[:, buffer_idx:buffer_idx + 3].

        Periodic boundary conditions are taken into account 
        when computing the alpha-weighted position.

        Parameters
        ----------
        appl_bc : bool
            Whether to apply kinetic boundary conditions before sorting.

        alpha : tuple | list | int | float
            For i=1,2,3 the sorting is according to alpha[i]*markers[:, i] + (1 - alpha[i])*markers[:, buffer_idx + i].
            If int or float then alpha = (alpha, alpha, alpha). alpha must be between 0 and 1. 

        do_test : bool
            Check if all markers are on the right process after sorting.
        """

        self.comm.Barrier()

        # before sorting, apply kinetic bc
        if apply_bc:
            self.apply_kinetic_bc()

        if isinstance(alpha, int) or isinstance(alpha, float):
            alpha = (alpha, alpha, alpha)

        # create new markers_to_be_sent array and make corresponding holes in markers array
        markers_to_be_sent, hole_inds_after_send, sorting_etas = sendrecv_determine_mtbs(
            self._markers,
            self._holes,
            self.domain_decomp,
            self.mpi_rank,
            self.vdim,
            self.bufferindex,
            alpha=alpha)

        # determine where to send markers_to_be_sent
        send_info, send_list = sendrecv_get_destinations(
            markers_to_be_sent, sorting_etas, self.domain_decomp, self.mpi_size)

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
            # assert self.phasespace_coords.size > 0, f'No particles on process {self.mpi_rank}, please rebalance, aborting ...'

        self.comm.Barrier()

    def initialize_weights(self):
        r"""
        Computes the initial weights

        .. math::

            w_{k0} := \frac{f^0(t, q_k(t)) }{s^0(t, q_k(t)) } = \frac{f^0(0, q_k(0)) }{s^0(0, q_k(0)) } = \frac{f^0_{\textnormal{in}}(q_{k0}) }{s^0_{\textnormal{in}}(q_{k0}) }

        from the initial distribution function :math:`f^0_{\textnormal{in}}` specified in the parmeter file
        and from the initial volume density :math:`s^n_{\textnormal{vol}}` specified in :meth:`~struphy.pic.base.Particles.draw_markers`.
        Moreover, it sets the corresponding columns for "w0", "s0" and "weights" in the markers array.
        If :attr:`~struphy.pic.base.Particles.control_variate` is True, the background :attr:`~struphy.pic.base.Particles.f0` is subtracted.
        """

        assert self.domain is not None, 'A domain is needed to initialize weights.'

        # compute s0 and save at vdim + 4
        self.sampling_density = self.s0(*self.phasespace_coords.T)

        # load distribution function (with given parameters or default parameters)
        bckgr_type = self.bckgr_params['type']
        bp_copy = copy.deepcopy(self.bckgr_params)
        pp_copy = copy.deepcopy(self.pert_params)

        if not isinstance(bckgr_type, list):
            bckgr_type = [bckgr_type]

        # For delta-f set markers only as perturbation
        if self.marker_params['type'] == 'delta_f':
            for fi in bckgr_type:
                # Take out background by setting its density to zero
                if fi in bp_copy:
                    bp_copy[fi]['n'] = 0.
                else:
                    bp_copy[fi] = {'n': 0.}

        # Get the initialization function and pass the correct arguments
        self._f_init = None
        for fi in bckgr_type:
            if fi[-2] == '_':
                fi_type = fi[:-2]
            else:
                fi_type = fi

            if self._f_init is None:
                self._f_init = getattr(maxwellians, fi_type)(
                    maxw_params=bp_copy[fi],
                    pert_params=pp_copy,
                    mhd_equil=self.mhd_equil,
                    braginskii_equil=self.braginskii_equil
                )
            else:
                self._f_init = self._f_init + getattr(maxwellians, fi_type)(
                    maxw_params=bp_copy[fi],
                    pert_params=pp_copy,
                    mhd_equil=self.mhd_equil,
                    braginskii_equil=self.braginskii_equil
                )
        # TODO: allow for different perturbations for different backgrounds

        # evaluate initial distribution function
        f_init = self.f_init(*self.f_coords.T)

        # if f_init is vol-form, transform to 0-form
        if self.pforms[0] == 'vol':
            f_init /= self.domain.jacobian_det(self.markers_wo_holes)

        if self.pforms[1] == 'vol':
            f_init /= self.f_init.velocity_jacobian_det(*self.f_jacobian_coords.T)

        # compute w0 and save at vdim + 5
        self.weights0 = f_init / self.sampling_density

        # compute weights
        if self._control_variate:
            self.update_weights()
        else:
            self.weights = self.weights0

    def update_weights(self):
        """
        Applies the control variate method, i.e. updates the time-dependent marker weights 
        according to the algorithm in :ref:`control_var`.
        The background :attr:`~struphy.pic.base.Particles.f0` is used for this.
        """

        f0 = self.f0(*self.f_coords.T)

        # if f_init is vol-form, transform to 0-form
        if self.pforms[0] == 'vol':
            f0 /= self.domain.jacobian_det(self.markers_wo_holes)

        if self.pforms[1] == 'vol':
            f0 /= self.f0.velocity_jacobian_det(*self.f_jacobian_coords.T)

        self.weights = self.weights0 - f0/self.sampling_density

    def binning(self, components, bin_edges):
        r""" Computes full-f and delta-f distribution functions via marker binning in logical space.
        Numpy's histogramdd is used, following the algorithm outlined in :ref:`binning`. 

        Parameters
        ----------
        components : list[bool]
            List of length 3 + vdim; an entry is True if the direction in phase space is to be binned.

        bin_edges : list[array]
            List of bin edges (resolution) having the length of True entries in components.

        Returns
        -------
        f_slice : array-like
            The reconstructed full-f distribution function.

        df_slice : array-like
            The reconstructed delta-f distribution function.
        """

        assert np.count_nonzero(components) == len(bin_edges)

        # volume of a bin
        bin_vol = 1.
        for be in bin_edges:
            bin_vol *= be[1] - be[0]

        # extend components list to number of columns of markers array
        _n = len(components)
        slicing = components + [False] * (self.markers.shape[1] - _n)

        # compute weights of histogram:
        _weights0 = self.weights0
        _weights = self.weights

        _weights /= self.domain.jacobian_det(self.markers_wo_holes)
        # _weights /= self.velocity_jacobian_det(*self.phasespace_coords.T)

        _weights0 /= self.domain.jacobian_det(self.markers_wo_holes)
        # _weights0 /= self.velocity_jacobian_det(*self.phasespace_coords.T)

        f_slice = np.histogramdd(self.markers_wo_holes[:, slicing],
                                 bins=bin_edges,
                                 weights=_weights0)[0]

        df_slice = np.histogramdd(self.markers_wo_holes[:, slicing],
                                  bins=bin_edges,
                                  weights=_weights)[0]

        f_slice /= self.n_mks * bin_vol
        df_slice /= self.n_mks * bin_vol

        return f_slice, df_slice

    def show_distribution_function(self, components, bin_edges):
        """
        1D and 2D plots of slices of the distribution function via marker binning.
        This routine is mainly for de-bugging.

        Parameters
        ----------
        components : list[bool]
            List of length 6 giving the directions in phase space in which to bin.

        bin_edges : list[array]
            List of bin edges (resolution) having the length of True entries in components.
        """

        import matplotlib.pyplot as plt

        n_dim = np.count_nonzero(components)

        assert n_dim == 1 or n_dim == 2, f'Distribution function can only be shown in 1D or 2D slices, not {n_dim}.'

        f_slice, df_slice = self.binning(components, bin_edges)

        bin_centers = [bi[:-1] + (bi[1] - bi[0])/2 for bi in bin_edges]

        labels = {
            0: r'$\eta_1$', 1: r'$\eta_2$', 2: r'$\eta_3$',
            3: '$v_1$', 4: '$v_2$', 5: '$v_3$'
        }
        indices = np.nonzero(components)[0]

        if n_dim == 1:
            plt.plot(bin_centers[0], f_slice)
            plt.xlabel(labels[indices[0]])
        else:
            plt.contourf(bin_centers[0], bin_centers[1], df_slice.T, levels=20)
            plt.colorbar()
            # plt.axis('square')
            plt.xlabel(labels[indices[0]])
            plt.ylabel(labels[indices[1]])

        plt.show()

    def apply_kinetic_bc(self, newton=False):
        """
        Apply boundary conditions to markers that are outside of the logical unit cube.

        Parameters
        ----------
        newton : bool
            Whether the shift due to boundary conditions should be computed 
            for a Newton step or for a strandard (explicit or Picard) step.
        """

        for axis, bc in enumerate(self.marker_params['bc']['type']):

            # determine particles outside of the logical unit cube
            self._is_outside_right[:] = self.markers[:, axis] > 1.
            self._is_outside_left[:] = self.markers[:, axis] < 0.

            self._is_outside_right[self.holes] = False
            self._is_outside_left[self.holes] = False

            self._is_outside[:] = np.logical_or(
                self._is_outside_right, self._is_outside_left)

            # indices or particles that are outside of the logical unit cube
            outside_inds = np.nonzero(self._is_outside)[0]

            # apply boundary conditions
            if bc == 'remove':

                if self.marker_params['bc']['remove']['boundary_transfer']:
                    outside_inds = self.boundary_transfer(self._is_outside)

                if self.marker_params['bc']['remove']['particle_refilling']:
                    outside_inds = self.particle_refilling(self._is_outside)

                self._markers[outside_inds, :-1] = -1.

                self._n_lost_markers += len(outside_inds)

            elif bc == 'periodic':
                self.markers[outside_inds, axis] = \
                    self.markers[outside_inds, axis] % 1.

                # set shift for alpha-weighted mid-point computation
                outside_right_inds = np.nonzero(self._is_outside_right)[0]
                outside_left_inds = np.nonzero(self._is_outside_left)[0]
                if newton:
                    self.markers[outside_right_inds,
                                self.bufferindex + 3 + self.vdim + axis] += 1.
                    self.markers[outside_left_inds,
                                self.bufferindex + 3 + self.vdim + axis] += -1.
                else:
                    self.markers[:, self.bufferindex + 3 + self.vdim + axis] = 0.
                    self.markers[outside_right_inds,
                                self.bufferindex + 3 + self.vdim + axis] = 1.
                    self.markers[outside_left_inds,
                                self.bufferindex + 3 + self.vdim + axis] = -1.

            elif bc == 'reflect':
                reflect(self.markers, self.domain.args_domain,
                        outside_inds, axis)

            else:
                raise NotImplementedError('Given bc_type is not implemented!')

    def boundary_transfer(self, is_outside):
        """
        Still draft. ONLY valid for the poloidal geometry with AdhocTorus equilibrium (eta1: clamped r-direction, eta2: periodic theta-direction). 

        When particles reach the inner boundary circle, transfer them to the opposite poloidal angle of the same magnetic flux surface.

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

        # phi_boundary_transfer = phi_loss - 2*q(r_loss)*theta_loss
        r_loss = self._markers[transfer_inds, 0] * \
            (1. - self._domain.params_map['a1']
             ) + self._domain.params_map['a1']

        self._markers[transfer_inds, 2] -= 2 * \
            self._mhd_equil.q_r(r_loss)*self._markers[transfer_inds, 1]

        # theta_boudary_transfer = - theta_loss
        self._markers[transfer_inds, 1] = 1. - self.markers[transfer_inds, 1]

        # mark the particle as done for multiple step pushers
        self._markers[transfer_inds, 11] = -1.

        is_outside[transfer_inds] = False
        outside_inds = np.nonzero(is_outside)[0]

        return outside_inds

    def particle_refilling(self, is_outside):
        """
        Still draft. ONLY valid for the poloidal geometry with AdhocTorus equilibrium (eta1: clamped r-direction, eta2: periodic theta-direction). 

        When particles reach the outter boundary of the poloidal plane, refills them to the opposite poloidal angle of the same magnetic flux surface.

        Parameters
        ----------
        """
        # sorting out particles which are outside of the poloidal plane
        smaller_than_rmin = self.markers[:, 0] > 1.

        # exclude holes
        smaller_than_rmin[self.holes] = False

        # indices or particles that are outside of the poloidal plane
        transfer_inds = np.nonzero(smaller_than_rmin)[0]

        self._markers[transfer_inds, 0] = 1. - 1e-8

        # phi_boundary_transfer = phi_loss - 2*q(r_loss)*theta_loss
        r_loss = self._markers[transfer_inds, 0] * \
            (1. - self._domain.params_map['a1']
             ) + self._domain.params_map['a1']

        self._markers[transfer_inds, 2] -= 2 * \
            self._mhd_equil.q_r(r_loss)*self._markers[transfer_inds, 1]

        # theta_boudary_transfer = - theta_loss
        self._markers[transfer_inds, 1] = 1. - self.markers[transfer_inds, 1]

        # mark the particle as done for multiple step pushers
        self._markers[transfer_inds, 11] = -1.

        is_outside[transfer_inds] = False
        outside_inds = np.nonzero(is_outside)[0]

        return outside_inds


def sendrecv_determine_mtbs(markers,
                            holes,
                            domain_decomp,
                            mpi_rank,
                            vdim,
                            buffer_index,
                            alpha: list | tuple | np.ndarray = (1., 1., 1.)):
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

        alpha : list | tuple
            For i=1,2,3 the sorting is according to alpha[i]*markers[:, i] + (1 - alpha[i])*markers[:, buffer_idx + i].
            alpha[i] must be between 0 and 1. 

        buffer_index : int
            The buffer index of the markers array.

    Returns
    -------
        markers_to_be_sent : array[float]
            Markers of shape (n_send, :) to be sent.

        hole_inds_after_send : array[int]
            Indices of empty columns in markers after send.

        sorting_etas : array[float]
            Eta-values of shape (n_send, :) according to which the sorting is performed.
    """
    # position that determines the sorting (including periodic shift of boundary conditions)
    if not isinstance(alpha, np.ndarray):
        alpha = np.array(alpha, dtype=float)
    assert alpha.size == 3
    assert np.all(alpha >= 0.) and np.all(alpha <= 1.)
    bi = buffer_index
    sorting_etas = np.mod(alpha*(markers[:, :3]
                                 + markers[:, bi + 3 + vdim:bi + 3 + vdim + 3])
                          + (1. - alpha)*markers[:, bi:bi + 3], 1.)

    # check which particles are on the current process domain
    is_on_proc_domain = np.logical_and(
        sorting_etas > domain_decomp[mpi_rank, 0::3],
        sorting_etas < domain_decomp[mpi_rank, 1::3])

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

    return markers_to_be_sent, hole_inds_after_send, sorting_etas[send_inds]


def sendrecv_get_destinations(markers_to_be_sent, sorting_etas, domain_decomp, mpi_size):
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
            sorting_etas > domain_decomp[i, 0::3],
            sorting_etas < domain_decomp[i, 1::3])

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
