from abc import ABCMeta

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

    Parameters
    ----------
    name : str
        Name of particle species.

    n_cols : int
        Number of columns (attributes) for each marker.

    **params : dict
        Marker parameters (defaults must be checked in the child classes).    
    """

    def __init__(self, name: str, n_cols: int, **params):

        self._name = name
        self._params = params

        # Assume full-f if type is not in parameters
        if 'type' in params.keys():
            if params['type'] == 'control_variate':
                self._use_control_variate = True
            else:
                self._use_control_variate = False
        else:
            self._use_control_variate = False

        self._domain_decomp = params['domain_array']

        assert params['comm'] is not None
        self._mpi_comm = params['comm']
        self._mpi_size = params['comm'].Get_size()
        self._mpi_rank = params['comm'].Get_rank()

        # number of cells on current process
        n_cells_loc = np.prod(
            self._domain_decomp[self._mpi_rank, 2::3], dtype=int)

        # total number of cells
        n_cells = np.sum(
            np.prod(self._domain_decomp[:, 2::3], axis=1, dtype=int))

        # number of markers to load on each process (depending on relative domain size)
        if params['ppc'] is not None:
            assert isinstance(params['ppc'], int)
            ppc = params['ppc']
            Np = ppc*n_cells
        else:
            Np = params['Np']
            assert isinstance(Np, int)
            ppc = Np/n_cells

        Np = int(Np)
        assert Np >= self._mpi_size

        n_mks_load = np.zeros(self._mpi_size, dtype=int)
        self._mpi_comm.Allgather(np.array([int(ppc*n_cells_loc)]), n_mks_load)

        # add deviation from Np to rank 0
        n_mks_load[0] += Np - np.sum(n_mks_load)

        # check if all markers are there
        assert np.sum(n_mks_load) == Np
        self._n_mks = Np

        # initialize markers array (3 x positions, 3 x velocities, weight, ...) with eps send/receive buffer
        n_mks_load_loc = n_mks_load[self._mpi_rank]

        markers_size = round(
            n_mks_load_loc*(1 + 1/np.sqrt(n_mks_load_loc) + params['eps']))

        self._markers = np.zeros((markers_size, n_cols), dtype=float)

        n_mks_load_cum_sum = np.cumsum(n_mks_load)

        loading_params = params['loading']

        # load markers from external .hdf5 file
        if loading_params['type'] == 'external':

            if self._mpi_rank == 0:
                file = h5py.File(loading_params['dir_markers'], 'r')

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
            if loading_params['type'] == 'pseudo_random':

                np.random.seed(loading_params['seed'])

                for i in range(self._mpi_size):
                    temp = np.random.rand(n_mks_load[i], 6)

                    if i == self._mpi_rank:
                        self._markers[:n_mks_load_loc, :6] = temp
                        break

                del temp

            # 2. plain sobol numbers with skip of first 1000 numbers
            elif loading_params['type'] == 'sobol_standard':

                self._markers[:n_mks_load_loc, :6] = sobol_seq.i4_sobol_generate(
                    6, n_mks_load_loc, 1000 + (n_mks_load_cum_sum - n_mks_load)[self._mpi_rank])

            # 3. symmetric sobol numbers in all 6 dimensions with skip of first 1000 numbers
            elif loading_params['type'] == 'sobol_antithetic':

                temp_markers = sobol_seq.i4_sobol_generate(
                    6, n_mks_load_loc//64, 1000 + (n_mks_load_cum_sum - n_mks_load)[self._mpi_rank]//64)

                sampling.set_particles_symmetric_3d_3v(
                    temp_markers, self._markers)

            # 4. Wrong specification
            else:
                raise ValueError(
                    'Specified particle loading method does not exist!')

            # inversion of Gaussian in velocity space
            for i in range(3):
                self._markers[:n_mks_load_loc, i + 3] = sp.erfinv(
                    2*self._markers[:n_mks_load_loc, i + 3] - 1)*loading_params['moments'][i + 3] + loading_params['moments'][i]

        # fill holes in markers array with -1
        self._markers[n_mks_load_loc:] = -1.

        # set markers ID in last column
        self._markers[:n_mks_load_loc, -1] = (n_mks_load_cum_sum - n_mks_load)[
            self._mpi_rank] + np.arange(n_mks_load_loc, dtype=float)

        # set specific initial condition for some particles
        if 'initial' in loading_params:
            specific_markers = loading_params['initial']

            counter = 0
            for i in range(len(specific_markers)):
                if i == int(self._markers[counter, -1]):

                    for j in range(6):
                        if specific_markers[i][j] is not None:
                            self._markers[counter, j] = specific_markers[i][j]

                    counter += 1

        # number of holes and markers on process
        self._holes = self._markers[:, 0] == -1.
        self._n_holes_loc = np.count_nonzero(self._holes)
        self._n_mks_loc = self._markers.shape[0] - self._n_holes_loc

        # load sampling density s3 (normalized to 1 in logical space!)
        Maxwellian6DUniform = getattr(analytical, 'Maxwellian6DUniform')

        self._s3 = Maxwellian6DUniform(n=1.,
                                       ux=loading_params['moments'][0],
                                       uy=loading_params['moments'][1],
                                       uz=loading_params['moments'][2],
                                       vthx=loading_params['moments'][3],
                                       vthy=loading_params['moments'][4],
                                       vthz=loading_params['moments'][5])

        # check if all particle positions are inside the unit cube [0, 1]^3
        assert np.all(~self._holes[:n_mks_load_loc]) and np.all(
            self._holes[n_mks_load_loc:])

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
        """ Total number of markers at loading stage.
        """
        return self._n_mks

    @property
    def n_mks_loc(self):
        """ Number of markers on process (without holes).
        """
        return self._n_mks_loc

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
    def s3(self):
        """ Sampling density function for markers (3-form, normalized to 1, constant moments).
        """
        return self._s3

    def s0(self, eta1, eta2, eta3, vx, vy, vz, domain, remove_holes=True):
        """ 
        Sampling density transformed from 3-form to 0-form (division by Jacobian determinant).

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        vx, vy, vz : array_like
            Velocity evaluation points.

        domain : struphy.geometry.domains
            Mapping info for evaluating metric coefficients.

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        """
        return domain.transform(self.s3(eta1, eta2, eta3, vx, vy, vz), self.markers, kind='3_to_0', remove_outside=remove_holes)

    def mpi_sort_markers(self, do_test=False):
        """ 
        Sorts markers according to domain decomposition.

        Parameters
        ----------
        do_test : bool
            Check if all markers are on the right process after sorting.
        """

        self.comm.Barrier()

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

            #print(self.mpi_rank, all_on_right_proc)
            assert all_on_right_proc

        self.comm.Barrier()

        #print(self.mpi_rank, self._n_mks_loc)

    def initialize_weights(self, fun_params, domain, bckgr_params=None):
        """
        Computes w0 = f0(t=0, eta(t=0), v(t=0)) / s0(t=0, eta(t=0), v(t=0)) from the initial
        distribution function and sets the corresponding columns for w0, s0 and weights in markers array.
        For the control variate method, the background is subtracted.

        Parameters
        ----------
        fun_params : dict
            Dictionary of the form {type : class_name, class_name : params_dict} defining the initial condition.

        domain : struphy.geometry.domains
            Mapping info for evaluating metric coefficients.

        bckgr_params : dict (optional)
            Dictionary of the form {type : class_name, class_name : params_dict} defining the background.
        """
        if self._use_control_variate:
            assert bckgr_params is not None, 'When control variate is used, background parameters must be given!'

        # compute s0
        self._markers[~self._holes, 7] = self.s0(*self._markers[:, :6].T,
                                                 domain)

        # load distribution function (with given parameters or default parameters)
        fun_name = fun_params['type']

        if fun_name in fun_params:
            f_init = getattr(analytical, fun_name)(**fun_params[fun_name])
        else:
            f_init = getattr(analytical, fun_name)()

        # compute w0
        self._markers[~self._holes, 8] = f_init(
            *self.markers_wo_holes[:, :6].T)/self.markers_wo_holes[:, 7]

        # set weights
        if self._use_control_variate:
            fun_name = bckgr_params['type']

            if fun_name in bckgr_params:
                f_bckgr = getattr(analytical, fun_name)(
                    **bckgr_params[fun_name])
            else:
                f_bckgr = getattr(analytical, fun_name)()

            self._markers[~self._holes, 6] = self.markers_wo_holes[:, 8] - \
                f_bckgr(*self.markers_wo_holes[:, :6].T) / \
                self.markers_wo_holes[:, 7]
        else:
            self._markers[~self._holes, 6] = self.markers_wo_holes[:, 8]

    def update_weights(self, f0):
        """
        Updates the marker weights according to w0 - control*f0(eta, v)/s0, where control=True or control=False.

        Parameters
        ----------
        f0 : callable
            The distribution function used as a control variate. Is called as f0(eta1, eta2, eta3, vx, vy, vz).
        """

        if self._use_control_variate:
            self._markers[~self._holes, 6] = self.markers_wo_holes[:, 8] - \
                f0(*self.markers_wo_holes[:, :6].T)/self.markers_wo_holes[:, 7]

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


class Particles6D(Particles):
    """
    A class for initializing particles in models that use the full 6D phase space.

    Parameters
    ----------
    name : str
        Name of the particle species.

    **params : dict
        Parameters for markers.
    """

    def __init__(self, name, **params):

        params_default = {'type': 'full_f',
                          'ppc': None,
                          'Np': 3,
                          'eps': .25,
                          'bc_type': ['periodic', 'periodic', 'periodic'],
                          'loading': {'type': 'pseudo:random', 'seed': 1234, 'dir_particles': None, 'moments': [0., 0., 0., 1., 1., 1.]},
                          'comm': None,
                          'domain_array': None
                          }

        params = set_defaults(params, params_default)

        super().__init__(name, 16, **params)


class Particles5D(Particles):
    """
    A class for initializing particles in guiding-center, drift-kinetic or gyro-kinetic models that use the 5D phase space.

         0     1     2                3                  4       
    guiding center position   parallel velocity   magnetic moment           

    Parameters
    ----------
    name : str
        Name of the particle species.

    params_markers : dict
        Parameters under key-word markers in the parameter file.

    domain_decomp : array[float]
        2d array of shape (comm_size, 6) defining the domain of each process.

    comm : Intracomm
        MPI communicator from mpi4py.MPI.Intracomm.
    """

    def __init__(self, name, params_markers, domain_decomp, comm):

        super().__init__(name, params_markers, domain_decomp, comm, 25)

    def save_magnetic_moment(self, derham, absB0):
        r"""
        Calculate magnetic moment of each particles :math:`\mu = \frac{m v_\perp^2}{2B}` and asign it into markers[:,4].
        """
        from struphy.pic.utilities_kernels import eval_magnetic_moment

        absB0.update_ghost_regions()

        # save the calculated magnetic moments in markers[:,4]
        T1, T2, T3 = derham.Vh_fem['0'].knots

        eval_magnetic_moment(self._markers,
                             np.array(derham.p), T1, T2, T3,
                             np.array(derham.Vh['0'].starts),
                             absB0._data)

    def save_magnetic_energy(self, derham, PB):
        r"""
        Calculate magnetic field energy at each particles' position and asign it into markers[:,5].
        """
        from struphy.pic.utilities_kernels import eval_magnetic_energy

        PB.update_ghost_regions()

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


def apply_kinetic_bc(markers, holes, domain, bc_type, comm):
    """
    Apply boundary conditions to markers that are outside of the logical unit cube.

    Parameters
    ----------
        markers : array[float]
            The markers array to which the boundary conditions shall be applied. Positions are the first three columns.

        holes : array[float]
            1d array of same length as number of rows of markers stating whether a row in markers is a hole or not.

        domain : struphy.geometry.domains
            All things mapping.

        bc_type : list[str]
            Kinetic boundary conditions in each direction.

        comm : Intracomm
            MPI communicator from mpi4py.MPI.Intracomm.
    """
    comm.Barrier()

    for axis, bc in enumerate(bc_type):

        # sorting out particles outside of the logical unit cube
        is_outside_cube = np.logical_or(markers[:, axis] > 1.,
                                        markers[:, axis] < 0.)

        # exclude holes
        is_outside_cube[holes] = False

        # indices or particles that are outside of the logical unit cube
        outside_inds = np.nonzero(is_outside_cube)[0]

        # apply boundary conditions
        if bc == 'remove':
            markers[outside_inds, :-1] = -1.

        elif bc == 'periodic':
            markers[outside_inds, axis] = (markers[outside_inds, axis]) % 1.

        elif bc == 'reflect':
            reflect(markers, *domain.args_map, outside_inds, axis)

        else:
            raise NotImplementedError('Given bc_type is not implemented!')

    comm.Barrier()
