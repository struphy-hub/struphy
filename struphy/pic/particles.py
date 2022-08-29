import numpy as np
import h5py
import scipy.special as sp

from struphy.pic import sampling, sobol_seq
from struphy.initial.initialize import KineticPerturbation


class Particles6D:
    """
    A class for initializing particles in models that use the full 6D phase space.

    Parameters
    ----------
        name : str
            Name of the particle species.

        domain: struphy.geometry.domains
            All things mapping.

        domain_array : array[float]
            2d array of shape (comm_size, 6) defining the domain of each process.

        params : dict
            Parameters under key-word markers in the parameter file.

        comm : Intracomm
            MPI communicator from mpi4py.MPI.Intracomm.
    """

    def __init__(self, name, params_markers, domain, domain_array, comm):

        self._name = name
        self._params = params_markers
        self._domain = domain
        self._domain_array = domain_array

        self._mpi_comm = comm
        self._mpi_size = comm.Get_size()
        self._mpi_rank = comm.Get_rank()

        # number of cells on current process
        n_cells_1 = int(self.domain_array[self.mpi_rank, 2])
        n_cells_2 = int(self.domain_array[self.mpi_rank, 5])
        n_cells_3 = int(self.domain_array[self.mpi_rank, 8])

        n_cells = n_cells_1*n_cells_2*n_cells_3

        # number of markers to load on each process (depending on relative size)
        n_mks_load = np.zeros(self.mpi_size, dtype=int)

        # gather n_mks_load info in order to load same markers for arbitrary number of processes (see below)
        self.comm.Allgather(
            np.array(params_markers['ppc']*n_cells), n_mks_load)

        # total number of cells and markers
        n_cells_all = 0

        for i in range(self.mpi_size):
            n_cells_1 = int(self.domain_array[i, 2])
            n_cells_2 = int(self.domain_array[i, 5])
            n_cells_3 = int(self.domain_array[i, 8])

            n_cells_all += n_cells_1*n_cells_2*n_cells_3

        self._n_mks = params_markers['ppc']*n_cells_all

        # initialize particle array (3 x positions, 3 x velocities and weight) with 25% send/receive buffer
        self._n_mks_loc_with_holes = round(
            n_mks_load[self.mpi_rank]*(1 + 1/np.sqrt(n_mks_load[self.mpi_rank]) + 0.25))

        self._markers = np.empty((self.n_mks_loc_with_holes, 9), dtype=float)

        # load markers from external .hdf5 file
        if self.params['loading']['type'] == 'external':

            n_mks_load_cum_sum = np.cumsum(n_mks_load)

            if self.mpi_rank == 0:
                file = h5py.File(self.params['loading']['dir_markers'], 'r')

                self._markers[:n_mks_load_cum_sum[0], :
                              ] = file['markers'][:n_mks_load_cum_sum[0], :]

                for i in range(1, self.mpi_size):
                    self.comm.Send(
                        file['markers'][n_mks_load_cum_sum[i - 1]:n_mks_load_cum_sum[i], :], dest=i, tag=123)

                file.close()
            else:
                recvbuf = np.zeros((n_mks_load[self.mpi_rank], 9), dtype=float)
                self.comm.Recv(recvbuf, source=0, tag=123)
                self._markers[:n_mks_load[self.mpi_rank], :] = recvbuf

        # load fresh markers
        else:

            # 1. standard random number generator (pseudo-random)
            if self.params['loading']['type'] == 'pseudo_random':

                np.random.seed(self.params['loading']['seed'])

                for i in range(self.mpi_size):
                    temp = np.random.rand(n_mks_load[i], 6)

                    if i == self.mpi_rank:
                        self._markers[:n_mks_load[i], :6] = temp
                        break

                del temp

            # 2. plain sobol numbers with skip of first 1000 numbers
            elif self.params['loading']['type'] == 'sobol_standard':

                n_mks_load_cum_sum = np.cumsum(n_mks_load)

                self._markers[:, :6] = sobol_seq.i4_sobol_generate(
                    6, n_mks_load[self.mpi_rank], 1000 + (np.cumsum(n_mks_load) - n_mks_load)[self.mpi_rank])

            # 3. symmetric sobol numbers in all 6 dimensions with skip of first 1000 numbers
            elif self._params['loading']['type'] == 'sobol_antithetic':

                n_mks_load_cum_sum = np.cumsum(n_mks_load)

                temp_markers = sobol_seq.i4_sobol_generate(
                    6, n_mks_load[self.mpi_rank]//64, 1000 + (np.cumsum(n_mks_load) - n_mks_load)[self.mpi_rank]//64)

                sampling.set_particles_symmetric_6d(
                    temp_markers, self._markers, self.n_mks_loc_with_holes)

            # 4. Wrong specification
            else:
                raise ValueError(
                    'Specified particle loading method does not exist!')

            # inversion of Gaussian in velocity space
            self._markers[:n_mks_load[self.mpi_rank], 3] = sp.erfinv(
                2*self._markers[:n_mks_load[self.mpi_rank], 3] - 1)*self.params['loading']['moms_params'][4] + self.params['loading']['moms_params'][1]
            self._markers[:n_mks_load[self.mpi_rank], 4] = sp.erfinv(
                2*self._markers[:n_mks_load[self.mpi_rank], 4] - 1)*self.params['loading']['moms_params'][5] + self.params['loading']['moms_params'][2]
            self._markers[:n_mks_load[self.mpi_rank], 5] = sp.erfinv(
                2*self._markers[:n_mks_load[self.mpi_rank], 5] - 1)*self.params['loading']['moms_params'][6] + self.params['loading']['moms_params'][3]

        # compute initial sampling density s0 at particle positions
        self._markers[:n_mks_load[self.mpi_rank], 7] = self.s0(
            self.markers[:n_mks_load[self.mpi_rank], :3], self.markers[:n_mks_load[self.mpi_rank], 3:6])

        # fill buffer in markers array with -1
        self._markers[n_mks_load[self.mpi_rank]:, :] = -1.

        # check if all particle positions are inside the unit cube [0, 1]^3
        assert not (np.any(self._markers[:n_mks_load[self.mpi_rank], :3] >= 1.) or np.any(
            self._markers[:n_mks_load[self.mpi_rank], :3] <= 0.))

        # number of markers on process
        self._n_mks_loc = np.count_nonzero(~(self.markers[:, 0] == -1.))

    @property
    def name(self):
        """Name of the kinetic species in DATA container."""
        return self._name

    @property
    def params(self):
        """Parameters for particle loading."""
        return self._params

    @property
    def n_mks(self):
        """Total number of markers (sum over n_mks_loc)."""
        return self._n_mks

    @property
    def n_mks_loc(self):
        """Number of markers on process (without holes)."""
        return self._n_mks_loc

    @property
    def n_mks_loc_with_holes(self):
        """Number of rows of self.markers (=number of marker slots on process, with holes)."""
        return self._n_mks_loc_with_holes

    @property
    def markers(self):
        """Numpy array of particle information. The i-th row holds the i-th marker's info:
        3 x positions, 3 x velocities, weights, s0 and w0."""
        return self._markers

    @property
    def domain(self):
        """Mapping from logical to physical space."""
        return self._domain

    @property
    def domain_array(self):
        """Array containing domain decomposition information."""
        return self._domain_array

    @property
    def comm(self):
        """MPI communicator."""
        return self._mpi_comm

    @property
    def mpi_rank(self):
        """Rank of current process."""
        return self._mpi_rank

    @property
    def mpi_size(self):
        """Number of MPI processes."""
        return self._mpi_size

    def s3(self, v):
        """
        Gaussian velocity distribution for sampling markers (normalized to 1, constant moments). 
        
        Parameters
        ----------
            v : array[float]
                2d array of shape (Np, 3).
        """

        assert len(v.shape) == 2

        vthx = self.params['loading']['moms_params'][4]
        vthy = self.params['loading']['moms_params'][5]
        vthz = self.params['loading']['moms_params'][6]

        Gx = np.exp(-(v[:, 0] - self.params['loading']['moms_params'][1])**2 / vthx**2) / vthx / np.sqrt(np.pi)
        Gy = np.exp(-(v[:, 1] - self.params['loading']['moms_params'][2])**2 / vthy**2) / vthy / np.sqrt(np.pi)
        Gz = np.exp(-(v[:, 2] - self.params['loading']['moms_params'][3])**2 / vthz**2) / vthz / np.sqrt(np.pi)
    
        return Gx*Gy*Gz

    def s0(self, eta, v):
        """Sampling distribution trasformed to 0-form."""
        assert eta.shape == v.shape

        s3_markers = self.s3(v)

        return self.domain.transform(s3_markers, eta[:, 0], eta[:, 1], eta[:, 2], '3_to_0', flat_eval=True)

    def send_recv_markers(self):
        """
        Sorts markers according to domain decomposition.
        """

        # create new markers_to_be_sent array and make corresponding holes in markers array
        markers_to_be_sent, holes = sendrecv_determine_mtbs(
            self._markers, self.domain_array, self.mpi_rank)

        # determine where to send markers_to_be_sent
        send_info, send_list = sendrecv_get_destinations(
            markers_to_be_sent, self.domain_array, self.mpi_size)

        # transpose send_info
        recv_info = sendrecv_all_to_all(send_info, self.comm)

        # send and receive markers
        sendrecv_markers(send_list, recv_info, holes,
                         self._markers, self.comm)

        # new number of markers on process
        self._n_mks_loc = np.count_nonzero(~(self.markers[:, 0] == -1.))

    def show_logical(self, save_dir=None):
        """
        Plots the markers on current process on the logical domain in the eta1-eta2-plane at eta3=0.

        Parameters
        ----------
            save_dir : string (optional)
                if given, the figure is saved at the given directory save_dir.
        """

        import matplotlib.pyplot as plt

        # find "true" particles in markers array (without holes)
        true_markers = ~(self.markers[:, 0] == -1.)

        plt.scatter(self.markers[true_markers, 0],
                    self.markers[true_markers, 1], s=1, color='b')

        # plot domain decomposition
        for i in range(self.mpi_size):

            e1 = np.linspace(self.domain_array[i, 0], self.domain_array[i, 1], int(
                self.domain_array[i, 2]) + 1)
            e2 = np.linspace(self.domain_array[i, 3], self.domain_array[i, 4], int(
                self.domain_array[i, 5]) + 1)

            E1, E2 = np.meshgrid(e1, e2, indexing='ij')

            # eta1-isolines
            first_line = plt.plot(E1[0, :], E2[0, :])

            for j in range(e1.size):
                plt.plot(E1[j, :], E2[j, :], color=first_line[0].get_color())

            # eta2-isolines
            for k in range(e2.size):
                plt.plot(E1[:, k], E2[:, k], color=first_line[0].get_color())

        plt.axis('square')

        plt.xlabel('$\eta_1$')
        plt.ylabel('$\eta_2$')

        if save_dir is not None:
            plt.savefig(save_dir)
        else:
            plt.show()

    def show_physical(self, save_dir=None):
        """
        Plots the particles on current process on the logical domain in the eta1-eta2-plane at eta3=0.

        Parameters
        ----------
            save_dir : string (optional)
                if given, the figure is saved at the given directory save_dir.
        """

        import matplotlib.pyplot as plt

        # find "true" particles in markers array (without holes)
        true_markers = ~(self.markers[:, 0] == -1.)

        X = self.domain.evaluate(self.markers[true_markers, 0], self.markers[true_markers, 1], np.zeros(
            self.markers[true_markers, 1].size, dtype=float), 'x', 'flat')
        Y = self.domain.evaluate(self.markers[true_markers, 0], self.markers[true_markers, 1], np.zeros(
            self.markers[true_markers, 1].size, dtype=float), 'y', 'flat')

        plt.scatter(X, Y, s=1, color='b')

        # plot domain decomposition
        for i in range(self.mpi_size):

            e1 = np.linspace(self.domain_array[i, 0], self.domain_array[i, 1], int(
                self.domain_array[i, 2]) + 1)
            e2 = np.linspace(self.domain_array[i, 3], self.domain_array[i, 4], int(
                self.domain_array[i, 5]) + 1)

            X = self.domain.evaluate(e1, e2, 0., 'x')
            Y = self.domain.evaluate(e1, e2, 0., 'y')

            # eta1-isolines
            first_line = plt.plot(X[0, :], Y[0, :])

            for j in range(e1.size):
                plt.plot(X[j, :], Y[j, :], color=first_line[0].get_color())

            # eta2-isolines
            for k in range(e2.size):
                plt.plot(X[:, k], Y[:, k], color=first_line[0].get_color())

        plt.axis('square')

        plt.xlabel('$x$')
        plt.ylabel('$y$')

        if save_dir is not None:
            plt.savefig(save_dir)
        else:
            plt.show()

    def initialize_weights(self, background_params, perturb_params):
        """
        Computes w0=f0(t=0, eta(t=0), v(t=0))/s0(t=0, eta(t=0), v(t=0)) from the initial conditions.

        Parameters
        ----------
            TODO
        """

        f_init = KineticPerturbation(background_params, perturb_params)

        # compute w0
        self._markers[:, 8] = f_init(
            self.markers[:, :3], self.markers[:, 3:6]) / self.markers[:, 7]

        # set weights
        self._markers[:, 6] = self.markers[:, 8]

    def update_weights(self, kinetic_background, use_control):
        """
        Computes the weight update w0-control*f0_back(eta, v)/s0, where control=True or control=False.

        Parameters
        ----------
            TODO
        """

        if use_control:
            self._markers[:, 6] = self._markers[:, 8] - kinetic_background.fh0_eq(
                self.markers[:, 0], self.markers[:, 1], self.markers[:, 2], self.markers[:, 3], self.markers[:, 4], self.markers[:, 5])
        else:
            self._markers[:, 6] = self.markers[:, 8]


class Particles5D:
    """
    A class for initializing particles in drift-kinetic or gyro-kinetic models that use the 5D phase space.

    Parameters
    ----------
        name : str
            Name of the particle species.

        domain: struphy.geometry.domains
            All things mapping.

        domain_array : array[float]
            2d array of shape (comm_size, 6) defining the domain of each process.

        params : dict
            Parameters under key-word markers in the parameter file.

        comm : Intracomm
            MPI communicator from mpi4py.MPI.Intracomm.
    """

    def __init__(self, name, params_markers, domain, domain_array, comm):

        # TODO
        pass


def sendrecv_determine_mtbs(markers, domain_array, mpi_rank):
    """
    Determine which markers have to be sent from current process and put them in a new array. 
    Corresponding entries in markers array become holes and are therfore set to -1.
    This can be done purely with numpy functions (fast, vectorized).

    Parameters
    ----------
        markers : array[float]
            Local markers array of shape (n_mks_loc_with_holes, 9). 

        domain_array : array[float]
            2d array of shape (mpi_size, 9) defining the domain of each process.

        mpi_rank : int
            Rank of MPI process.

    Returns
    -------
        markers_to_be_sent : array[float]
            Markers of shape (n_send, 9) to be sent.

        holes : array[int]
            Indices of empty columns in markers after send.
    """

    # check which particles are in a certain interval (e.g. the process domain)
    conds = np.logical_and(
        markers[:, :3] > domain_array[mpi_rank, ::3], markers[:, :3] < domain_array[mpi_rank, 1::3])
    conds_m1 = markers[:, 0] == -1.

    # to stay on the current process, all three columns must be True
    stay = np.all(conds, axis=1)

    # True values can stay on the process, False must be sent, already empty rows (-1) cannot be sent
    holes = np.nonzero(~stay)[0]
    send_inds = np.nonzero(~stay[~conds_m1])[0]

    # New array for sending particles.
    # TODO: do not create new array, but just return send_inds?
    # Careful: just markers[send_ids] already creates a new array in memory
    markers_to_be_sent = markers[send_inds]

    # set new holes to -1
    markers[send_inds] = -1.

    return markers_to_be_sent, holes


def sendrecv_get_destinations(markers_to_be_sent, domain_array, mpi_size):
    """
    Determine to which process particles have to be sent.

    Parameters
    ----------
        markers_to_be_sent : array[float]
            Markers of shape (n_send, 9) to be sent.

        domain_array : array[float]
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
            markers_to_be_sent[:, :3] > domain_array[i, ::3], markers_to_be_sent[:, :3] < domain_array[i, 1::3])

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


def sendrecv_markers(send_list, recv_info, holes, markers, comm):
    """
    Use non-blocking communication. In-place modification of markers

    Parameters
    ----------
        send_list : list[array]
            Markers to be sent to i-th process.

        recv_info : array[int]
            Amount of markers to be received from i-th process.

        holes : array[int]
            Indices of empty rows in markers array after send.

        markers : array[float]
            Local markers array of shape (n_mks_loc_with_holes, 9). 

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

            recvbufs += [np.zeros((N_recv, 9), dtype=float)]
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

                    markers[holes[first_hole[i] +
                                  np.arange(recv_info[i])]] = recvbufs[i]

                    test_reqs.pop()
                    reqs[i] = None
