import copy
import os
from abc import ABCMeta, abstractmethod

import h5py
import numpy as np
import scipy.special as sp
from mpi4py import MPI
from mpi4py.MPI import Intracomm
from sympy.ntheory import factorint

from struphy.fields_background import equils
from struphy.fields_background.base import FluidEquilibrium, FluidEquilibriumWithB
from struphy.fields_background.equils import set_defaults
from struphy.fields_background.projected_equils import ProjectedFluidEquilibrium
from struphy.geometry.base import Domain
from struphy.io.output_handling import DataContainer
from struphy.kinetic_background import maxwellians
from struphy.pic import sampling_kernels, sobol_seq
from struphy.pic.pushing.pusher_args_kernels import MarkerArguments
from struphy.pic.pushing.pusher_utilities_kernels import reflect
from struphy.pic.sorting_kernels import (
    flatten_index,
    initialize_neighbours,
    put_particles_in_boxes_kernel,
    reassign_boxes,
    sort_boxed_particles,
)
from struphy.pic.sph_eval_kernels import (
    box_based_evaluation,
    box_based_evaluation_3d,
    naive_evaluation,
    naive_evaluation_3d,
    periodic_distance,
)


class Particles(metaclass=ABCMeta):
    """
    Base class for particle species.

    The marker information is stored in a 2D numpy array,
    see `Tutorial on PIC data structures <https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_08_data_structures.html#PIC-data-structures>`_.

    Parameters
    ----------
    name : str
        Name of particle species.

    Np : int
        Number of particles.

    bc : list
        Either 'remove', 'reflect', 'periodic' or 'refill' in each direction.

    loading : str
        Drawing of markers; either 'pseudo_random', 'sobol_standard',
        'sobol_antithetic', 'external' or 'restart'.

    eps : float
        Size of buffer (as fraction of total size, default=.25) in markers array.

    type : str
        Either 'full_f' (default), 'control_variate' or 'delta_f'.

    loading_params : dict
        Parameterts for loading, see defaults below.

    bc_refill : list
        Either 'inner' or 'outer'.

    sorting_params : dict
        Sorting boxes size parameters.

    comm : mpi4py.MPI.Intracomm
        MPI communicator (within a clone if domain cloning is used, otherwise MPI.COMM_WORLD)

    inter_comm : mpi4py.MPI.Intracomm
        MPI communicator (between clones if domain cloning is used, otherwise None)

    domain : Domain
        Struphy domain object.

    equil : FluidEquilibrium
        Struphy fluid equilibrium object.

    bckgr_params : dict
        Kinetic background parameters.

    pert_params : dict
        Kinetic perturbation parameters.

    domain_array : np.array
        Holds info on the domain decomposition, see :class:`~struphy.feec.psydac_derham.Derham`.

    ppc : int
        Particles per cell (optional).

    projected_equil : ProjectedFluidEquilibrium
        Struphy fluid equilibrium projected into a discrete Derham complex.

    """

    def __init__(
        self,
        name: str,
        Np: int,
        bc: list,
        loading: str,
        *,
        type: str = "full_f",
        eps: float = 0.25,
        loading_params: dict = None,
        bc_refill: str = None,
        sorting_params: dict = None,
        equation_params: dict = None,
        comm: Intracomm = None,
        inter_comm: Intracomm = None,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
        bckgr_params: dict = None,
        pert_params: dict = None,
        domain_array: np.ndarray = None,
        ppc: int = None,
        projected_equil: ProjectedFluidEquilibrium = None,
    ):
        self._name = name

        assert type in (
            "full_f",
            "control_variate",
            "delta_f",
        )
        assert loading in (
            "pseudo_random",
            "sobol_standard",
            "sobol_antithetic",
            "external",
            "restart",
        )
        for bci in bc:
            assert bci in ("remove", "reflect", "periodic", "refill")
            if bci == "reflect":
                assert domain is not None, "Reflecting boundary conditions require a domain."

        if bc_refill is not None:
            for bc_refilli in bc_refill:
                assert bc_refilli in ("outer", "inner")

        self._type = type
        self._loading = loading
        self._bc = bc
        self._bc_refill = bc_refill
        self._eps = eps

        # check for mpi communicator
        self._mpi_comm = comm
        if self.mpi_comm is None:
            self._mpi_size = 1
            self._mpi_rank = 0
        else:
            self._mpi_size = self.mpi_comm.Get_size()
            self._mpi_rank = self.mpi_comm.Get_rank()

        # check for domain cloning
        self._inter_comm = inter_comm
        if self.inter_comm is None:
            self._Nclones = 1
            self._clone_rank = 0
        else:
            self._Nclones = self.inter_comm.Get_size()
            self._clone_rank = self.inter_comm.Get_rank()

        # domain decomposition for MPI
        if domain_array is None:
            self._domain_decomp = self._get_domain_decomp()
        else:
            self._domain_decomp = domain_array

        # total number of cells (equal to mpi_size if no grid)
        n_cells = np.sum(np.prod(self._domain_decomp[:, 2::3], axis=1, dtype=int))

        # total number of markers (Np) and particles per cell (ppc)
        if ppc is None:
            self._Np = int(Np)
            self._ppc = Np / n_cells
        else:
            self._ppc = ppc
            self._Np = int(ppc * n_cells)

        assert self.Np >= self.mpi_size

        self._domain = domain
        self._equil = equil
        self._projected_equil = projected_equil

        # background and perturbations
        if bckgr_params is None:
            bckgr_params = {"Maxwellian3D": {}, "pforms": [None, None]}

        # background p-form description in [eta, v] (None means 0-form, "vol" means volume form -> divide by det)
        self._pforms = bckgr_params.pop("pforms", [None, None])

        self._bckgr_params = bckgr_params
        self._pert_params = pert_params

        # default loading parameters
        loading_params_default = {
            "seed": 1234,
            "dir_particles": None,
            "moments": None,
            "spatial": "uniform",
            "initial": None,
        }

        self._loading_params = set_defaults(
            loading_params,
            loading_params_default,
        )

        if self.loading_params["moments"] is None:
            self.auto_sampling_params()

        self._equation_params = equation_params

        # create marker array
        self.create_marker_array()

        # allocate arrays for sorting
        n_rows = self.markers.shape[0]
        self._is_outside_right = np.zeros(n_rows, dtype=bool)
        self._is_outside_left = np.zeros(n_rows, dtype=bool)
        self._is_outside = np.zeros(n_rows, dtype=bool)

        # check if control variate
        self._control_variate = self.type == "control_variate"

        # set background function
        self._f0 = None
        for fi, maxw_params in bckgr_params.items():
            if fi[-2] == "_":
                fi_type = fi[:-2]
            else:
                fi_type = fi

            # SPH case: f0 is set to n0
            if self.loading_params["moments"] == "degenerate":
                eq_class = getattr(equils, fi_type)
                eq_class.domain = self.domain
                if self._f0 is None:
                    self._f0 = lambda eta: eq_class.n0(*eta)
                else:
                    self._f0 = self._f0 + (lambda eta: eq_class.n0(*eta))
            # default case
            else:
                if self._f0 is None:
                    self._f0 = getattr(maxwellians, fi_type)(
                        maxw_params=maxw_params,
                        equil=equil,
                    )
                else:
                    self._f0 = self._f0 + getattr(maxwellians, fi_type)(
                        maxw_params=maxw_params,
                        equil=equil,
                    )

        # set coordinates of the background distribution
        if self.loading_params["moments"] != "degenerate" and self.f0.coords == "constants_of_motion":
            # Particles6D
            if self.vdim == 3:
                assert self.n_cols_diagnostics >= 7, (
                    f"In case of the distribution '{self.f0}' with Particles6D, minimum number of n_cols_diagnostics is 7!"
                )

                self._f_coords_index = self.index["com"]["6D"]
                self._f_jacobian_coords_index = self.index["pos+energy"]["6D"]

            # Particles5D
            elif self.vdim == 2:
                assert self.n_cols_diagnostics >= 3, (
                    f"In case of the distribution '{self.f0}' with Particles5D, minimum number of n_cols_diagnostics is 3!"
                )

                self._f_coords_index = self.index["com"]["5D"]
                self._f_jacobian_coords_index = self.index["pos+energy"]["5D"]

        if self.loading_params["moments"] == "degenerate":
            self._f_coords_index = self.index["coords"]
            self._f_jacobian_coords_index = self.index["coords"]
        else:
            if self.f0.coords != "constants_of_motion":
                self._f_coords_index = self.index["coords"]
                self._f_jacobian_coords_index = self.index["coords"]

        # Marker arguments for kernels
        self._args_markers = MarkerArguments(
            self.markers,
            self.vdim,
            self.first_pusher_idx,
        )

        # Have at least 3 spare places in markers array
        assert self.args_markers.first_free_idx + 2 < self.n_cols - 1, (
            f"{self.args_markers.first_free_idx + 2} is not smaller than {self.n_cols - 1 = }; not enough columns in marker array !!"
        )

        # initialize the sorting
        self._sorting_params = sorting_params
        self._initialized_sorting = False
        if sorting_params is not None:
            self._sorting_boxes = self.SortingBoxes(
                sorting_params["nx"],
                sorting_params["ny"],
                sorting_params["nz"],
                sorting_params["communicate"],
                self._markers,
                eps=sorting_params["eps"],
            )
            if sorting_params["communicate"]:
                self._get_neighbouring_proc()

            self._initialized_sorting = True
            self._argsort_array = np.zeros(self._markers.shape[0], dtype=int)

        # create buffers for mpi_sort_markers
        self._sorting_etas = np.zeros(self._markers.shape, dtype=float)
        self._is_on_proc_domain = np.zeros((self._markers.shape[0], 3), dtype=bool)
        self._can_stay = np.zeros(self._markers.shape[0], dtype=bool)
        self._reqs = [None] * self.mpi_size
        self._recvbufs = [None] * self.mpi_size
        self._send_to_i = [None] * self.mpi_size
        self._send_list = [None] * self.mpi_size

    @classmethod
    @abstractmethod
    def default_bckgr_params(cls):
        """Dictionary holding the minimal information of the default background.

        Must contain at least a keyword 'type' with corresponding value a valid choice of background.
        """
        pass

    @abstractmethod
    def svol(self, eta1, eta2, eta3, *v):
        r"""Marker sampling distribution function :math:`s^\textrm{vol}` as a volume form, see :ref:`monte_carlo`."""
        pass

    @abstractmethod
    def s0(self, eta1, eta2, eta3, *v, remove_holes=True):
        r"""Marker sampling distribution function :math:`s^0` as 0-form, see :ref:`monte_carlo`."""
        pass

    @property
    @abstractmethod
    def n_cols(self):
        """Number of columns in the :attr:`~struphy.pic.base.Particles.markers` array."""
        pass

    @property
    @abstractmethod
    def vdim(self):
        """Dimension of the velocity space."""
        pass

    @property
    @abstractmethod
    def first_diagnostics_idx(self):
        """Starting buffer marker index number for diagnostics."""
        pass

    @property
    @abstractmethod
    def first_pusher_idx(self):
        """Starting buffer marker index number for pusher."""
        pass

    @property
    @abstractmethod
    def n_cols_diagnostics(self):
        """Number of the diagnostics columns."""
        pass

    @property
    @abstractmethod
    def n_cols_aux(self):
        """Number of the auxiliary columns."""
        pass

    @property
    def kinds(self):
        """Name of the class."""
        return self.__class__.__name__

    @property
    def name(self):
        """Name of the kinetic species in DATA container."""
        return self._name

    @property
    def type(self):
        """Compzation of weights: 'full_f', 'control_variate' or 'delta_f'."""
        return self._type

    @property
    def loading(self):
        """Type of particle loading."""
        return self._loading

    @property
    def bc(self):
        """List of particle boundary conditions in each direction."""
        return self._bc

    @property
    def bc_refill(self):
        """How to re-enter particles if bc is 'refill'."""
        return self._bc_refill

    @property
    def Np(self):
        """Total number of markers/particles."""
        return self._Np

    @property
    def ppc(self):
        """Particles per cell (=Np if no grid is present)."""
        return self._ppc

    @property
    def eps(self):
        """Relative size of buffer in markers array."""
        return self._eps

    @property
    def mpi_comm(self):
        """MPI communicator."""
        return self._mpi_comm

    @property
    def mpi_size(self):
        """Number of MPI processes."""
        return self._mpi_size

    @property
    def mpi_rank(self):
        """Rank of current process."""
        return self._mpi_rank

    @property
    def inter_comm(self):
        """MPI communicator between clones."""
        return self._inter_comm

    @property
    def Nclones(self):
        """Number of clones."""
        return self._Nclones

    @property
    def clone_rank(self):
        """Clone rank of current process."""
        return self._clone_rank

    @property
    def bckgr_params(self):
        """Kinetic background parameters."""
        return self._bckgr_params

    @property
    def pert_params(self):
        """Kinetic perturbation parameters."""
        return self._pert_params

    @property
    def loading_params(self):
        """Parameters for marker loading."""
        return self._loading_params

    @property
    def sorting_params(self):
        """Sorting boxes size parameters."""
        return self._sorting_params

    @property
    def equation_params(self):
        """Parameters appearing in model equation due to Struphy normalization."""
        return self._equation_params

    @property
    def f_init(self):
        assert hasattr(self, "_f_init"), AttributeError(
            'The method "initialize_weights" has not yet been called.',
        )
        return self._f_init

    @property
    def f0(self):
        assert hasattr(self, "_f0"), AttributeError(
            "No background distribution available, maybe this is a full-f model?",
        )
        return self._f0

    @property
    def control_variate(self):
        """Boolean for whether to use the :ref:`control_var` during time stepping."""
        return self._control_variate

    @property
    def domain_decomp(self):
        """
        A 2d array[float] of shape (comm.Get_size(), 9). The row index denotes the process number and
        for n=0,1,2:

            * domain_decomp[i, 3*n + 0] holds the LEFT domain boundary of process i in direction eta_(n+1).
            * domain_decomp[i, 3*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).
            * domain_decomp[i, 3*n + 2] holds the number of cells of process i in direction eta_(n+1).
        """
        return self._domain_decomp

    @property
    def n_mks(self):
        """Total number of markers."""
        return self._n_mks

    @property
    def n_mks_loc(self):
        """Number of markers on process (without holes)."""
        return self._n_mks_loc

    @property
    def n_mks_load(self):
        """Array of number of markers on each process at loading stage"""
        return self._n_mks_load

    @property
    def markers(self):
        """2D numpy array holding the marker information, including holes.
        The i-th row holds the i-th marker info.

        ===== ============== ======================= ======= ====== ====== ========== === === ===
        index  | 0 | 1 | 2 | | 3 | ... | 3+(vdim-1)|  3+vdim 4+vdim 5+vdim >=6+vdim   ... -2  -1
        ===== ============== ======================= ======= ====== ====== ========== === === ===
        value position (eta)    velocities           weight   s0     w0      other    ... box ID
        ===== ============== ======================= ======= ====== ====== ========== === === ===

        The column indices referring to different attributes can be obtained from
        :attr:`~struphy.pic.base.Particles.index`.
        """
        return self._markers

    @property
    def holes(self):
        """Array of booleans stating if an entry in the markers array is a hole or not."""
        return self._holes

    @property
    def ghost_particles(self):
        """Array of booleans stating if an entry in the markers array is a ghost particle or not."""
        return self._ghost_particles

    @property
    def n_holes_loc(self):
        """Number of holes on process (= marker.shape[0] - n_mks_loc)."""
        return self._n_holes_loc

    @property
    def markers_wo_holes(self):
        """Array holding the marker information, excluding holes. The i-th row holds the i-th marker info."""
        return self.markers[~self.holes]

    @property
    def markers_wo_holes_and_ghost(self):
        """Array holding the marker information, excluding holes. The i-th row holds the i-th marker info."""
        return self.markers[~np.logical_or(self.holes, self.ghost_particles)]

    @property
    def domain(self):
        """From :mod:`struphy.geometry.domains`."""
        return self._domain

    @property
    def equil(self):
        """From :mod:`struphy.fields_background.equils`."""
        return self._equil

    @property
    def projected_equil(self):
        """MHD equilibrium projected on 3d Derham sequence with commuting projectors."""
        return self._projected_equil

    @property
    def lost_markers(self):
        """Array containing the last infos of removed markers"""
        return self._lost_markers

    @property
    def n_lost_markers(self):
        """Number of removed particles."""
        return self._n_lost_markers

    @property
    def index(self):
        """Dict holding the column indices referring to specific marker parameters (coordinates)."""
        out = {}
        out["pos"] = slice(0, 3)  # positions
        out["vel"] = slice(3, 3 + self.vdim)  # velocities
        out["coords"] = slice(0, 3 + self.vdim)  # phasespace_coords
        out["com"] = {}
        out["com"]["6D"] = slice(12, 15)  # constants of motion (Particles6D)
        out["com"]["5D"] = slice(8, 11)  # constants of motion (Particles5D)
        out["pos+energy"] = {}
        out["pos+energy"]["6D"] = slice(9, 13)  # positions + energy
        out["pos+energy"]["5D"] = list(range(0, 3)) + [8]  # positions + energy
        out["weights"] = 3 + self.vdim  # weights
        out["s0"] = 4 + self.vdim  # sampling_density
        out["w0"] = 5 + self.vdim  # weights0
        out["box"] = -2  # sorting box index
        out["ids"] = -1  # marker_inds
        return out

    @property
    def positions(self):
        """Array holding the marker positions in logical space, excluding holes. The i-th row holds the i-th marker info."""
        return self.markers[~np.logical_or(self.holes, self.ghost_particles), self.index["pos"]]

    @positions.setter
    def positions(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, 3)
        self._markers[~np.logical_or(self.holes, self.ghost_particles), self.index["pos"]] = new

    @property
    def velocities(self):
        """Array holding the marker velocities in logical space, excluding holes. The i-th row holds the i-th marker info."""
        return self.markers[~self.holes, self.index["vel"]]

    @velocities.setter
    def velocities(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, self.vdim)
        self._markers[~self.holes, self.index["vel"]] = new

    @property
    def phasespace_coords(self):
        """Array holding the marker velocities in logical space, excluding holes. The i-th row holds the i-th marker info."""
        return self.markers[~self.holes, self.index["coords"]]

    @phasespace_coords.setter
    def phasespace_coords(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, 3 + self.vdim)
        self._markers[~self.holes, self.index["coords"]] = new

    @property
    def weights(self):
        """Array holding the current marker weights, excluding holes. The i-th row holds the i-th marker info."""
        return self.markers[~self.holes, self.index["weights"]]

    @weights.setter
    def weights(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[~self.holes, self.index["weights"]] = new

    @property
    def sampling_density(self):
        """Array holding the current marker 0form sampling density s0, excluding holes. The i-th row holds the i-th marker info."""
        return self.markers[~self.holes, self.index["s0"]]

    @sampling_density.setter
    def sampling_density(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[~self.holes, self.index["s0"]] = new

    @property
    def weights0(self):
        """Array holding the initial marker weights, excluding holes. The i-th row holds the i-th marker info."""
        return self.markers[~self.holes, self.index["w0"]]

    @weights0.setter
    def weights0(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[~self.holes, self.index["w0"]] = new

    @property
    def marker_ids(self):
        """Array holding the marker id's on the current process."""
        return self.markers[~self.holes, self.index["ids"]]

    @marker_ids.setter
    def marker_ids(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[~self.holes, self.index["ids"]] = new

    @property
    def pforms(self):
        """Tuple of size 2; each entry must be either "vol" or None, defining the p-form
        (space and velocity, respectively) of f_init.
        """
        return self._pforms

    @property
    def spatial(self):
        """Drawing particles uniformly on the unit cube('uniform') or on the disc('disc')"""
        return self._spatial

    @property
    def f_coords_index(self):
        """Dict holding the column indices referring to coords of the distribution fuction."""
        return self._f_coords_index

    @property
    def f_jacobian_coords_index(self):
        """Dict holding the column indices referring to coords of the velocity jacobian determinant of the distribution fuction."""
        return self._f_jacobian_coords_index

    @property
    def f_coords(self):
        """Coordinates of the distribution function."""
        return self.markers[~self.holes, self.f_coords_index]

    @f_coords.setter
    def f_coords(self, new):
        assert isinstance(new, np.ndarray)
        self.markers[~self.holes, self.f_coords_index] = new

    @property
    def args_markers(self):
        """Collection of mandatory arguments for pusher kernels."""
        return self._args_markers

    @property
    def f_jacobian_coords(self):
        """Coordinates of the velocity jacobian determinant of the distribution fuction."""
        if isinstance(self.f_jacobian_coords_index, list):
            return self.markers[np.ix_(~self.holes, self.f_jacobian_coords_index)]
        else:
            return self.markers[~self.holes, self.f_jacobian_coords_index]

    @f_jacobian_coords.setter
    def f_jacobian_coords(self, new):
        assert isinstance(new, np.ndarray)
        if isinstance(self.f_jacobian_coords_index, list):
            self.markers[
                np.ix_(
                    ~self.holes,
                    self.f_jacobian_coords_index,
                )
            ] = new
        else:
            self.markers[~self.holes, self.f_jacobian_coords_index] = new

    @property
    def sorting_boxes(self):
        return self._sorting_boxes

    def _get_domain_decomp(self):
        """
        Compute domain decomposition for mesh-less methods (no Derham object).

        Returns
        -------
        dom_arr : np.ndarray
            A 2d array of shape (#MPI processes, 9). The row index denotes the process rank. The columns are for n=0,1,2:
                - arr[i, 3*n + 0] holds the LEFT domain boundary of process i in direction eta_(n+1).
                - arr[i, 3*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).
                - arr[i, 3*n + 2] holds the number of cells of process i in direction eta_(n+1).
        """

        dom_arr = np.zeros((self.mpi_size, 9), dtype=float)

        # factorize mpi size
        factors = factorint(self.mpi_size)
        factors_vec = []
        for fac, multiplicity in factors.items():
            for m in range(multiplicity):
                factors_vec += [fac]

        # processes in each direction
        nprocs = [1, 1, 1]
        for m, fac in enumerate(factors_vec):
            mm = m % 3
            nprocs[mm] *= fac

        assert np.prod(nprocs) == self.mpi_size

        # domain decomposition
        breaks = [np.linspace(0.0, 1.0, nproc + 1) for nproc in nprocs]

        # fill domain array
        for n in range(self.mpi_size):
            # determine (ijk box index) corresponding to n (inverse flattening)
            i = n // (nprocs[1] * nprocs[2])
            nn = n % (nprocs[1] * nprocs[2])
            j = nn // nprocs[2]
            k = nn % nprocs[2]

            dom_arr[n, 0] = breaks[0][i]
            dom_arr[n, 1] = breaks[0][i + 1]
            dom_arr[n, 2] = 1
            dom_arr[n, 3] = breaks[1][j]
            dom_arr[n, 4] = breaks[1][j + 1]
            dom_arr[n, 5] = 1
            dom_arr[n, 6] = breaks[2][k]
            dom_arr[n, 7] = breaks[2][k + 1]
            dom_arr[n, 8] = 1

        return dom_arr

    def create_marker_array(self):
        """Create marker array :attr:`~struphy.pic.base.Particles.markers`."""

        # number of cells on current process
        n_cells_loc = np.prod(
            self.domain_decomp[self.mpi_rank, 2::3],
            dtype=int,
        )

        # array of number of markers on each process at loading stage
        self._n_mks_load = np.zeros(self.mpi_size, dtype=int)

        if self.mpi_comm is not None:
            self.mpi_comm.Allgather(
                np.array([int(self.ppc * n_cells_loc)]),
                self._n_mks_load,
            )
        else:
            self._n_mks_load[0] = int(self.ppc * n_cells_loc)

        # add deviation from Np to rank 0
        self._n_mks_load[0] += self.Np - np.sum(self._n_mks_load)

        # check if all markers are there
        assert np.sum(self._n_mks_load) == self.Np
        self._n_mks = self.Np

        # number of markers on the local process at loading stage
        n_mks_load_loc = self._n_mks_load[self._mpi_rank]

        # create markers array (3 x positions, vdim x velocities, weight, s0, w0, ..., ID) with eps send/receive buffer
        n_rows = round(
            n_mks_load_loc * (1 + 1 / np.sqrt(n_mks_load_loc) + self.eps),
        )
        self._markers = np.zeros((n_rows, self.n_cols), dtype=float)

        # create array container (3 x positions, vdim x velocities, weight, s0, w0, ID) for removed markers
        self._n_lost_markers = 0
        self._lost_markers = np.zeros((int(n_rows * 0.5), 10), dtype=float)

    def draw_markers(self, sort: "bool" = True, verbose=True):
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

        An initial sorting will be performed if sort is given as True (default) and sorting_params were given to the init.

        Parameters
        ----------
        sort : Bool
            Wether to sort the particules in boxes after initial drawing (only if sorting params were passed)
            
        verbose : bool
            Show info on screen.
        """

        # number of markers on the local process at loading stage
        n_mks_load_loc = self.n_mks_load[self.mpi_rank]

        # fill holes in markers array with -1 (all holes are at end of array at loading stage)
        self._markers[n_mks_load_loc:] = -1.0

        # number of holes and markers on process
        self._holes = self.markers[:, 0] == -1.0
        self._ghost_particles = self.markers[:, -1] == -2.0
        self._n_holes_loc = np.count_nonzero(self._holes)
        self._n_mks_loc = self.markers.shape[0] - self._n_holes_loc

        # cumulative sum of number of markers on each process at loading stage.
        n_mks_load_cum_sum = np.cumsum(self.n_mks_load)

        if self.mpi_rank == 0 and verbose:
            print("\nMARKERS:")
            print(("name:").ljust(25), self.name)
            print(("Np:").ljust(25), self.Np)
            print(("bc:").ljust(25), self.bc)
            print(("loading:").ljust(25), self.loading)
            print(("type:").ljust(25), self.type)
            print(("domain_decomp[0]:").ljust(25), self.domain_decomp[0])
            print(("ppc:").ljust(25), self.ppc)
            print(("bc_refill:").ljust(25), self.bc_refill)
            print(("sorting_params:").ljust(25), self.sorting_params)

        # load markers from external .hdf5 file
        if self.loading == "external":
            if self.mpi_rank == 0:
                file = h5py.File(
                    self.loading_params["dir_external"],
                    "r",
                )
                print(f"\nLoading markers from file: {file}")

                self._markers[
                    : n_mks_load_cum_sum[0],
                    :,
                ] = file["markers"][: n_mks_load_cum_sum[0], :]

                for i in range(1, self._mpi_size):
                    self._mpi_comm.Send(
                        file["markers"][n_mks_load_cum_sum[i - 1] : n_mks_load_cum_sum[i], :],
                        dest=i,
                        tag=123,
                    )

                file.close()
            else:
                recvbuf = np.zeros(
                    (n_mks_load_loc, self.markers.shape[1]),
                    dtype=float,
                )
                self._mpi_comm.Recv(recvbuf, source=0, tag=123)
                self._markers[:n_mks_load_loc, :] = recvbuf

        # load markers from restart .hdf5 file
        elif self.loading == "restart":
            import struphy.utils.utils as utils

            # Read struphy state file
            state = utils.read_state()

            o_path = state["o_path"]

            if self.loading_params["dir_particles_abs"] is None:
                data_path = os.path.join(
                    o_path,
                    self.loading_params["dir_particles"],
                )
            else:
                data_path = self.loading_params["dir_particles_abs"]

            data = DataContainer(data_path, comm=self.mpi_comm)

            self.markers[:, :] = data.file["restart/" + self.loading_params["key"]][-1, :, :]

        # load fresh markers
        else:
            if self.mpi_rank == 0 and verbose:
                print("\nLoading fresh markers:")
                for key, val in self.loading_params.items():
                    print((key + " :").ljust(25), val)

            # 1. standard random number generator (pseudo-random)
            # TODO: assumes all clones have same number of particles
            if self.loading == "pseudo_random":
                # set seed
                _seed = self.loading_params["seed"]
                if _seed is not None:
                    np.random.seed(_seed)

                # counting integers
                num_loaded_particles_loc = 0  # number of particles alreday loaded (local)
                num_loaded_particles = 0  # number of particles already loaded (each clone)
                chunk_size = 10000  # TODO: number of particle chunk
                total_num_particles_to_load = np.sum(self.n_mks_load)

                while num_loaded_particles < int(total_num_particles_to_load * self.Nclones):
                    # Generate a chunk of random particles
                    num_to_add = min(chunk_size, int(total_num_particles_to_load * self.Nclones) - num_loaded_particles)
                    temp = np.random.rand(num_to_add, 3 + self.vdim)

                    # check which particles are on the current process domain
                    is_on_proc_domain = np.logical_and(
                        temp[:, :3] > self.domain_decomp[self.mpi_rank, 0::3],
                        temp[:, :3] < self.domain_decomp[self.mpi_rank, 1::3],
                    )

                    valid_idx = np.nonzero(np.all(is_on_proc_domain, axis=1))[0]

                    valid_particles = temp[valid_idx]
                    valid_particles = np.array_split(valid_particles, self.Nclones)[self.clone_rank]
                    num_valid = valid_particles.shape[0]

                    # Add the valid particles to the phasespace_coords array
                    self.markers[
                        num_loaded_particles_loc : num_loaded_particles_loc + num_valid,
                        : 3 + self.vdim,
                    ] = valid_particles
                    num_loaded_particles += num_to_add
                    num_loaded_particles_loc += num_valid

                # make sure all particles are loaded
                #assert np.sum(self.n_mks_load) == int(num_loaded_particles / self.Nclones)

                # set new n_mks_load
                self.n_mks_load[self.mpi_rank] = num_loaded_particles_loc
                n_mks_load_loc = num_loaded_particles_loc

                if self.mpi_comm is not None:
                    self.mpi_comm.Allgather(self._n_mks_load[self.mpi_rank], self._n_mks_load)

                n_mks_load_cum_sum = np.cumsum(self.n_mks_load)

                #assert np.sum(self.n_mks_load) == int(num_loaded_particles / self.Nclones)

                # set new holes in markers array to -1
                self.markers[num_loaded_particles_loc:, :] = -1.0
                self.update_holes()

                del temp

            # 2. plain sobol numbers with skip of first 1000 numbers
            elif self.loading == "sobol_standard":
                self.phasespace_coords = sobol_seq.i4_sobol_generate(
                    3 + self.vdim,
                    n_mks_load_loc,
                    1000 + (n_mks_load_cum_sum - self.n_mks_load)[self._mpi_rank],
                )

            # 3. symmetric sobol numbers in all 6 dimensions with skip of first 1000 numbers
            elif self.loading == "sobol_antithetic":
                assert self.vdim == 3, NotImplementedError(
                    '"sobol_antithetic" requires vdim=3 at the moment.',
                )

                temp_markers = sobol_seq.i4_sobol_generate(
                    3 + self.vdim,
                    n_mks_load_loc // 64,
                    1000 + (n_mks_load_cum_sum - self.n_mks_load)[self._mpi_rank] // 64,
                )

                sampling_kernels.set_particles_symmetric_3d_3v(
                    temp_markers,
                    self.markers,
                )

            # 4. Wrong specification
            else:
                raise ValueError(
                    "Specified particle loading method does not exist!",
                )

            # initial velocities - SPH case: v(0) = u(x(0)) for given velocity u(x)
            if self.loading_params["moments"] == "degenerate":
                self._f_init = None
                for fi, params in self.bckgr_params.items():
                    if fi[-2] == "_":
                        fi_type = fi[:-2]
                    else:
                        fi_type = fi

                    if self._f_init is None:
                        self._f_init = getattr(equils, fi_type)(**params)
                    else:
                        self._f_init = self._f_init + getattr(equils, fi_type)(**params)
                self.velocities = np.array(self._f_init.u_xyz(*self.phasespace_coords[:, 0:3].T)).T

            else:
                # inverse transform sampling in velocity space
                u_mean = np.array(self.loading_params["moments"][: self.vdim])
                v_th = np.array(self.loading_params["moments"][self.vdim :])

                # Particles6D: (1d Maxwellian, 1d Maxwellian, 1d Maxwellian)
                if self.vdim == 3:
                    self.velocities = (
                        sp.erfinv(
                            2 * self.velocities - 1,
                        )
                        * np.sqrt(2)
                        * v_th
                        + u_mean
                    )
                # Particles5D: (1d Maxwellian, polar Maxwellian as volume-form)
                elif self.vdim == 2:
                    self._markers[:n_mks_load_loc, 3] = (
                        sp.erfinv(
                            2 * self.velocities[:, 0] - 1,
                        )
                        * np.sqrt(2)
                        * v_th[0]
                        + u_mean[0]
                    )

                    self._markers[:n_mks_load_loc, 4] = (
                        np.sqrt(
                            -1 * np.log(1 - self.velocities[:, 1]),
                        )
                        * np.sqrt(2)
                        * v_th[1]
                        + u_mean[1]
                    )
                elif self.vdim == 0:
                    pass
                else:
                    raise NotImplementedError(
                        "Inverse transform sampling of given vdim is not implemented!",
                    )

            # inversion method for drawing uniformly on the disc
            self._spatial = self.loading_params["spatial"]
            if self._spatial == "disc":
                self._markers[:n_mks_load_loc, 0] = np.sqrt(
                    self._markers[:n_mks_load_loc, 0],
                )
            else:
                assert self._spatial == "uniform", f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.'

            # set markers ID in last column
            self.marker_ids = (n_mks_load_cum_sum - self.n_mks_load)[self._mpi_rank] + np.arange(
                n_mks_load_loc, dtype=float
            )

            # set specific initial condition for some particles
            if self.loading_params["initial"] is not None:
                specific_markers = self.loading_params["initial"]

                counter = 0
                for i in range(len(specific_markers)):
                    if i == int(self.markers[counter, -1]):
                        for j in range(3 + self.vdim):
                            if specific_markers[i][j] is not None:
                                self._markers[
                                    counter,
                                    j,
                                ] = specific_markers[i][j]

                        counter += 1

            # check if all particle positions are inside the unit cube [0, 1]^3
            n_mks_load_loc = self._n_mks_load[self._mpi_rank]

            assert np.all(~self._holes[:n_mks_load_loc]) and np.all(
                self._holes[n_mks_load_loc:],
            )

        if self._initialized_sorting and sort:
            if self.mpi_rank == 0 and verbose:
                print("Sorting the markers after initial draw")
            self.mpi_sort_markers()
            self.do_sort()

    def mpi_sort_markers(
        self,
        apply_bc: bool = True,
        alpha: tuple | list | int | float = 1.0,
        do_test=False,
    ):
        """
        Sorts markers according to MPI domain decomposition.

        Markers are sent to the process corresponding to the alpha-weighted position
        alpha*markers[:, 0:3] + (1 - alpha)*markers[:, first_pusher_idx:first_pusher_idx + 3].

        Periodic boundary conditions are taken into account
        when computing the alpha-weighted position.

        Parameters
        ----------
        appl_bc : bool
            Whether to apply kinetic boundary conditions before sorting.

        alpha : tuple | list | int | float
            For i=1,2,3 the sorting is according to alpha[i]*markers[:, i] + (1 - alpha[i])*markers[:, first_pusher_idx + i].
            If int or float then alpha = (alpha, alpha, alpha). alpha must be between 0 and 1.

        do_test : bool
            Check if all markers are on the right process after sorting.
        """
        self.mpi_comm.Barrier()

        # before sorting, apply kinetic bc
        if apply_bc:
            self.apply_kinetic_bc()

        if isinstance(alpha, int) or isinstance(alpha, float):
            alpha = (alpha, alpha, alpha)

        # create new markers_to_be_sent array and make corresponding holes in markers array
        hole_inds_after_send, send_inds = self.sendrecv_determine_mtbs(alpha=alpha)

        # determine where to send markers_to_be_sent
        send_info = self.sendrecv_get_destinations(send_inds)

        # set new holes in markers array to -1
        self.markers[send_inds] = -1.0

        # transpose send_info
        recv_info = self.sendrecv_all_to_all(send_info)

        # send and receive markers
        self.sendrecv_markers(recv_info, hole_inds_after_send)

        # new holes and new number of holes and markers on process
        self.update_holes()

        # check if all markers are on the right process after sorting
        if do_test:
            all_on_right_proc = np.all(
                np.logical_and(
                    self.positions > self.domain_decomp[self.mpi_rank, 0::3],
                    self.positions < self.domain_decomp[self.mpi_rank, 1::3],
                ),
            )

            assert all_on_right_proc
            # assert self.phasespace_coords.size > 0, f'No particles on process {self.mpi_rank}, please rebalance, aborting ...'

        self.mpi_comm.Barrier()

    def initialize_weights(
        self,
        *,
        bckgr_params=None,
        pert_params=None,
    ):
        r"""
        Computes the initial weights

        .. math::

            w_{k0} := \frac{f^0(t, q_k(t)) }{s^0(t, q_k(t)) } = \frac{f^0(0, q_k(0)) }{s^0(0, q_k(0)) } = \frac{f^0_{\textnormal{in}}(q_{k0}) }{s^0_{\textnormal{in}}(q_{k0}) }

        from the initial distribution function :math:`f^0_{\textnormal{in}}` specified in the parmeter file
        and from the initial volume density :math:`s^n_{\textnormal{vol}}` specified in :meth:`~struphy.pic.base.Particles.draw_markers`.
        Moreover, it sets the corresponding columns for "w0", "s0" and "weights" in the markers array.
        If :attr:`~struphy.pic.base.Particles.control_variate` is True, the background :attr:`~struphy.pic.base.Particles.f0` is subtracted.

        Parameters
        ----------
        bckgr_params : dict
            Kinetic background parameters.

        pert_params : dict
            Kinetic perturbation parameters for initial condition.
        """

        assert self.domain is not None, "A domain is needed to initialize weights."

        # set background paramters
        if bckgr_params is not None:
            self._bckgr_params = bckgr_params

        # set perturbation paramters
        if pert_params is not None:
            self._pert_params = pert_params

        # compute s0 and save at vdim + 4
        self.sampling_density = self.s0(*self.phasespace_coords.T)

        # load distribution function (with given parameters or default parameters)
        bp_copy = copy.deepcopy(self.bckgr_params)
        pp_copy = copy.deepcopy(self.pert_params)

        # Prepare delta-f perturbation parameters
        if self.type == "delta_f":
            for fi in bp_copy:
                if fi[-2] == "_":
                    fi_type = fi[:-2]
                else:
                    fi_type = fi

                if pp_copy is not None:
                    # Set background to zero (if "use_background_n" in perturbation params is set to false or not in keys)
                    if fi in pp_copy:
                        if "use_background_n" in pp_copy[fi]:
                            if not pp_copy[fi]["use_background_n"]:
                                bp_copy[fi]["n"] = 0.0
                        else:
                            bp_copy[fi]["n"] = 0.0
                    else:
                        bp_copy[fi]["n"] = 0.0

        # Get the initialization function and pass the correct arguments
        if self.loading_params["moments"] != "degenerate":
            # In SPH case f_init is set in draw_markers
            self._f_init = None
            for fi, maxw_params in bp_copy.items():
                if fi[-2] == "_":
                    fi_type = fi[:-2]
                else:
                    fi_type = fi

                pert_params = pp_copy
                if pp_copy is not None:
                    if fi in pp_copy:
                        pert_params = pp_copy[fi]

                if self._f_init is None:
                    self._f_init = getattr(maxwellians, fi_type)(
                        maxw_params=maxw_params,
                        pert_params=pert_params,
                        equil=self.equil,
                    )
                else:
                    self._f_init = self._f_init + getattr(maxwellians, fi_type)(
                        maxw_params=maxw_params,
                        pert_params=pert_params,
                        equil=self.equil,
                    )

        # evaluate initial distribution function
        if self.loading_params["moments"] == "degenerate":
            f_init = self.f_init.n0(self.f_coords)

        else:
            f_init = self.f_init(*self.f_coords.T)

        # if f_init is vol-form, transform to 0-form
        if self.pforms[0] == "vol":
            f_init /= self.domain.jacobian_det(self.markers_wo_holes)

        if self.pforms[1] == "vol":
            f_init /= self.f_init.velocity_jacobian_det(
                *self.f_jacobian_coords.T,
            )
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

        # in case of CanonicalMaxwellian, evaluate constants_of_motion
        if self.f0.coords == "constants_of_motion":
            self.save_constants_of_motion()

        f0 = self.f0(*self.f_coords.T)

        # if f_init is vol-form, transform to 0-form
        if self.pforms[0] == "vol":
            f0 /= self.domain.jacobian_det(self.markers_wo_holes)

        if self.pforms[1] == "vol":
            f0 /= self.f0.velocity_jacobian_det(*self.f_jacobian_coords.T)

        self.weights = self.weights0 - f0 / self.sampling_density

    def binning(self, components, bin_edges):
        r"""Computes full-f and delta-f distribution functions via marker binning in logical space.
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
        bin_vol = 1.0
        for be in bin_edges:
            bin_vol *= be[1] - be[0]

        # extend components list to number of columns of markers array
        _n = len(components)
        slicing = components + [False] * (self.markers.shape[1] - _n)

        # compute weights of histogram:
        _weights0 = self.weights0
        _weights = self.weights

        _weights /= self.domain.jacobian_det(self.markers_wo_holes, remove_outside=False)
        # _weights /= self.velocity_jacobian_det(*self.phasespace_coords.T)

        _weights0 /= self.domain.jacobian_det(self.markers_wo_holes, remove_outside=False)
        # _weights0 /= self.velocity_jacobian_det(*self.phasespace_coords.T)

        f_slice = np.histogramdd(
            self.markers_wo_holes[:, slicing],
            bins=bin_edges,
            weights=_weights0,
        )[0]

        df_slice = np.histogramdd(
            self.markers_wo_holes[:, slicing],
            bins=bin_edges,
            weights=_weights,
        )[0]

        # Initialize the total number of markers
        n_mks_tot = np.array([self.n_mks])

        if self.Nclones > 1:
            self.inter_comm.Allreduce(
                MPI.IN_PLACE,
                n_mks_tot,
                op=MPI.SUM,
            )

        f_slice /= n_mks_tot * bin_vol
        df_slice /= n_mks_tot * bin_vol

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

        assert n_dim == 1 or n_dim == 2, f"Distribution function can only be shown in 1D or 2D slices, not {n_dim}."

        f_slice, df_slice = self.binning(components, bin_edges)

        bin_centers = [bi[:-1] + (bi[1] - bi[0]) / 2 for bi in bin_edges]

        labels = {
            0: r"$\eta_1$",
            1: r"$\eta_2$",
            2: r"$\eta_3$",
            3: "$v_1$",
            4: "$v_2$",
            5: "$v_3$",
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

        for axis, bc in enumerate(self.bc):
            # determine particles outside of the logical unit cube
            self._is_outside_right[:] = self.markers[:, axis] > 1.0
            self._is_outside_left[:] = self.markers[:, axis] < 0.0

            self._is_outside_right[self.holes] = False
            self._is_outside_right[self.ghost_particles] = False
            self._is_outside_left[self.holes] = False
            self._is_outside_left[self.ghost_particles] = False

            self._is_outside[:] = np.logical_or(
                self._is_outside_right,
                self._is_outside_left,
            )

            # indices or particles that are outside of the logical unit cube
            outside_inds = np.nonzero(self._is_outside)[0]

            if len(outside_inds) == 0:
                continue

            # apply boundary conditions
            if bc == "remove":
                if self.bc_refill is not None:
                    self.particle_refilling()

                self._markers[self._is_outside, :-1] = -1.0
                self._n_lost_markers += len(np.nonzero(self._is_outside)[0])

            elif bc == "periodic":
                self.markers[outside_inds, axis] = self.markers[outside_inds, axis] % 1.0

                # set shift for alpha-weighted mid-point computation
                outside_right_inds = np.nonzero(self._is_outside_right)[0]
                outside_left_inds = np.nonzero(self._is_outside_left)[0]
                if newton:
                    self.markers[
                        outside_right_inds,
                        self.first_pusher_idx + 3 + self.vdim + axis,
                    ] += 1.0
                    self.markers[
                        outside_left_inds,
                        self.first_pusher_idx + 3 + self.vdim + axis,
                    ] += -1.0
                else:
                    self.markers[
                        :,
                        self.first_pusher_idx + 3 + self.vdim + axis,
                    ] = 0.0
                    self.markers[
                        outside_right_inds,
                        self.first_pusher_idx + 3 + self.vdim + axis,
                    ] = 1.0
                    self.markers[
                        outside_left_inds,
                        self.first_pusher_idx + 3 + self.vdim + axis,
                    ] = -1.0

            elif bc == "reflect":
                self.markers[self._is_outside_left, axis] = 1e-4
                self.markers[self._is_outside_right, axis] = 1 - 1e-4

                reflect(
                    self.markers,
                    self.domain.args_domain,
                    outside_inds,
                    axis,
                )

                self.markers[self._is_outside, self.first_pusher_idx] = -1.0

            else:
                raise NotImplementedError("Given bc_type is not implemented!")

    def auto_sampling_params(self):
        """Automatically determine sampling parameters from the background given"""
        ns = []
        us = []
        vths = []

        for fi, params in self.bckgr_params.items():
            if fi[-2] == "_":
                fi_type = fi[:-2]
            else:
                fi_type = fi

            us.append([])
            vths.append([])

            bckgr = getattr(maxwellians, fi_type)
            default_maxw_params = bckgr.default_maxw_params()

            for key in default_maxw_params:
                if key[0] == "n":
                    if key in params:
                        ns += [params[key]]
                    else:
                        ns += [1.0]

                elif key[0] == "u":
                    if key in params:
                        us[-1] += [params[key]]
                    else:
                        us[-1] += [0.0]

                elif key[0] == "v":
                    if key in params:
                        vths[-1] += [params[key]]
                    else:
                        vths[-1] += [1.0]

        assert len(ns) == len(us) == len(vths)

        ns = np.array(ns)
        us = np.array(us)
        vths = np.array(vths)

        new_moments = []

        new_moments += [*np.mean(us, axis=0)]
        new_moments += [*(np.max(vths, axis=0) + np.max(np.abs(us), axis=0) - np.mean(us, axis=0))]

        self.loading_params["moments"] = new_moments

    def particle_refilling(self):
        r"""
        When particles move outside of the domain, refills them.
        TODO: Currently only valid for HollowTorus geometry with AdhocTorus equilibrium.

        In case of guiding-center orbit, refills particles at the opposite poloidal angle of the same magnetic flux surface.

        .. math::

            \theta_\text{refill} &= - \theta_\text{loss}
            \\
            \phi_\text{refill} &= -2 q(r_\text{loss}) \theta_\text{loss}

        In case of full orbit, refills particles at the same gyro orbit until their guiding-centers are also outside of the domain.
        When their guiding-centers also reach at the boundary, refills them as we did with guiding-center orbit.
        """

        for kind in self.bc_refill:
            # sorting out particles which are out of the domain
            if kind == "inner":
                outside_inds = np.nonzero(self._is_outside_left)[0]
                self.markers[outside_inds, 0] = 1e-4
                r_loss = self._domain.params_map["a1"]

            else:
                outside_inds = np.nonzero(self._is_outside_right)[0]
                self.markers[outside_inds, 0] = 1 - 1e-4
                r_loss = 1.0

            if len(outside_inds) == 0:
                continue

            # in case of Particles6D, do gyro boundary transfer
            if self.vdim == 3:
                gyro_inside_inds = self.gyro_transfer(outside_inds)

                # mark the particle as done for multiple step pushers
                self.markers[outside_inds[gyro_inside_inds], self.first_pusher_idx] = -1.0
                self._is_outside[outside_inds[gyro_inside_inds]] = False

                # exclude particles whose guiding center positions are still inside.
                if len(gyro_inside_inds) > 0:
                    outside_inds = outside_inds[~gyro_inside_inds]

            # do phi boundary transfer = phi_loss - 2*q(r_loss)*theta_loss
            self.markers[outside_inds, 2] -= 2 * self.equil.q_r(r_loss) * self.markers[outside_inds, 1]

            # theta_boudary_transfer = - theta_loss
            self.markers[outside_inds, 1] = 1.0 - self.markers[outside_inds, 1]

            # mark the particle as done for multiple step pushers
            self.markers[outside_inds, self.first_pusher_idx] = -1.0
            self._is_outside[outside_inds] = False

    def gyro_transfer(self, outside_inds):
        r"""Refills particles at the same gyro orbit.
        Their perpendicular velocity directions are also changed accordingly:

        First, refills the particles at the other side of the cross point (between gyro circle and domain boundary),

        .. math::

            \theta_\text{refill} = \theta_\text{gc} - \left(\theta_\text{loss} - \theta_\text{gc} \right) \,.

        Then changes the direction of the perpendicular velocity,

        .. math::

            \vec{v}_{\perp, \text{refill}} = \frac{\vec{\rho}_g}{|\vec{\rho}_g|} \times \vec{b}_0 |\vec{v}_{\perp, \text{loss}}| \,,

        where :math:`\vec{\rho}_g = \vec{x}_\text{refill} - \vec{X}_\text{gc}` is the cartesian radial vector.

        Parameters
        ----------
        outside_inds : np.array (int)
            An array of indices of particles which are outside of the domain.

        Returns
        -------
        out : np.array (bool)
            An array of indices of particles where its guiding centers are outside of the domain.
        """

        # incoming markers must be "Particles6D".
        assert self.vdim == 3

        # TODO: currently assumes periodic boundary condition along poloidal and toroidal angle
        self.markers[outside_inds, 1:3] = self.markers[outside_inds, 1:3] % 1

        v = self.markers[outside_inds, 3:6].T

        # eval cartesian equilibrium magnetic field at the marker positions
        assert isinstance(self.equil, FluidEquilibriumWithB), "Gyro transfer function needs a magnetic background."
        b_cart, xyz = self.equil.b_cart(self.markers[outside_inds, :])

        # calculate magnetic field amplitude and normalized magnetic field
        absB0 = np.sqrt(b_cart[0] ** 2 + b_cart[1] ** 2 + b_cart[2] ** 2)
        norm_b_cart = b_cart / absB0

        # calculate parallel and perpendicular velocities
        v_parallel = np.einsum("ij,ij->j", v, norm_b_cart)
        v_perp = np.cross(norm_b_cart, np.cross(v, norm_b_cart, axis=0), axis=0)
        v_perp_square = np.sqrt(v_perp[0] ** 2 + v_perp[1] ** 2 + v_perp[2] ** 2)

        assert np.all(np.isclose(v_perp, v - norm_b_cart * v_parallel))

        # calculate Larmor radius
        Larmor_r = np.cross(norm_b_cart, v_perp, axis=0) / absB0 * self._epsilon

        # transform cartesian coordinates to logical coordinates
        # TODO: currently only possible with the geomoetry where its inverse map is defined.
        assert hasattr(self.domain, "inverse_map")

        xyz -= Larmor_r

        gc_etas = self.domain.inverse_map(*xyz, bounded=False)

        # gyro transfer
        self.markers[outside_inds, 1] = (gc_etas[1] - (self.markers[outside_inds, 1] - gc_etas[1]) % 1) % 1

        new_xyz = self.domain(self.markers[outside_inds, :])

        # eval cartesian equilibrium magnetic field at the marker positions
        b_cart = self.equil.b_cart(self.markers[outside_inds, :])[0]

        # calculate magnetic field amplitude and normalized magnetic field
        absB0 = np.sqrt(b_cart[0] ** 2 + b_cart[1] ** 2 + b_cart[2] ** 2)
        norm_b_cart = b_cart / absB0

        Larmor_r = new_xyz - xyz
        Larmor_r /= np.sqrt(Larmor_r[0] ** 2 + Larmor_r[1] ** 2 + Larmor_r[2] ** 2)

        new_v_perp = np.cross(Larmor_r, norm_b_cart, axis=0) * v_perp_square

        self.markers[outside_inds, 3:6] = (norm_b_cart * v_parallel).T + new_v_perp.T

        return np.logical_and(1.0 > gc_etas[0], gc_etas[0] > 0.0)

    class SortingBoxes:
        """Boxes used for the sorting of the particles.

        Represented as a 2D array of integers,
        each line of the array corespond to one box,
        and all the non (-1) entries of line i are the particles in the i-th box

        Parameters
        ----------
        nx : int
            number of boxes in the x direction.

        ny : int
            number of boxes in the y direction.

        nz : int
            number of boxes in the z direction.

        communicate : bool
            indicate if the particles in the neighbouring boxes are communicate (via MPI).

        markers : 2D numpy.array
            marker array of the particles.

        box_index : int
            Column index of the particles array to store the box number, counted from
            the end (e.g. -2 for the second-to-last).

        eps : float
            additional buffer space in the size of the boxes"""

        def __init__(
            self,
            nx: "int",
            ny: "int",
            nz: "int",
            communicate: "bool",
            markers: "float[:,:]",
            box_index: "int" = -2,
            eps: "float" = 0.1,
        ):
            assert isinstance(nx, int)
            assert isinstance(ny, int)
            assert isinstance(nz, int)
            self._nx = nx
            self._ny = ny
            self._nz = nz
            self._communicate = communicate
            self._markers = markers
            self._box_index = box_index
            self._eps = eps
            self._set_boxes()
            if self._communicate:
                self._set_boundary_boxes()

        @property
        def nx(self):
            return self._nx

        @property
        def ny(self):
            return self._ny

        @property
        def nz(self):
            return self._nz

        @property
        def communicate(self):
            return self._communicate

        @property
        def box_index(self):
            return self._box_index

        def _set_boxes(self):
            """ "(Re)set the box structure."""
            n_particles = self._markers.shape[0]
            self._n_boxes = (self._nx + 2) * (self._ny + 2) * (self._nz + 2)
            n_box_in = self._nx * self._ny * self._nz
            n_mkr = int(n_particles / n_box_in) + 1
            eps = self._eps
            n_rows = round(
                n_mkr * (1 + 1 / np.sqrt(n_mkr) + eps),
            )
            # cartesian boxes
            self._boxes = np.zeros((self._n_boxes, n_rows), dtype=int)
            self._next_index = np.zeros((self._n_boxes + 1), dtype=int)
            self._cumul_next_index = np.zeros((self._n_boxes + 2), dtype=int)
            self._neighbours = np.zeros((self._n_boxes, 27), dtype=int)
            initialize_neighbours(self._neighbours, self.nx, self.ny, self.nz)
            # exit()
            # A particle on box i only sees particles in boxes that belong to neighbours[i]
            self._swap_line_1 = np.zeros(self._markers.shape[1])
            self._swap_line_2 = np.zeros(self._markers.shape[1])

        def _set_boundary_boxes(self):
            """Gather all the boxes that are part of a boundary"""

            # x boundary
            # negative direction
            self._bnd_boxes_x_m = []
            # positive direction
            self._bnd_boxes_x_p = []
            for j in range(1, self.ny + 1):
                for k in range(1, self.nz + 1):
                    self._bnd_boxes_x_m.append(flatten_index(1, j, k, self.nx, self.ny, self.nz))
                    self._bnd_boxes_x_p.append(flatten_index(self.nx, j, k, self.nx, self.ny, self.nz))

            # y boundary
            # negative direction
            self._bnd_boxes_y_m = []
            # positive direction
            self._bnd_boxes_y_p = []
            for i in range(1, self.nx + 1):
                for k in range(1, self.nz + 1):
                    self._bnd_boxes_y_m.append(flatten_index(i, 1, k, self.nx, self.ny, self.nz))
                    self._bnd_boxes_y_p.append(flatten_index(i, self.ny, k, self.nx, self.ny, self.nz))

            # z boundary
            # negative direction
            self._bnd_boxes_z_m = []
            # positive direction
            self._bnd_boxes_z_p = []
            for i in range(1, self.nx + 1):
                for j in range(1, self.ny + 1):
                    self._bnd_boxes_z_m.append(flatten_index(i, j, 1, self.nx, self.ny, self.nz))
                    self._bnd_boxes_z_p.append(flatten_index(i, j, self.nz, self.nx, self.ny, self.nz))

            # x-y edges
            self._bnd_boxes_x_m_y_m = []
            self._bnd_boxes_x_m_y_p = []
            self._bnd_boxes_x_p_y_m = []
            self._bnd_boxes_x_p_y_p = []
            for k in range(1, self.nz + 1):
                self._bnd_boxes_x_m_y_m.append(flatten_index(1, 1, k, self.nx, self.ny, self.nz))
                self._bnd_boxes_x_m_y_p.append(flatten_index(1, self.ny, k, self.nx, self.ny, self.nz))
                self._bnd_boxes_x_p_y_m.append(flatten_index(self.nx, 1, k, self.nx, self.ny, self.nz))
                self._bnd_boxes_x_p_y_p.append(flatten_index(self.nx, self.ny, k, self.nx, self.ny, self.nz))

            # x-z edges
            self._bnd_boxes_x_m_z_m = []
            self._bnd_boxes_x_m_z_p = []
            self._bnd_boxes_x_p_z_m = []
            self._bnd_boxes_x_p_z_p = []
            for j in range(1, self.ny + 1):
                self._bnd_boxes_x_m_z_m.append(flatten_index(1, j, 1, self.nx, self.ny, self.nz))
                self._bnd_boxes_x_m_z_p.append(flatten_index(1, j, self.nz, self.nx, self.ny, self.nz))
                self._bnd_boxes_x_p_z_m.append(flatten_index(self.nx, j, 1, self.nx, self.ny, self.nz))
                self._bnd_boxes_x_p_z_p.append(flatten_index(self.nx, j, self.nz, self.nx, self.ny, self.nz))

            # y-z edges
            self._bnd_boxes_y_m_z_m = []
            self._bnd_boxes_y_m_z_p = []
            self._bnd_boxes_y_p_z_m = []
            self._bnd_boxes_y_p_z_p = []
            for i in range(1, self.nx + 1):
                self._bnd_boxes_y_m_z_m.append(flatten_index(i, 1, 1, self.nx, self.ny, self.nz))
                self._bnd_boxes_y_m_z_p.append(flatten_index(i, 1, self.nz, self.nx, self.ny, self.nz))
                self._bnd_boxes_y_p_z_m.append(flatten_index(i, self.ny, 1, self.nx, self.ny, self.nz))
                self._bnd_boxes_y_p_z_p.append(flatten_index(i, self.ny, self.nz, self.nx, self.ny, self.nz))

            # corners
            self._bnd_boxes_x_m_y_m_z_m = [flatten_index(1, 1, 1, self.nx, self.ny, self.nz)]
            self._bnd_boxes_x_m_y_m_z_p = [flatten_index(1, 1, self.nz, self.nx, self.ny, self.nz)]
            self._bnd_boxes_x_m_y_p_z_m = [flatten_index(1, self.ny, 1, self.nx, self.ny, self.nz)]
            self._bnd_boxes_x_m_y_p_z_p = [flatten_index(1, self.ny, self.nz, self.nx, self.ny, self.nz)]
            self._bnd_boxes_x_p_y_m_z_m = [flatten_index(self.nx, 1, 1, self.nx, self.ny, self.nz)]
            self._bnd_boxes_x_p_y_m_z_p = [flatten_index(self.nx, 1, self.nz, self.nx, self.ny, self.nz)]
            self._bnd_boxes_x_p_y_p_z_m = [flatten_index(self.nx, self.ny, 1, self.nx, self.ny, self.nz)]
            self._bnd_boxes_x_p_y_p_z_p = [flatten_index(self.nx, self.ny, self.nz, self.nx, self.ny, self.nz)]

    def sort_boxed_particles_numpy(self):
        """Sort the particles by box using numpy.sort."""
        sorting_axis = self._sorting_boxes.box_index
        self._argsort_array[:] = self._markers[:, sorting_axis].argsort()
        self._markers[:, :] = self._markers[self._argsort_array]

    def put_particles_in_boxes(self):
        """Assign the right box to the particles and the list of the particles to each box.
        If sorting_boxes was instantiated with communicate=True, then the particles in the
        neighbouring boxes of neighbours processors or also communicated"""
        self.remove_ghost_particles()

        put_particles_in_boxes_kernel(
            self._markers,
            self.holes,
            self._sorting_boxes.nx,
            self._sorting_boxes.ny,
            self._sorting_boxes.nz,
            self._sorting_boxes._boxes,
            self._sorting_boxes._next_index,
            self.domain_decomp[self.mpi_rank],
        )

        if self.sorting_boxes.communicate:
            self.communicate_boxes()

            reassign_boxes(self._markers, self.holes, self._sorting_boxes._boxes, self._sorting_boxes._next_index)

            self.update_ghost_particles()

    def do_sort(self):
        """Assign the particles to boxes and then sort them."""
        nx = self._sorting_boxes.nx
        ny = self._sorting_boxes.ny
        nz = self._sorting_boxes.nz
        nboxes = (nx + 2) * (ny + 2) * (nz + 2)

        self.put_particles_in_boxes()

        # We could either use numpy routine or kernel to sort
        # Kernel seems to be 3x faster
        # self.sort_boxed_particles_numpy()

        sort_boxed_particles(
            self._markers,
            self._sorting_boxes._swap_line_1,
            self._sorting_boxes._swap_line_2,
            nboxes + 1,
            self._sorting_boxes._next_index,
            self._sorting_boxes._cumul_next_index,
        )

        if self._sorting_boxes.communicate:
            self.update_ghost_particles()

    def remove_ghost_particles(self):
        self.update_ghost_particles()
        new_holes = np.nonzero(self.ghost_particles)
        self._markers[new_holes, :] = -1.0
        self.update_holes()

    def determine_send_markers_box(self):
        """Determine which markers belong to boxes that are at the boundary and put them in a new array"""
        # Faces
        self._markers_x_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_m)
        self._markers_x_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_p)
        self._markers_y_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_y_m)
        self._markers_y_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_y_p)
        self._markers_z_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_z_m)
        self._markers_z_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_z_p)

        # Adjust box number
        self._markers_x_m[:, self._sorting_boxes.box_index] += self._sorting_boxes.nx
        self._markers_x_p[:, self._sorting_boxes.box_index] -= self._sorting_boxes.nx
        self._markers_y_m[:, self._sorting_boxes.box_index] += (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
        self._markers_y_p[:, self._sorting_boxes.box_index] -= (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
        self._markers_z_m[:, self._sorting_boxes.box_index] += (
            (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_z_p[:, self._sorting_boxes.box_index] -= (
            (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )

        # Put last index to -2 to indicate that they should not move
        self._markers_x_m[:, -1] = -2.0
        self._markers_x_p[:, -1] = -2.0
        self._markers_y_m[:, -1] = -2.0
        self._markers_y_p[:, -1] = -2.0
        self._markers_z_m[:, -1] = -2.0
        self._markers_z_p[:, -1] = -2.0

        # Edges x-y
        self._markers_x_m_y_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_m_y_m)
        self._markers_x_m_y_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_m_y_p)
        self._markers_x_p_y_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_p_y_m)
        self._markers_x_p_y_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_p_y_p)

        # Adjust box number
        self._markers_x_m_y_m[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx + (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
        )
        self._markers_x_m_y_p[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx - (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
        )
        self._markers_x_p_y_m[:, self._sorting_boxes.box_index] += (
            -self._sorting_boxes.nx + (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
        )
        self._markers_x_p_y_p[:, self._sorting_boxes.box_index] += (
            -self._sorting_boxes.nx - (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
        )

        # Put first last index to -2 to indicate that they should not move
        self._markers_x_m_y_m[:, -1] = -2.0
        self._markers_x_m_y_p[:, -1] = -2.0
        self._markers_x_p_y_m[:, -1] = -2.0
        self._markers_x_p_y_p[:, -1] = -2.0

        # Edges x-z
        self._markers_x_m_z_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_m_z_m)
        self._markers_x_m_z_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_m_z_p)
        self._markers_x_p_z_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_p_z_m)
        self._markers_x_p_z_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_p_z_p)

        # Adjust box number
        self._markers_x_m_z_m[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx
            + (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_m_z_p[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx
            - (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_p_z_m[:, self._sorting_boxes.box_index] += (
            -self._sorting_boxes.nx
            + (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_p_z_p[:, self._sorting_boxes.box_index] += (
            -self._sorting_boxes.nx
            - (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )

        # Put first last index to -2 to indicate that they should not move
        self._markers_x_m_z_m[:, -1] = -2.0
        self._markers_x_m_z_p[:, -1] = -2.0
        self._markers_x_p_z_m[:, -1] = -2.0
        self._markers_x_p_z_p[:, -1] = -2.0

        # Edges y-z
        self._markers_y_m_z_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_y_m_z_m)
        self._markers_y_m_z_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_y_m_z_p)
        self._markers_y_p_z_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_y_p_z_m)
        self._markers_y_p_z_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_y_p_z_p)

        # Adjust box number
        self._markers_y_m_z_m[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx + 2
        ) * self._sorting_boxes.ny + (self._sorting_boxes.nx + 2) * (
            self._sorting_boxes.ny + 2
        ) * self._sorting_boxes.nz
        self._markers_y_m_z_p[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx + 2
        ) * self._sorting_boxes.ny - (self._sorting_boxes.nx + 2) * (
            self._sorting_boxes.ny + 2
        ) * self._sorting_boxes.nz
        self._markers_y_p_z_m[:, self._sorting_boxes.box_index] += (
            -(self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            + (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_y_p_z_p[:, self._sorting_boxes.box_index] += (
            -(self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            - (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )

        # Put first last index to -2 to indicate that they should not move
        self._markers_y_m_z_m[:, -1] = -2.0
        self._markers_y_m_z_p[:, -1] = -2.0
        self._markers_y_p_z_m[:, -1] = -2.0
        self._markers_y_p_z_p[:, -1] = -2.0

        # Corners
        self._markers_x_m_y_m_z_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_m_y_m_z_m)
        self._markers_x_m_y_m_z_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_m_y_m_z_p)
        self._markers_x_m_y_p_z_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_m_y_p_z_m)
        self._markers_x_m_y_p_z_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_m_y_p_z_p)
        self._markers_x_p_y_m_z_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_p_y_m_z_m)
        self._markers_x_p_y_m_z_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_p_y_m_z_p)
        self._markers_x_p_y_p_z_m = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_p_y_p_z_m)
        self._markers_x_p_y_p_z_p = self.determine_marker_in_box(self._sorting_boxes._bnd_boxes_x_p_y_p_z_p)

        # Adjust box number
        self._markers_x_m_y_m_z_m[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx
            + (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            + (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_m_y_m_z_p[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx
            + (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            - (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_m_y_p_z_m[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx
            - (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            + (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_m_y_p_z_p[:, self._sorting_boxes.box_index] += (
            self._sorting_boxes.nx
            - (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            - (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_p_y_m_z_m[:, self._sorting_boxes.box_index] += (
            -self._sorting_boxes.nx
            + (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            + (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_p_y_m_z_p[:, self._sorting_boxes.box_index] += (
            -self._sorting_boxes.nx
            + (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            - (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_p_y_p_z_m[:, self._sorting_boxes.box_index] += (
            -self._sorting_boxes.nx
            - (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            + (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        self._markers_x_p_y_p_z_p[:, self._sorting_boxes.box_index] += (
            -self._sorting_boxes.nx
            - (self._sorting_boxes.nx + 2) * self._sorting_boxes.ny
            - (self._sorting_boxes.nx + 2) * (self._sorting_boxes.ny + 2) * self._sorting_boxes.nz
        )
        # Put first last index to -2 to indicate that they should not move
        self._markers_x_m_y_m_z_m[:, -1] = -2.0
        self._markers_x_m_y_m_z_p[:, -1] = -2.0
        self._markers_x_m_y_p_z_m[:, -1] = -2.0
        self._markers_x_m_y_p_z_p[:, -1] = -2.0
        self._markers_x_p_y_m_z_m[:, -1] = -2.0
        self._markers_x_p_y_m_z_p[:, -1] = -2.0
        self._markers_x_p_y_p_z_m[:, -1] = -2.0
        self._markers_x_p_y_p_z_p[:, -1] = -2.0

    def determine_marker_in_box(self, list_boxes):
        """Determine the markers that belong to a certain box and put them in an array"""
        indices = []
        for i in list_boxes:
            indices += list(self._sorting_boxes._boxes[i][self._sorting_boxes._boxes[i] != -1])

        indices = np.array(indices, dtype=int)
        markers_in_box = self.markers[indices]
        return markers_in_box

    def get_destinations_box(self):
        """Find the destination proc for the particles to communicate for the box structure."""
        self._send_info_box = np.zeros(self.mpi_size, dtype=int)
        self._send_list_box = [np.zeros((0, self._markers.shape[1]))] * self.mpi_size

        # Faces
        self._send_info_box[self._x_m_proc] += len(self._markers_x_m)
        self._send_list_box[self._x_m_proc] = np.concatenate((self._send_list_box[self._x_m_proc], self._markers_x_m))

        self._send_info_box[self._x_p_proc] += len(self._markers_x_p)
        self._send_list_box[self._x_p_proc] = np.concatenate((self._send_list_box[self._x_p_proc], self._markers_x_p))

        self._send_info_box[self._y_m_proc] += len(self._markers_y_m)
        self._send_list_box[self._y_m_proc] = np.concatenate((self._send_list_box[self._y_m_proc], self._markers_y_m))

        self._send_info_box[self._y_p_proc] += len(self._markers_y_p)
        self._send_list_box[self._y_p_proc] = np.concatenate((self._send_list_box[self._y_p_proc], self._markers_y_p))

        self._send_info_box[self._z_m_proc] += len(self._markers_z_m)
        self._send_list_box[self._z_m_proc] = np.concatenate((self._send_list_box[self._z_m_proc], self._markers_z_m))

        self._send_info_box[self._z_p_proc] += len(self._markers_z_p)
        self._send_list_box[self._z_p_proc] = np.concatenate((self._send_list_box[self._z_p_proc], self._markers_z_p))

        # x-y edges
        self._send_info_box[self._x_m_y_m_proc] += len(self._markers_x_m_y_m)
        self._send_list_box[self._x_m_y_m_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_m_proc], self._markers_x_m_y_m)
        )

        self._send_info_box[self._x_m_y_p_proc] += len(self._markers_x_m_y_p)
        self._send_list_box[self._x_m_y_p_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_p_proc], self._markers_x_m_y_p)
        )

        self._send_info_box[self._x_p_y_m_proc] += len(self._markers_x_p_y_m)
        self._send_list_box[self._x_p_y_m_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_m_proc], self._markers_x_p_y_m)
        )

        self._send_info_box[self._x_p_y_p_proc] += len(self._markers_x_p_y_p)
        self._send_list_box[self._x_p_y_p_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_p_proc], self._markers_x_p_y_p)
        )

        # x-z edges
        self._send_info_box[self._x_m_z_m_proc] += len(self._markers_x_m_z_m)
        self._send_list_box[self._x_m_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_m_z_m_proc], self._markers_x_m_z_m)
        )

        self._send_info_box[self._x_m_z_p_proc] += len(self._markers_x_m_z_p)
        self._send_list_box[self._x_m_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_m_z_p_proc], self._markers_x_m_z_p)
        )

        self._send_info_box[self._x_p_z_m_proc] += len(self._markers_x_p_z_m)
        self._send_list_box[self._x_p_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_p_z_m_proc], self._markers_x_p_z_m)
        )

        self._send_info_box[self._x_p_z_p_proc] += len(self._markers_x_p_z_p)
        self._send_list_box[self._x_p_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_p_z_p_proc], self._markers_x_p_z_p)
        )

        # y-z edges
        self._send_info_box[self._y_m_z_m_proc] += len(self._markers_y_m_z_m)
        self._send_list_box[self._y_m_z_m_proc] = np.concatenate(
            (self._send_list_box[self._y_m_z_m_proc], self._markers_y_m_z_m)
        )

        self._send_info_box[self._y_m_z_p_proc] += len(self._markers_y_m_z_p)
        self._send_list_box[self._y_m_z_p_proc] = np.concatenate(
            (self._send_list_box[self._y_m_z_p_proc], self._markers_y_m_z_p)
        )

        self._send_info_box[self._y_p_z_m_proc] += len(self._markers_y_p_z_m)
        self._send_list_box[self._y_p_z_m_proc] = np.concatenate(
            (self._send_list_box[self._y_p_z_m_proc], self._markers_y_p_z_m)
        )

        self._send_info_box[self._y_p_z_p_proc] += len(self._markers_y_p_z_p)
        self._send_list_box[self._y_p_z_p_proc] = np.concatenate(
            (self._send_list_box[self._y_p_z_p_proc], self._markers_y_p_z_p)
        )

        # corners
        self._send_info_box[self._x_m_y_m_z_m_proc] += len(self._markers_x_m_y_m_z_m)
        self._send_list_box[self._x_m_y_m_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_m_z_m_proc], self._markers_x_m_y_m_z_m)
        )

        self._send_info_box[self._x_m_y_m_z_p_proc] += len(self._markers_x_m_y_m_z_p)
        self._send_list_box[self._x_m_y_m_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_m_z_p_proc], self._markers_x_m_y_m_z_p)
        )

        self._send_info_box[self._x_m_y_p_z_m_proc] += len(self._markers_x_m_y_p_z_m)
        self._send_list_box[self._x_m_y_p_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_p_z_m_proc], self._markers_x_m_y_p_z_m)
        )

        self._send_info_box[self._x_m_y_p_z_p_proc] += len(self._markers_x_m_y_p_z_p)
        self._send_list_box[self._x_m_y_p_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_p_z_p_proc], self._markers_x_m_y_p_z_p)
        )

        self._send_info_box[self._x_p_y_m_z_m_proc] += len(self._markers_x_p_y_m_z_m)
        self._send_list_box[self._x_p_y_m_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_m_z_m_proc], self._markers_x_p_y_m_z_m)
        )

        self._send_info_box[self._x_p_y_m_z_p_proc] += len(self._markers_x_p_y_m_z_p)
        self._send_list_box[self._x_p_y_m_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_m_z_p_proc], self._markers_x_p_y_m_z_p)
        )

        self._send_info_box[self._x_p_y_p_z_m_proc] += len(self._markers_x_p_y_p_z_m)
        self._send_list_box[self._x_p_y_p_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_p_z_m_proc], self._markers_x_p_y_p_z_m)
        )

        self._send_info_box[self._x_p_y_p_z_p_proc] += len(self._markers_x_p_y_p_z_p)
        self._send_list_box[self._x_p_y_p_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_p_z_p_proc], self._markers_x_p_y_p_z_p)
        )

    def self_communication_boxes(self):
        """Communicate the particle in case a process is it's own neighbour (in case of periodicity with low number of procs)"""

        if self._send_info_box[self.mpi_rank] > 0:
            self.update_holes()
            holes_inds = np.nonzero(self._holes)[0]
            self._markers[holes_inds[np.arange(self._send_info_box[self.mpi_rank])]] = self._send_list_box[
                self.mpi_rank
            ]

    def communicate_boxes(self):
        self.determine_send_markers_box()
        self.get_destinations_box()
        self.self_communication_boxes()
        self.mpi_comm.Barrier()
        self.sendrecv_all_to_all_boxes()
        self.update_holes()
        self.sendrecv_markers_boxes()
        self.update_holes()

    def sendrecv_all_to_all_boxes(self):
        """
        Distribute info on how many markers will be sent/received to/from each process via all-to-all
        for the communication of particles in boundary boxes.
        """

        self._recv_info_box = np.zeros(self.mpi_comm.Get_size(), dtype=int)

        self.mpi_comm.Alltoall(self._send_info_box, self._recv_info_box)

    def sendrecv_markers_boxes(self):
        """
        Use non-blocking communication. In-place modification of markers
        for the communication of particles in boundary boxes.
        """

        # i-th entry holds the number (not the index) of the first hole to be filled by data from process i
        first_hole = np.cumsum(self._recv_info_box) - self._recv_info_box
        hole_inds = np.nonzero(self._holes)[0]
        # Initialize send and receive commands
        reqs = []
        recvbufs = []
        for i, (data, N_recv) in enumerate(zip(self._send_list_box, list(self._recv_info_box))):
            if i == self.mpi_comm.Get_rank():
                reqs += [None]
                recvbufs += [None]
            else:
                self.mpi_comm.Isend(data, dest=i, tag=self.mpi_comm.Get_rank())

                recvbufs += [np.zeros((N_recv, self._markers.shape[1]), dtype=float)]
                reqs += [self.mpi_comm.Irecv(recvbufs[-1], source=i, tag=i)]

        # Wait for buffer, then put markers into holes
        test_reqs = [False] * (self._recv_info_box.size - 1)
        while len(test_reqs) > 0:
            # loop over all receive requests
            for i, req in enumerate(reqs):
                if req is None:
                    continue
                else:
                    # check if data has been received
                    if req.Test():
                        self._markers[hole_inds[first_hole[i] + np.arange(self._recv_info_box[i])]] = recvbufs[i]

                        test_reqs.pop()
                        reqs[i] = None

        self.mpi_comm.Barrier()

    def _get_neighbouring_proc(self):
        """Find the neighbouring processes for the sending of boxes"""
        dd = self.domain_decomp
        # Determine which proc are on which side
        x_l = dd[self.mpi_rank][0]
        x_r = dd[self.mpi_rank][1]
        y_l = dd[self.mpi_rank][3]
        y_r = dd[self.mpi_rank][4]
        z_l = dd[self.mpi_rank][6]
        z_r = dd[self.mpi_rank][7]
        for i in range(self.mpi_size):
            xl_i = dd[i][0]
            xr_i = dd[i][1]
            yl_i = dd[i][3]
            yr_i = dd[i][4]
            zl_i = dd[i][6]
            zr_i = dd[i][7]

            # Faces

            # Process on the left (minus axis) in the x direction
            if (
                abs(periodic_distance(yl_i, y_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_r)) < 1e-5
                and abs(periodic_distance(zl_i, z_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_r)) < 1e-5
                and abs(periodic_distance(xr_i, x_l)) < 1e-5
            ):
                self._x_m_proc = i

            # Process on the right (plus axis) in the x direction
            if (
                abs(periodic_distance(yl_i, y_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_r)) < 1e-5
                and abs(periodic_distance(zl_i, z_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_r)) < 1e-5
                and abs(periodic_distance(xl_i, x_r)) < 1e-5
            ):
                self._x_p_proc = i

            # Process on the left (minus axis) in the y direction
            if (
                abs(periodic_distance(xl_i, x_l)) < 1e-5
                and abs(periodic_distance(xr_i, x_r)) < 1e-5
                and abs(periodic_distance(zl_i, z_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_r)) < 1e-5
                and abs(periodic_distance(yr_i, y_l)) < 1e-5
            ):
                self._y_m_proc = i

            # Process on the right (plus axis) in the y direction
            if (
                abs(periodic_distance(xl_i, x_l)) < 1e-5
                and abs(periodic_distance(xr_i, x_r)) < 1e-5
                and abs(periodic_distance(zl_i, z_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_r)) < 1e-5
                and abs(periodic_distance(yl_i, y_r)) < 1e-5
            ):
                self._y_p_proc = i

            # Process on the left (minus axis) in the z direction
            if (
                abs(periodic_distance(xl_i, x_l)) < 1e-5
                and abs(periodic_distance(xr_i, x_r)) < 1e-5
                and abs(periodic_distance(yl_i, y_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_r)) < 1e-5
                and abs(periodic_distance(zr_i, z_l)) < 1e-5
            ):
                self._z_m_proc = i

            # Process on the right (plus axis) in the z direction
            if (
                abs(periodic_distance(xl_i, x_l)) < 1e-5
                and abs(periodic_distance(xr_i, x_r)) < 1e-5
                and abs(periodic_distance(yl_i, y_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_r)) < 1e-5
                and abs(periodic_distance(zl_i, z_r)) < 1e-5
            ):
                self._z_p_proc = i

            # Edges

            # Process on the left in x and left in y axis
            if (
                abs(periodic_distance(zl_i, z_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_r)) < 1e-5
                and abs(periodic_distance(xr_i, x_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_l)) < 1e-5
            ):
                self._x_m_y_m_proc = i

            # Process on the left in x and right in y axis
            if (
                abs(periodic_distance(zl_i, z_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_r)) < 1e-5
                and abs(periodic_distance(xr_i, x_l)) < 1e-5
                and abs(periodic_distance(yl_i, y_r)) < 1e-5
            ):
                self._x_m_y_p_proc = i

            # Process on the right in x and left in y axis
            if (
                abs(periodic_distance(zl_i, z_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_r)) < 1e-5
                and abs(periodic_distance(xl_i, x_r)) < 1e-5
                and abs(periodic_distance(yr_i, y_l)) < 1e-5
            ):
                self._x_p_y_m_proc = i

            # Process on the right in x and right in y axis
            if (
                abs(periodic_distance(zl_i, z_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_r)) < 1e-5
                and abs(periodic_distance(xl_i, x_r)) < 1e-5
                and abs(periodic_distance(yl_i, y_r)) < 1e-5
            ):
                self._x_p_y_p_proc = i

            # Process on the left in x and left in z axis
            if (
                abs(periodic_distance(yl_i, y_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_r)) < 1e-5
                and abs(periodic_distance(xr_i, x_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_l)) < 1e-5
            ):
                self._x_m_z_m_proc = i

            # Process on the left in x and right in z axis
            if (
                abs(periodic_distance(yl_i, y_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_r)) < 1e-5
                and abs(periodic_distance(xr_i, x_l)) < 1e-5
                and abs(periodic_distance(zl_i, z_r)) < 1e-5
            ):
                self._x_m_z_p_proc = i

            # Process on the right in x and left in z axis
            if (
                abs(periodic_distance(yl_i, y_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_r)) < 1e-5
                and abs(periodic_distance(xl_i, x_r)) < 1e-5
                and abs(periodic_distance(zr_i, z_l)) < 1e-5
            ):
                self._x_p_z_m_proc = i

            # Process on the right in x and right in z axis
            if (
                abs(periodic_distance(yl_i, y_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_r)) < 1e-5
                and abs(periodic_distance(xl_i, x_r)) < 1e-5
                and abs(periodic_distance(zl_i, z_r)) < 1e-5
            ):
                self._x_p_z_p_proc = i

            # Process on the left in y and left in z axis
            if (
                abs(periodic_distance(xl_i, x_l)) < 1e-5
                and abs(periodic_distance(xr_i, x_r)) < 1e-5
                and abs(periodic_distance(yr_i, y_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_l)) < 1e-5
            ):
                self._y_m_z_m_proc = i

            # Process on the left in y and right in z axis
            if (
                abs(periodic_distance(xl_i, x_l)) < 1e-5
                and abs(periodic_distance(xr_i, x_r)) < 1e-5
                and abs(periodic_distance(yr_i, y_l)) < 1e-5
                and abs(periodic_distance(zl_i, z_r)) < 1e-5
            ):
                self._y_m_z_p_proc = i

            # Process on the right in y and left in z axis
            if (
                abs(periodic_distance(xl_i, x_l)) < 1e-5
                and abs(periodic_distance(xr_i, x_r)) < 1e-5
                and abs(periodic_distance(yl_i, y_r)) < 1e-5
                and abs(periodic_distance(zr_i, z_l)) < 1e-5
            ):
                self._y_p_z_m_proc = i

            # Process on the right in y and right in z axis
            if (
                abs(periodic_distance(xl_i, x_l)) < 1e-5
                and abs(periodic_distance(xr_i, x_r)) < 1e-5
                and abs(periodic_distance(yl_i, y_r)) < 1e-5
                and abs(periodic_distance(zl_i, z_r)) < 1e-5
            ):
                self._y_p_z_p_proc = i

            # Corners

            # Process on the left in x, left in y and left in z axis
            if (
                abs(periodic_distance(xr_i, x_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_l)) < 1e-5
            ):
                self._x_m_y_m_z_m_proc = i

            # Process on the left in x, left in y and right in z axis
            if (
                abs(periodic_distance(xr_i, x_l)) < 1e-5
                and abs(periodic_distance(yr_i, y_l)) < 1e-5
                and abs(periodic_distance(zl_i, z_r)) < 1e-5
            ):
                self._x_m_y_m_z_p_proc = i

            # Process on the left in x, right in y and left in z axis
            if (
                abs(periodic_distance(xr_i, x_l)) < 1e-5
                and abs(periodic_distance(yl_i, y_r)) < 1e-5
                and abs(periodic_distance(zr_i, z_l)) < 1e-5
            ):
                self._x_m_y_p_z_m_proc = i

            # Process on the left in x, right in y and right in z axis
            if (
                abs(periodic_distance(xr_i, x_l)) < 1e-5
                and abs(periodic_distance(yl_i, y_r)) < 1e-5
                and abs(periodic_distance(zl_i, z_r)) < 1e-5
            ):
                self._x_m_y_p_z_p_proc = i

            # Process on the right in x, left in y and left in z axis
            if (
                abs(periodic_distance(xl_i, x_r)) < 1e-5
                and abs(periodic_distance(yr_i, y_l)) < 1e-5
                and abs(periodic_distance(zr_i, z_l)) < 1e-5
            ):
                self._x_p_y_m_z_m_proc = i

            # Process on the right in x, left in y and right in z axis
            if (
                abs(periodic_distance(xl_i, x_r)) < 1e-5
                and abs(periodic_distance(yr_i, y_l)) < 1e-5
                and abs(periodic_distance(zl_i, z_r)) < 1e-5
            ):
                self._x_p_y_m_z_p_proc = i

            # Process on the right in x, right in y and left in z axis
            if (
                abs(periodic_distance(xl_i, x_r)) < 1e-5
                and abs(periodic_distance(yl_i, y_r)) < 1e-5
                and abs(periodic_distance(zr_i, z_l)) < 1e-5
            ):
                self._x_p_y_p_z_m_proc = i

            # Process on the right in x, right in y and right in z axis
            if (
                abs(periodic_distance(xl_i, x_r)) < 1e-5
                and abs(periodic_distance(yl_i, y_r)) < 1e-5
                and abs(periodic_distance(zl_i, z_r)) < 1e-5
            ):
                self._x_p_y_p_z_p_proc = i

    def eval_density_fun(self, eta1, eta2, eta3, index, out=None, fast=True, h=0.2):
        """Evaluate the function defined at the `index` of the particles
        at points given by eta1, eta2, eta3. This is done evaluating smoothed version of the
        sum of Dirac delta-functions given by the values at the particle position

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        index : int
            At which index of the markers array are located the value of the function to evaluate.

        out : array_like
            Output will be store in this array. A new array is created if not provided.

        fast : bool
            If true, uses an optimized evaluation algorithm taking advantage of the box structure.
            This assume that the boxes are bigger then the radius used for the smoothing kernel.

        h : float
            Radius of the smoothing kernel to use.
        """

        assert np.shape(eta1) == np.shape(eta2)
        assert np.shape(eta1) == np.shape(eta3)
        if out is not None:
            assert np.shape(eta1) == np.shape(out)
        else:
            out = np.zeros_like(eta1)

        if fast:
            self.put_particles_in_boxes()
            if len(np.shape(eta1)) == 1:
                box_based_evaluation(
                    eta1,
                    eta2,
                    eta3,
                    self._markers,
                    self._sorting_boxes.nx,
                    self._sorting_boxes.ny,
                    self._sorting_boxes.nz,
                    self._sorting_boxes._boxes,
                    self._sorting_boxes._neighbours,
                    self.domain_decomp[self.mpi_rank],
                    self.holes,
                    index,
                    h,
                    out,
                )

            elif len(np.shape(eta1)) == 3:
                # meshgrid format
                box_based_evaluation_3d(
                    eta1,
                    eta2,
                    eta3,
                    self._markers,
                    self._sorting_boxes.nx,
                    self._sorting_boxes.ny,
                    self._sorting_boxes.nz,
                    self._sorting_boxes._boxes,
                    self._sorting_boxes._neighbours,
                    self.domain_decomp[self.mpi_rank],
                    self.holes,
                    index,
                    h,
                    out,
                )
            out /= self.n_mks
        else:
            if len(np.shape(eta1)) == 1:
                naive_evaluation(eta1, eta2, eta3, self._markers, self.holes, index, h, out)

            elif len(np.shape(eta1)) == 3:
                # meshgrid format
                naive_evaluation_3d(eta1, eta2, eta3, self._markers, self.holes, index, h, out)
            out /= self.n_mks
        return out

    def update_holes(self):
        """Compute new holes, new number of holes and markers on process"""
        self._holes[:] = self.markers[:, 0] == -1.0
        self._n_holes_loc = np.count_nonzero(self.holes)
        self._n_mks_loc = self.markers.shape[0] - self._n_holes_loc

    def update_ghost_particles(self):
        """Compute new particles that belong to boundary processes needed for sph evaluation"""
        self._ghost_particles[:] = self.markers[:, -1] == -2.0

    def sendrecv_determine_mtbs(
        self,
        alpha: list | tuple | np.ndarray = (1.0, 1.0, 1.0),
    ):
        """
        Determine which markers have to be sent from current process and put them in a new array.
        Corresponding rows in markers array become holes and are therefore set to -1.
        This can be done purely with numpy functions (fast, vectorized).

        Parameters
        ----------
            alpha : list | tuple
                For i=1,2,3 the sorting is according to alpha[i]*markers[:, i] + (1 - alpha[i])*markers[:, first_pusher_idx + i].
                alpha[i] must be between 0 and 1.

        Returns
        -------
            hole_inds_after_send : array[int]
                Indices of empty columns in markers after send.

            sorting_etas : array[float]
                Eta-values of shape (n_send, :) according to which the sorting is performed.
        """
        # position that determines the sorting (including periodic shift of boundary conditions)
        if not isinstance(alpha, np.ndarray):
            alpha = np.array(alpha, dtype=float)
        assert alpha.size == 3
        assert np.all(alpha >= 0.0) and np.all(alpha <= 1.0)
        bi = self.first_pusher_idx
        self._sorting_etas = np.mod(
            alpha * (self.markers[:, :3] + self.markers[:, bi + 3 + self.vdim : bi + 3 + self.vdim + 3])
            + (1.0 - alpha) * self.markers[:, bi : bi + 3],
            1.0,
        )

        # check which particles are on the current process domain
        self._is_on_proc_domain = np.logical_and(
            self._sorting_etas > self.domain_decomp[self.mpi_rank, 0::3],
            self._sorting_etas < self.domain_decomp[self.mpi_rank, 1::3],
        )

        # to stay on the current process, all three columns must be True
        self._can_stay = np.all(self._is_on_proc_domain, axis=1)

        # holes can stay, too
        self._can_stay[self.holes] = True

        # True values can stay on the process, False must be sent, already empty rows (-1) cannot be sent
        send_inds = np.nonzero(~self._can_stay)[0]

        hole_inds_after_send = np.nonzero(np.logical_or(~self._can_stay, self.holes))[0]

        return hole_inds_after_send, send_inds

    def sendrecv_get_destinations(self, send_inds):
        """
        Determine to which process particles have to be sent.

        Parameters
        ----------
            send_inds : array[int]
                 Indices of particles which will be sent.
        Returns
        -------
            send_info : array[int]
                Amount of particles sent to i-th process.
        """

        # One entry for each process
        send_info = np.zeros(self.mpi_size, dtype=int)

        # TODO: do not loop over all processes, start with neighbours and work outwards (using while)
        for i in range(self.mpi_size):
            conds = np.logical_and(
                self._sorting_etas[send_inds] > self.domain_decomp[i, 0::3],
                self._sorting_etas[send_inds] < self.domain_decomp[i, 1::3],
            )

            self._send_to_i[i] = np.nonzero(np.all(conds, axis=1))[0]
            send_info[i] = self._send_to_i[i].size

            self._send_list[i] = self.markers[send_inds][self._send_to_i[i]]

        return send_info

    def sendrecv_all_to_all(self, send_info):
        """
        Distribute info on how many markers will be sent/received to/from each process via all-to-all.

        Parameters
        ----------
            send_info : array[int]
                Amount of markers to be sent to i-th process.

        Returns
        -------
            recv_info : array[int]
                Amount of marticles to be received from i-th process.
        """

        recv_info = np.zeros(self.mpi_size, dtype=int)

        self.mpi_comm.Alltoall(send_info, recv_info)

        return recv_info

    def sendrecv_markers(self, recv_info, hole_inds_after_send):
        """
        Use non-blocking communication. In-place modification of markers

        Parameters
        ----------
            recv_info : array[int]
                Amount of markers to be received from i-th process.

            hole_inds_after_send : array[int]
                Indices of empty rows in markers after send.
        """

        # i-th entry holds the number (not the index) of the first hole to be filled by data from process i
        first_hole = np.cumsum(recv_info) - recv_info

        # Initialize send and receive commands
        for i, (data, N_recv) in enumerate(zip(self._send_list, list(recv_info))):
            if i == self.mpi_rank:
                self._reqs[i] = None
                self._recvbufs[i] = None
            else:
                self.mpi_comm.Isend(data, dest=i, tag=self.mpi_rank)

                self._recvbufs[i] = np.zeros((N_recv, self.markers.shape[1]), dtype=float)
                self._reqs[i] = self.mpi_comm.Irecv(self._recvbufs[i], source=i, tag=i)

        # Wait for buffer, then put markers into holes
        test_reqs = [False] * (recv_info.size - 1)
        while len(test_reqs) > 0:
            # loop over all receive requests
            for i, req in enumerate(self._reqs):
                if req is None:
                    continue
                else:
                    # check if data has been received
                    if req.Test():
                        self.markers[hole_inds_after_send[first_hole[i] + np.arange(recv_info[i])]] = self._recvbufs[i]

                        test_reqs.pop()
                        self._reqs[i] = None
