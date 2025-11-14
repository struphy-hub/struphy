import copy
import os
import warnings
from abc import ABCMeta, abstractmethod

import h5py
import numpy as np
import scipy.special as sp
from line_profiler import profile
from mpi4py import MPI
from mpi4py.MPI import Intracomm
from sympy.ntheory import factorint

from struphy.bsplines.bsplines import quadrature_grid
from struphy.fields_background import equils
from struphy.fields_background.base import FluidEquilibrium, FluidEquilibriumWithB, NumericalFluidEquilibrium
from struphy.fields_background.equils import set_defaults
from struphy.fields_background.projected_equils import ProjectedFluidEquilibrium
from struphy.geometry.base import Domain
from struphy.geometry.utilities import TransformedPformComponent
from struphy.initial.base import Perturbation
from struphy.io.options import OptsLoading
from struphy.io.output_handling import DataContainer
from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments
from struphy.kinetic_background.base import KineticBackground, Maxwellian
from struphy.pic import sampling_kernels, sobol_seq
from struphy.pic.pushing.pusher_utilities_kernels import reflect
from struphy.pic.sorting_kernels import (
    assign_box_to_each_particle,
    assign_particles_to_boxes,
    flatten_index,
    initialize_neighbours,
    sort_boxed_particles,
)
from struphy.pic.sph_eval_kernels import (
    box_based_evaluation_flat,
    box_based_evaluation_meshgrid,
    distance,
    naive_evaluation_flat,
    naive_evaluation_meshgrid,
)
from struphy.pic.utilities import (
    BoundaryParameters,
    LoadingParameters,
    WeightsParameters,
)
from struphy.utils import utils
from struphy.utils.clone_config import CloneConfig


class Particles(metaclass=ABCMeta):
    r"""
    Base class for particle species.

    The marker information is stored in a 2D numpy array.
    In ``markers[ip, j]`` The row index ``ip`` refers to a specific particle,
    the column index ``j`` to its attributes.
    The columns are indexed as follows:

    * ``0:3``: position in the logical unit cube (:math:`\boldsymbol \eta_p \in [0, 1]^3`)
    * ``3:3 + vdim``: velocities
    * ``3 + vdim``: (time-dependent) weight :math:`w_k(t)`
    * ``4 + vdim``: PDF :math:`s^0 = s^3/\sqrt g` at particle position
    * ``5 + vdim``: initial weight :math:`w_0`
    * ``6 + vdim <= j < -2``: buffer indices; see attributes ``first_diagnostics_idx``, ``first_pusher_idx`` and ``first_free_idx`` below
    * ``-2``: number of the sorting box the particle is in
    * ``-1``: particle ID

    Parameters
    ----------
    comm_world : Intracomm
        World MPI communicator.

    clone_config : CloneConfig
        Manages the configuration for clone-based (copied grids) parallel processing using MPI.

    domain_decomp : tuple
        The first entry is a domain_array (see :attr:`~struphy.feec.psydac_derham.Derham.domain_array`) and
        the second entry is the number of MPI processes in each direction.

    mpi_dims_mask: list | tuple of bool
            True if the dimension is to be used in the domain decomposition (=default for each dimension).
            If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.

    boxes_per_dim : tuple
        Number of boxes in each logical direction (n_eta1, n_eta2, n_eta3).

    box_bufsize : float
        Between 0 and 1, relative buffer size for box array (default = 0.25).

    type : str
        Either 'full_f' (default), 'delta_f' or 'sph'.

    name : str
        Name of particle species.

    loading_params : LoadingParameters
        Parameterts for particle loading.

    weights_params : WeightsParameters
        Parameters for particle weights.

    boundary_params : BoundaryParameters
        Parameters for particle boundary conditions.

    bufsize : float
        Size of buffer (as multiple of total size, default=.25) in markers array.

    domain : Domain
        Struphy domain object.

    equil : FluidEquilibrium
        Struphy fluid equilibrium object.

    projected_equil : ProjectedFluidEquilibrium
        Struphy fluid equilibrium projected into a discrete Derham complex.

    background : KineticBackground
        Kinetic background.

    initial_condition : KineticBackground
        Kinetic initial condition.

    n_as_volume_form: bool
        Whether the number density n is given as a volume form or scalar function (=default).

    perturbations : Perturbation | list
        Kinetic perturbation parameters.

    equation_params : dict
        Normalization parameters (epsilon, alpha, ...)

    verbose : bool
        Show some more Particle info.
    """

    def __init__(
        self,
        comm_world: Intracomm = None,
        clone_config: CloneConfig = None,
        domain_decomp: tuple = None,
        mpi_dims_mask: tuple | list = None,
        boxes_per_dim: tuple | list = None,
        box_bufsize: float = 2.0,
        type: str = "full_f",
        name: str = "some_name",
        loading_params: LoadingParameters = None,
        weights_params: WeightsParameters = None,
        boundary_params: BoundaryParameters = None,
        bufsize: float = 0.25,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
        projected_equil: ProjectedFluidEquilibrium = None,
        background: KineticBackground | FluidEquilibrium = None,
        initial_condition: KineticBackground = None,
        perturbations: dict[str, Perturbation] = None,
        n_as_volume_form: bool = False,
        equation_params: dict = None,
        verbose: bool = False,
    ):
        self._clone_config = clone_config
        if self.clone_config is None:
            self._mpi_comm = comm_world
            self._num_clones = 1
            self._clone_id = 0
        else:
            self._mpi_comm = self.clone_config.sub_comm
            self._num_clones = self.clone_config.num_clones
            self._clone_id = self.clone_config.clone_id

        # defaults
        if loading_params is None:
            loading_params = LoadingParameters()

        if weights_params is None:
            weights_params = WeightsParameters()

        if boundary_params is None:
            boundary_params = BoundaryParameters()

        # other parameters
        self._name = name
        self._loading_params = loading_params
        self._weights_params = weights_params
        self._boundary_params = boundary_params
        self._domain = domain
        self._equil = equil
        self._projected_equil = projected_equil
        self._equation_params = equation_params

        # check for mpi communicator (i.e. sub_comm of clone)
        if self.mpi_comm is None:
            self._mpi_size = 1
            self._mpi_rank = 0
        else:
            self._mpi_size = self.mpi_comm.Get_size()
            self._mpi_rank = self.mpi_comm.Get_rank()

        # domain decomposition (MPI) and cell information
        self._boxes_per_dim = boxes_per_dim
        self._box_bufsize = box_bufsize
        self._mpi_dims_mask = mpi_dims_mask
        if domain_decomp is None:
            self._domain_array, self._nprocs = self._get_domain_decomp(mpi_dims_mask)
        else:
            self._domain_array = domain_decomp[0]
            self._nprocs = domain_decomp[1]

        # total number of cells (equal to mpi_size if no grid)
        n_cells = np.sum(np.prod(self.domain_array[:, 2::3], axis=1, dtype=int)) * self.num_clones
        # if verbose:
        #     print(f"\n{self.mpi_rank = }, {n_cells = }")

        # total number of boxes
        if self.boxes_per_dim is None:
            n_boxes = self.mpi_size * self.num_clones
        else:
            assert all([nboxes >= nproc for nboxes, nproc in zip(self.boxes_per_dim, self.nprocs)]), (
                f"There must be at least one box {self.boxes_per_dim = } on each process {self.nprocs = } in each direction."
            )
            assert all([nboxes % nproc == 0 for nboxes, nproc in zip(self.boxes_per_dim, self.nprocs)]), (
                f"Number of boxes {self.boxes_per_dim = } must be divisible by number of processes {self.nprocs = } in each direction."
            )
            n_boxes = np.prod(self.boxes_per_dim, dtype=int) * self.num_clones

        # if verbose:
        #     print(f"\n{self.mpi_rank = }, {n_boxes = }")

        # total number of markers (Np) and particles per cell (ppc)
        Np = self.loading_params.Np
        ppc = self.loading_params.ppc
        ppb = self.loading_params.ppb
        if Np is not None:
            self._Np = int(Np)
            self._ppc = self.Np / n_cells
            self._ppb = self.Np / n_boxes
        elif ppc is not None:
            self._ppc = ppc
            self._Np = int(self.ppc * n_cells)
            self._ppb = self.Np / n_boxes
        elif ppb is not None:
            self._ppb = ppb
            self._Np = int(self.ppb * n_boxes)
            self._ppc = self.Np / n_cells

        assert self.Np >= self.mpi_size

        # create marker array
        self._bufsize = bufsize
        self._allocate_marker_array()

        # boundary conditions
        bc = boundary_params.bc
        bc_refill = boundary_params.bc_refill
        if bc is None:
            bc = ["periodic", "periodic", "periodic"]

        for bci in bc:
            assert bci in ("remove", "reflect", "periodic", "refill")
            if bci == "reflect":
                assert domain is not None, "Reflecting boundary conditions require a domain."

        if bc_refill is not None:
            for bc_refilli in bc_refill:
                assert bc_refilli in ("outer", "inner")

        self._bc = bc
        self._periodic_axes = [axis for axis, b_c in enumerate(bc) if b_c == "periodic"]
        self._reflect_axes = [axis for axis, b_c in enumerate(bc) if b_c == "reflect"]
        self._remove_axes = [axis for axis, b_c in enumerate(bc) if b_c == "remove"]
        self._bc_refill = bc_refill

        bc_sph = boundary_params.bc_sph
        if bc_sph is None:
            bc_sph = [bci if bci == "periodic" else "mirror" for bci in self.bc]

        for bci in bc_sph:
            assert bci in ("periodic", "mirror", "fixed")
        self._bc_sph = bc_sph

        # particle type
        assert type in ("full_f", "delta_f", "sph")
        self._type = type

        # initialize sorting boxes
        self._verbose = verbose
        self._initialize_sorting_boxes()

        # particle loading parameters
        self._loading = loading_params.loading
        self._spatial = loading_params.spatial

        # weights
        self._reject_weights = weights_params.reject_weights
        self._threshold = weights_params.threshold
        self._control_variate = weights_params.control_variate

        # background
        if background is None:
            raise ValueError("A background function must be passed to Particles.")
        else:
            self._background = background

        # background p-form description in [eta, v] (False means 0-form, True means volume form -> divide by det)
        if isinstance(background, FluidEquilibrium):
            self._is_volume_form = (n_as_volume_form, False)
        else:
            self._is_volume_form = (
                n_as_volume_form,
                self.background.volume_form,
            )

        # set background function
        self._set_background_function()
        self._set_background_coordinates()

        # perturbation parameters (needed for fluid background)
        self._perturbations = perturbations

        # initial condition
        if initial_condition is None:
            self._initial_condition = self.background
        else:
            self._initial_condition = initial_condition

        # for loading
        # if self.loading_params["moments"] is None and self.type != "sph" and isinstance(self.bckgr_params, dict):
        self._generate_sampling_moments()

        # create buffers for mpi_sort_markers
        if self.mpi_comm is not None:
            self._sorting_etas = np.zeros(self.markers.shape, dtype=float)
            self._is_on_proc_domain = np.zeros((self.markers.shape[0], 3), dtype=bool)
            self._can_stay = np.zeros(self.markers.shape[0], dtype=bool)
            self._reqs = [None] * self.mpi_size
            self._recvbufs = [None] * self.mpi_size
            self._send_to_i = [None] * self.mpi_size
            self._send_list = [None] * self.mpi_size

    @classmethod
    @abstractmethod
    def default_background(cls):
        """The default background (of type Maxwellian)."""
        pass

    @abstractmethod
    def svol(self, eta1, eta2, eta3, *v):
        r"""Marker sampling distribution function :math:`s^\textrm{vol}` as a volume form, see :ref:`monte_carlo`."""
        pass

    @abstractmethod
    def s0(self, eta1, eta2, eta3, *v, flat_eval=False, remove_holes=True):
        r"""Marker sampling distribution function :math:`s^0` as 0-form, see :ref:`monte_carlo`."""
        pass

    @property
    @abstractmethod
    def vdim(self):
        """Dimension of the velocity space."""
        pass

    @property
    @abstractmethod
    def n_cols_diagnostics(self):
        """Number of columns for storing diagnostics for each marker."""
        pass

    @property
    @abstractmethod
    def n_cols_aux(self):
        """Number of auxiliary columns for each marker (e.g. for storing evaluation data)."""
        pass

    @property
    def first_diagnostics_idx(self):
        """Starting index for diagnostics columns:
        after 3 positions, vdim velocities, weight, s0 and w0."""
        return 3 + self.vdim + 3

    @property
    def first_pusher_idx(self):
        """Starting index for storing initial conditions for a Pusher call."""
        return self.first_diagnostics_idx + self.n_cols_diagnostics

    @property
    def n_cols_pusher(self):
        """Dimension of the phase space (for storing initial conditions for a Pusher call)."""
        return 3 + self.vdim

    @property
    def first_shift_idx(self):
        """First index for storing shifts due to boundary conditions in eta-space."""
        return self.first_pusher_idx + self.n_cols_pusher

    @property
    def n_cols_shift(self):
        """Number of columns for storing shifts due to boundary conditions in eta-space."""
        return 3

    @property
    def residual_idx(self):
        """Column for storing the residual in iterative pushers."""
        return self.first_shift_idx + self.n_cols_shift

    @property
    def first_free_idx(self):
        """First index for storing auxiliary quantities for each particle."""
        return self.residual_idx + 1

    @property
    def n_cols(self):
        """Total umber of columns in markers array.
        The last 2 columns refer to box number and particle ID, respectively."""
        return self.first_free_idx + self.n_cols_aux + 2

    @property
    def n_rows(self):
        """Total number of rows in markers array."""
        if not hasattr(self, "_n_rows"):
            input("\nWarning: marker array not yet created, creating now ...")
            self._allocate_marker_array()
        return self._n_rows

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
        """Particle type: 'full_f', 'delta_f' or 'sph'."""
        return self._type

    @property
    def loading(self) -> OptsLoading:
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
    def bc_sph(self):
        """List of boundary conditions for sph evaluation in each direction."""
        return self._bc_sph

    @property
    def Np(self):
        """Total number of markers/particles, from user input."""
        return self._Np

    @property
    def Np_per_clone(self):
        """Array where i-th entry corresponds to the number of loaded particles on clone i.
        (This is not necessarily the number of valid markers per clone, see self.n_mks_on_each_clone)."""
        return self._Np_per_clone

    @property
    def ppc(self):
        """Particles per cell (=Np if no grid is present)."""
        return self._ppc

    @property
    def ppb(self):
        """Particles per sorting box."""
        return self._ppb

    @property
    def bufsize(self):
        """Relative size of buffer in markers array."""
        return self._bufsize

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
    def clone_config(self):
        """Manages the configuration for clone-based (copied grids) parallel processing using MPI."""
        return self._clone_config

    @property
    def num_clones(self):
        """Total number of clones."""
        return self._num_clones

    @property
    def clone_id(self):
        """Clone id of current process."""
        return self._clone_id

    @property
    def background(self) -> KineticBackground:
        """Kinetic background."""
        return self._background

    @property
    def perturbations(self) -> dict[str, Perturbation]:
        """Kinetic perturbations, keys are the names of moments of the distribution function ("n", "u1", etc.)."""
        return self._perturbations

    @property
    def loading_params(self) -> LoadingParameters:
        return self._loading_params

    @property
    def weights_params(self) -> WeightsParameters:
        return self._weights_params

    @property
    def boundary_params(self) -> BoundaryParameters:
        """Parameters for marker loading."""
        return self._boundary_params

    @property
    def reject_weights(self):
        """Whether to reect weights below threshold."""
        return self._reject_weights

    @property
    def threshold(self):
        """Threshold for rejecting weights."""
        return self._threshold

    @property
    def boxes_per_dim(self):
        """Tuple, number of sorting boxes per dimension."""
        return self._boxes_per_dim

    @property
    def verbose(self):
        """Show some more particles info."""
        return self._verbose

    @property
    def equation_params(self):
        """Parameters appearing in model equation due to Struphy normalization."""
        return self._equation_params

    @property
    def initial_condition(self) -> KineticBackground:
        """Kinetic initial condition"""
        return self._initial_condition

    @property
    def f_init(self):
        """Callable initial condition (background + perturbation).
        For kinetic models this is a Maxwellian.
        For SPH models this is a :class:`~struphy.fields_background.base.FluidEquilibrium`."""
        assert hasattr(self, "_f_init"), AttributeError(
            'The method "_set_initial_condition" has not yet been called.',
        )
        return self._f_init

    @property
    def u_init(self):
        """Callable initial condition (background + perturbation) for the Cartesian velocity
        in SPH models."""
        assert hasattr(self, "_u_init"), AttributeError(
            'The method "_set_initial_condition" has not yet been called.',
        )
        return self._u_init

    @property
    def f0(self) -> Maxwellian:
        assert hasattr(self, "_f0"), AttributeError(
            "No background distribution available, please run self._set_background_function()",
        )
        return self._f0

    @property
    def control_variate(self):
        """Boolean for whether to use the :ref:`control_var` during time stepping."""
        return self._control_variate

    @property
    def domain_array(self):
        """
        A 2d array[float] of shape (comm.Get_size(), 9). The row index denotes the process number and
        for n=0,1,2:

            * domain_array[i, 3*n + 0] holds the LEFT domain boundary of process i in direction eta_(n+1).
            * domain_array[i, 3*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).
            * domain_array[i, 3*n + 2] holds the number of cells of process i in direction eta_(n+1).
        """
        return self._domain_array

    @property
    def mpi_dims_mask(self):
        """3-list | tuple; True if the dimension is to be used in the domain decomposition (=default for each dimension).
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed."""
        return self._mpi_dims_mask

    @property
    def nprocs(self):
        """Number of MPI processes in each dimension."""
        return self._nprocs

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
        """Array of booleans stating if an entry in the markers array is a hole."""
        if not hasattr(self, "_holes"):
            self._holes = self.markers[:, 0] == -1.0
        return self._holes

    @property
    def ghost_particles(self):
        """Array of booleans stating if an entry in the markers array is a ghost particle."""
        if not hasattr(self, "_ghost_particles"):
            self._ghost_particles = self.markers[:, -1] == -2.0
        return self._ghost_particles

    @property
    def markers_wo_holes(self):
        """Array holding the marker information, excluding holes. The i-th row holds the i-th marker info."""
        return self.markers[~self.holes]

    @property
    def markers_wo_holes_and_ghost(self):
        """Array holding the marker information, excluding holes and ghosts (only valid markers). The i-th row holds the i-th marker info."""
        return self.markers[self.valid_mks]

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
        out["s0"] = 4 + self.vdim  # sampling density at t=0
        out["w0"] = 5 + self.vdim  # weights at t=0
        out["box"] = -2  # sorting box index
        out["ids"] = -1  # marker_inds
        return out

    @property
    def valid_mks(self):
        """Array of booleans stating if an entry in the markers array is a true local particle (not a hole or ghost)."""
        if not hasattr(self, "_valid_mks"):
            self._valid_mks = ~np.logical_or(self.holes, self.ghost_particles)
        return self._valid_mks

    def update_valid_mks(self):
        self._valid_mks[:] = ~np.logical_or(self.holes, self.ghost_particles)

    @property
    def n_mks_loc(self):
        """Number of valid markers on process (without holes and ghosts)."""
        return np.count_nonzero(self.valid_mks)

    @property
    def n_mks_on_each_proc(self):
        """Array where i-th entry corresponds to the number of valid markers on i-th process (without holes and ghosts)."""
        return self._gather_scalar_in_subcomm_array(self.n_mks_loc)

    @property
    def n_mks_on_clone(self):
        """Number of valid markers on current clone (without holes and ghosts)."""
        return np.sum(self.n_mks_on_each_proc)

    @property
    def n_mks_on_each_clone(self):
        """Number of valid markers on current clone (without holes and ghosts)."""
        return self._gather_scalar_in_intercomm_array(self.n_mks_on_clone)

    @property
    def n_mks_global(self):
        """Number of valid markers on current clone (without holes and ghosts)."""
        return np.sum(self.n_mks_on_each_clone)

    @property
    def positions(self):
        """Array holding the marker positions in logical space. The i-th row holds the i-th marker info."""
        return self.markers[self.valid_mks, self.index["pos"]]

    @positions.setter
    def positions(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, 3)
        self._markers[self.valid_mks, self.index["pos"]] = new

    @property
    def velocities(self):
        """Array holding the marker velocities in logical space. The i-th row holds the i-th marker info."""
        return self.markers[self.valid_mks, self.index["vel"]]

    @velocities.setter
    def velocities(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, self.vdim), f"{self.n_mks_loc = } and {self.vdim = } but {new.shape = }"
        self._markers[self.valid_mks, self.index["vel"]] = new

    @property
    def phasespace_coords(self):
        """Array holding the marker positions and velocities in logical space. The i-th row holds the i-th marker info."""
        return self.markers[self.valid_mks, self.index["coords"]]

    @phasespace_coords.setter
    def phasespace_coords(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc, 3 + self.vdim)
        self._markers[self.valid_mks, self.index["coords"]] = new

    @property
    def weights(self):
        """Array holding the current marker weights. The i-th row holds the i-th marker info."""
        return self.markers[self.valid_mks, self.index["weights"]]

    @weights.setter
    def weights(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[self.valid_mks, self.index["weights"]] = new

    @property
    def sampling_density(self):
        """Array holding the current marker 0form sampling density s0. The i-th row holds the i-th marker info."""
        return self.markers[self.valid_mks, self.index["s0"]]

    @sampling_density.setter
    def sampling_density(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[self.valid_mks, self.index["s0"]] = new

    @property
    def weights0(self):
        """Array holding the initial marker weights. The i-th row holds the i-th marker info."""
        return self.markers[self.valid_mks, self.index["w0"]]

    @weights0.setter
    def weights0(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[self.valid_mks, self.index["w0"]] = new

    @property
    def marker_ids(self):
        """Array holding the marker id's on the current process."""
        return self.markers[self.valid_mks, self.index["ids"]]

    @marker_ids.setter
    def marker_ids(self, new):
        assert isinstance(new, np.ndarray)
        assert new.shape == (self.n_mks_loc,)
        self._markers[self.valid_mks, self.index["ids"]] = new

    @property
    def is_volume_form(self):
        """Tuple of size 2 for (position, velocity), defining the p-form representation of f_init: True means volume-form, False means 0-form."""
        return self._is_volume_form

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
        return self.markers[self.valid_mks, self.f_coords_index]

    @f_coords.setter
    def f_coords(self, new):
        assert isinstance(new, np.ndarray)
        self.markers[self.valid_mks, self.f_coords_index] = new

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
        if not hasattr(self, "_sorting_boxes"):
            self._initialize_sorting_boxes()
        return self._sorting_boxes

    @property
    def tesselation(self):
        """Tesselation of the current process domain."""
        return self._tesselation

    @classmethod
    def ker_dct(self):
        """Available smoothing kernels, numbers must be multiplies of 100."""
        return {
            "trigonometric_1d": 100,
            "gaussian_1d": 110,
            "linear_1d": 120,
            "trigonometric_2d": 340,
            "gaussian_2d": 350,
            "linear_2d": 360,
            "trigonometric_3d": 670,
            "gaussian_3d": 680,
            "linear_isotropic_3d": 690,
            "linear_3d": 700,
        }

    def _get_domain_decomp(self, mpi_dims_mask: tuple | list = None):
        """
        Compute domain decomposition for mesh-less methods (no Derham object).

        Parameters
        ----------
        mpi_dims_mask: list | tuple of bool
            True if the dimension is to be used in the domain decomposition (=default for each dimension).
            If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.

        Returns
        -------
        dom_arr : np.ndarray
            A 2d array of shape (#MPI processes, 9). The row index denotes the process rank. The columns are for n=0,1,2:
                - arr[i, 3*n + 0] holds the LEFT domain boundary of process i in direction eta_(n+1).
                - arr[i, 3*n + 1] holds the RIGHT domain boundary of process i in direction eta_(n+1).
                - arr[i, 3*n + 2] holds the number of cells of process i in direction eta_(n+1).

        nprocs : tuple
            The number of processes in each direction.
        """
        if mpi_dims_mask is None:
            mpi_dims_mask = [True, True, True]

        dom_arr = np.zeros((self.mpi_size, 9), dtype=float)

        # factorize mpi size
        factors = factorint(self.mpi_size)
        factors_vec = []
        for fac, multiplicity in factors.items():
            for m in range(multiplicity):
                factors_vec += [fac]

        # processes in each direction
        skip_dims = False
        boxes_per_dim = (1, 1, 1)
        if self.boxes_per_dim is not None:
            boxes_per_dim = self.boxes_per_dim
            if not all([bpd == 1 for bpd in self.boxes_per_dim]):
                skip_dims = True

        nprocs = [1, 1, 1]
        for m, fac in enumerate(factors_vec):
            mm = m % 3
            while (boxes_per_dim[mm] == 1 and skip_dims) or not mpi_dims_mask[mm]:
                mm = (mm + 1) % 3
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

        return dom_arr, tuple(nprocs)

    def _set_background_function(self):
        self._f0 = self.background

        # if isinstance(self.background, FluidEquilibrium):
        #     self._f0 = self.background
        # else:
        #     self._f0 = copy.deepcopy(self.background)
        #     self.f0.add_perturbation = False

        # self._f0 = None
        # if isinstance(self.bckgr_params, FluidEquilibrium):
        #     self._f0 = self.bckgr_params
        # else:
        #     for bckgr in self.backgrounds:
        #         # SPH case: f0 is set to a FluidEquilibrium
        #         if self.type == "sph":
        #             _eq = getattr(equils, fi_type)(**maxw_params)
        #             if not isinstance(_eq, NumericalFluidEquilibrium):
        #                 _eq.domain = self.domain
        #             if self._f0 is None:
        #                 self._f0 = _eq
        #             else:
        #                 raise NotImplementedError("Summation of fluid backgrounds not yet implemented.")
        #                 # self._f0 = self._f0 + (lambda e1, e2, e3: _eq.n0(e1, e2, e3))
        #         # default case
        #         else:
        #             if self._f0 is None:
        #                 self._f0 = bckgr
        #             else:
        #                 self._f0 = self._f0 + bckgr

    def _set_background_coordinates(self):
        if self.type != "sph" and self.f0.coords == "constants_of_motion":
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

        if self.type == "sph":
            self._f_coords_index = self.index["coords"]
            self._f_jacobian_coords_index = self.index["coords"]
        else:
            if self.f0.coords == "constants_of_motion":
                self._f_coords_index = self.index["com"]
                self._f_jacobian_coords_index = self.index["pos+energy"]

            else:
                self._f_coords_index = self.index["coords"]
                self._f_jacobian_coords_index = self.index["coords"]

    def _n_mks_load_and_Np_per_clone(self):
        """Return two arrays: 1) an array of sub_comm.size where the i-th entry corresponds to the number of markers drawn on process i,
        and 2) an array of size num_clones where the i-th entry corresponds to the number of markers on clone i."""
        # number of cells on current process
        n_cells_loc = np.prod(
            self.domain_array[self.mpi_rank, 2::3],
            dtype=int,
        )

        # array of number of markers on each process at loading stage
        if self.clone_config is not None:
            _n_cells_clone = np.sum(np.prod(self.domain_array[:, 2::3], axis=1, dtype=int))
            _n_mks_load_tot = self.clone_config.get_Np_clone(self.Np)
            _ppc = _n_mks_load_tot / _n_cells_clone
        else:
            _n_mks_load_tot = self.Np
            _ppc = self.ppc

        n_mks_load = self._gather_scalar_in_subcomm_array(int(_ppc * n_cells_loc))

        # add deviation from Np to rank 0
        n_mks_load[0] += _n_mks_load_tot - np.sum(n_mks_load)

        # check if all markers are there
        assert np.sum(n_mks_load) == _n_mks_load_tot

        # Np on each clone
        Np_per_clone = self._gather_scalar_in_intercomm_array(_n_mks_load_tot)
        assert np.sum(Np_per_clone) == self.Np

        return n_mks_load, Np_per_clone

    def _allocate_marker_array(self):
        """Create marker array :attr:`~struphy.pic.base.Particles.markers`."""
        if not hasattr(self, "_n_mks_load"):
            self._n_mks_load, self._Np_per_clone = self._n_mks_load_and_Np_per_clone()

        # number of markers on the local process at loading stage
        n_mks_load_loc = self.n_mks_load[self._mpi_rank]
        bufsize = self.bufsize + 1.0 / np.sqrt(n_mks_load_loc)

        # allocate markers array (3 x positions, vdim x velocities, weight, s0, w0, ..., ID) with buffer
        self._n_rows = round(n_mks_load_loc * (1 + bufsize))
        self._markers = np.zeros((self.n_rows, self.n_cols), dtype=float)

        # allocate auxiliary arrays
        self._holes = np.zeros(self.n_rows, dtype=bool)
        self._ghost_particles = np.zeros(self.n_rows, dtype=bool)
        self._valid_mks = np.zeros(self.n_rows, dtype=bool)
        self._is_outside_right = np.zeros(self.n_rows, dtype=bool)
        self._is_outside_left = np.zeros(self.n_rows, dtype=bool)
        self._is_outside = np.zeros(self.n_rows, dtype=bool)

        # create array container (3 x positions, vdim x velocities, weight, s0, w0, ID) for removed markers
        self._n_lost_markers = 0
        self._lost_markers = np.zeros((int(self.n_rows * 0.5), 10), dtype=float)

        # arguments for kernels
        self._args_markers = MarkerArguments(
            self.markers,
            self.valid_mks,
            self.Np,
            self.vdim,
            self.index["weights"],
            self.first_diagnostics_idx,
            self.first_pusher_idx,
            self.first_shift_idx,
            self.residual_idx,
            self.first_free_idx,
        )

        # Have at least 3 spare places in markers array
        assert self.args_markers.first_free_idx + 2 < self.n_cols - 1, (
            f"{self.args_markers.first_free_idx + 2} is not smaller than {self.n_cols - 1 = }; not enough columns in marker array !!"
        )

    def _initialize_sorting_boxes(self):
        """Initializes the sorting boxes.

        Each MPI process has exactly the same box structure and numbering.
        For instance, if boxes_per_dim = (16, 1, 1) and there are 2 MPI processes,
        each process would get 8 boxes in the first direction.
        Hence boxes_per_dim has to be divisible by the number of ranks in each direction.
        """

        self._initialized_sorting = False
        if self.boxes_per_dim is not None:
            # split boxes across MPI processes
            nboxes = [nboxes // nproc for nboxes, nproc in zip(self.boxes_per_dim, self.nprocs)]

            # check whether this process touches the domain boundary
            is_domain_boundary = {}
            x_l = self.domain_array[self.mpi_rank, 0]
            x_r = self.domain_array[self.mpi_rank, 1]
            y_l = self.domain_array[self.mpi_rank, 3]
            y_r = self.domain_array[self.mpi_rank, 4]
            z_l = self.domain_array[self.mpi_rank, 6]
            z_r = self.domain_array[self.mpi_rank, 7]
            is_domain_boundary["x_m"] = x_l == 0.0
            is_domain_boundary["x_p"] = x_r == 1.0
            is_domain_boundary["y_m"] = y_l == 0.0
            is_domain_boundary["y_p"] = y_r == 1.0
            is_domain_boundary["z_m"] = z_l == 0.0
            is_domain_boundary["z_p"] = z_r == 1.0

            self._sorting_boxes = self.SortingBoxes(
                self.markers.shape,
                self.type == "sph",
                nx=nboxes[0],
                ny=nboxes[1],
                nz=nboxes[2],
                bc_sph=self.bc_sph,
                is_domain_boundary=is_domain_boundary,
                comm=self.mpi_comm,
                verbose=False,
                box_bufsize=self._box_bufsize,
            )

            if self.sorting_boxes.communicate:
                self._get_neighbouring_proc()

            self._initialized_sorting = True

        else:
            self._sorting_boxes = None

    def _generate_sampling_moments(self):
        """Automatically determine moments for sampling distribution (Gaussian) from the given background."""

        if self.loading_params.moments is None:
            self.loading_params.moments = tuple([0.0] * self.vdim + [1.0] * self.vdim)

        # TODO: reformulate this function with KineticBackground methods

        # ns = []
        # us = []
        # vths = []

        # for fi, params in self.bckgr_params.items():
        #     if fi[-2] == "_":
        #         fi_type = fi[:-2]
        #     else:
        #         fi_type = fi

        #     us.append([])
        #     vths.append([])

        #     bckgr = getattr(maxwellians, fi_type)

        #     for key in default_maxw_params:
        #         if key[0] == "n":
        #             if key in params:
        #                 ns += [params[key]]
        #             else:
        #                 ns += [1.0]

        #         elif key[0] == "u":
        #             if key in params:
        #                 us[-1] += [params[key]]
        #             else:
        #                 us[-1] += [0.0]

        #         elif key[0] == "v":
        #             if key in params:
        #                 vths[-1] += [params[key]]
        #             else:
        #                 vths[-1] += [1.0]

        # assert len(ns) == len(us) == len(vths)

        # ns = np.array(ns)
        # us = np.array(us)
        # vths = np.array(vths)

        # Use the mean of shifts and thermal velocity such that outermost shift+thermal is
        # new shift + new thermal
        # mean_us = np.mean(us, axis=0)
        # us_ext = us + vths * np.where(us >= 0, 1, -1)
        # us_ext_dist = us_ext - mean_us[None, :]
        # new_vths = np.max(np.abs(us_ext_dist), axis=0)

        # new_moments = []

        # new_moments += [*mean_us]
        # new_moments += [*new_vths]
        # new_moments = [float(moment) for moment in new_moments]

        # self.loading_params["moments"] = new_moments

    def _set_initial_condition(self):
        if self.type != "sph":
            self._f_init = self.initial_condition
        else:
            # Get the initialization function and pass the correct arguments
            assert isinstance(self.f0, FluidEquilibrium)
            self._u_init = self.f0.u_cart

            if self.perturbations is not None:
                for (
                    moment,
                    pert,
                ) in self.perturbations.items():  # only one perturbation is taken into account at the moment
                    assert isinstance(moment, str)
                    if pert is None:
                        continue
                    assert isinstance(pert, Perturbation)

                    if moment == "n":
                        _fun = TransformedPformComponent(
                            pert,
                            pert.given_in_basis,
                            "0",
                            comp=pert.comp,
                            domain=self.domain,
                        )
                    elif moment == "u1":
                        _fun = TransformedPformComponent(
                            pert,
                            pert.given_in_basis,
                            "v",
                            comp=pert.comp,
                            domain=self.domain,
                        )
                        _fun_cart = lambda e1, e2, e3: self.domain.push(_fun, e1, e2, e3, kind="v")
                        self._u_init = lambda e1, e2, e3: self.f0.u_cart(e1, e2, e3)[0] + _fun_cart(e1, e2, e3)
                        # TODO: add other velocity components
            else:
                _fun = None

            def _f_init(*etas, flat_eval=False):
                if len(etas) == 1:
                    if _fun is None:
                        out = self.f0.n0(etas[0])
                    else:
                        out = self.f0.n0(etas[0]) + _fun(*etas[0].T)
                else:
                    assert len(etas) == 3
                    E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
                        etas[0],
                        etas[1],
                        etas[2],
                        flat_eval=flat_eval,
                    )

                    out0 = self.f0.n0(E1, E2, E3)

                    if _fun is None:
                        out = out0
                    else:
                        out1 = _fun(E1, E2, E3)
                        assert out0.shape == out1.shape
                        out = out0 + out1

                    if flat_eval:
                        out = np.squeeze(out)

                return out

            self._f_init = _f_init

    def _load_external(
        self,
        n_mks_load_loc: int,
        n_mks_load_cum_sum: np.ndarray,
    ):
        """Load markers from external .hdf5 file.

        Parameters
        ----------
        n_mks_load_loc: int
            Number of markers on the local process at loading stage.

        n_mks_load_cum_sum: np.ndarray
            Cumulative sum of number of markers on each process at loading stage.
        """
        if self.mpi_rank == 0:
            file = h5py.File(
                self.loading_params.dir_external,
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

    def _load_restart(self):
        """Load markers from restart .hdf5 file."""
        # Read struphy state file
        state = utils.read_state()

        o_path = state["o_path"]

        if self.loading_params.dir_particles_abs is None:
            data_path = os.path.join(
                o_path,
                self.loading_params.dir_particles,
            )
        else:
            data_path = self.loading_params.dir_particles_abs

        data = DataContainer(data_path, comm=self.mpi_comm)
        self._markers[:, :] = data.file["restart/" + self.loading_params.restart_key][-1, :, :]

    def _load_tesselation(self, n_quad: int = 1):
        """
        Load markers on a grid defined by the center-of-mass points of a tesselation.

        Parameters
        ----------
        n_quad: int
            Number of quadrature points for the Gauss-Legendre quadrature for cell averages.
        """
        self._tesselation = Tesselation(
            self.ppb,
            comm=self.mpi_comm,
            domain_array=self.domain_array,
            sorting_boxes=self.sorting_boxes,
        )
        eta1, eta2, eta3 = self.tesselation.draw_markers()
        self._markers[: eta1.size, 0] = eta1
        self._markers[: eta2.size, 1] = eta2
        self._markers[: eta3.size, 2] = eta3
        self.update_valid_mks()

    def draw_markers(
        self,
        sort: bool = True,
        verbose: bool = True,
    ):
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
        # Np_per_clone_loc = self.Np_per_clone[self.clone_id]

        # fill holes in markers array with -1 (all holes are at end of array at loading stage)
        self._markers[n_mks_load_loc:] = -1.0

        # number of holes and markers on process
        self.update_holes()
        self.update_ghost_particles()

        # cumulative sum of number of markers on each process at loading stage.
        n_mks_load_cum_sum = np.cumsum(self.n_mks_load)
        Np_per_clone_cum_sum = np.cumsum(self.Np_per_clone)
        _first_marker_id = (Np_per_clone_cum_sum - self.Np_per_clone)[self.clone_id] + (
            n_mks_load_cum_sum - self.n_mks_load
        )[self._mpi_rank]

        if self.mpi_rank == 0 and verbose:
            print("\nMARKERS:")
            print(("name:").ljust(25), self.name)
            print(("Np:").ljust(25), self.Np)
            print(("ppc:").ljust(25), self.ppc)
            print(("ppb:").ljust(25), self.ppb)
            print(("bc:").ljust(25), self.bc)
            print(("bc_refill:").ljust(25), self.bc_refill)
            print(("loading:").ljust(25), self.loading)
            print(("type:").ljust(25), self.type)
            print(("control_variate:").ljust(25), self.control_variate)
            print(("domain_array[0]:").ljust(25), self.domain_array[0])
            print(("boxes_per_dim:").ljust(25), self.boxes_per_dim)
            print(("mpi_dims_mask:").ljust(25), self.mpi_dims_mask)

        if self.loading == "external":
            self._load_external()
        elif self.loading == "restart":
            self._load_restart()
        elif self.loading == "tesselation":
            self._load_tesselation()
            if self.type == "sph":
                self._set_initial_condition()
                self.velocities = np.array(self.u_init(self.positions)[0]).T
            # set markers ID in last column
            self.marker_ids = _first_marker_id + np.arange(n_mks_load_loc, dtype=float)
        else:
            if self.mpi_rank == 0 and verbose:
                print("\nLoading fresh markers:")
                for key, val in self.loading_params.__dict__.items():
                    print((key + " :").ljust(25), val)

            # 1. standard random number generator (pseudo-random)
            if self.loading == "pseudo_random":
                # set seed
                _seed = self.loading_params.seed
                if _seed is not None:
                    np.random.seed(_seed)

                # counting integers
                num_loaded_particles_loc = 0  # number of particles alreday loaded (local)
                num_loaded_particles_glob = 0  # number of particles already loaded (each clone)
                chunk_size = 10000  # TODO: number of particle chunk

                # Total number of markers to draw (sum over all clones)
                while num_loaded_particles_glob < int(self.Np):
                    # Generate a chunk of random particles
                    num_to_add_glob = min(chunk_size, int(self.Np) - num_loaded_particles_glob)
                    temp = np.random.rand(num_to_add_glob, 3 + self.vdim)
                    # check which particles are on the current process domain
                    is_on_proc_domain = np.logical_and(
                        temp[:, :3] > self.domain_array[self.mpi_rank, 0::3],
                        temp[:, :3] < self.domain_array[self.mpi_rank, 1::3],
                    )
                    valid_idx = np.nonzero(np.all(is_on_proc_domain, axis=1))[0]
                    valid_particles = temp[valid_idx]
                    valid_particles = np.array_split(valid_particles, self.num_clones)[self.clone_id]
                    num_valid = valid_particles.shape[0]

                    # Add the valid particles to the phasespace_coords array
                    self._markers[
                        num_loaded_particles_loc : num_loaded_particles_loc + num_valid,
                        : 3 + self.vdim,
                    ] = valid_particles
                    num_loaded_particles_glob += num_to_add_glob
                    num_loaded_particles_loc += num_valid

                # make sure all particles are loaded
                assert self.Np == int(num_loaded_particles_glob), f"{self.Np = }, {int(num_loaded_particles_glob) = }"

                # set new n_mks_load
                self._gather_scalar_in_subcomm_array(num_loaded_particles_loc, out=self.n_mks_load)
                n_mks_load_loc = self.n_mks_load[self.mpi_rank]
                n_mks_load_cum_sum = np.cumsum(self.n_mks_load)

                # set new holes in markers array to -1
                self._markers[num_loaded_particles_loc:] = -1.0
                self.update_holes()

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
            if self.type == "sph":
                self._set_initial_condition()
                self.velocities = np.array(self.u_init(self.positions)[0]).T
            else:
                # inverse transform sampling in velocity space
                u_mean = np.array(self.loading_params.moments[: self.vdim])
                v_th = np.array(self.loading_params.moments[self.vdim :])

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
            if self.spatial == "disc":
                self._markers[:n_mks_load_loc, 0] = np.sqrt(
                    self._markers[:n_mks_load_loc, 0],
                )
            else:
                assert self.spatial == "uniform", f'Spatial drawing must be "uniform" or "disc", is {self.spatial}.'

            self.marker_ids = _first_marker_id + np.arange(n_mks_load_loc, dtype=float)

            # set specific initial condition for some particles
            if self.loading_params.specific_markers is not None:
                specific_markers = self.loading_params.specific_markers

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
            n_mks_load_loc = self.n_mks_load[self._mpi_rank]

            assert np.all(~self.holes[:n_mks_load_loc])
            assert np.all(self.holes[n_mks_load_loc:])

        if self._initialized_sorting and sort:
            if self.mpi_rank == 0 and verbose:
                print("Sorting the markers after initial draw")
            self.mpi_sort_markers()
            self.do_sort()

    @profile
    def mpi_sort_markers(
        self,
        apply_bc: bool = True,
        alpha: tuple | list | int | float = 1.0,
        do_test: bool = False,
        remove_ghost: bool = True,
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

        remove_ghost : bool
            Remove ghost particles before send.
        """
        if remove_ghost:
            self.remove_ghost_particles()

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
        self._markers[send_inds] = -1.0

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
                    self.positions > self.domain_array[self.mpi_rank, 0::3],
                    self.positions < self.domain_array[self.mpi_rank, 1::3],
                ),
            )

            assert all_on_right_proc
            # assert self.phasespace_coords.size > 0, f'No particles on process {self.mpi_rank}, please rebalance, aborting ...'

        self.mpi_comm.Barrier()

    def initialize_weights(
        self,
        *,
        bckgr_params: dict = None,
        pert_params: dict = None,
        # reject_weights: bool = False,
        # threshold: float = 1e-8,
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

        if self.loading == "tesselation":
            if not self.is_volume_form[0]:
                fvol = TransformedPformComponent([self.f_init], "0", "3", domain=self.domain)
            else:
                fvol = self.f_init
            cell_avg = self.tesselation.cell_averages(fvol, n_quad=self.loading_params.n_quad)
            self.weights0 = cell_avg.flatten()
        else:
            assert self.domain is not None, "A domain is needed to initialize weights."

            # set initial condition
            if bckgr_params is not None:
                self._bckgr_params = bckgr_params

            if pert_params is not None:
                self._pert_params = pert_params

            if self.type != "sph":
                self._set_initial_condition()

            # evaluate initial distribution function
            if self.type == "sph":
                f_init = self.f_init(self.positions)
            else:
                f_init = self.f_init(*self.f_coords.T)

            # if f_init is vol-form, transform to 0-form
            if self.is_volume_form[0]:
                f_init /= self.domain.jacobian_det(self.positions)

            if self.is_volume_form[1]:
                f_init /= self.f_init.velocity_jacobian_det(
                    *self.f_jacobian_coords.T,
                )

            # compute s0 and save at vdim + 4
            self.sampling_density = self.s0(*self.phasespace_coords.T, flat_eval=True)

            # compute w0 and save at vdim + 5
            self.weights0 = f_init / self.sampling_density

        if self.reject_weights:
            reject = self.markers[:, self.index["w0"]] < self.threshold
            self._markers[reject] = -1.0
            self.update_holes()
            self.reset_marker_ids()
            print(
                f"\nWeights < {self.threshold} have been rejected, number of valid markers on process {self.mpi_rank} is {self.n_mks_loc}."
            )

        # compute (time-dependent) weights at vdim + 3
        if self.control_variate:
            self.update_weights()
        else:
            self.weights = self.weights0

    @profile
    def update_weights(self):
        """
        Applies the control variate method, i.e. updates the time-dependent marker weights
        according to the algorithm in :ref:`control_var`.
        The background :attr:`~struphy.pic.base.Particles.f0` is used for this.
        """

        if self.type == "sph":
            f0 = self.f0.n0(self.positions)
        else:
            # in case of CanonicalMaxwellian, evaluate constants_of_motion
            if self.f0.coords == "constants_of_motion":
                self.save_constants_of_motion()
            f0 = self.f0(*self.f_coords.T)

        # if f_init is vol-form, transform to 0-form
        if self.is_volume_form[0]:
            f0 /= self.domain.jacobian_det(self.positions)

        if self.is_volume_form[1]:
            f0 /= self.f0.velocity_jacobian_det(*self.f_jacobian_coords.T)

        self.weights = self.weights0 - f0 / self.sampling_density

    def reset_marker_ids(self):
        """Reset the marker ids (last column in marker array) according to the current distribution of particles.
        The first marker on rank 0 gets the id '0', the last marker on the last rank gets the id 'n_mks_global - 1'."""
        n_mks_proc_cumsum = np.cumsum(self.n_mks_on_each_proc)
        n_mks_clone_cumsum = np.cumsum(self.n_mks_on_each_clone)
        first_marker_id = (n_mks_clone_cumsum - self.n_mks_on_each_clone)[self.clone_id] + (
            n_mks_proc_cumsum - self.n_mks_on_each_proc
        )[self.mpi_rank]
        self.marker_ids = first_marker_id + np.arange(self.n_mks_loc, dtype=int)

    @profile
    def binning(
        self,
        components: tuple[bool],
        bin_edges: tuple[np.ndarray],
        divide_by_jac: bool = True,
    ):
        r"""Computes full-f and delta-f distribution functions via marker binning in logical space.
        Numpy's histogramdd is used, following the algorithm outlined in :ref:`binning`.

        Parameters
        ----------
        components : tuple[bool]
            List of length 3 + vdim; an entry is True if the direction in phase space is to be binned.

        bin_edges : tuple[array]
            List of bin edges (resolution) having the length of True entries in components.

        divide_by_jac : boll
            Whether to divide the weights by the Jacobian determinant for binning.

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

        if divide_by_jac:
            _weights /= self.domain.jacobian_det(self.positions, remove_outside=False)
            # _weights /= self.velocity_jacobian_det(*self.phasespace_coords.T)

            _weights0 /= self.domain.jacobian_det(self.positions, remove_outside=False)
            # _weights0 /= self.velocity_jacobian_det(*self.phasespace_coords.T)

        f_slice = np.histogramdd(
            self.markers_wo_holes_and_ghost[:, slicing],
            bins=bin_edges,
            weights=_weights0,
        )[0]

        df_slice = np.histogramdd(
            self.markers_wo_holes_and_ghost[:, slicing],
            bins=bin_edges,
            weights=_weights,
        )[0]

        f_slice /= self.Np * bin_vol
        df_slice /= self.Np * bin_vol

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

    def _find_outside_particles(self, axis):
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

        return outside_inds

    @profile
    def apply_kinetic_bc(self, newton=False):
        """
        Apply boundary conditions to markers that are outside of the logical unit cube.

        Parameters
        ----------
        newton : bool
            Whether the shift due to boundary conditions should be computed
            for a Newton step or for a strandard (explicit or Picard) step.
        """

        # apply boundary conditions
        for axis in self._remove_axes:
            outside_inds = self._find_outside_particles(axis)

            if len(outside_inds) == 0:
                continue

            if self.bc_refill is not None:
                self.particle_refilling()

            self._markers[self._is_outside, :-1] = -1.0
            self._n_lost_markers += len(np.nonzero(self._is_outside)[0])

        for axis in self._periodic_axes:
            outside_inds = self._find_outside_particles(axis)

            if len(outside_inds) == 0:
                continue

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

        # put all coordinate inside the unit cube (avoid wrong Jacobian evaluations)
        outside_inds_per_axis = {}
        for axis in self._reflect_axes:
            outside_inds = self._find_outside_particles(axis)

            self.markers[self._is_outside_left, axis] *= -1.0
            self.markers[self._is_outside_right, axis] *= -1.0
            self.markers[self._is_outside_right, axis] += 2.0

            self.markers[self._is_outside, self.first_pusher_idx] = -1.0

            outside_inds_per_axis[axis] = outside_inds

        for axis in self._reflect_axes:
            if len(outside_inds_per_axis[axis]) == 0:
                continue
            # flip velocity
            reflect(
                self.markers,
                self.domain.args_domain,
                outside_inds_per_axis[axis],
                axis,
            )

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
                r_loss = self.domain.params["a1"]

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

        Boxes are represented as a 2D array of integers, where
        each line coresponds to one box, and all entries of line i that are not -1
        correspond to a particles in the i-th box.

        Parameters
        ----------
        markers_shape : tuple
            shape of 2D marker array.

        is_sph : bool
            True if particle type is "sph".

        nx : int
            number of boxes in the x direction.

        ny : int
            number of boxes in the y direction.

        nz : int
            number of boxes in the z direction.

        bc_sph : list
            Boundary condition for sph density evaluation.
            Either 'periodic', 'mirror' or 'fixed' in each direction.

        is_domain_boundary: dict
            Has two booleans for each direction; True when the boundary of the MPI process is a domain boundary.

        comm : Intracomm
            MPI communicator or None.

        box_index : int
            Column index of the particles array to store the box number, counted from
            the end (e.g. -2 for the second-to-last).

        box_bufsize : float
            additional buffer space in the size of the boxes"""

        def __init__(
            self,
            markers_shape: tuple,
            is_sph: bool,
            *,
            nx: int = 1,
            ny: int = 1,
            nz: int = 1,
            bc_sph: list = None,
            is_domain_boundary: dict = None,
            comm: Intracomm = None,
            box_index: "int" = -2,
            box_bufsize: "float" = 2.0,
            verbose: str = False,
        ):
            self._markers_shape = markers_shape
            self._nx = nx
            self._ny = ny
            self._nz = nz
            self._comm = comm
            self._box_index = box_index
            self._box_bufsize = box_bufsize
            self._verbose = verbose

            if bc_sph is None:
                bc_sph = ["periodic"] * 3
            self._bc_sph = bc_sph

            if is_domain_boundary is None:
                is_domain_boundary = {}
                is_domain_boundary["x_m"] = True
                is_domain_boundary["x_p"] = True
                is_domain_boundary["y_m"] = True
                is_domain_boundary["y_p"] = True
                is_domain_boundary["z_m"] = True
                is_domain_boundary["z_p"] = True

            self._is_domain_boundary = is_domain_boundary

            if comm is None:
                self._rank = 0
            else:
                self._rank = comm.Get_rank()

            self._set_boxes()

            self._communicate = is_sph

            if self.communicate:
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
        def comm(self):
            return self._comm

        @property
        def box_index(self):
            return self._box_index

        @property
        def boxes(self):
            if not hasattr(self, "_boxes"):
                self._set_boxes()
            return self._boxes

        @property
        def neighbours(self):
            if not hasattr(self, "_neighbours"):
                self._set_boxes()
            return self._neighbours

        @property
        def communicate(self):
            return self._communicate

        @property
        def is_domain_boundary(self):
            """Dict with two booleans for each direction (e.g. 'x_m' and 'x_p'); True when the boundary of the MPI process is a domain boundary (0.0 or 1.0)."""
            return self._is_domain_boundary

        @property
        def bc_sph(self):
            """List of boundary conditions for sph evaluation in each direction."""
            return self._bc_sph

        @property
        def bc_sph_index_shifts(self):
            """Dictionary holding the index shifts of box number for ghost particles in each direction."""
            if not hasattr(self, "_bc_sph_index_shifts"):
                self._compute_sph_index_shifts()
            return self._bc_sph_index_shifts

        def _compute_sph_index_shifts(self):
            """The index shifts are applied to ghost particles to indicate their new box after sending."""
            self._bc_sph_index_shifts = {}
            self._bc_sph_index_shifts["x_m"] = flatten_index(self.nx, 0, 0, self.nx, self.ny, self.nz)
            self._bc_sph_index_shifts["x_p"] = flatten_index(self.nx, 0, 0, self.nx, self.ny, self.nz)
            self._bc_sph_index_shifts["y_m"] = flatten_index(0, self.ny, 0, self.nx, self.ny, self.nz)
            self._bc_sph_index_shifts["y_p"] = flatten_index(0, self.ny, 0, self.nx, self.ny, self.nz)
            self._bc_sph_index_shifts["z_m"] = flatten_index(0, 0, self.nz, self.nx, self.ny, self.nz)
            self._bc_sph_index_shifts["z_p"] = flatten_index(0, 0, self.nz, self.nx, self.ny, self.nz)

            if self.bc_sph[0] in ("mirror", "fixed"):
                if self.is_domain_boundary["x_m"]:
                    self._bc_sph_index_shifts["x_m"] = flatten_index(-1, 0, 0, self.nx, self.ny, self.nz)
                if self.is_domain_boundary["x_p"]:
                    self._bc_sph_index_shifts["x_p"] = flatten_index(-1, 0, 0, self.nx, self.ny, self.nz)

            if self.bc_sph[1] in ("mirror", "fixed"):
                if self.is_domain_boundary["y_m"]:
                    self._bc_sph_index_shifts["y_m"] = flatten_index(0, -1, 0, self.nx, self.ny, self.nz)
                if self.is_domain_boundary["y_p"]:
                    self._bc_sph_index_shifts["y_p"] = flatten_index(0, -1, 0, self.nx, self.ny, self.nz)

            if self.bc_sph[2] in ("mirror", "fixed"):
                if self.is_domain_boundary["z_m"]:
                    self._bc_sph_index_shifts["z_m"] = flatten_index(0, 0, -1, self.nx, self.ny, self.nz)
                if self.is_domain_boundary["z_p"]:
                    self._bc_sph_index_shifts["z_p"] = flatten_index(0, 0, -1, self.nx, self.ny, self.nz)

        def _set_boxes(self):
            """ "(Re)set the box structure."""
            self._n_boxes = (self._nx + 2) * (self._ny + 2) * (self._nz + 2)
            n_box_in = self._nx * self._ny * self._nz

            n_particles = self._markers_shape[0]
            n_mkr = int(n_particles / n_box_in) + 1
            n_cols = round(
                n_mkr * (1 + 1 / np.sqrt(n_mkr) + self._box_bufsize),
            )

            # cartesian boxes
            self._boxes = np.zeros((self._n_boxes + 1, n_cols), dtype=int)

            # TODO: there is still a bug here
            # the row number in self._boxes should not be n_boxes + 1; this is just a temporary fix to avoid an error that I dont understand.
            # Must be fixed soon!

            self._next_index = np.zeros((self._n_boxes + 1), dtype=int)
            self._cumul_next_index = np.zeros((self._n_boxes + 2), dtype=int)
            self._neighbours = np.zeros((self._n_boxes, 27), dtype=int)

            # A particle on box i only sees particles in boxes that belong to neighbours[i]
            initialize_neighbours(self._neighbours, self.nx, self.ny, self.nz)
            # print(f"{self._rank = }\n{self._neighbours = }")

            self._swap_line_1 = np.zeros(self._markers_shape[1])
            self._swap_line_2 = np.zeros(self._markers_shape[1])

        def _set_boundary_boxes(self):
            """Gather all the boxes that are part of a boundary"""
            gather_x_boxes = self.nx > 1
            gather_y_boxes = self.ny > 1
            gather_z_boxes = self.nz > 1

            # x boundary
            # negative direction
            self._bnd_boxes_x_m = []
            # positive direction
            self._bnd_boxes_x_p = []

            if gather_x_boxes:
                for j in range(1, self.ny + 1):
                    for k in range(1, self.nz + 1):
                        self._bnd_boxes_x_m.append(flatten_index(1, j, k, self.nx, self.ny, self.nz))
                        self._bnd_boxes_x_p.append(flatten_index(self.nx, j, k, self.nx, self.ny, self.nz))

            if self._verbose:
                print(f"eta1 boundary on {self._rank = }:\n{self._bnd_boxes_x_m = }\n{self._bnd_boxes_x_p = }")

            # y boundary
            # negative direction
            self._bnd_boxes_y_m = []
            # positive direction
            self._bnd_boxes_y_p = []

            if gather_y_boxes:
                for i in range(1, self.nx + 1):
                    for k in range(1, self.nz + 1):
                        self._bnd_boxes_y_m.append(flatten_index(i, 1, k, self.nx, self.ny, self.nz))
                        self._bnd_boxes_y_p.append(flatten_index(i, self.ny, k, self.nx, self.ny, self.nz))

            if self._verbose:
                print(f"eta2 boundary on {self._rank = }:\n{self._bnd_boxes_y_m = }\n{self._bnd_boxes_y_p = }")

            # z boundary
            # negative direction
            self._bnd_boxes_z_m = []
            # positive direction
            self._bnd_boxes_z_p = []

            if gather_z_boxes:
                for i in range(1, self.nx + 1):
                    for j in range(1, self.ny + 1):
                        self._bnd_boxes_z_m.append(flatten_index(i, j, 1, self.nx, self.ny, self.nz))
                        self._bnd_boxes_z_p.append(flatten_index(i, j, self.nz, self.nx, self.ny, self.nz))

            if self._verbose:
                print(f"eta3 boundary on {self._rank = }:\n{self._bnd_boxes_z_m = }\n{self._bnd_boxes_z_p = }")

            # x-y edges
            self._bnd_boxes_x_m_y_m = []
            self._bnd_boxes_x_m_y_p = []
            self._bnd_boxes_x_p_y_m = []
            self._bnd_boxes_x_p_y_p = []

            if gather_x_boxes and gather_y_boxes:
                for k in range(1, self.nz + 1):
                    self._bnd_boxes_x_m_y_m.append(flatten_index(1, 1, k, self.nx, self.ny, self.nz))
                    self._bnd_boxes_x_m_y_p.append(flatten_index(1, self.ny, k, self.nx, self.ny, self.nz))
                    self._bnd_boxes_x_p_y_m.append(flatten_index(self.nx, 1, k, self.nx, self.ny, self.nz))
                    self._bnd_boxes_x_p_y_p.append(flatten_index(self.nx, self.ny, k, self.nx, self.ny, self.nz))

            if self._verbose:
                print(
                    (
                        f"eta1-eta2 edge on {self._rank = }:\n{self._bnd_boxes_x_m_y_m = }"
                        f"\n{self._bnd_boxes_x_m_y_p = }"
                        f"\n{self._bnd_boxes_x_p_y_m = }"
                        f"\n{self._bnd_boxes_x_p_y_p = }"
                    )
                )

            # x-z edges
            self._bnd_boxes_x_m_z_m = []
            self._bnd_boxes_x_m_z_p = []
            self._bnd_boxes_x_p_z_m = []
            self._bnd_boxes_x_p_z_p = []

            if gather_x_boxes and gather_z_boxes:
                for j in range(1, self.ny + 1):
                    self._bnd_boxes_x_m_z_m.append(flatten_index(1, j, 1, self.nx, self.ny, self.nz))
                    self._bnd_boxes_x_m_z_p.append(flatten_index(1, j, self.nz, self.nx, self.ny, self.nz))
                    self._bnd_boxes_x_p_z_m.append(flatten_index(self.nx, j, 1, self.nx, self.ny, self.nz))
                    self._bnd_boxes_x_p_z_p.append(flatten_index(self.nx, j, self.nz, self.nx, self.ny, self.nz))

            if self._verbose:
                print(
                    (
                        f"eta1-eta3 edge on {self._rank = }:\n{self._bnd_boxes_x_m_z_m = }"
                        f"\n{self._bnd_boxes_x_m_z_p = }"
                        f"\n{self._bnd_boxes_x_p_z_m = }"
                        f"\n{self._bnd_boxes_x_p_z_p = }"
                    )
                )

            # y-z edges
            self._bnd_boxes_y_m_z_m = []
            self._bnd_boxes_y_m_z_p = []
            self._bnd_boxes_y_p_z_m = []
            self._bnd_boxes_y_p_z_p = []

            if gather_y_boxes and gather_z_boxes:
                for i in range(1, self.nx + 1):
                    self._bnd_boxes_y_m_z_m.append(flatten_index(i, 1, 1, self.nx, self.ny, self.nz))
                    self._bnd_boxes_y_m_z_p.append(flatten_index(i, 1, self.nz, self.nx, self.ny, self.nz))
                    self._bnd_boxes_y_p_z_m.append(flatten_index(i, self.ny, 1, self.nx, self.ny, self.nz))
                    self._bnd_boxes_y_p_z_p.append(flatten_index(i, self.ny, self.nz, self.nx, self.ny, self.nz))

            if self._verbose:
                print(
                    (
                        f"eta2-eta3 edge on {self._rank = }:\n{self._bnd_boxes_y_m_z_m = }"
                        f"\n{self._bnd_boxes_y_m_z_p = }"
                        f"\n{self._bnd_boxes_y_p_z_m = }"
                        f"\n{self._bnd_boxes_y_p_z_p = }"
                    )
                )

            # corners
            self._bnd_boxes_x_m_y_m_z_m = []
            self._bnd_boxes_x_m_y_m_z_p = []
            self._bnd_boxes_x_m_y_p_z_m = []
            self._bnd_boxes_x_p_y_m_z_m = []
            self._bnd_boxes_x_m_y_p_z_p = []
            self._bnd_boxes_x_p_y_m_z_p = []
            self._bnd_boxes_x_p_y_p_z_m = []
            self._bnd_boxes_x_p_y_p_z_p = []

            if gather_x_boxes and gather_y_boxes and gather_z_boxes:
                self._bnd_boxes_x_m_y_m_z_m = [flatten_index(1, 1, 1, self.nx, self.ny, self.nz)]
                self._bnd_boxes_x_m_y_m_z_p = [flatten_index(1, 1, self.nz, self.nx, self.ny, self.nz)]
                self._bnd_boxes_x_m_y_p_z_m = [flatten_index(1, self.ny, 1, self.nx, self.ny, self.nz)]
                self._bnd_boxes_x_p_y_m_z_m = [flatten_index(self.nx, 1, 1, self.nx, self.ny, self.nz)]
                self._bnd_boxes_x_m_y_p_z_p = [flatten_index(1, self.ny, self.nz, self.nx, self.ny, self.nz)]
                self._bnd_boxes_x_p_y_m_z_p = [flatten_index(self.nx, 1, self.nz, self.nx, self.ny, self.nz)]
                self._bnd_boxes_x_p_y_p_z_m = [flatten_index(self.nx, self.ny, 1, self.nx, self.ny, self.nz)]
                self._bnd_boxes_x_p_y_p_z_p = [flatten_index(self.nx, self.ny, self.nz, self.nx, self.ny, self.nz)]

            if self._verbose:
                print(
                    (
                        f"corners on {self._rank = }:\n{self._bnd_boxes_x_m_y_m_z_m = }"
                        f"\n{self._bnd_boxes_x_m_y_m_z_p = }"
                        f"\n{self._bnd_boxes_x_m_y_p_z_m = }"
                        f"\n{self._bnd_boxes_x_p_y_m_z_m = }"
                        f"\n{self._bnd_boxes_x_m_y_p_z_p = }"
                        f"\n{self._bnd_boxes_x_p_y_m_z_p = }"
                        f"\n{self._bnd_boxes_x_p_y_p_z_m = }"
                        f"\n{self._bnd_boxes_x_p_y_p_z_p = }"
                    )
                )

    def _sort_boxed_particles_numpy(self):
        """Sort the particles by box using numpy.argsort."""
        sorting_axis = self._sorting_boxes.box_index

        if not hasattr(self, "_argsort_array"):
            self._argsort_array = np.zeros(self.markers.shape[0], dtype=int)
        self._argsort_array[:] = self._markers[:, sorting_axis].argsort()

        self._markers[:, :] = self._markers[self._argsort_array]

    @profile
    def put_particles_in_boxes(self):
        """Assign the right box to the particles and the list of the particles to each box.
        If sorting_boxes was instantiated with an MPI comm, then the particles in the
        neighbouring boxes of neighbours processors or also communicated"""
        self.remove_ghost_particles()

        assign_box_to_each_particle(
            self.markers,
            self.holes,
            self._sorting_boxes.nx,
            self._sorting_boxes.ny,
            self._sorting_boxes.nz,
            self.domain_array[self.mpi_rank],
        )

        self.check_and_assign_particles_to_boxes()

        if self.sorting_boxes.communicate:
            self.communicate_boxes(verbose=self.verbose)
            self.check_and_assign_particles_to_boxes()
            self.update_ghost_particles()

        # if self.verbose:
        #     valid_box_ids = np.nonzero(self._sorting_boxes._boxes[:, 0] != -1)[0]
        #     print(f"Boxes holding at least one particle: {valid_box_ids}")
        #     for i in valid_box_ids:
        #         n_mks_box = np.count_nonzero(self._sorting_boxes._boxes[i] != -1)
        #         print(f"Number of markers in box {i} is {n_mks_box}")

    def check_and_assign_particles_to_boxes(self):
        """Check whether the box array has enough columns (detect load imbalance wrt to sorting boxes),
        and then assigne the particles to boxes."""

        bcount = np.bincount(np.int64(self.markers_wo_holes[:, -2]))
        max_in_box = np.max(bcount)
        if max_in_box > self._sorting_boxes.boxes.shape[1]:
            warnings.warn(
                f'Strong load imbalance detected in sorting boxes: \
max number of markers in a box ({max_in_box}) on rank {self.mpi_rank} \
exceeds the column-size of the box array ({self._sorting_boxes.boxes.shape[1]}). \
Increasing the value of "box_bufsize" in the markers parameters for the next run.'
            )
            self.mpi_comm.Abort()

        assign_particles_to_boxes(
            self.markers,
            self.holes,
            self._sorting_boxes._boxes,
            self._sorting_boxes._next_index,
        )

    @profile
    def do_sort(self, use_numpy_argsort=False):
        """Assign the particles to boxes and then sort them."""
        nx = self._sorting_boxes.nx
        ny = self._sorting_boxes.ny
        nz = self._sorting_boxes.nz
        nboxes = (nx + 2) * (ny + 2) * (nz + 2)

        self.put_particles_in_boxes()

        if use_numpy_argsort:
            self._sort_boxed_particles_numpy()
        else:
            sort_boxed_particles(
                self._markers,
                self._sorting_boxes._swap_line_1,
                self._sorting_boxes._swap_line_2,
                nboxes + 1,
                self._sorting_boxes._next_index,
                self._sorting_boxes._cumul_next_index,
            )

        if self.sorting_boxes.communicate:
            self.update_ghost_particles()

    def remove_ghost_particles(self):
        self.update_ghost_particles()
        new_holes = np.nonzero(self.ghost_particles)
        self._markers[new_holes] = -1.0
        self.update_holes()

    def prepare_ghost_particles(self):
        """Markers for boundary conditions and MPI communication.

        Does the following:
        1. determine which markers belong to boxes that are at the boundary and put these markers in a new array (e.g. markers_x_m)
        2. set their last index to -2 to indicate that they will be "ghost particles" after sending
        3. set their new box number (boundary conditions enter here)
        4. optional: mirror position for boundary conditions
        """
        shifts = self.sorting_boxes.bc_sph_index_shifts
        # if self.verbose:
        #     print(f"{self.sorting_boxes.bc_sph_index_shifts = }")

        ## Faces

        # ghost marker arrays
        self._markers_x_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_m)
        self._markers_x_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_p)
        self._markers_y_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_y_m)
        self._markers_y_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_y_p)
        self._markers_z_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_z_m)
        self._markers_z_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_z_p)

        # Put last index to -2 to indicate that they are ghosts on the new process
        self._markers_x_m[:, -1] = -2.0
        self._markers_x_p[:, -1] = -2.0
        self._markers_y_m[:, -1] = -2.0
        self._markers_y_p[:, -1] = -2.0
        self._markers_z_m[:, -1] = -2.0
        self._markers_z_p[:, -1] = -2.0

        # Adjust box number
        self._markers_x_m[:, self._sorting_boxes.box_index] += shifts["x_m"]
        self._markers_x_p[:, self._sorting_boxes.box_index] -= shifts["x_p"]
        self._markers_y_m[:, self._sorting_boxes.box_index] += shifts["y_m"]
        self._markers_y_p[:, self._sorting_boxes.box_index] -= shifts["y_p"]
        self._markers_z_m[:, self._sorting_boxes.box_index] += shifts["z_m"]
        self._markers_z_p[:, self._sorting_boxes.box_index] -= shifts["z_p"]

        # Mirror position for boundary condition
        if self.bc_sph[0] in ("mirror", "fixed"):
            self._mirror_particles(
                "_markers_x_m", "_markers_x_p", is_domain_boundary=self.sorting_boxes.is_domain_boundary
            )

        if self.bc_sph[1] in ("mirror", "fixed"):
            self._mirror_particles(
                "_markers_y_m", "_markers_y_p", is_domain_boundary=self.sorting_boxes.is_domain_boundary
            )

        if self.bc_sph[2] in ("mirror", "fixed"):
            self._mirror_particles(
                "_markers_z_m", "_markers_z_p", is_domain_boundary=self.sorting_boxes.is_domain_boundary
            )

        ## Edges x-y

        # ghost marker arrays
        self._markers_x_m_y_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_m_y_m)
        self._markers_x_m_y_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_m_y_p)
        self._markers_x_p_y_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_p_y_m)
        self._markers_x_p_y_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_p_y_p)

        # Put last index to -2 to indicate that they are ghosts on the new process
        self._markers_x_m_y_m[:, -1] = -2.0
        self._markers_x_m_y_p[:, -1] = -2.0
        self._markers_x_p_y_m[:, -1] = -2.0
        self._markers_x_p_y_p[:, -1] = -2.0

        # Adjust box number
        self._markers_x_m_y_m[:, self._sorting_boxes.box_index] += shifts["x_m"] + shifts["y_m"]
        self._markers_x_m_y_p[:, self._sorting_boxes.box_index] += shifts["x_m"] - shifts["y_p"]
        self._markers_x_p_y_m[:, self._sorting_boxes.box_index] += -shifts["x_p"] + shifts["y_m"]
        self._markers_x_p_y_p[:, self._sorting_boxes.box_index] += -shifts["x_p"] - shifts["y_p"]

        # Mirror position for boundary condition
        if self.bc_sph[0] in ("mirror", "fixed") or self.bc_sph[1] in ("mirror", "fixed"):
            self._mirror_particles(
                "_markers_x_m_y_m",
                "_markers_x_m_y_p",
                "_markers_x_p_y_m",
                "_markers_x_p_y_p",
                is_domain_boundary=self.sorting_boxes.is_domain_boundary,
            )

        ## Edges x-z

        # ghost marker arrays
        self._markers_x_m_z_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_m_z_m)
        self._markers_x_m_z_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_m_z_p)
        self._markers_x_p_z_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_p_z_m)
        self._markers_x_p_z_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_p_z_p)

        # Put last index to -2 to indicate that they are ghosts on the new process
        self._markers_x_m_z_m[:, -1] = -2.0
        self._markers_x_m_z_p[:, -1] = -2.0
        self._markers_x_p_z_m[:, -1] = -2.0
        self._markers_x_p_z_p[:, -1] = -2.0

        # Adjust box number
        self._markers_x_m_z_m[:, self._sorting_boxes.box_index] += shifts["x_m"] + shifts["z_m"]
        self._markers_x_m_z_p[:, self._sorting_boxes.box_index] += shifts["x_m"] - shifts["z_p"]
        self._markers_x_p_z_m[:, self._sorting_boxes.box_index] += -shifts["x_p"] + shifts["z_m"]
        self._markers_x_p_z_p[:, self._sorting_boxes.box_index] += -shifts["x_p"] - shifts["z_p"]

        # Mirror position for boundary condition
        if self.bc_sph[0] in ("mirror", "fixed") or self.bc_sph[2] in ("mirror", "fixed"):
            self._mirror_particles(
                "_markers_x_m_z_m",
                "_markers_x_m_z_p",
                "_markers_x_p_z_m",
                "_markers_x_p_z_p",
                is_domain_boundary=self.sorting_boxes.is_domain_boundary,
            )

        ## Edges y-z

        # ghost marker arrays
        self._markers_y_m_z_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_y_m_z_m)
        self._markers_y_m_z_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_y_m_z_p)
        self._markers_y_p_z_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_y_p_z_m)
        self._markers_y_p_z_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_y_p_z_p)

        # Put last index to -2 to indicate that they are ghosts on the new process
        self._markers_y_m_z_m[:, -1] = -2.0
        self._markers_y_m_z_p[:, -1] = -2.0
        self._markers_y_p_z_m[:, -1] = -2.0
        self._markers_y_p_z_p[:, -1] = -2.0

        # Adjust box number
        self._markers_y_m_z_m[:, self._sorting_boxes.box_index] += shifts["y_m"] + shifts["z_m"]
        self._markers_y_m_z_p[:, self._sorting_boxes.box_index] += shifts["y_m"] - shifts["z_p"]
        self._markers_y_p_z_m[:, self._sorting_boxes.box_index] += -shifts["y_p"] + shifts["z_m"]
        self._markers_y_p_z_p[:, self._sorting_boxes.box_index] += -shifts["y_p"] - shifts["z_p"]

        # Mirror position for boundary condition
        if self.bc_sph[1] in ("mirror", "fixed") or self.bc_sph[2] in ("mirror", "fixed"):
            self._mirror_particles(
                "_markers_y_m_z_m",
                "_markers_y_m_z_p",
                "_markers_y_p_z_m",
                "_markers_y_p_z_p",
                is_domain_boundary=self.sorting_boxes.is_domain_boundary,
            )

        ## Corners

        # ghost marker arrays
        self._markers_x_m_y_m_z_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_m_y_m_z_m)
        self._markers_x_m_y_m_z_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_m_y_m_z_p)
        self._markers_x_m_y_p_z_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_m_y_p_z_m)
        self._markers_x_m_y_p_z_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_m_y_p_z_p)
        self._markers_x_p_y_m_z_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_p_y_m_z_m)
        self._markers_x_p_y_m_z_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_p_y_m_z_p)
        self._markers_x_p_y_p_z_m = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_p_y_p_z_m)
        self._markers_x_p_y_p_z_p = self.determine_markers_in_box(self._sorting_boxes._bnd_boxes_x_p_y_p_z_p)

        # Put last index to -2 to indicate that they are ghosts on the new process
        self._markers_x_m_y_m_z_m[:, -1] = -2.0
        self._markers_x_m_y_m_z_p[:, -1] = -2.0
        self._markers_x_m_y_p_z_m[:, -1] = -2.0
        self._markers_x_m_y_p_z_p[:, -1] = -2.0
        self._markers_x_p_y_m_z_m[:, -1] = -2.0
        self._markers_x_p_y_m_z_p[:, -1] = -2.0
        self._markers_x_p_y_p_z_m[:, -1] = -2.0
        self._markers_x_p_y_p_z_p[:, -1] = -2.0

        # Adjust box number
        self._markers_x_m_y_m_z_m[:, self._sorting_boxes.box_index] += shifts["x_m"] + shifts["y_m"] + shifts["z_m"]
        self._markers_x_m_y_m_z_p[:, self._sorting_boxes.box_index] += shifts["x_m"] + shifts["y_m"] - shifts["z_p"]
        self._markers_x_m_y_p_z_m[:, self._sorting_boxes.box_index] += shifts["x_m"] - shifts["y_p"] + shifts["z_m"]
        self._markers_x_m_y_p_z_p[:, self._sorting_boxes.box_index] += shifts["x_m"] - shifts["y_p"] - shifts["z_p"]
        self._markers_x_p_y_m_z_m[:, self._sorting_boxes.box_index] += -shifts["x_p"] + shifts["y_m"] + shifts["z_m"]
        self._markers_x_p_y_m_z_p[:, self._sorting_boxes.box_index] += -shifts["x_p"] + shifts["y_m"] - shifts["z_p"]
        self._markers_x_p_y_p_z_m[:, self._sorting_boxes.box_index] += -shifts["x_p"] - shifts["y_p"] + shifts["z_m"]
        self._markers_x_p_y_p_z_p[:, self._sorting_boxes.box_index] += -shifts["x_p"] - shifts["y_p"] - shifts["z_p"]

        # Mirror position for boundary condition
        if any([bci in ("mirror", "fixed") for bci in self.bc_sph]):
            self._mirror_particles(
                "_markers_x_m_y_m_z_m",
                "_markers_x_m_y_m_z_p",
                "_markers_x_m_y_p_z_m",
                "_markers_x_m_y_p_z_p",
                "_markers_x_p_y_m_z_m",
                "_markers_x_p_y_m_z_p",
                "_markers_x_p_y_p_z_m",
                "_markers_x_p_y_p_z_p",
                is_domain_boundary=self.sorting_boxes.is_domain_boundary,
            )

    def _mirror_particles(self, *marker_array_names, is_domain_boundary=None):
        self._fixed_markers_set = {}

        for arr_name in marker_array_names:
            assert isinstance(arr_name, str)
            arr = getattr(self, arr_name)

            if arr.size == 0:
                continue

            # x-direction
            if self.bc_sph[0] in ("mirror", "fixed"):
                if "x_m" in arr_name and is_domain_boundary["x_m"]:
                    arr[:, 0] *= -1.0
                    if self.bc_sph[0] == "fixed" and arr_name not in self._fixed_markers_set:
                        boundary_values = self.f_init(
                            *arr[:, :3].T, flat_eval=True
                        )  # evaluation outside of the unit cube - maybe not working for all f_init!
                        arr[:, self.index["weights"]] = -boundary_values / self.s0(
                            *arr[:, :3].T,
                            flat_eval=True,
                            remove_holes=False,
                        )
                        self._fixed_markers_set[arr_name] = True
                elif "x_p" in arr_name and is_domain_boundary["x_p"]:
                    arr[:, 0] = 2.0 - arr[:, 0]
                    if self.bc_sph[0] == "fixed" and arr_name not in self._fixed_markers_set:
                        boundary_values = self.f_init(
                            *arr[:, :3].T, flat_eval=True
                        )  # evaluation outside of the unit cube - maybe not working for all f_init!
                        arr[:, self.index["weights"]] = -boundary_values / self.s0(
                            *arr[:, :3].T,
                            flat_eval=True,
                            remove_holes=False,
                        )
                        self._fixed_markers_set[arr_name] = True

            # y-direction
            if self.bc_sph[1] in ("mirror", "fixed"):
                if "y_m" in arr_name and is_domain_boundary["y_m"]:
                    arr[:, 1] *= -1.0
                    if self.bc_sph[1] == "fixed" and arr_name not in self._fixed_markers_set:
                        boundary_values = self.f_init(
                            *arr[:, :3].T, flat_eval=True
                        )  # evaluation outside of the unit cube - maybe not working for all f_init!
                        arr[:, self.index["weights"]] = -boundary_values / self.s0(
                            *arr[:, :3].T,
                            flat_eval=True,
                            remove_holes=False,
                        )
                        self._fixed_markers_set[arr_name] = True
                elif "y_p" in arr_name and is_domain_boundary["y_p"]:
                    arr[:, 1] = 2.0 - arr[:, 1]
                    if self.bc_sph[1] == "fixed" and arr_name not in self._fixed_markers_set:
                        boundary_values = self.f_init(
                            *arr[:, :3].T, flat_eval=True
                        )  # evaluation outside of the unit cube - maybe not working for all f_init!
                        arr[:, self.index["weights"]] = -boundary_values / self.s0(
                            *arr[:, :3].T,
                            flat_eval=True,
                            remove_holes=False,
                        )
                        self._fixed_markers_set[arr_name] = True

            # z-direction
            if self.bc_sph[2] in ("mirror", "fixed"):
                if "z_m" in arr_name and is_domain_boundary["z_m"]:
                    arr[:, 2] *= -1.0
                    if self.bc_sph[2] == "fixed" and arr_name not in self._fixed_markers_set:
                        boundary_values = self.f_init(
                            *arr[:, :3].T, flat_eval=True
                        )  # evaluation outside of the unit cube - maybe not working for all f_init!
                        arr[:, self.index["weights"]] = -boundary_values / self.s0(
                            *arr[:, :3].T,
                            flat_eval=True,
                            remove_holes=False,
                        )
                        self._fixed_markers_set[arr_name] = True
                elif "z_p" in arr_name and is_domain_boundary["z_p"]:
                    arr[:, 2] = 2.0 - arr[:, 2]
                    if self.bc_sph[2] == "fixed" and arr_name not in self._fixed_markers_set:
                        boundary_values = self.f_init(
                            *arr[:, :3].T, flat_eval=True
                        )  # evaluation outside of the unit cube - maybe not working for all f_init!
                        arr[:, self.index["weights"]] = -boundary_values / self.s0(
                            *arr[:, :3].T,
                            flat_eval=True,
                            remove_holes=False,
                        )
                        self._fixed_markers_set[arr_name] = True

    def determine_markers_in_box(self, list_boxes):
        """Determine the markers that belong to a certain box (list of boxes) and put them in an array"""
        indices = []
        for i in list_boxes:
            indices += list(self._sorting_boxes._boxes[i][self._sorting_boxes._boxes[i] != -1])

        indices = np.array(indices, dtype=int)
        markers_in_box = self.markers[indices]
        return markers_in_box

    def get_destinations_box(self):
        """Find the destination proc for the particles to communicate for the box structure."""
        self._send_info_box = np.zeros(self.mpi_size, dtype=int)
        self._send_list_box = [np.zeros((0, self.n_cols))] * self.mpi_size

        # Faces
        # if self._x_m_proc is not None:
        self._send_info_box[self._x_m_proc] += len(self._markers_x_m)
        self._send_list_box[self._x_m_proc] = np.concatenate((self._send_list_box[self._x_m_proc], self._markers_x_m))

        # if self._x_p_proc is not None:
        self._send_info_box[self._x_p_proc] += len(self._markers_x_p)
        self._send_list_box[self._x_p_proc] = np.concatenate((self._send_list_box[self._x_p_proc], self._markers_x_p))

        # if self._y_m_proc is not None:
        self._send_info_box[self._y_m_proc] += len(self._markers_y_m)
        self._send_list_box[self._y_m_proc] = np.concatenate((self._send_list_box[self._y_m_proc], self._markers_y_m))

        # if self._y_p_proc is not None:
        self._send_info_box[self._y_p_proc] += len(self._markers_y_p)
        self._send_list_box[self._y_p_proc] = np.concatenate((self._send_list_box[self._y_p_proc], self._markers_y_p))

        # if self._z_m_proc is not None:
        self._send_info_box[self._z_m_proc] += len(self._markers_z_m)
        self._send_list_box[self._z_m_proc] = np.concatenate((self._send_list_box[self._z_m_proc], self._markers_z_m))

        # if self._z_p_proc is not None:
        self._send_info_box[self._z_p_proc] += len(self._markers_z_p)
        self._send_list_box[self._z_p_proc] = np.concatenate((self._send_list_box[self._z_p_proc], self._markers_z_p))

        # x-y edges
        # if self._x_m_y_m_proc is not None:
        self._send_info_box[self._x_m_y_m_proc] += len(self._markers_x_m_y_m)
        self._send_list_box[self._x_m_y_m_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_m_proc], self._markers_x_m_y_m)
        )

        # if self._x_m_y_p_proc is not None:
        self._send_info_box[self._x_m_y_p_proc] += len(self._markers_x_m_y_p)
        self._send_list_box[self._x_m_y_p_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_p_proc], self._markers_x_m_y_p)
        )

        # if self._x_p_y_m_proc is not None:
        self._send_info_box[self._x_p_y_m_proc] += len(self._markers_x_p_y_m)
        self._send_list_box[self._x_p_y_m_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_m_proc], self._markers_x_p_y_m)
        )

        # if self._x_p_y_p_proc is not None:
        self._send_info_box[self._x_p_y_p_proc] += len(self._markers_x_p_y_p)
        self._send_list_box[self._x_p_y_p_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_p_proc], self._markers_x_p_y_p)
        )

        # x-z edges
        # if self._x_m_z_m_proc is not None:
        self._send_info_box[self._x_m_z_m_proc] += len(self._markers_x_m_z_m)
        self._send_list_box[self._x_m_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_m_z_m_proc], self._markers_x_m_z_m)
        )

        # if self._x_m_z_p_proc is not None:
        self._send_info_box[self._x_m_z_p_proc] += len(self._markers_x_m_z_p)
        self._send_list_box[self._x_m_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_m_z_p_proc], self._markers_x_m_z_p)
        )

        # if self._x_p_z_m_proc is not None:
        self._send_info_box[self._x_p_z_m_proc] += len(self._markers_x_p_z_m)
        self._send_list_box[self._x_p_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_p_z_m_proc], self._markers_x_p_z_m)
        )

        # if self._x_p_z_p_proc is not None:
        self._send_info_box[self._x_p_z_p_proc] += len(self._markers_x_p_z_p)
        self._send_list_box[self._x_p_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_p_z_p_proc], self._markers_x_p_z_p)
        )

        # y-z edges
        # if self._y_m_z_m_proc is not None:
        self._send_info_box[self._y_m_z_m_proc] += len(self._markers_y_m_z_m)
        self._send_list_box[self._y_m_z_m_proc] = np.concatenate(
            (self._send_list_box[self._y_m_z_m_proc], self._markers_y_m_z_m)
        )

        # if self._y_m_z_p_proc is not None:
        self._send_info_box[self._y_m_z_p_proc] += len(self._markers_y_m_z_p)
        self._send_list_box[self._y_m_z_p_proc] = np.concatenate(
            (self._send_list_box[self._y_m_z_p_proc], self._markers_y_m_z_p)
        )

        # if self._y_p_z_m_proc is not None:
        self._send_info_box[self._y_p_z_m_proc] += len(self._markers_y_p_z_m)
        self._send_list_box[self._y_p_z_m_proc] = np.concatenate(
            (self._send_list_box[self._y_p_z_m_proc], self._markers_y_p_z_m)
        )

        # if self._y_p_z_p_proc is not None:
        self._send_info_box[self._y_p_z_p_proc] += len(self._markers_y_p_z_p)
        self._send_list_box[self._y_p_z_p_proc] = np.concatenate(
            (self._send_list_box[self._y_p_z_p_proc], self._markers_y_p_z_p)
        )

        # corners
        # if self._x_m_y_m_z_m_proc is not None:
        self._send_info_box[self._x_m_y_m_z_m_proc] += len(self._markers_x_m_y_m_z_m)
        self._send_list_box[self._x_m_y_m_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_m_z_m_proc], self._markers_x_m_y_m_z_m)
        )

        # if self._x_m_y_m_z_p_proc is not None:
        self._send_info_box[self._x_m_y_m_z_p_proc] += len(self._markers_x_m_y_m_z_p)
        self._send_list_box[self._x_m_y_m_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_m_z_p_proc], self._markers_x_m_y_m_z_p)
        )

        # if self._x_m_y_p_z_m_proc is not None:
        self._send_info_box[self._x_m_y_p_z_m_proc] += len(self._markers_x_m_y_p_z_m)
        self._send_list_box[self._x_m_y_p_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_p_z_m_proc], self._markers_x_m_y_p_z_m)
        )

        # if self._x_m_y_p_z_p_proc is not None:
        self._send_info_box[self._x_m_y_p_z_p_proc] += len(self._markers_x_m_y_p_z_p)
        self._send_list_box[self._x_m_y_p_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_m_y_p_z_p_proc], self._markers_x_m_y_p_z_p)
        )

        # if self._x_p_y_m_z_m_proc is not None:
        self._send_info_box[self._x_p_y_m_z_m_proc] += len(self._markers_x_p_y_m_z_m)
        self._send_list_box[self._x_p_y_m_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_m_z_m_proc], self._markers_x_p_y_m_z_m)
        )

        # if self._x_p_y_m_z_p_proc is not None:
        self._send_info_box[self._x_p_y_m_z_p_proc] += len(self._markers_x_p_y_m_z_p)
        self._send_list_box[self._x_p_y_m_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_m_z_p_proc], self._markers_x_p_y_m_z_p)
        )

        # if self._x_p_y_p_z_m_proc is not None:
        self._send_info_box[self._x_p_y_p_z_m_proc] += len(self._markers_x_p_y_p_z_m)
        self._send_list_box[self._x_p_y_p_z_m_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_p_z_m_proc], self._markers_x_p_y_p_z_m)
        )

        # if self._x_p_y_p_z_p_proc is not None:
        self._send_info_box[self._x_p_y_p_z_p_proc] += len(self._markers_x_p_y_p_z_p)
        self._send_list_box[self._x_p_y_p_z_p_proc] = np.concatenate(
            (self._send_list_box[self._x_p_y_p_z_p_proc], self._markers_x_p_y_p_z_p)
        )

    def self_communication_boxes(self):
        """Communicate the particles in case a process is it's own neighbour
        (in case of periodicity with low number of procs/boxes)"""

        if self._send_info_box[self.mpi_rank] > 0:
            self.update_holes()
            holes_inds = np.nonzero(self.holes)[0]

            if holes_inds.size < self._send_info_box[self.mpi_rank]:
                warnings.warn(
                    f'Strong load imbalance detected: \
number of holes ({holes_inds.size}) on rank {self.mpi_rank} \
is smaller than number of incoming particles ({self._send_info_box[self.mpi_rank]}). \
Increasing the value of "bufsize" in the markers parameters for the next run.'
                )
                self.mpi_comm.Abort()

                # _tmp = self.markers.copy()
                # _n_rows_old = _tmp.shape[0]
                # print(f"old: {self.markers.shape = }")
                # self._bufsize *= 2.0
                # self._allocate_marker_array()
                # print(f"new: {self.markers.shape = }\n")
                # self.markers[:] = -1.0
                # self.markers[:_n_rows_old] = _tmp
                # self.update_holes()
                # self.update_ghost_particles()
                # self.update_valid_mks()
                # holes_inds = np.nonzero(self.holes)[0]

            self.markers[holes_inds[np.arange(self._send_info_box[self.mpi_rank])]] = self._send_list_box[self.mpi_rank]

    @profile
    def communicate_boxes(self, verbose=False):
        # if verbose:
        #     n_valid = np.count_nonzero(self.valid_mks)
        #     n_holes = np.count_nonzero(self.holes)
        #     n_ghosts = np.count_nonzero(self.ghost_particles)
        #     print(f"before communicate_boxes: {self.mpi_rank = }, {n_valid = } {n_holes = }, {n_ghosts = }")

        self.prepare_ghost_particles()
        self.get_destinations_box()
        self.self_communication_boxes()
        self.update_holes()
        if self.mpi_comm is not None:
            self.mpi_comm.Barrier()
            self.sendrecv_all_to_all_boxes()
            self.sendrecv_markers_boxes()
            self.update_holes()
        self.update_ghost_particles()

        # if verbose:
        #     n_valid = np.count_nonzero(self.valid_mks)
        #     n_holes = np.count_nonzero(self.holes)
        #     n_ghosts = np.count_nonzero(self.ghost_particles)
        #     print(f"after communicate_boxes: {self.mpi_rank = }, {n_valid = }, {n_holes = }, {n_ghosts = }")

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
                        if hole_inds.size < first_hole[i] + self._recv_info_box[i]:
                            warnings.warn(
                                f'Strong load imbalance detected: \
number of holes ({hole_inds.size}) on rank {self.mpi_rank} \
is smaller than number of incoming particles ({first_hole[i] + self._recv_info_box[i]}). \
Increasing the value of "bufsize" in the markers parameters for the next run.'
                            )
                            self.mpi_comm.Abort()
                            # exit()

                        self._markers[hole_inds[first_hole[i] + np.arange(self._recv_info_box[i])]] = recvbufs[i]

                        test_reqs.pop()
                        reqs[i] = None

        self.mpi_comm.Barrier()

    def _get_neighbouring_proc(self):
        """Find the neighbouring processes for the sending of boxes.

        The left (right) neighbour in direction 1 is called x_m_proc (x_p_proc), etc.
        By default every process is its own neighbour.
        """
        # Faces
        self._x_m_proc = None
        self._x_p_proc = None
        self._y_m_proc = None
        self._y_p_proc = None
        self._z_m_proc = None
        self._z_p_proc = None
        # Edges
        self._x_m_y_m_proc = None
        self._x_m_y_p_proc = None
        self._x_p_y_m_proc = None
        self._x_p_y_p_proc = None
        self._x_m_z_m_proc = None
        self._x_m_z_p_proc = None
        self._x_p_z_m_proc = None
        self._x_p_z_p_proc = None
        self._y_m_z_m_proc = None
        self._y_m_z_p_proc = None
        self._y_p_z_m_proc = None
        self._y_p_z_p_proc = None
        # Corners
        self._x_m_y_m_z_m_proc = None
        self._x_m_y_m_z_p_proc = None
        self._x_m_y_p_z_m_proc = None
        self._x_p_y_m_z_m_proc = None
        self._x_m_y_p_z_p_proc = None
        self._x_p_y_m_z_p_proc = None
        self._x_p_y_p_z_m_proc = None
        self._x_p_y_p_z_p_proc = None

        # periodicitiy for distance computation
        periodic1 = self.bc_sph[0] == "periodic"
        periodic2 = self.bc_sph[1] == "periodic"
        periodic3 = self.bc_sph[2] == "periodic"

        # Determine which proc are on which side
        dd = self.domain_array
        rank = self.mpi_rank

        x_l = dd[rank][0]
        x_r = dd[rank][1]
        y_l = dd[rank][3]
        y_r = dd[rank][4]
        z_l = dd[rank][6]
        z_r = dd[rank][7]
        for i in range(self.mpi_size):
            xl_i = dd[i][0]
            xr_i = dd[i][1]
            yl_i = dd[i][3]
            yr_i = dd[i][4]
            zl_i = dd[i][6]
            zr_i = dd[i][7]

            is_same_x_l = abs(distance(xl_i, x_l, periodic1)) < 1e-5
            is_same_x_r = abs(distance(xr_i, x_r, periodic1)) < 1e-5
            is_same_y_l = abs(distance(yl_i, y_l, periodic2)) < 1e-5
            is_same_y_r = abs(distance(yr_i, y_r, periodic2)) < 1e-5
            is_same_z_l = abs(distance(zl_i, z_l, periodic3)) < 1e-5
            is_same_z_r = abs(distance(zr_i, z_r, periodic3)) < 1e-5

            is_neigh_x_l = abs(distance(xr_i, x_l, periodic1)) < 1e-5
            is_neigh_x_r = abs(distance(xl_i, x_r, periodic1)) < 1e-5
            is_neigh_y_l = abs(distance(yr_i, y_l, periodic2)) < 1e-5
            is_neigh_y_r = abs(distance(yl_i, y_r, periodic2)) < 1e-5
            is_neigh_z_l = abs(distance(zr_i, z_l, periodic3)) < 1e-5
            is_neigh_z_r = abs(distance(zl_i, z_r, periodic3)) < 1e-5

            # Faces

            # Process on the left (minus axis) in the x direction
            if is_same_y_l and is_same_y_r and is_same_z_l and is_same_z_r and is_neigh_x_l:
                self._x_m_proc = i

            # Process on the right (plus axis) in the x direction
            if is_same_y_l and is_same_y_r and is_same_z_l and is_same_z_r and is_neigh_x_r:
                self._x_p_proc = i

            # Process on the left (minus axis) in the y direction
            if is_same_x_l and is_same_x_r and is_same_z_l and is_same_z_r and is_neigh_y_l:
                self._y_m_proc = i

            # Process on the right (plus axis) in the y direction
            if is_same_x_l and is_same_x_r and is_same_z_l and is_same_z_r and is_neigh_y_r:
                self._y_p_proc = i

            # Process on the left (minus axis) in the z direction
            if is_same_x_l and is_same_x_r and is_same_y_l and is_same_y_r and is_neigh_z_l:
                self._z_m_proc = i

            # Process on the right (plus axis) in the z direction
            if is_same_x_l and is_same_x_r and is_same_y_l and is_same_y_r and is_neigh_z_r:
                self._z_p_proc = i

            # Edges

            # Process on the left in x and left in y axis
            if is_same_z_l and is_same_z_r and is_neigh_x_l and is_neigh_y_l:
                self._x_m_y_m_proc = i

            # Process on the left in x and right in y axis
            if is_same_z_l and is_same_z_r and is_neigh_x_l and is_neigh_y_r:
                self._x_m_y_p_proc = i

            # Process on the right in x and left in y axis
            if is_same_z_l and is_same_z_r and is_neigh_x_r and is_neigh_y_l:
                self._x_p_y_m_proc = i

            # Process on the right in x and right in y axis
            if is_same_z_l and is_same_z_r and is_neigh_x_r and is_neigh_y_r:
                self._x_p_y_p_proc = i

            # Process on the left in x and left in z axis
            if is_same_y_l and is_same_y_r and is_neigh_x_l and is_neigh_z_l:
                self._x_m_z_m_proc = i

            # Process on the left in x and right in z axis
            if is_same_y_l and is_same_y_r and is_neigh_x_l and is_neigh_z_r:
                self._x_m_z_p_proc = i

            # Process on the right in x and left in z axis
            if is_same_y_l and is_same_y_r and is_neigh_x_r and is_neigh_z_l:
                self._x_p_z_m_proc = i

            # Process on the right in x and right in z axis
            if is_same_y_l and is_same_y_r and is_neigh_x_r and is_neigh_z_r:
                self._x_p_z_p_proc = i

            # Process on the left in y and left in z axis
            if is_same_x_l and is_same_x_r and is_neigh_y_l and is_neigh_z_l:
                self._y_m_z_m_proc = i

            # Process on the left in y and right in z axis
            if is_same_x_l and is_same_x_r and is_neigh_y_l and is_neigh_z_r:
                self._y_m_z_p_proc = i

            # Process on the right in y and left in z axis
            if is_same_x_l and is_same_x_r and is_neigh_y_r and is_neigh_z_l:
                self._y_p_z_m_proc = i

            # Process on the right in y and right in z axis
            if is_same_x_l and is_same_x_r and is_neigh_y_r and is_neigh_z_r:
                self._y_p_z_p_proc = i

            # Corners

            # Process on the left in x, left in y and left in z axis
            if is_neigh_x_l and is_neigh_y_l and is_neigh_z_l:
                self._x_m_y_m_z_m_proc = i

            # Process on the left in x, left in y and right in z axis
            if is_neigh_x_l and is_neigh_y_l and is_neigh_z_r:
                self._x_m_y_m_z_p_proc = i

            # Process on the left in x, right in y and left in z axis
            if is_neigh_x_l and is_neigh_y_r and is_neigh_z_l:
                self._x_m_y_p_z_m_proc = i

            # Process on the left in x, right in y and right in z axis
            if is_neigh_x_l and is_neigh_y_r and is_neigh_z_r:
                self._x_m_y_p_z_p_proc = i

            # Process on the right in x, left in y and left in z axis
            if is_neigh_x_r and is_neigh_y_l and is_neigh_z_l:
                self._x_p_y_m_z_m_proc = i

            # Process on the right in x, left in y and right in z axis
            if is_neigh_x_r and is_neigh_y_l and is_neigh_z_r:
                self._x_p_y_m_z_p_proc = i

            # Process on the right in x, right in y and left in z axis
            if is_neigh_x_r and is_neigh_y_r and is_neigh_z_l:
                self._x_p_y_p_z_m_proc = i

            # Process on the right in x, right in y and right in z axis
            if is_neigh_x_r and is_neigh_y_r and is_neigh_z_r:
                self._x_p_y_p_z_p_proc = i

        # set empty faces in x
        if self._x_m_proc is None:
            self._x_m_proc = rank
        if self._x_p_proc is None:
            self._x_p_proc = rank

        # set empty faces in y
        if self._y_m_proc is None:
            self._y_m_proc = rank
        if self._y_p_proc is None:
            self._y_p_proc = rank

        # set empty faces in z
        if self._z_m_proc is None:
            self._z_m_proc = rank
        if self._z_p_proc is None:
            self._z_p_proc = rank

        # set empty edges in xy
        if self._x_m_y_m_proc is None:
            if self._x_m_proc == rank:
                self._x_m_y_m_proc = self._y_m_proc
            elif self._y_m_proc == rank:
                self._x_m_y_m_proc = self._x_m_proc

        if self._x_m_y_p_proc is None:
            if self._x_m_proc == rank:
                self._x_m_y_p_proc = self._y_p_proc
            elif self._y_p_proc == rank:
                self._x_m_y_p_proc = self._x_m_proc

        if self._x_p_y_m_proc is None:
            if self._x_p_proc == rank:
                self._x_p_y_m_proc = self._y_m_proc
            elif self._y_m_proc == rank:
                self._x_p_y_m_proc = self._x_p_proc

        if self._x_p_y_p_proc is None:
            if self._x_p_proc == rank:
                self._x_p_y_p_proc = self._y_p_proc
            elif self._y_p_proc == rank:
                self._x_p_y_p_proc = self._x_p_proc

        # set empty edges in xz
        if self._x_m_z_m_proc is None:
            if self._x_m_proc == rank:
                self._x_m_z_m_proc = self._z_m_proc
            elif self._z_m_proc == rank:
                self._x_m_z_m_proc = self._x_m_proc

        if self._x_m_z_p_proc is None:
            if self._x_m_proc == rank:
                self._x_m_z_p_proc = self._z_p_proc
            elif self._z_p_proc == rank:
                self._x_m_z_p_proc = self._x_m_proc

        if self._x_p_z_m_proc is None:
            if self._x_p_proc == rank:
                self._x_p_z_m_proc = self._z_m_proc
            elif self._z_m_proc == rank:
                self._x_p_z_m_proc = self._x_p_proc

        if self._x_p_z_p_proc is None:
            if self._x_p_proc == rank:
                self._x_p_z_p_proc = self._z_p_proc
            elif self._z_p_proc == rank:
                self._x_p_z_p_proc = self._x_p_proc

        # set empty edges in yz
        if self._y_m_z_m_proc is None:
            if self._y_m_proc == rank:
                self._y_m_z_m_proc = self._z_m_proc
            elif self._z_m_proc == rank:
                self._y_m_z_m_proc = self._y_m_proc

        if self._y_m_z_p_proc is None:
            if self._y_m_proc == rank:
                self._y_m_z_p_proc = self._z_p_proc
            elif self._z_p_proc == rank:
                self._y_m_z_p_proc = self._y_m_proc

        if self._y_p_z_m_proc is None:
            if self._y_p_proc == rank:
                self._y_p_z_m_proc = self._z_m_proc
            elif self._z_m_proc == rank:
                self._y_p_z_m_proc = self._y_p_proc

        if self._y_p_z_p_proc is None:
            if self._y_p_proc == rank:
                self._y_p_z_p_proc = self._z_p_proc
            elif self._z_p_proc == rank:
                self._y_p_z_p_proc = self._y_p_proc

        # set empty corners
        if self._x_m_y_m_z_m_proc is None:
            if self._x_m_proc == rank:
                if self._y_m_proc == rank:
                    self._x_m_y_m_z_m_proc = self._z_m_proc
                elif self._z_m_proc == rank:
                    self._x_m_y_m_z_m_proc = self._y_m_proc
            elif self._y_m_proc == rank:
                if self._x_m_proc == rank:
                    self._x_m_y_m_z_m_proc = self._z_m_proc
                elif self._z_m_proc == rank:
                    self._x_m_y_m_z_m_proc = self._x_m_proc
            elif self._z_m_proc == rank:
                if self._x_m_proc == rank:
                    self._x_m_y_m_z_m_proc = self._y_m_proc
                elif self._y_m_proc == rank:
                    self._x_m_y_m_z_m_proc = self._x_m_proc

        if self._x_m_y_m_z_p_proc is None:
            if self._x_m_proc == rank:
                if self._y_m_proc == rank:
                    self._x_m_y_m_z_p_proc = self._z_p_proc
                elif self._z_p_proc == rank:
                    self._x_m_y_m_z_p_proc = self._y_m_proc
            elif self._y_m_proc == rank:
                if self._x_m_proc == rank:
                    self._x_m_y_m_z_p_proc = self._z_p_proc
                elif self._z_p_proc == rank:
                    self._x_m_y_m_z_p_proc = self._x_m_proc
            elif self._z_p_proc == rank:
                if self._x_m_proc == rank:
                    self._x_m_y_m_z_p_proc = self._y_m_proc
                elif self._y_m_proc == rank:
                    self._x_m_y_m_z_p_proc = self._x_m_proc

        if self._x_m_y_p_z_m_proc is None:
            if self._x_m_proc == rank:
                if self._y_p_proc == rank:
                    self._x_m_y_p_z_m_proc = self._z_m_proc
                elif self._z_m_proc == rank:
                    self._x_m_y_p_z_m_proc = self._y_p_proc
            elif self._y_p_proc == rank:
                if self._x_m_proc == rank:
                    self._x_m_y_p_z_m_proc = self._z_m_proc
                elif self._z_m_proc == rank:
                    self._x_m_y_p_z_m_proc = self._x_m_proc
            elif self._z_m_proc == rank:
                if self._x_m_proc == rank:
                    self._x_m_y_p_z_m_proc = self._y_p_proc
                elif self._y_p_proc == rank:
                    self._x_m_y_p_z_m_proc = self._x_m_proc

        if self._x_m_y_p_z_p_proc is None:
            if self._x_m_proc == rank:
                if self._y_p_proc == rank:
                    self._x_m_y_p_z_p_proc = self._z_p_proc
                elif self._z_p_proc == rank:
                    self._x_m_y_p_z_p_proc = self._y_p_proc
            elif self._y_p_proc == rank:
                if self._x_m_proc == rank:
                    self._x_m_y_p_z_p_proc = self._z_p_proc
                elif self._z_p_proc == rank:
                    self._x_m_y_p_z_p_proc = self._x_m_proc
            elif self._z_p_proc == rank:
                if self._x_m_proc == rank:
                    self._x_m_y_p_z_p_proc = self._y_p_proc
                elif self._y_p_proc == rank:
                    self._x_m_y_p_z_p_proc = self._x_m_proc

        if self._x_p_y_m_z_m_proc is None:
            if self._x_p_proc == rank:
                if self._y_m_proc == rank:
                    self._x_p_y_m_z_m_proc = self._z_m_proc
                elif self._z_m_proc == rank:
                    self._x_p_y_m_z_m_proc = self._y_m_proc
            elif self._y_m_proc == rank:
                if self._x_p_proc == rank:
                    self._x_p_y_m_z_m_proc = self._z_m_proc
                elif self._z_m_proc == rank:
                    self._x_p_y_m_z_m_proc = self._x_p_proc
            elif self._z_m_proc == rank:
                if self._x_p_proc == rank:
                    self._x_p_y_m_z_m_proc = self._y_m_proc
                elif self._y_m_proc == rank:
                    self._x_p_y_m_z_m_proc = self._x_p_proc

        if self._x_p_y_m_z_p_proc is None:
            if self._x_p_proc == rank:
                if self._y_m_proc == rank:
                    self._x_p_y_m_z_p_proc = self._z_p_proc
                elif self._z_p_proc == rank:
                    self._x_p_y_m_z_p_proc = self._y_m_proc
            elif self._y_m_proc == rank:
                if self._x_p_proc == rank:
                    self._x_p_y_m_z_p_proc = self._z_p_proc
                elif self._z_p_proc == rank:
                    self._x_p_y_m_z_p_proc = self._x_p_proc
            elif self._z_p_proc == rank:
                if self._x_p_proc == rank:
                    self._x_p_y_m_z_p_proc = self._y_m_proc
                elif self._y_m_proc == rank:
                    self._x_p_y_m_z_p_proc = self._x_p_proc

        if self._x_p_y_p_z_m_proc is None:
            if self._x_p_proc == rank:
                if self._y_p_proc == rank:
                    self._x_p_y_p_z_m_proc = self._z_m_proc
                elif self._z_m_proc == rank:
                    self._x_p_y_p_z_m_proc = self._y_p_proc
            elif self._y_p_proc == rank:
                if self._x_p_proc == rank:
                    self._x_p_y_p_z_m_proc = self._z_m_proc
                elif self._z_m_proc == rank:
                    self._x_p_y_p_z_m_proc = self._x_p_proc
            elif self._z_m_proc == rank:
                if self._x_p_proc == rank:
                    self._x_p_y_p_z_m_proc = self._y_p_proc
                elif self._y_p_proc == rank:
                    self._x_p_y_p_z_m_proc = self._x_p_proc

        if self._x_p_y_p_z_p_proc is None:
            if self._x_p_proc == rank:
                if self._y_p_proc == rank:
                    self._x_p_y_p_z_p_proc = self._z_p_proc
                elif self._z_p_proc == rank:
                    self._x_p_y_p_z_p_proc = self._y_p_proc
            elif self._y_p_proc == rank:
                if self._x_p_proc == rank:
                    self._x_p_y_p_z_p_proc = self._z_p_proc
                elif self._z_p_proc == rank:
                    self._x_p_y_p_z_p_proc = self._x_p_proc
            elif self._z_p_proc == rank:
                if self._x_p_proc == rank:
                    self._x_p_y_p_z_p_proc = self._y_p_proc
                elif self._y_p_proc == rank:
                    self._x_p_y_p_z_p_proc = self._x_p_proc

    def eval_density(
        self,
        eta1,
        eta2,
        eta3,
        h1,
        h2,
        h3,
        kernel_type="gaussian_1d",
        derivative=0,
        fast=True,
    ):
        """Density function as 0-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points (flat or meshgrid evaluation).

        h1, h2, h3 : float
            Support radius of the smoothing kernel in each dimension.

        kernel_type : str
            Name of the smoothing kernel to be used.

        derivative: int
            0: no kernel derivative
            1: first component of grad
            2: second component of grad
            3: third component of grad

        fast : bool
            True: box-based evaluation, False: naive evaluation.

        Returns
        -------
        out : array-like
            Same size as eta1.
        -------
        """
        return self.eval_sph(
            eta1,
            eta2,
            eta3,
            self.index["weights"],
            kernel_type=kernel_type,
            derivative=derivative,
            h1=h1,
            h2=h2,
            h3=h3,
            fast=fast,
        )

    def eval_sph(
        self,
        eta1: np.ndarray,
        eta2: np.ndarray,
        eta3: np.ndarray,
        index: int,
        out: np.ndarray = None,
        fast: bool = True,
        kernel_type: str = "gaussian_1d",
        derivative: int = "0",
        h1: float = 0.1,
        h2: float = 0.1,
        h3: float = 0.1,
    ):
        r"""Perform an SPH evaluation of a function :math:`b: [0, 1]^3 \to \mathbb R` in the following sense:

        .. math::

            b(\boldsymbol \eta_i) = \frac 1N \sum_k \beta_k W_h(\boldsymbol \eta_i - \boldsymbol \eta_k)\,.

        The coefficients :math:`\beta_k` must be stored at ``self.markers[k, index]``.
        The possible choices for :math:`W_h` are listed in :mod:`~struphy.pic.sph_smoothing_kernels`
        and in :meth:`~struphy.pic.base.Particles.ker_dct`.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        index : int
            At which index of the markers array are located the the coefficients :math:`a_k`.

        out : array_like
            Output will be store in this array. A new array is created if not provided.

        fast : bool
            If true, uses an optimized evaluation algorithm taking advantage of the box structure.
            This assume that the boxes are bigger then the radius used for the smoothing kernel.

        kernel_type : str
            Name of the smoothing kernel, see :mod:`~struphy.pic.sph_smoothing_kernels`
            and :meth:`~struphy.pic.base.Particles.ker_dct`.

        derivative: int
            0: no kernel derivative
            1: first component of grad
            2: second component of grad
            3: third component of grad

        h1, h2, h3 : float
            Radius of the smoothing kernel in each dimension.
        """
        _shp = np.shape(eta1)
        assert _shp == np.shape(eta2) == np.shape(eta3)
        if out is not None:
            assert _shp == np.shape(out)
        else:
            out = np.zeros_like(eta1)

        assert derivative in {0, 1, 2, 3}, f"derivative must be 0, 1, 2 or 3, but is {derivative}."

        ker_id = self.ker_dct()[kernel_type]
        ker_id += derivative

        # for the moment we always assume periodicity for the evaluation near the boundary, TODO: fill ghost boxes with suitable markers for other bcs?
        periodic1, periodic2, periodic3 = [True] * 3  # [bci == "periodic" for bci in self.bc]

        if fast:
            self.put_particles_in_boxes()

            if len(_shp) == 1:
                func = box_based_evaluation_flat
            elif len(_shp) == 3:
                if _shp[0] > 1:
                    assert eta1[0, 0, 0] != eta1[1, 0, 0], "Meshgrids must be obtained with indexing='ij'!"
                if _shp[1] > 1:
                    assert eta2[0, 0, 0] != eta2[0, 1, 0], "Meshgrids must be obtained with indexing='ij'!"
                if _shp[2] > 1:
                    assert eta3[0, 0, 0] != eta3[0, 0, 1], "Meshgrids must be obtained with indexing='ij'!"
                func = box_based_evaluation_meshgrid

            func(
                self.args_markers,
                eta1,
                eta2,
                eta3,
                self.sorting_boxes.nx,
                self.sorting_boxes.ny,
                self.sorting_boxes.nz,
                self.domain_array[self.mpi_rank],
                self.sorting_boxes.boxes,
                self.sorting_boxes.neighbours,
                self.holes,
                periodic1,
                periodic2,
                periodic3,
                index,
                ker_id,
                h1,
                h2,
                h3,
                out,
            )
        else:
            if len(_shp) == 1:
                func = naive_evaluation_flat
            elif len(_shp) == 3:
                func = naive_evaluation_meshgrid
            func(
                self.args_markers,
                eta1,
                eta2,
                eta3,
                self.holes,
                periodic1,
                periodic2,
                periodic3,
                index,
                ker_id,
                h1,
                h2,
                h3,
                out,
            )
        return out

    def update_holes(self):
        """Compute new holes, new number of holes and markers on process"""
        self._holes[:] = self.markers[:, 0] == -1.0
        self.update_valid_mks()

    def update_ghost_particles(self):
        """Compute new particles that belong to boundary processes needed for sph evaluation"""
        self._ghost_particles[:] = self.markers[:, -1] == -2.0
        self.update_valid_mks()

    ### MPI comm for domain decomposition ###

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
            self._sorting_etas > self.domain_array[self.mpi_rank, 0::3],
            self._sorting_etas < self.domain_array[self.mpi_rank, 1::3],
        )

        # to stay on the current process, all three columns must be True
        self._can_stay = np.all(self._is_on_proc_domain, axis=1)

        # holes and ghosts can stay, too
        self._can_stay[self.holes] = True
        self._can_stay[self.ghost_particles] = True

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
                self._sorting_etas[send_inds] > self.domain_array[i, 0::3],
                self._sorting_etas[send_inds] < self.domain_array[i, 1::3],
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
                        if hole_inds_after_send.size < first_hole[i] + recv_info[i]:
                            warnings.warn(
                                f'Strong load imbalance detected: \
number of holes ({hole_inds_after_send.size}) on rank {self.mpi_rank} \
is smaller than number of incoming particles ({first_hole[i] + recv_info[i]}). \
Increasing the value of "bufsize" in the markers parameters for the next run.'
                            )
                            self.mpi_comm.Abort()

                        self.markers[hole_inds_after_send[first_hole[i] + np.arange(recv_info[i])]] = self._recvbufs[i]

                        test_reqs.pop()
                        self._reqs[i] = None

    def _gather_scalar_in_subcomm_array(self, scalar: int, out: np.ndarray = None):
        """Return an array of length sub_comm.size, where the i-th entry corresponds to the value
        of the scalar on process i.

        Parameters
        ----------
        scalar : int
            The scalar value on each process.

        out : np.ndarray
            The returned array (optional).
        """
        if out is None:
            _tmp = np.zeros(self.mpi_size, dtype=int)
        else:
            assert out.size == self.mpi_size
            _tmp = out

        _tmp[self.mpi_rank] = scalar

        if self.mpi_comm is not None:
            self.mpi_comm.Allgather(
                _tmp[self.mpi_rank],
                _tmp,
            )

        return _tmp

    def _gather_scalar_in_intercomm_array(self, scalar: int, out: np.ndarray = None):
        """Return an array of length inter_comm.size, where the i-th entry corresponds to the value
        of the scalar on clone i.

        Parameters
        ----------
        scalar : int
            The scalar value on each clone.

        out : np.ndarray
            The returned array (optional).
        """
        if out is None:
            _tmp = np.zeros(self.num_clones, dtype=int)
        else:
            assert out.size == self.num_clones
            _tmp = out

        _tmp[self.clone_id] = scalar

        if self.clone_config is not None:
            self.clone_config.inter_comm.Allgather(
                _tmp[self.clone_id],
                _tmp,
            )

        return _tmp


class Tesselation:
    """
    Make a tesselation of the simulation domain into tiles of equal size.

    Parameters
    ----------
    tiles_pb : int
        Number of equally sized tiles per box defined in sorting boxes (there is 1 box if sorting_boxes=None).
        This is equal to particels per box (ppb) when used for SPH markers.

    comm : Intracomm
        MPI communicator.

    domain_array : np.ndarray
        A 2d array[float] of shape (comm.Get_size(), 9) holding info on the domain decomposition.

    sorting_boxes : Particles.SortingBoxes
        Box info for SPH evaluations.
    """

    def __init__(
        self,
        tiles_pb: int | float,
        *,
        comm: Intracomm = None,
        domain_array: np.ndarray = None,
        sorting_boxes: Particles.SortingBoxes = None,
    ):
        if isinstance(tiles_pb, int):
            self._tiles_pb = tiles_pb
        else:
            if tiles_pb == int(tiles_pb):
                self._tiles_pb = int(tiles_pb)
            else:
                self._tiles_pb = int(tiles_pb + 1)

        if comm is None:
            self._rank = 0
        else:
            self._rank = comm.Get_rank()
            assert domain_array is not None

        if domain_array is None:
            self._starts = np.zeros(3)
            self._ends = np.ones(3)
        else:
            self._starts = domain_array[self.rank, 0::3]
            self._ends = domain_array[self.rank, 1::3]

        if sorting_boxes is None:
            self._boxes_per_dim = [1, 1, 1]
        else:
            self._boxes_per_dim = [
                sorting_boxes.nx,
                sorting_boxes.ny,
                sorting_boxes.nz,
            ]

        self._box_widths = [(ri - le) / nb for ri, le, nb in zip(self._ends, self._starts, self.boxes_per_dim)]

        n_boxes = 1
        for nbi in self.boxes_per_dim:
            n_boxes *= nbi

        if n_boxes == 1:
            self._dims_mask = [True] * 3
        else:
            self._dims_mask = np.array(self.boxes_per_dim) > 1

        min_tiles = 2 ** np.count_nonzero(self.dims_mask)
        assert self.tiles_pb >= min_tiles, (
            f"At least {min_tiles} tiles per sorting box is enforced, but you have {self.tiles_pb}!"
        )

        self._n_tiles = n_boxes * self.tiles_pb

        self.get_tiles()

    def get_tiles(self):
        """Compute tesselation of a single sorting box."""
        # factorize tiles per box
        factors = factorint(self.tiles_pb)
        factors_vec = []
        for fac, multiplicity in factors.items():
            for m in range(multiplicity):
                factors_vec += [fac]

        # print(f'{self.tiles_pb = }')
        # print(f'{factors_vec = }')
        # print(f'{self.dims_mask = }')

        # tiles in one sorting box
        self._nt_per_dim = np.array([1, 1, 1])
        _ids = np.nonzero(self._dims_mask)[0]
        for fac in factors_vec:
            _nt = self.nt_per_dim[self._dims_mask]
            d = _ids[np.argmin(_nt)]
            self._nt_per_dim[d] *= fac
            # print(f'{_nt = }, {d = }, {self.nt_per_dim = }')

        assert np.prod(self.nt_per_dim) == self.tiles_pb

        # tiles between [0, box_width] in each direction
        self._tile_breaks = [np.linspace(0.0, bw, nt + 1) for bw, nt in zip(self.box_widths, self.nt_per_dim)]
        self._tile_midpoints = [(np.roll(tbs, -1)[:-1] + tbs[:-1]) / 2 for tbs in self.tile_breaks]
        self._tile_volume = 1.0
        for tb in self.tile_breaks:
            self._tile_volume *= tb[1]

    def draw_markers(self):
        """Draw markers on the tile midpoints."""
        _, eta1 = self._tile_output_arrays()
        eta2 = np.zeros_like(eta1)
        eta3 = np.zeros_like(eta1)

        nt_x, nt_y, nt_z = self.nt_per_dim

        for i in range(self.boxes_per_dim[0]):
            x_midpoints = self._get_midpoints(i, 0)
            for j in range(self.boxes_per_dim[1]):
                y_midpoints = self._get_midpoints(j, 1)
                for k in range(self.boxes_per_dim[2]):
                    z_midpoints = self._get_midpoints(k, 2)

                    xx, yy, zz = np.meshgrid(
                        x_midpoints,
                        y_midpoints,
                        z_midpoints,
                        indexing="ij",
                    )

                    eta1[
                        i * nt_x : (i + 1) * nt_x,
                        j * nt_y : (j + 1) * nt_y,
                        k * nt_z : (k + 1) * nt_z,
                    ] = xx

                    eta2[
                        i * nt_x : (i + 1) * nt_x,
                        j * nt_y : (j + 1) * nt_y,
                        k * nt_z : (k + 1) * nt_z,
                    ] = yy

                    eta3[
                        i * nt_x : (i + 1) * nt_x,
                        j * nt_y : (j + 1) * nt_y,
                        k * nt_z : (k + 1) * nt_z,
                    ] = zz

        return eta1.flatten(), eta2.flatten(), eta3.flatten()

    def _get_quad_pts(self, n_quad=None):
        """Compute the quadrature points and weights in a single tile."""
        if n_quad is None:
            n_quad = [1, 1, 1]
        elif isinstance(n_quad, int):
            n_quad = [n_quad] * 3

        self._tile_quad_pts = []
        self._tile_quad_wts = []
        for nq, tb in zip(n_quad, self.tile_breaks):
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(nq)
            pts, wts = quadrature_grid(tb[:2], pts_loc, wts_loc)
            self._tile_quad_pts += [pts[0]]
            self._tile_quad_wts += [wts[0]]

    def cell_averages(self, fun, n_quad=None):
        """Compute cell averages of fun over all tiles on current process.

        Parameters
        ----------
        fun: callable
            Some callable function.
        """
        self._get_quad_pts(n_quad=n_quad)
        # print(f'{self.tile_quad_pts = }')

        single_box_out, out = self._tile_output_arrays()

        nt_x, nt_y, nt_z = self.nt_per_dim

        for i in range(self.boxes_per_dim[0]):
            x_pts = self._get_box_quad_pts(i, 0)
            for j in range(self.boxes_per_dim[1]):
                y_pts = self._get_box_quad_pts(j, 1)
                for k in range(self.boxes_per_dim[2]):
                    z_pts = self._get_box_quad_pts(k, 2)

                    xx, yy, zz = np.meshgrid(
                        x_pts.flatten(),
                        y_pts.flatten(),
                        z_pts.flatten(),
                        indexing="ij",
                    )

                    fun_vals = fun(xx, yy, zz)

                    sampling_kernels.tile_int_kernel(
                        fun_vals,
                        *self.tile_quad_wts,
                        single_box_out,
                    )

                    single_box_out /= self.tile_volume

                    out[
                        i * nt_x : (i + 1) * nt_x,
                        j * nt_y : (j + 1) * nt_y,
                        k * nt_z : (k + 1) * nt_z,
                    ] = single_box_out
        return out

    def _tile_output_arrays(self):
        """Returns two 3d arrays filled with zeros:
        * the first with one entry for each tile on one sorting box
        * the second with one entry for each tile on current process
        """
        # self._quad_pts = [np.zeros((nt, nq)).flatten() for nt, nq in zip(self.nt_per_dim, self.tile_quad_pts)]
        single_box_out = np.zeros(self.nt_per_dim)
        out = np.tile(single_box_out, self.boxes_per_dim)
        return single_box_out, out

    def _get_midpoints(self, i: int, dim: int):
        """Compute all tile midpoints within one sorting box."""
        xl = self.starts[dim] + i * self.box_widths[dim]
        return xl + self.tile_midpoints[dim]

    def _get_box_quad_pts(self, i: int, dim: int):
        """Compute all quadrature points for cell averages within the i-th sorting box in direction dim.

        Parameters
        ----------
        i : int
            Index of the box, starting at 0.

        dim : int
            Direction, either 0, 1, or 2.

        Returns
        -------
        x_pts : np.array
            2d array of shape (n_tiles_pb, n_tile_quad_pts)
        """
        xl = self.starts[dim] + i * self.box_widths[dim]
        x_tile_breaks = xl + self.tile_breaks[dim][:-1]
        x_tile_pts = self.tile_quad_pts[dim]
        x_pts = np.tile(x_tile_breaks, (x_tile_pts.size, 1)).T + x_tile_pts
        return x_pts

    @property
    def tiles_pb(self):
        """Number of equally sized tiles per sorting box."""
        return self._tiles_pb

    @property
    def n_tiles(self):
        """Total number of tiles on current process."""
        return self._n_tiles

    @property
    def nt_per_dim(self):
        """3-list of number of equally sized tiles per sorting box per direction."""
        return self._nt_per_dim

    @property
    def starts(self):
        """3-list of domain starts (left boundaries) on current process."""
        return self._starts

    @property
    def ends(self):
        """3-list of domain ends (right boundaries) on current process."""
        return self._ends

    @property
    def tile_breaks(self):
        """3-list of tile break points within the single sorting box [0.0, sorting_box_width], in each direction."""
        return self._tile_breaks

    @property
    def tile_midpoints(self):
        """3-list of tile midpoints within the single sorting box [0.0, sorting_box_width], in each direction."""
        return self._tile_midpoints

    @property
    def tile_volume(self):
        """Volume of a single tile."""
        return self._tile_volume

    @property
    def tile_quad_pts(self):
        """3-list of quadrature points (GL) within a single tile, in each direction."""
        return self._tile_quad_pts

    @property
    def tile_quad_wts(self):
        """3-list of quadrature weights (GL) within a single tile, in each direction."""
        return self._tile_quad_wts

    @property
    def rank(self):
        """Current process rank."""
        return self._rank

    @property
    def boxes_per_dim(self):
        """Sorting boxes per direction."""
        return self._boxes_per_dim

    @property
    def box_widths(self):
        """3-list of sorting box widths in each direction."""
        return self._box_widths

    @property
    def dims_mask(self):
        """Boolean array of size 3; entry is True if direction participates in tesselation."""
        return self._dims_mask
