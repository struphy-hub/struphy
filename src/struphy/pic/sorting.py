import logging

import numpy as np

try:
    from mpi4py.MPI import Intracomm
except ModuleNotFoundError:

    class Intracomm:
        x = None


import cunumpy as xp

from struphy.pic.sorting_kernels import flatten_index, initialize_neighbours

logger = logging.getLogger("struphy")


class SortingBoxes:
    """Boxes used for the sorting of the particles.

    The simulation domain of one MPI process is divided into a Cartesian grid of
    ``nx * ny * nz`` boxes, surrounded by one extra layer of ghost boxes on each side
    (hence ``(nx + 2) * (ny + 2) * (nz + 2)`` boxes in total). Sorting particles into
    boxes allows :meth:`~struphy.pic.base.Particles.eval_density` and related routines
    to only loop over particles in neighbouring boxes instead of all particles, which is
    essential for SPH-type evaluations.

    Boxes are represented as a 2D array of integers (:attr:`boxes`), where
    each line corresponds to one box, and all entries of line i that are not -1
    correspond to a particle in the i-th box. The mapping from a 3D box index
    ``(n1, n2, n3)`` to the flat row index used in :attr:`boxes` is computed by
    :func:`~struphy.pic.sorting_kernels.flatten_index`.

    For "sph" particles (``is_sph=True``), the ghost-box layer is filled with copies of
    particles from the neighbouring MPI process (or mirrored/periodic copies at a domain
    boundary), which requires MPI communication; see :attr:`communicate`.

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
        Either 'periodic', 'mirror', 'fixed' or 'noslip' in each direction.

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
    ):
        self._markers_shape = markers_shape
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._comm = comm
        self._box_index = box_index
        self._box_bufsize = box_bufsize

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
        """Number of (non-ghost) boxes in the x direction on this MPI process."""
        return self._nx

    @property
    def ny(self):
        """Number of (non-ghost) boxes in the y direction on this MPI process."""
        return self._ny

    @property
    def nz(self):
        """Number of (non-ghost) boxes in the z direction on this MPI process."""
        return self._nz

    @property
    def comm(self):
        """MPI communicator used to exchange ghost-box particles, or None."""
        return self._comm

    @property
    def box_index(self):
        """Column index of the markers array holding each particle's box number, counted from the end."""
        return self._box_index

    @property
    def boxes(self):
        """2D int array of shape (n_boxes + 1, n_cols); row i lists the marker-array row indices
        (padded with -1) of the particles currently sorted into box i."""
        if not hasattr(self, "_boxes"):
            self._set_boxes()
        return self._boxes

    @property
    def neighbours(self):
        """2D int array of shape (n_boxes, 27); row i lists the box indices of the 26 boxes
        surrounding box i (plus box i itself), i.e. the only boxes a particle in box i can interact with."""
        if not hasattr(self, "_neighbours"):
            self._set_boxes()
        return self._neighbours

    @property
    def communicate(self):
        """True if ghost boxes must be filled via MPI communication (only needed for "sph" particles)."""
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
        """Compute, for each of the six faces (x_m, x_p, y_m, y_p, z_m, z_p), the box-index
        offset to add to a particle's box number when it is turned into a ghost particle sent
        across that face (used by :meth:`~struphy.pic.base.Particles._prepare_ghost_particles`).

        The default (periodic) shift moves a particle by a full row of boxes (``nx``, ``ny``
        or ``nz``, via :func:`~struphy.pic.sorting_kernels.flatten_index`) so that it lands in
        the ghost-box layer on the opposite side. For 'mirror', 'fixed' or 'noslip' boundary
        conditions, the shift is instead a single box (magnitude 1) and is only applied on
        faces where this MPI process actually touches the physical domain boundary
        (see :attr:`is_domain_boundary`); interior MPI faces always keep the periodic shift.
        """
        self._bc_sph_index_shifts = {}
        self._bc_sph_index_shifts["x_m"] = flatten_index(self.nx, 0, 0, self.nx, self.ny, self.nz)
        self._bc_sph_index_shifts["x_p"] = flatten_index(self.nx, 0, 0, self.nx, self.ny, self.nz)
        self._bc_sph_index_shifts["y_m"] = flatten_index(0, self.ny, 0, self.nx, self.ny, self.nz)
        self._bc_sph_index_shifts["y_p"] = flatten_index(0, self.ny, 0, self.nx, self.ny, self.nz)
        self._bc_sph_index_shifts["z_m"] = flatten_index(0, 0, self.nz, self.nx, self.ny, self.nz)
        self._bc_sph_index_shifts["z_p"] = flatten_index(0, 0, self.nz, self.nx, self.ny, self.nz)

        if self.bc_sph[0] in ("mirror", "fixed", "noslip"):
            if self.is_domain_boundary["x_m"]:
                self._bc_sph_index_shifts["x_m"] = flatten_index(-1, 0, 0, self.nx, self.ny, self.nz)
            if self.is_domain_boundary["x_p"]:
                self._bc_sph_index_shifts["x_p"] = flatten_index(-1, 0, 0, self.nx, self.ny, self.nz)

        if self.bc_sph[1] in ("mirror", "fixed", "noslip"):
            if self.is_domain_boundary["y_m"]:
                self._bc_sph_index_shifts["y_m"] = flatten_index(0, -1, 0, self.nx, self.ny, self.nz)
            if self.is_domain_boundary["y_p"]:
                self._bc_sph_index_shifts["y_p"] = flatten_index(0, -1, 0, self.nx, self.ny, self.nz)

        if self.bc_sph[2] in ("mirror", "fixed", "noslip"):
            if self.is_domain_boundary["z_m"]:
                self._bc_sph_index_shifts["z_m"] = flatten_index(0, 0, -1, self.nx, self.ny, self.nz)
            if self.is_domain_boundary["z_p"]:
                self._bc_sph_index_shifts["z_p"] = flatten_index(0, 0, -1, self.nx, self.ny, self.nz)

    def _set_boxes(self):
        """(Re)allocate the box structure: the :attr:`boxes` array (sized to hold an
        estimated maximum number of particles per box, plus a buffer), the bookkeeping
        arrays used while sorting (``_next_index``, ``_cumul_next_index``, swap lines),
        and the :attr:`neighbours` array."""
        self._n_boxes = (self._nx + 2) * (self._ny + 2) * (self._nz + 2)
        n_box_in = self._nx * self._ny * self._nz

        n_particles = self._markers_shape[0]
        n_mkr = int(n_particles / n_box_in) + 1
        # scalar box-sizing estimate, not physics data; the rest of this box
        # structure is host-resident (see below), and round() doesn't accept
        # a CuPy 0-d array, so this must stay plain math regardless of backend.
        n_cols = round(
            n_mkr * (1 + 1 / np.sqrt(n_mkr) + self._box_bufsize),
        )

        # cartesian boxes (extra last row stores holes/outside particles); host-resident,
        # read/written directly by the compiled, host-only sorting kernels
        self._boxes = np.full((self._n_boxes + 1, n_cols), -1, dtype=int)
        self._next_index = np.zeros((self._n_boxes + 1), dtype=int)
        self._cumul_next_index = np.zeros((self._n_boxes + 2), dtype=int)
        self._neighbours = np.zeros((self._n_boxes, 27), dtype=int)

        # A particle on box i only sees particles in boxes that belong to neighbours[i]
        initialize_neighbours(self._neighbours, self.nx, self.ny, self.nz)
        # logger.info(f"{self._rank = }\n{self._neighbours = }")

        self._swap_line_1 = np.zeros(self._markers_shape[1])
        self._swap_line_2 = np.zeros(self._markers_shape[1])

    def _set_boundary_boxes(self):
        """Collect the (flat) indices of all non-ghost boxes that lie on the outer surface
        of this MPI process's box grid, grouped by which face(s)/edge(s)/corner they belong
        to (e.g. :attr:`~_bnd_boxes_x_m`, :attr:`~_bnd_boxes_x_m_y_m`, :attr:`~_bnd_boxes_x_m_y_m_z_m`, ...).

        These groups are used by :meth:`~struphy.pic.base.Particles._prepare_ghost_particles`
        to find, for each of the 6 faces, 12 edges and 8 corners of the process domain, the
        particles that must be turned into ghost particles and sent to the corresponding
        neighbouring process (or folded back for a physical domain boundary)."""
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

        logger.debug(f"eta1 boundary on {self._rank =}:\n{self._bnd_boxes_x_m =}\n{self._bnd_boxes_x_p =}")

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

        logger.debug(f"eta2 boundary on {self._rank =}:\n{self._bnd_boxes_y_m =}\n{self._bnd_boxes_y_p =}")

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

        logger.debug(f"eta3 boundary on {self._rank =}:\n{self._bnd_boxes_z_m =}\n{self._bnd_boxes_z_p =}")

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

        logger.debug(
            (
                f"eta1-eta2 edge on {self._rank =}:\n{self._bnd_boxes_x_m_y_m =}"
                f"\n{self._bnd_boxes_x_m_y_p =}"
                f"\n{self._bnd_boxes_x_p_y_m =}"
                f"\n{self._bnd_boxes_x_p_y_p =}"
            ),
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

        logger.debug(
            (
                f"eta1-eta3 edge on {self._rank =}:\n{self._bnd_boxes_x_m_z_m =}"
                f"\n{self._bnd_boxes_x_m_z_p =}"
                f"\n{self._bnd_boxes_x_p_z_m =}"
                f"\n{self._bnd_boxes_x_p_z_p =}"
            ),
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

        logger.debug(
            (
                f"eta2-eta3 edge on {self._rank =}:\n{self._bnd_boxes_y_m_z_m =}"
                f"\n{self._bnd_boxes_y_m_z_p =}"
                f"\n{self._bnd_boxes_y_p_z_m =}"
                f"\n{self._bnd_boxes_y_p_z_p =}"
            ),
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

        logger.debug(
            (
                f"corners on {self._rank =}:\n{self._bnd_boxes_x_m_y_m_z_m =}"
                f"\n{self._bnd_boxes_x_m_y_m_z_p =}"
                f"\n{self._bnd_boxes_x_m_y_p_z_m =}"
                f"\n{self._bnd_boxes_x_p_y_m_z_m =}"
                f"\n{self._bnd_boxes_x_m_y_p_z_p =}"
                f"\n{self._bnd_boxes_x_p_y_m_z_p =}"
                f"\n{self._bnd_boxes_x_p_y_p_z_m =}"
                f"\n{self._bnd_boxes_x_p_y_p_z_p =}"
            ),
        )
