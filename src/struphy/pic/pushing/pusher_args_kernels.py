# from numpy import copy
from numpy import empty


class MarkerArguments:
    """Holds arguments pertaining to :class:`~struphy.pic.base.Particles`
    passed to particle kernels.

    Paramaters
    ----------
    markers : array[float]
        Markers array.

    Np : int
        Total number of particles.

    vdim : int
        Dimension of velocity space.

    weight_idx : int
        Column index of particle weight.

    first_diagnostics_idx : int
        Starting index for diagnostics columns:
        after 3 positions, vdim velocities, weight, s0 and w0.

    first_pusher_idx : int
        Starting buffer marker index number for pusher.

    first_shift_idx : int
        First index for storing shifts due to boundary conditions in eta-space.

    residual_idx: int
        Column for storing the residual in iterative pushers.

    first_free_idx : int
        First index for storing auxiliary quantities for each particle.
    """

    def __init__(
        self,
        markers: "float[:, :]",
        valid_mks: "bool[:]",
        Np: int,
        vdim: int,
        weight_idx: int,
        first_diagnostics_idx: int,
        first_pusher_idx: int,
        first_shift_idx: int,
        residual_idx: int,
        first_free_idx: int,
    ):
        self.markers = markers
        self.valid_mks = valid_mks
        self.Np = Np
        self.vdim = vdim
        self.weight_idx = weight_idx
        self.n_markers = markers.shape[0]

        # useful indices
        self.first_diagnostics_idx = first_diagnostics_idx
        self.first_init_idx = first_pusher_idx
        self.first_shift_idx = first_shift_idx  # starting idx for eta-shifts due to boundary conditions
        self.residual_idx = residual_idx  # residual in iterative solvers
        self.first_free_idx = first_free_idx  # index after which auxiliary saving is possible

        # only used for Particles5D
        self.energy_idx = 8  # particle energy
        self.mu_idx = 9  # particle magnetic moment
        self.toroidalmom_idx = 10  # particle toroidal momentum


class DerhamArguments:
    """Holds the mandatory arguments pertaining to :class:`~struphy.feec.psydac_derham.Derham` passed to particle pusher kernels.

    Paramaters
    ----------
    pn : array[int]
        Spline degrees of :class:`~struphy.feec.psydac_derham.Derham`.

    tn1, tn2, tn3 : array[float]
        Knot sequences of :class:`~struphy.feec.psydac_derham.Derham`.

    starts : array[int]
        Start indices (current MPI process) of :class:`~struphy.feec.psydac_derham.Derham`.
    """

    def __init__(
        self,
        pn: "int[:]",
        tn1: "float[:]",
        tn2: "float[:]",
        tn3: "float[:]",
        starts: "int[:]",
    ):
        self.pn = pn
        self.tn1 = tn1
        self.tn2 = tn2
        self.tn3 = tn3
        self.starts = starts

        self.bn1 = empty(pn[0] + 1, dtype=float)
        self.bn2 = empty(pn[1] + 1, dtype=float)
        self.bn3 = empty(pn[2] + 1, dtype=float)
        self.bd1 = empty(pn[0], dtype=float)
        self.bd2 = empty(pn[1], dtype=float)
        self.bd3 = empty(pn[2], dtype=float)


class DomainArguments:
    """Holds the mandatory arguments pertaining to :class:`~struphy.geometry.base.Domain` passed to particle pusher kernels.

    Paramaters
    ----------
    kind : int
        Mapping identifier of :class:`~struphy.geometry.base.Domain`.

    params : array[float]
        Mapping parameters of :class:`~struphy.geometry.base.Domain`.

    p : array[int]
        Spline degrees of :class:`~struphy.geometry.base.Domain`.

    t1, t2, t3 : array[float]
        Knot sequences of :class:`~struphy.geometry.base.Domain`.

    ind1, ind2, ind3 : array[float]
        Indices of non-vanishing splines in format (number of mapping grid cells, p + 1) of :class:`~struphy.geometry.base.Domain`.

    cx, cy, cz : array[float]
        Spline coefficients (control points) of :class:`~struphy.geometry.base.Domain`.
    """

    def __init__(
        self,
        kind_map: int,
        params: "float[:]",
        p: "int[:]",
        t1: "float[:]",
        t2: "float[:]",
        t3: "float[:]",
        ind1: "int[:,:]",
        ind2: "int[:,:]",
        ind3: "int[:,:]",
        cx: "float[:,:,:]",
        cy: "float[:,:,:]",
        cz: "float[:,:,:]",
    ):
        self.kind_map = kind_map
        self.params = params
        self.p = p
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.ind1 = ind1
        self.ind2 = ind2
        self.ind3 = ind3
        self.cx = cx
        self.cy = cy
        self.cz = cz
