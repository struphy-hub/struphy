class LocalProjectorsArguments:
    """Holds the mandatory arguments pertaining to :class:`~struphy.feec.projectors.CommutingProjectorLocal` passed to solve kernels.

    Paramaters
    ----------
    space_key : int
        Determines the space into which the projector projects. It must be one of the following values
        0, 1, 2, 3 or 4. 4 stands for H1vec.

    IoH : 1d bool array
        Determines if we have interpolation or histopolation in each spatial direction. False means interpolation, True means histopolation.

    shift: 1d int array
        array of 3 ints, each one denotes the amout by which we must shift the indices to loop around the quasi-points for each spatial direction.

    original_size: 1d int array
        Number of total quasi-interpolation points (or quasi-histopolation intervals) in each direction

    index_translation1: 1d int array
        Array which translates for the e1 direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation1[index_global]

    index_translation2: 1d int array
        Array which translates for the e2 direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation2[index_global]

    index_translation3: 1d int array
        Array which translates for the e3 direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation3[index_global]

    starts: 1d int array
        Array with the StencilVector start indices for each MPI rank.

    ends: 1d int array
        Array with the StencilVector end indices for each MPI rank.

    pds: 1d int array
        Array with the StencilVector pads for each MPI rank.

    B_nbasis: 1d int array
        Array with the number of B-splines in each dimension.

    periodic: 1d bool array
        Array that tell us if the splines are periodic or clamped in each dimension.

    p : 1d int array
        Degree of the B-splines in each direction.

    wij0: 2d float array
        Array with the histopolation or interpolation geometric weights for the first direction.

    wij1: 2d float array
        Array with the histopolation or interpolation geometric weights for the second direction.

    wij2: 2d float array
        Array with the histopolation or interpolation geometric weights for the third direction.

    wts0 : 2d float array
        Gauss-Legandre quadrature weights for the intergrals in the e1 direction.

    wts1 : 2d float array
        Gauss-Legandre quadrature weights for the intergrals in the e2 direction.

    wts2 : 2d float array
        Gauss-Legandre quadrature weights for the intergrals in the e3 direction.

    inv_index_translation0: 1d int array
        Array which translates for the e1 direction from the local indices to the global indices. index_global = inv_index_translation0[index_local]

    inv_index_translation1: 1d int array
        Array which translates for the e2 direction from the local indices to the global indices. index_global = inv_index_translation1[index_local]

    inv_index_translation2: 1d int array
        Array which translates for the e3 direction from the local indices to the global indices. index_global = inv_index_translation2[index_local]
    """

    def __init__(
        self,
        space_key: int,
        IoH: "bool[:]",
        shift: "int[:]",
        original_size: "int[:]",
        index_translation1: "int[:]",
        index_translation2: "int[:]",
        index_translation3: "int[:]",
        starts: "int[:]",
        ends: "int[:]",
        pds: "int[:]",
        B_nbasis: "int[:]",
        periodic: "bool[:]",
        p: "int[:]",
        wij0: "float[:,:]",
        wij1: "float[:,:]",
        wij2: "float[:,:]",
        wts0: "float[:,:]",
        wts1: "float[:,:]",
        wts2: "float[:,:]",
        inv_index_translation0: "int[:]",
        inv_index_translation1: "int[:]",
        inv_index_translation2: "int[:]",
    ):
        self.space_key = space_key
        self.IoH = IoH
        self.shift = shift
        self.original_size = original_size
        self.index_translation1 = index_translation1
        self.index_translation2 = index_translation2
        self.index_translation3 = index_translation3
        self.starts = starts
        self.ends = ends
        self.pds = pds
        self.B_nbasis = B_nbasis
        self.periodic = periodic
        self.p = p
        self.wij0 = wij0
        self.wij1 = wij1
        self.wij2 = wij2
        self.wts0 = wts0
        self.wts1 = wts1
        self.wts2 = wts2
        self.inv_index_translation0 = inv_index_translation0
        self.inv_index_translation1 = inv_index_translation1
        self.inv_index_translation2 = inv_index_translation2
