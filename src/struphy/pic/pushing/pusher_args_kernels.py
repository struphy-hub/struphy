class MarkerArguments:
    '''Holds the mandatory arguments pertaining to :class:`~struphy.pic.base.Particles` 
    passed to particle pusher kernels.
    
    Paramaters
    ----------
    markers : array[float]
        Markers array.
        
    vdim : int
        Dimension of velocity space.
        
    buffer_idx : int
        Buffer index of markers array.
    '''
        
    def __init__(self,
                 markers: 'float[:, :]',
                 vdim: int,
                 buffer_idx: int
                 ):

        self.markers = markers
        self.vdim = vdim
        self.buffer_idx = buffer_idx
        self.n_markers = markers.shape[0]
        
        # useful indices
        self.shift_idx = buffer_idx + 3 + vdim # starting idx for eta-shifts due to boundary conditions
        self.residual_idx = self.shift_idx + 3 # residual in iterative solvers
        self.first_free_idx = self.residual_idx + 1 # index after which auxiliary saving is possible
        
        # only used for Particles5D
        self.energy_idx = 8 # particle energy
        self.mu_idx = 9 # particle magnetic moment
        self.toroidalmom_idx = 10 # particle toroidal momentum


class DerhamArguments:
    '''Holds the mandatory arguments pertaining to :class:`~struphy.feec.psydac_derham.Derham` passed to particle pusher kernels.
    
    Paramaters
    ----------
    pn : array[int]
        Spline degrees of :class:`~struphy.feec.psydac_derham.Derham`.
        
    tn1, tn2, tn3 : array[float]
        Knot sequences of :class:`~struphy.feec.psydac_derham.Derham`.
        
    starts : array[int]
        Start indices (current MPI process) of :class:`~struphy.feec.psydac_derham.Derham`.
    '''
        
    def __init__(self,
                 pn: 'int[:]',
                 tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                 starts: 'int[:]',
                 bn1: 'float[:]',
                 bn2: 'float[:]',
                 bn3: 'float[:]',
                 bd1: 'float[:]',
                 bd2: 'float[:]',
                 bd3: 'float[:]',
                 ):

        self.pn = pn
        self.tn1 = tn1
        self.tn2 = tn2
        self.tn3 = tn3
        self.starts = starts
        self.bn1 = bn1
        self.bn2 = bn2
        self.bn3 = bn3
        self.bd1 = bd1
        self.bd2 = bd2
        self.bd3 = bd3
        
        
class DomainArguments:
    '''Holds the mandatory arguments pertaining to :class:`~struphy.geometry.base.Domain` passed to particle pusher kernels.
    
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
    '''
        
    def __init__(self,
                 kind_map: int, 
                 params: 'float[:]',
                 p: 'int[:]', 
                 t1: 'float[:]', t2: 'float[:]', t3: 'float[:]',
                 ind1: 'int[:,:]', ind2: 'int[:,:]', ind3: 'int[:,:]',
                 cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]'
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