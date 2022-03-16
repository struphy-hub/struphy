
import struphy.pic.kinetic_extended.fB_massless_pusher as pusher
import time

class Substep_1:
    '''The first substep of fB formulation
    
    Parameters
    ----------
        DOMAIN : obj
            Domain object from geometry/domain_3d.

        SPACES : obj
            FEEC self.SPACES.

        GATHER : obj
            Particle gather object
        KIN    : obj
            obj storing information of particles
        MHD    : obj
            obj storing information of MHD variables
        MPI_COMM: obj
            communitator of MPI
        temperature: double
            electron temperature
        SHAPE   : obj
            obj storing information of smoothed delta functions
        TEMP    : obj
            obj stroing all temp arrays used in the simulations
    '''

    def __init__(self, DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM, temperature, SHAPE, TEMP):

        self.DOMAIN     = DOMAIN
        self.SPACES     = SPACES
        self.Np_loc     = KIN.Np_loc
        self.Np         = KIN.Np
        self.GATHER     = GATHER
        self.MHD        = MHD
        self.mpi_rank   = MPI_COMM.Get_rank()
        self.MPI_COMM   = MPI_COMM
        self.temperature= temperature
        self.SHAPE      = SHAPE
        self.TEMP       = TEMP



    def push(self, dt, EN, KIN, MPI_COMM):
        # xv substep
        timea = time.time()
    
        pusher.step1_pushx(self.SPACES.Nel, KIN.particles_loc, self.Np_loc, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        time1 = time.time()
        self.GATHER.func_gather_quadrature(KIN.particles_loc, self.Np_loc, self.Np, MPI_COMM)
        time2 = time.time()
        print('check_gather_time', time2 - time1)
        self.GATHER.func_quadrature_log()
        MPI_COMM.Barrier()

        pusher.step1_pushv(self.GATHER.index_shapex, self.GATHER.index_shapey, self.GATHER.index_shapez, self.GATHER.index_diffx, self.GATHER.index_diffy, self.GATHER.index_diffz, self.temperature, self.SHAPE.p, self.SPACES.n_quad, self.SPACES.pts[0], self.SPACES.pts[1], self.SPACES.pts[2], self.SPACES.Nel, self.SHAPE.size, self.KIN.particles_loc, self.Np_loc, self.Np, dt, self.GATHER.quadrature_log, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        #now delta-f version of pushv is only for the case that equilibrium of density is independent of position, ie. constant.
        pusher.step1_pushx(self.SPACES.Nel, KIN.particles_loc, self.Np_loc, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        timeb = time.time()

        print('1_step_time_used', timeb - timea)
        # ==== calculate inv of density =======
        self.GATHER.func_gather_quadrature(KIN.particles_loc, self.Np_loc, self.Np, MPI_COMM)
        if mpi_rank == 0:
            self.GATHER.func_quadrature_inverse(LO_inv)

        self.GATHER.func_quadrature_log()
        self.GATHER.func_gather_grid(KIN.particles_loc, self.Np_loc, self.Np, MPI_COMM)
        MPI_COMM.Barrier()
        # ============ energy calsulations ==================
        EN.cal_total(KIN, MPI_COMM)
        if self.mpi_rank == 0:
            print('total_energy_step1', EN.total[0])
