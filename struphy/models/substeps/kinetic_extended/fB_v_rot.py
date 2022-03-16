
import time
from mpi4py import MPI
from struphy.pic.kinetic_extended import fB_massless_pusher
from struphy.pic.kinetic_extended import fB_energy

class Substep_5:
    '''The substep v rotation of fB formulation
    Parameters
        ----------
        DOMAIN : obj, 
            Domain object from geometry/domain_3d.

        SPACES : obj, 
            FEEC self.SPACES., store information of tensor products of B-splines

        GATHER : obj, 
            Particle gather object

        KIN    : obj, 
            obj storing information of particles

        MHD    : obj, 
            obj storing information of MHD variables

        MPI_COMM: obj, 
            communicator of MPI
    '''


    def __init__(self, DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM):

        self.DOMAIN     = DOMAIN
        self.SPACES     = SPACES
        self.Np_loc     = KIN.Np_loc
        self.Np         = KIN.Np
        self.GATHER     = GATHER
        self.MHD        = MHD
        self.mpi_rank   = MPI_COMM.Get_rank()


    def push(self, dt, EN, KIN, MPI_COMM):
        # particles rotating in the given magnetic field
        timea = time.time()
        fB_massless_pusher.rotation(self.SPACES.NbaseN, self.SPACES.NbaseD, self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2], self.SPACES.Nel, KIN.particles_loc, dt, self.MHD.b1, self.MHD.b2, self.MHD.b3, self.SPACES.p, self.Np_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        timeb = time.time()

        print('rotation_step_time_used', timeb - timea)
        MPI_COMM.Barrier()

        # ============ energy calsulations ==================
        EN.cal_total(KIN, MPI_COMM)
        if self.mpi_rank == 0:
            print('total_energy_step5', EN.total[0])

        MPI_COMM.Barrier()