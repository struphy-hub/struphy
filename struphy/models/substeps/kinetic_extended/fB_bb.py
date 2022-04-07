import scipy.sparse as spa
import numpy as np
import time

#===============================================================
class Substep_3:
    '''
        The substep bb of fB formulation
        Parameters
        ----------
            LINEAR OPERATORS: obj, 
                store linear operators

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

            temperature: double, 
                electron temperature

            SHAPE   : obj, 
                obj storing information of smoothed delta functions

            TEMP    : obj, 
                obj stroing all temp arrays used in the simulations

            control : int, 
                delta f method or not 
    '''
    def __init__(self, LINEAR_OPERATORS, DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM, temperature, SHAPE, TEMP, control):

        self.DOMAIN     = DOMAIN
        self.SPACES     = SPACES
        self.MPI_COMM   = MPI_COMM
        self.Np         = KIN.Np
        self.Np_loc     = KIN.Np_loc
        self.LO         = LINEAR_OPERATORS
        self.TEMP       = TEMP
        self.GATHER     = GATHER
        self.temperature= temperature
        self.mpi_rank   = MPI_COMM.Get_rank()
        self.MHD        = MHD
        self.Ntot_1form = SPACES.Ntot_1form
        self.N_1form    = SPACES.Nbase_1form
        self.N_2form    = SPACES.Nbase_2form

    def update(self, dt, EN, KIN, M1_PRE, MPI_COMM):
        # bb substep

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1
    
        timea = time.time()
        if self.mpi_rank == 0:
            self.GATHER.func_quadrature_inverse(self.TEMP.LO_inv)

        if self.mpi_rank ==0:

            self.TEMP.b1_old[:,:,:] = self.MHD.b1[:,:,:]
            self.TEMP.b2_old[:,:,:] = self.MHD.b2[:,:,:]
            self.TEMP.b3_old[:,:,:] = self.MHD.b3[:,:,:]

            self.TEMP.b1_iter[:,:,:] = self.MHD.b1[:,:,:]
            self.TEMP.b2_iter[:,:,:] = self.MHD.b2[:,:,:]
            self.TEMP.b3_iter[:,:,:] = self.MHD.b3[:,:,:]

            inputvector = np.concatenate( (self.TEMP.b1_old.flatten(), self.TEMP.b2_old.flatten(), self.TEMP.b3_old.flatten()) )

            for loop in range(30):
                right_bb = self.LO.linearoperator_right_step3(self.TEMP.twoform_temp1_long, self.TEMP.twoform_temp2_long, self.TEMP.twoform_temp3_long, self.TEMP.temp_twoform1, self.TEMP.temp_twoform2, self.TEMP.temp_twoform3, self.SPACES.indN, self.SPACES.indD, self.SPACES, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, dt, inputvector, 0.5*(self.TEMP.b1_old+self.TEMP.b1_iter), 0.5*(self.TEMP.b2_old+self.TEMP.b2_iter), 0.5*(self.TEMP.b3_old+self.TEMP.b3_iter), self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_inv, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3)
                #print('check_rightbb', right_bb)
                BB = spa.linalg.LinearOperator(shape=(self.Ntot_1form[0]+self.Ntot_1form[1]+self.Ntot_1form[2], self.Ntot_1form[0]+self.Ntot_1form[1]+self.Ntot_1form[2]), matvec=lambda x: self.LO.linearoperator_step3(self.TEMP.twoform_temp1_long, self.TEMP.twoform_temp2_long, self.TEMP.twoform_temp3_long, self.TEMP.temp_twoform1, self.TEMP.temp_twoform2, self.TEMP.temp_twoform3, self.SPACES.indN, self.SPACES.indD, self.SPACES, dt, x, 0.5*(self.TEMP.b1_old+self.TEMP.b1_iter), 0.5*(self.TEMP.b2_old+self.TEMP.b2_iter), 0.5*(self.TEMP.b3_old+self.TEMP.b3_iter), self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_inv, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3))
                self.TEMP.oneform_temp1_long[:], self.TEMP.oneform_temp2_long[:], self.TEMP.oneform_temp3_long[:] = np.split(spa.linalg.gmres(BB, right_bb, x0 =np.concatenate( ( 0.5 * (self.TEMP.b1_old + self.TEMP.b1_iter).flatten(), 0.5 * (self.TEMP.b2_old + self.TEMP.b2_iter).flatten(), 0.5 * (self.TEMP.b3_old+self.TEMP.b3_iter).flatten() )), tol = 10**(-12), M=M1_PRE, callback=count_iters)[0], [self.Ntot_1form[0], self.Ntot_1form[0] + self.Ntot_1form[1]])

                #===================================================
                self.MHD.b1[:,:,:] = self.TEMP.oneform_temp1_long.reshape(self.N_1form[0])
                self.MHD.b2[:,:,:] = self.TEMP.oneform_temp2_long.reshape(self.N_1form[1])
                self.MHD.b3[:,:,:] = self.TEMP.oneform_temp3_long.reshape(self.N_1form[2])

                print('iteration error', np.linalg.norm((self.MHD.b1 - self.TEMP.b1_iter).flatten()) + np.linalg.norm((self.MHD.b2 - self.TEMP.b2_iter).flatten()) + np.linalg.norm((self.MHD.b3 - self.TEMP.b3_iter).flatten()))
                if np.linalg.norm((self.MHD.b1 - self.TEMP.b1_iter).flatten()) + np.linalg.norm((self.MHD.b2 - self.TEMP.b2_iter).flatten()) + np.linalg.norm((self.MHD.b3 - self.TEMP.b3_iter).flatten()) < 10**(-11):
                    break

                self.TEMP.b1_iter[:,:,:] = self.MHD.b1[:,:,:]
                self.TEMP.b2_iter[:,:,:] = self.MHD.b2[:,:,:]
                self.TEMP.b3_iter[:,:,:] = self.MHD.b3[:,:,:]

        MPI_COMM.Barrier()
        MPI_COMM.Bcast(self.TEMP.b1_iter,   root=0)
        MPI_COMM.Bcast(self.TEMP.b2_iter,   root=0)
        MPI_COMM.Bcast(self.TEMP.b3_iter,   root=0)
        MPI_COMM.Bcast(self.MHD.b1,        root=0)
        MPI_COMM.Bcast(self.MHD.b2,        root=0)
        MPI_COMM.Bcast(self.MHD.b3,        root=0)

        timeb = time.time()
        
        print('3_step_time_used', timeb - timea)

        MPI_COMM.Barrier()
        # ============ energy calsulations ==================
        EN.cal_total(KIN, MPI_COMM)
        if self.mpi_rank == 0:
            print('total_energy_step3', EN.total[0])
        