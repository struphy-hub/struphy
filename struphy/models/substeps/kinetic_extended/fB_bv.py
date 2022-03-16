from pkgutil import ModuleInfo
import scipy.sparse as spa
import numpy as np
import time
import struphy.pic.kinetic_extended.bvfastpush as bvfastpush
import struphy.feec.projectors.shape_pro_local.shape_local_projector_kernel as loc_proj_ker
import struphy.feec.projectors.shape_pro_local.shape_L2_projector_kernel as L2_smoothed_ker
#===============================================================
class Substep_4:
    '''The substep bv of fB formulation
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
            ommunicator of MPI

        temperature: double, 
            electron temperature

        SHAPE   : obj, 
            obj storing information of smoothed delta functions

        TEMP    : obj, 
            obj stroing all temp arrays used in the simulations

        control : int, 
            delta f method or not 

        ACCBV   : obj, 
            obj storing accmulation method from particles
    '''
    def __init__(self, LINEAR_OPERATORS, DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM, temperature, SHAPE, TEMP, control, ACCBV):

        self.DOMAIN     = DOMAIN
        self.SPACES     = SPACES
        self.MPI_COMM   = MPI_COMM
        self.Np         = KIN.Np
        self.Np_loc     = KIN.Np_loc
        self.LO         = LINEAR_OPERATORS
        self.TEMP       = TEMP
        self.GATHER     = GATHER
        self.temperature= temperature
        self.ACCBV      = ACCBV
        self.mpi_rank   = MPI_COMM.Get_rank()
        self.MHD        = MHD
        self.Ntot_1form = SPACES.Ntot_1form
        self.Ntot_2form = SPACES.Ntot_2form
        self.Nbase_1form= SPACES.Nbase_1form
        self.Nbase_2form= SPACES.Nbase_2form



    def update(self, dt, M1_PRE, EN, KIN, MPI_COMM):
        # bv substep &  L2 projector is used for the current term & delta function is used for f

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        # current accumulation (all processes) 
    
        # use projection 
        if self.mpi_rank == 0:
            self.GATHER.func_quadrature_inverse(self.TEMP.LO_inv)

        MPI_COMM.Barrier()
        # ==========================================================================================
        self.TEMP.b1_old[:,:,:] = self.MHD.b1[:,:,:]
        self.TEMP.b2_old[:,:,:] = self.MHD.b2[:,:,:]
        self.TEMP.b3_old[:,:,:] = self.MHD.b3[:,:,:]

        self.TEMP.b1_iter[:,:,:] = self.MHD.b1[:,:,:]
        self.TEMP.b2_iter[:,:,:] = self.MHD.b2[:,:,:]
        self.TEMP.b3_iter[:,:,:] = self.MHD.b3[:,:,:]

        # When using cg method, tol has an strong impact on the energy conservation
        time1 = time.time()
        self.ACCBV.accumulate_substep4(KIN.particles_loc, MPI_COMM)
        time2 = time.time()
        print('check_assemble_time', time2 - time1)
        MPI_COMM.Barrier()
    
        if self.mpi_rank == 0:
            mat, vec = self.ACCBV.assemble_substep4(self.Np)
            timea = time.time()
            while True:
                self.LO.substep4_pre(self.TEMP.df_det, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], 0.5*(self.TEMP.b1_old + self.TEMP.b1_iter), 0.5*(self.TEMP.b2_old + self.TEMP.b2_iter), 0.5*(self.TEMP.b3_old + self.TEMP.b3_iter), self.SPACES, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, self.TEMP.LO_inv, self.DOMAIN.kind_map, self.DOMAIN.params_map)
            
                right_bv = self.LO.substep4_linear_operator_right(self.ACCBV, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.DFI_11, self.TEMP.DFI_12, self.TEMP.DFI_13, self.TEMP.DFI_21, self.TEMP.DFI_22, self.TEMP.DFI_23, self.TEMP.DFI_31, self.TEMP.DFI_32, self.TEMP.DFI_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.M1, M1_PRE, self.SPACES.C, mat, self.TEMP.b1_old, self.TEMP.b2_old, self.TEMP.b3_old, self.SPACES, self.SPACES.Ntot_2form, self.SPACES.Ntot_1form, self.SPACES.Nbase_2form,self.SPACES.Nbase_1form, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, vec, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map)
                #print('check_rightbv', right_bv)
                #if control == True:
                #    right_bv[:] = right_bv[:] + dt * cv.bv_right(p, indN, indD, Nel, G_inv_11, G_inv_12, G_inv_13, G_inv_22, G_inv_23, G_inv_33, DFI_11, DFI_12, DFI_13, DFI_21, DFI_22, DFI_23, DFI_31, DFI_32, DFI_33, df_det, Jeqx, Jeqy, Jeqz, temp_dft, temp_generate_weight1, temp_generate_weight2, temp_generate_weight3, temp_twoform1, temp_twoform2, temp_twoform3, (b1_old + b1_iter)/2.0, (b2_old + b2_iter)/2.0, (b3_old + b3_iter)/2.0, LO_inv, LO_b1, LO_b2, LO_b3, tensor_space_FEM)
                BV = spa.linalg.LinearOperator(shape=(self.Ntot_1form[0]+self.Ntot_1form[1]+self.Ntot_1form[2], self.Ntot_1form[0]+self.Ntot_1form[1]+self.Ntot_1form[2]), matvec=lambda x: self.LO.substep4_linear_operator(self.ACCBV, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.DFI_11, self.TEMP.DFI_12, self.TEMP.DFI_13, self.TEMP.DFI_21, self.TEMP.DFI_22, self.TEMP.DFI_23, self.TEMP.DFI_31, self.TEMP.DFI_32, self.TEMP.DFI_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.M1, M1_PRE, self.SPACES.C, mat, x, self.SPACES, self.SPACES.Ntot_2form, self.SPACES.Ntot_1form, self.SPACES.Nbase_2form, self.SPACES.Nbase_1form, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map)  )

                temp1, temp2, temp3 = np.split(spa.linalg.cg(BV, right_bv, x0 =np.concatenate( ( 0.5 * (self.TEMP.b1_old + self.TEMP.b1_iter).flatten(), 0.5 * (self.TEMP.b2_old + self.TEMP.b2_iter).flatten(), 0.5 * (self.TEMP.b3_old+self.TEMP.b3_iter).flatten() )), tol = 10**(-13), M=M1_PRE)[0], [self.Ntot_1form[0], self.Ntot_1form[0] + self.Ntot_1form[1]]   )
                #print('check_temp3', temp3)
                self.MHD.b1[:,:,:] = temp1.reshape(self.Nbase_1form[0])
                self.MHD.b2[:,:,:] = temp2.reshape(self.Nbase_1form[1])
                self.MHD.b3[:,:,:] = temp3.reshape(self.Nbase_1form[2])

                print('iteration error', np.linalg.norm((self.MHD.b1 - self.TEMP.b1_iter).flatten()) + np.linalg.norm((self.MHD.b2 - self.TEMP.b2_iter).flatten()) + np.linalg.norm((self.MHD.b3 - self.TEMP.b3_iter).flatten()))
                if np.linalg.norm((self.MHD.b1 - self.TEMP.b1_iter).flatten()) + np.linalg.norm((self.MHD.b2 - self.TEMP.b2_iter).flatten()) + np.linalg.norm((self.MHD.b3 - self.TEMP.b3_iter).flatten()) < 10** (-12): 
                    break
                self.TEMP.b1_iter[:,:,:] = self.MHD.b1[:,:,:]
                self.TEMP.b2_iter[:,:,:] = self.MHD.b2[:,:,:]
                self.TEMP.b3_iter[:,:,:] = self.MHD.b3[:,:,:]

            timeb = time.time()
            print('4_step_time_used', timeb - timea)
        MPI_COMM.Barrier()
        MPI_COMM.Bcast(self.MHD.b1,   root=0)
        MPI_COMM.Bcast(self.MHD.b2,   root=0)
        MPI_COMM.Bcast(self.MHD.b3,   root=0)
        MPI_COMM.Barrier()
    
        time1 = time.time()
        if self.mpi_rank == 0:
            temp1, temp2, temp3 = np.split(self.LO.substep4_pusher_field(self.ACCBV, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.DFI_11, self.TEMP.DFI_12, self.TEMP.DFI_13, self.TEMP.DFI_21, self.TEMP.DFI_22, self.TEMP.DFI_23, self.TEMP.DFI_31, self.TEMP.DFI_32, self.TEMP.DFI_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.M1, M1_PRE, self.SPACES.C, mat, 0.5*(self.TEMP.b1_old + self.TEMP.b1_iter), 0.5*(self.TEMP.b2_old + self.TEMP.b2_iter), 0.5*(self.TEMP.b3_old + self.TEMP.b3_iter), self.SPACES, self.SPACES.Ntot_2form, self.SPACES.Nbase_2form, self.SPACES.Nbase_1form, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, vec, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map), [self.Ntot_1form[0], self.Ntot_1form[0] + self.Ntot_1form[1]]  ) 
            self.TEMP.b1_iter[:,:,:] = temp1.reshape(self.Nbase_1form[0])
            self.TEMP.b2_iter[:,:,:] = temp2.reshape(self.Nbase_1form[1])
            self.TEMP.b3_iter[:,:,:] = temp3.reshape(self.Nbase_1form[2])

        MPI_COMM.Barrier()
        MPI_COMM.Bcast(self.TEMP.b1_iter,   root=0)
        MPI_COMM.Bcast(self.TEMP.b2_iter,   root=0)
        MPI_COMM.Bcast(self.TEMP.b3_iter,   root=0)
        MPI_COMM.Barrier()
        bvfastpush.bvfastpusher(self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.T[0], self.SPACES.T[1], self.SPACES.T[2], self.SPACES.Nel, self.SPACES.p, self.Np_loc, self.TEMP.b1_iter, self.TEMP.b2_iter, self.TEMP.b3_iter, dt, KIN.particles_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        MPI_COMM.Barrier()
    
    

        MPI_COMM.Barrier()
        # ============ energy calsulations ==================
        EN.cal_total(KIN, MPI_COMM)
        if self.mpi_rank == 0:
            print('total_energy_step4', EN.total[0])

        MPI_COMM.Barrier()




    
    def substep_4_local_projection(self, dt, M1_PRE, EN, KIN, loc_shape, MPI_COMM):
        # bv substep

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        # current accumulation (all processes) 
    
        # use projection 
        if self.mpi_rank == 0:
            self.GATHER.func_quadrature_inverse(self.TEMP.LO_inv)

        MPI_COMM.Barrier()
        # ==========================================================================================
        self.TEMP.b1_old[:,:,:] = self.MHD.b1[:,:,:]
        self.TEMP.b2_old[:,:,:] = self.MHD.b2[:,:,:]
        self.TEMP.b3_old[:,:,:] = self.MHD.b3[:,:,:]

        self.TEMP.b1_iter[:,:,:] = self.MHD.b1[:,:,:]
        self.TEMP.b2_iter[:,:,:] = self.MHD.b2[:,:,:]
        self.TEMP.b3_iter[:,:,:] = self.MHD.b3[:,:,:]

        # When using cg method, tol has an strong impact on the energy conservation
        time1 = time.time()
        loc_shape.S_pi_1(KIN.particles_loc, self.Np, self.DOMAIN)
        time2 = time.time()
        if self.mpi_rank == 0:
            print('check_time_local_projection', time2 - time1)
        time1 = time.time()
        loc_shape.accumulate_1_form(MPI_COMM)
        time2 = time.time()
        if self.mpi_rank == 0:
            print('check_time_accumulation', time2 - time1)

        if self.mpi_rank == 0:
            time1 = time.time()
            mat, vec = loc_shape.assemble_1_form(self.SPACES)
            time2 = time.time()
            print('check_time_assemble', time2 - time1)
            timea = time.time()
            while True:
                self.LO.substep4_pre(self.TEMP.df_det, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], 0.5*(self.TEMP.b1_old + self.TEMP.b1_iter), 0.5*(self.TEMP.b2_old + self.TEMP.b2_iter), 0.5*(self.TEMP.b3_old + self.TEMP.b3_iter), self.SPACES, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, self.TEMP.LO_inv, self.DOMAIN..kind_map, self.DOMAIN.params_map)
            
                right_bv = self.LO.substep4_localproj_right(self.ACCBV, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.M1, M1_PRE, self.SPACES.C, mat, self.TEMP.b1_old, self.TEMP.b2_old, self.TEMP.b3_old, self.SPACES, self.SPACES.Ntot_2form, self.SPACES.Ntot_1form, self.SPACES.Nbase_2form, self.SPACES.Nbase_1form, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, vec, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map)
                #print('check_rightbv', right_bv)
                #if control == True:
                #    right_bv[:] = right_bv[:] + dt * cv.bv_right(p, indN, indD, Nel, G_inv_11, G_inv_12, G_inv_13, G_inv_22, G_inv_23, G_inv_33, DFI_11, DFI_12, DFI_13, DFI_21, DFI_22, DFI_23, DFI_31, DFI_32, DFI_33, df_det, Jeqx, Jeqy, Jeqz, temp_dft, temp_generate_weight1, temp_generate_weight2, temp_generate_weight3, temp_twoform1, temp_twoform2, temp_twoform3, (b1_old + b1_iter)/2.0, (b2_old + b2_iter)/2.0, (b3_old + b3_iter)/2.0, LO_inv, LO_b1, LO_b2, LO_b3, tensor_space_FEM)
                BV = spa.linalg.LinearOperator(shape=(self.SPACES.Ntot_1form[0]+self.SPACES.Ntot_1form[1]+self.SPACES.Ntot_1form[2], self.SPACES.Ntot_1form[0]+self.SPACES.Ntot_1form[1]+self.SPACES.Ntot_1form[2]), matvec=lambda x: self.LO.substep4_localproj_linear_operator(self.DOMAIN, self.Np, loc_shape, KIN.particles_loc, self.ACCBV, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.M1, M1_PRE, self.SPACES.C, mat, x, self.SPACES, self.SPACES.Ntot_2form, self.SPACES.Ntot_1form, self.SPACES.Nbase_2form, self.SPACES.Nbase_1form, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map)  )

                temp1, temp2, temp3 = np.split(spa.linalg.cg(BV, right_bv, x0 =np.concatenate( ( 0.5 * (self.TEMP.b1_old + self.TEMP.b1_iter).flatten(), 0.5 * (self.TEMP.b2_old + self.TEMP.b2_iter).flatten(), 0.5 * (self.TEMP.b3_old+self.TEMP.b3_iter).flatten() )), tol = 10**(-14), M=M1_PRE)[0], [self.SPACES.Ntot_1form[0], self.SPACES.Ntot_1form[0] + self.SPACES.Ntot_1form[1]]   )
                #print('check_temp3', temp3)
                self.MHD.b1[:,:,:] = temp1.reshape(self.SPACES.Nbase_1form[0])
                self.MHD.b2[:,:,:] = temp2.reshape(self.SPACES.Nbase_1form[1])
                self.MHD.b3[:,:,:] = temp3.reshape(self.SPACES.Nbase_1form[2])

                print('iteration error', np.linalg.norm((self.MHD.b1 - self.TEMP.b1_iter).flatten()) + np.linalg.norm((self.MHD.b2 - self.TEMP.b2_iter).flatten()) + np.linalg.norm((self.MHD.b3 - self.TEMP.b3_iter).flatten()))
                if np.linalg.norm((self.MHD.b1 - self.TEMP.b1_iter).flatten()) + np.linalg.norm((self.MHD.b2 - self.TEMP.b2_iter).flatten()) + np.linalg.norm((self.MHD.b3 - self.TEMP.b3_iter).flatten()) < 10** (-12): 
                    break
                self.TEMP.b1_iter[:,:,:] = self.MHD.b1[:,:,:]
                self.TEMP.b2_iter[:,:,:] = self.MHD.b2[:,:,:]
                self.TEMP.b3_iter[:,:,:] = self.MHD.b3[:,:,:]

            timeb = time.time()
            print('4_step_time_used', timeb - timea)
        MPI_COMM.Barrier()
        MPI_COMM.Bcast(self.MHD.b1,   root=0)
        MPI_COMM.Bcast(self.MHD.b2,   root=0)
        MPI_COMM.Bcast(self.MHD.b3,   root=0)
        MPI_COMM.Barrier()
        time1 = time.time()
        if self.mpi_rank == 0:
            temp1, temp2, temp3 = np.split(self.LO.substep4_localproj_pusher_field(self.ACCBV, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.M1, M1_PRE, self.SPACES.C, mat, 0.5*(self.TEMP.b1_old + self.TEMP.b1_iter), 0.5*(self.TEMP.b2_old + self.TEMP.b2_iter), 0.5*(self.TEMP.b3_old + self.TEMP.b3_iter), self.SPACES, self.SPACES.Ntot_2form, self.SPACES.Nbase_2form, self.SPACES.Nbase_1form, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, vec, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map), [self.SPACES.Ntot_1form[0], self.SPACES.Ntot_1form[0] + self.SPACES.Ntot_1form[1]]  ) 
            self.TEMP.b1_iter[:,:,:] = temp1.reshape(self.SPACES.Nbase1_form[0])
            self.TEMP.b2_iter[:,:,:] = temp2.reshape(self.SPACES.Nbase_1form[1])
            self.TEMP.b3_iter[:,:,:] = temp3.reshape(self.SPACES.Nbase_1form[2])

        MPI_COMM.Barrier()
        MPI_COMM.Bcast(self.TEMP.b1_iter,   root=0)
        MPI_COMM.Bcast(self.TEMP.b2_iter,   root=0)
        MPI_COMM.Bcast(self.TEMP.b3_iter,   root=0)
        MPI_COMM.Barrier()
        time1 = time.time()
        loc_proj_ker.bv_localproj_push(dt, self.TEMP.b1_iter, self.TEMP.b2_iter, self.TEMP.b3_iter, loc_shape.pts[0][0], loc_shape.pts[1][0], loc_shape.pts[2][0], loc_shape.wts[0][0], loc_shape.wts[1][0], loc_shape.wts[2][0], self.Np, loc_shape.n_quad, p, loc_shape.Nel, loc_shape.p_shape, loc_shape.p_size, KIN.particles_loc, loc_shape.lambdas_1_11, loc_shape.lambdas_1_12, loc_shape.lambdas_1_13, loc_shape.lambdas_1_21, loc_shape.lambdas_1_22, loc_shape.lambdas_1_23, loc_shape.lambdas_1_31, loc_shape.lambdas_1_32, loc_shape.lambdas_1_33, loc_shape.num_cell, loc_shape.coeff_i[0], loc_shape.coeff_i[1], loc_shape.coeff_i[2], loc_shape.coeff_h[0], loc_shape.coeff_h[1], loc_shape.coeff_h[2], loc_shape.NbaseN, loc_shape.NbaseD, loc_shape.related, self.Np_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        time2 = time.time()
        if self.mpi_rank == 0:
            print('check_time_push', time2 - time1)
        #bvfastpush.bvfastpusher(indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], Nel, p, Np_loc, b1_iter, b2_iter, b3_iter, dt, particles_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
        MPI_COMM.Barrier()
        # ============ energy calsulations ==================
        EN.cal_total(KIN, MPI_COMM)
        if self.mpi_rank == 0:
            print('total_energy_step4', EN.total[0])

        MPI_COMM.Barrier()

        



    def substep_4_smoothed_delta_L2(self, dt, M1_PRE, EN, L2_smoothed_shape, KIN, MPI_COMM):
        # bv substep smoothed delta function + L2 projector

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        # current accumulation (all processes) 
    
        # use projection 
        if self.mpi_rank == 0:
            self.GATHER.func_quadrature_inverse(self.TEMP.LO_inv)

        MPI_COMM.Barrier()
        # ==========================================================================================
        self.TEMP.b1_old[:,:,:] = self.MHD.b1[:,:,:]
        self.TEMP.b2_old[:,:,:] = self.MHD.b2[:,:,:]
        self.TEMP.b3_old[:,:,:] = self.MHD.b3[:,:,:]

        self.TEMP.b1_iter[:,:,:] = self.MHD.b1[:,:,:]
        self.TEMP.b2_iter[:,:,:] = self.MHD.b2[:,:,:]
        self.TEMP.b3_iter[:,:,:] = self.MHD.b3[:,:,:]

        # When using cg method, tol has an strong impact on the energy conservation
        time1 = time.time()
        L2_smoothed_shape.S_pi_1(KIN.particles_loc, self.Np, self.DOMAIN)
        time2 = time.time()
        if self.mpi_rank == 0:
            print('check_time_l2_smoothed_projection', time2 - time1)
        time1 = time.time()
        L2_smoothed_shape.accumulate_1_form(MPI_COMM)
        time2 = time.time()
        if self.mpi_rank == 0:
            print('check_time_accumulation', time2 - time1)
    
        if self.mpi_rank == 0:
            mat, vec = L2_smoothed_shape.assemble_1_form(self.SPACES)
            print('check_time_assemble', time2 - time1)
            timea = time.time()
            while True:
                self.LO.substep4_pre(self.TEMP.df_det, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], 0.5*(self.TEMP.b1_old + self.TEMP.b1_iter), 0.5*(self.TEMP.b2_old + self.TEMP.b2_iter), 0.5*(self.TEMP.b3_old + self.TEMP.b3_iter), self.SPACES, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, self.TEMP.LO_inv, self.DOMAIN.kind_map, self.DOMAIN.params_map)
            
                right_bv = self.LO.substep4_linear_operator_right(self.ACCBV, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.M1, M1_PRE, self.SPACES.C, mat, self.TEMP.b1_old, self.TEMP.b2_old, self.TEMP.b3_old, self.tensor_space_FEM, self.SPACES.Ntot_2form, self.SPACES.Ntot_1form, self.SPACES.Nbase_2form, self.SPACES.Nbase_1form, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, vec, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map)
                #print('check_rightbv', right_bv)
                #if self.control == True:
                #    right_bv[:] = right_bv[:] + dt * cv.bv_right(p, indN, indD, Nel, G_inv_11, G_inv_12, G_inv_13, G_inv_22, G_inv_23, G_inv_33, DFI_11, DFI_12, DFI_13, DFI_21, DFI_22, DFI_23, DFI_31, DFI_32, DFI_33, df_det, Jeqx, Jeqy, Jeqz, temp_dft, temp_generate_weight1, temp_generate_weight2, temp_generate_weight3, temp_twoform1, temp_twoform2, temp_twoform3, (b1_old + b1_iter)/2.0, (b2_old + b2_iter)/2.0, (b3_old + b3_iter)/2.0, LO_inv, LO_b1, LO_b2, LO_b3, tensor_space_FEM)
                BV = spa.linalg.LinearOperator(shape=(self.SPACES.Ntot_1form[0]+self.SPACES.Ntot_1form[1]+self.SPACES.Ntot_1form[2], self.SPACES.Ntot_1form[0]+self.SPACES.Ntot_1form[1]+self.SPACES.Ntot_1form[2]), matvec=lambda x: self.LO.substep4_linear_operator(self.ACCBV, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.M1, M1_PRE, self.SPACES.C, mat, x, self.SPACES, self.SPACES.Ntot_2form, self.SPACES.Ntot_1form, self.SPACES.Nbase_2form, self.SPACES.Nbase_1form, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, dt, self.DOMAIN.kind_map, self.DOMAIN.params_map)  )

                temp1, temp2, temp3 = np.split(spa.linalg.cg(BV, right_bv, x0 =np.concatenate( ( 0.5 * (self.TEMP.b1_old + self.TEMP.b1_iter).flatten(), 0.5 * (self.TEMP.b2_old + self.TEMP.b2_iter).flatten(), 0.5 * (self.TEMP.b3_old+self.TEMP.b3_iter).flatten() )), tol = 10**(-10), M=M1_PRE)[0], [self.SPACES.Ntot_1form[0], self.SPACES.Ntot_1form[0] + self.SPACES.Ntot_1form[1]]   )
                #print('check_temp3', temp3)
                self.MHD.b1[:,:,:] = temp1.reshape(self.SPACES.Nbase_1form[0])
                self.MHD.b2[:,:,:] = temp2.reshape(self.SPACES.Nbase_1form[1])
                self.MHD.b3[:,:,:] = temp3.reshape(self.SPACES.Nbase_1form[2])

                print('iteration error', np.linalg.norm((self.MHD.b1 - self.TEMP.b1_iter).flatten()) + np.linalg.norm((self.MHD.b2 - self.TEMP.b2_iter).flatten()) + np.linalg.norm((self.MHD.b3 - self.TEMP.b3_iter).flatten()))
                if np.linalg.norm((self.MHD.b1 - self.TEMP.b1_iter).flatten()) + np.linalg.norm((self.MHD.b2 - self.TEMP.b2_iter).flatten()) + np.linalg.norm((self.MHD.b3 - self.TEMP.b3_iter).flatten()) < 10** (-12): 
                    break
                self.TEMP.b1_iter[:,:,:] = self.MHD.b1[:,:,:]
                self.TEMP.b2_iter[:,:,:] = self.MHD.b2[:,:,:]
                self.TEMP.b3_iter[:,:,:] = self.MHD.b3[:,:,:]

            timeb = time.time()
            print('4_step_time_used', timeb - timea)
        MPI_COMM.Barrier()
        MPI_COMM.Bcast(self.MHD.b1,   root=0)
        MPI_COMM.Bcast(self.MHD.b2,   root=0)
        MPI_COMM.Bcast(self.MHD.b3,   root=0)
        MPI_COMM.Barrier()
    
        time1 = time.time()
        if self.mpi_rank == 0:
            temp1, temp2, temp3 = np.split(self.LO.substep4_pusher_field(self.ACCBV, self.TEMP.temp_dft, self.TEMP.temp_generate_weight1, self.TEMP.temp_generate_weight3, self.TEMP.G_inv_11, self.TEMP.G_inv_12, self.TEMP.G_inv_13, self.TEMP.G_inv_22, self.TEMP.G_inv_23, self.TEMP.G_inv_33, self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], self.SPACES.M1, M1_PRE, self.SPACES.C, mat, 0.5*(self.TEMP.b1_old + self.TEMP.b1_iter), 0.5*(self.TEMP.b2_old + self.TEMP.b2_iter), 0.5*(self.TEMP.b3_old + self.TEMP.b3_iter), self.SPACES, self.SPACES.Ntot_2form, self.SPACES.Nbase_2form, self.SPACES.Nbase_1form, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, vec, dt, self.DOMAIN.kind_map, self.Domain.params_map), [self.SPACES.Ntot_1form[0], self.SPACES.Ntot_1form[0] + self.SPACES.Ntot_1form[1]]  ) 
            self.TEMP.b1_iter[:,:,:] = temp1.reshape(self.SPACES.Nbase_1form[0])
            self.TEMP.b2_iter[:,:,:] = temp2.reshape(self.SPACES.Nbase_1form[1])
            self.TEMP.b3_iter[:,:,:] = temp3.reshape(self.SPACES.Nbase_1form[2])

        MPI_COMM.Barrier()
        MPI_COMM.Bcast(self.TEMP.b1_iter,   root=0)
        MPI_COMM.Bcast(self.TEMP.b2_iter,   root=0)
        MPI_COMM.Bcast(self.TEMP.b3_iter,   root=0)
        MPI_COMM.Barrier()
        L2_smoothed_ker.bvpushltwo(self.SPACES.indN[0], self.SPACES.indN[1], self.SPACES.indN[2], self.SPACES.indD[0], self.SPACES.indD[1], self.SPACES.indD[2], dt, self.TEMP.b1_iter, self.TEMP.b2_iter, self.TEMP.b3_iter, self.SPACES.basisN[0], self.SPACES.basisN[1], self.SPACES.basisN[2], self.SPACES.basisD[0], self.SPACES.basisD[1], self.SPACES.basisD[2], self.SPACES.pts[0], self.SPACES.pts[1], self.SPACES.pts[2], self.SPACES.wts[0], self.SPACES.wts[1], self.SPACES.wts[2], self.Np, self.SPACES.n_quad, self.SPACES.p, self.SPACES.Nel, self.SHAPE.p_shape, self.SHAPE.p_size, KIN.particles_loc, L2_smoothed_shape.lambdas_1_11, L2_smoothed_shape.lambdas_1_12, L2_smoothed_shape.lambdas_1_13, L2_smoothed_shape.lambdas_1_21, L2_smoothed_shape.lambdas_1_22, L2_smoothed_shape.lambdas_1_23, L2_smoothed_shape.lambdas_1_31, L2_smoothed_shape.lambdas_1_32, L2_smoothed_shape.lambdas_1_33, self.SPACES.NbaseN, self.SPACES.NbaseD, L2_smoothed_shape.related, self.Np_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        MPI_COMM.Barrier()
    
    

        # ============ energy calsulations ==================
        EN.cal_total(KIN, MPI_COMM)
        if self.mpi_rank == 0:
            print('total_energy_step4', EN.total[0])

        MPI_COMM.Barrier()
