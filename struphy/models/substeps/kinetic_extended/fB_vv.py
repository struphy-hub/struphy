
import struphy.pic.kinetic_extended.fB_massless_pusher as pusher
import time
import struphy.feec.projectors.shape_pro_local.shape_local_projector_kernel as loc_proj_ker
from mpi4py import MPI

class Substep_2:
    '''The substep vv of fB formulation
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

        ACCVV   : obj, 
            obj storing accmulation method from particles

        M2_PRE  : operator, 
            preconditoiner of M2
        M1_PRE  : operator, 
            preconditoiner of M1
    '''

    def __init__(self, LINEAR_OPERATORS, DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM, temperature, SHAPE, TEMP, ACCUMVV, control, M2_PRE, M1_PRE):

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
        self.ACCUMVV    = ACCUMVV
        self.control    = control
        self.M2_PRE     = M2_PRE
        self.M1_PRE     = M1_PRE
        self.LO         = LINEAR_OPERATORS
        self.ACCUMVV    = ACCUMVV
        self.indN    = SPACES.indN
        self.indD    = SPACES.indD
        self.Nel     = SPACES.Nel
        self.p       = SPACES.p
        self.T       = SPACES.T
        self.DOMAIN  = DOMAIN




    def push_proj_RK4(self, dt, EN, KIN, tol, MPI_COMM):
        # vv substep
        # L2 projector is used.
        timea = time.time()

        if self.mpi_rank == 0:
            self.LO.linearoperator_pre_step_vv(self.SPACES, self.TEMP.df_det, self.TEMP.DFIT_11, self.TEMP.DFIT_12, self.TEMP.DFIT_13, self.TEMP.DFIT_21, self.TEMP.DFIT_22, self.TEMP.DFIT_23, self.TEMP.DFIT_31, self.TEMP.DFIT_32, self.TEMP.DFIT_33, self.M2_PRE, self.SPACES.M2, self.MHD.b1, self.MHD.b2, self.MHD.b3, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_inv, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3)
        #march.projection_RK4(LO, tol, self.GATHER.gather_grid, self.Np_loc, self.MHD.b1, self.MHD.b2, self.MHD.b3, self.SPACES.Nel, self.SPACES.p, self.DOMAIN, self.TEMP.temp_particle, self.SPACES, self.SPACES.indN, self.SPACES.indD, dt, self.Np, self.ACCUMVV, self.SPACES.M2, self.M2_PRE, KIN.particles_loc, MPI_COMM, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_inv, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, self.control)
        #march.scatter_gather_RK4(Np_loc, b1, b2, b3, Nel, p, domain, temp_particle, tensor_space_FEM, indN, indD, dt, Np, acc_step_vv, tensor_space_FEM.M2, M2_PRE, particles_loc, mpi_comm, LO_w1, LO_w2, LO_w3, LO_r1, LO_r2, LO_r3, LO_inv, LO_b1, LO_b2, LO_b3, control)
        #============= first stage =================================
        self.ACCUMVV.mid_particles[:,:] = 0.0
        self.ACCUMVV.accumulate_substep_vv(KIN.particles_loc, MPI_COMM, 1, dt)
        if self.mpi_rank == 0:
            self.LO.linearoperator_step_vv(self.M2_PRE, self.SPACES.M2, self.M1_PRE, self.SPACES.M1, self.TEMP, self.ACCUMVV)

        MPI_COMM.Bcast(self.ACCUMVV.coe1, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe2, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe3, root=0) 

        pusher.pushvv(self.indN[0], self.indN[1], self.indN[2], self.indD[0], self.indD[1], self.indD[2], self.ACCUMVV.stage1_out_loc, self.ACCUMVV.coe1, self.ACCUMVV.coe2, self.ACCUMVV.coe3, self.Np_loc, self.Nel, self.p, self.T[0], self.T[1], self.T[2], KIN.particles_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        #if control == True:
        #    cv.vv_right(1, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        MPI_COMM.Barrier()

        # ============= second stage =================================
        self.ACCUMVV.accumulate_substep_vv(KIN.particles_loc, MPI_COMM, 2, dt)
        MPI_COMM.Barrier()
        if self.mpi_rank == 0:
            self.LO.linearoperator_step_vv(self.M2_PRE, self.SPACES.M2, self.M1_PRE, self.SPACES.M1, self.TEMP, self.ACCUMVV)

        MPI_COMM.Bcast(self.ACCUMVV.coe1, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe2, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe3, root=0) 

        pusher.pushvv(self.indN[0], self.indN[1], self.indN[2], self.indD[0], self.indD[1], self.indD[2], self.ACCUMVV.stage2_out_loc, self.ACCUMVV.coe1, self.ACCUMVV.coe2, self.ACCUMVV.coe3, self.Np_loc, self.Nel, self.p, self.T[0], self.T[1], self.T[2], KIN.particles_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        #if control == True:
            #cv.vv_right(2, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        MPI_COMM.Barrier()

        # ==============third stage======================
        self.ACCUMVV.accumulate_substep_vv(KIN.particles_loc, MPI_COMM, 3, dt)
        if self.mpi_rank == 0:
            self.LO.linearoperator_step_vv(self.M2_PRE, self.SPACES.M2, self.M1_PRE, self.SPACES.M1, self.TEMP, self.ACCUMVV)

        MPI_COMM.Bcast(self.ACCUMVV.coe1, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe2, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe3, root=0) 

        pusher.pushvv(self.indN[0], self.indN[1], self.indN[2], self.indD[0], self.indD[1], self.indD[2], self.ACCUMVV.stage3_out_loc, self.ACCUMVV.coe1, self.ACCUMVV.coe2, self.ACCUMVV.coe3, self.Np_loc, self.Nel, self.p, self.T[0], self.T[1], self.T[2], KIN.particles_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN,self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)

        MPI_COMM.Barrier()
        #if control == True:
        #    cv.vv_right(3, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        
        timeb = time.time()
        print('2_step_time_used', timeb - timea)
        MPI_COMM.Barrier()

        # =============fourth stage====================
        self.ACCUMVV.accumulate_substep_vv(KIN.particles_loc, MPI_COMM, 4, dt)
        MPI_COMM.Barrier()
        if self.mpi_rank == 0:
            self.LO.linearoperator_step_vv(self.M2_PRE, self.SPACES.M2, self.M1_PRE, self.SPACES.M1, self.TEMP, self.ACCUMVV)

        MPI_COMM.Barrier()
        MPI_COMM.Bcast(self.ACCUMVV.coe1, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe2, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe3, root=0) 

        pusher.pushvv(self.indN[0], self.indN[1], self.indN[2], self.indD[0], self.indD[1], self.indD[2], self.ACCUMVV.stage4_out_loc, self.ACCUMVV.coe1, self.ACCUMVV.coe2, self.ACCUMVV.coe3, self.Np_loc, self.Nel, self.p, self.T[0], self.T[1], self.T[2], KIN.particles_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        #if control == True:
        #    cv.vv_right(4, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        MPI_COMM.Barrier()

        pusher.rkfinal(KIN.particles_loc, self.ACCUMVV.stage1_out_loc, self.ACCUMVV.stage2_out_loc, self.ACCUMVV.stage3_out_loc, self.ACCUMVV.stage4_out_loc, self.Np_loc, dt)
        MPI_COMM.Barrier()
        # ============ energy calsulations ==================
        EN.cal_total(KIN, MPI_COMM)
        if self.mpi_rank == 0:
            print('total_energy_step2', EN.total[0])


    def push_proj_RK2(self, dt, EN, KIN, tol, MPI_COMM):
        # vv substep
        # L2 projector is used.
        timea = time.time()

        if self.mpi_rank == 0:
            self.LO.linearoperator_pre_step_vv(self.SPACES, self.TEMP.df_det, self.TEMP.DFIT_11, self.TEMP.DFIT_12, self.TEMP.DFIT_13, self.TEMP.DFIT_21, self.TEMP.DFIT_22, self.TEMP.DFIT_23, self.TEMP.DFIT_31, self.TEMP.DFIT_32, self.TEMP.DFIT_33, self.M2_PRE, self.SPACES.M2, self.MHD.b1, self.MHD.b2, self.MHD.b3, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_inv, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3)
        #march.projection_RK4(LO, tol, self.GATHER.gather_grid, self.Np_loc, self.MHD.b1, self.MHD.b2, self.MHD.b3, self.SPACES.Nel, self.SPACES.p, self.DOMAIN, self.TEMP.temp_particle, self.SPACES, self.SPACES.indN, self.SPACES.indD, dt, self.Np, self.ACCUMVV, self.SPACES.M2, self.M2_PRE, KIN.particles_loc, MPI_COMM, self.TEMP.LO_w1, self.TEMP.LO_w2, self.TEMP.LO_w3, self.TEMP.LO_r1, self.TEMP.LO_r2, self.TEMP.LO_r3, self.TEMP.LO_inv, self.TEMP.LO_b1, self.TEMP.LO_b2, self.TEMP.LO_b3, self.control)
        #march.scatter_gather_RK4(Np_loc, b1, b2, b3, Nel, p, domain, temp_particle, tensor_space_FEM, indN, indD, dt, Np, acc_step_vv, tensor_space_FEM.M2, M2_PRE, particles_loc, mpi_comm, LO_w1, LO_w2, LO_w3, LO_r1, LO_r2, LO_r3, LO_inv, LO_b1, LO_b2, LO_b3, control)
        #============= first stage =================================
        self.ACCUMVV.mid_particles[:,:] = 0.0
        self.ACCUMVV.accumulate_substep_vv(KIN.particles_loc, MPI_COMM, 1, dt)
        if self.mpi_rank == 0:
            self.LO.linearoperator_step_vv(self.M2_PRE, self.SPACES.M2, self.M1_PRE, self.SPACES.M1, self.TEMP, self.ACCUMVV)

        MPI_COMM.Bcast(self.ACCUMVV.coe1, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe2, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe3, root=0) 

        pusher.pushvv(self.indN[0], self.indN[1], self.indN[2], self.indD[0], self.indD[1], self.indD[2], self.ACCUMVV.stage1_out_loc, self.ACCUMVV.coe1, self.ACCUMVV.coe2, self.ACCUMVV.coe3, self.Np_loc, self.Nel, self.p, self.T[0], self.T[1], self.T[2], KIN.particles_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        #if control == True:
        #    cv.vv_right(1, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        MPI_COMM.Barrier()

        # ============= second stage =================================
        self.ACCUMVV.accumulate_substep_vv(KIN.particles_loc, MPI_COMM, 2, dt)
        MPI_COMM.Barrier()
        if self.mpi_rank == 0:
            self.LO.linearoperator_step_vv(self.M2_PRE, self.SPACES.M2, self.M1_PRE, self.SPACES.M1, self.TEMP, self.ACCUMVV)

        MPI_COMM.Bcast(self.ACCUMVV.coe1, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe2, root=0) 
        MPI_COMM.Bcast(self.ACCUMVV.coe3, root=0) 

        pusher.pushvv(self.indN[0], self.indN[1], self.indN[2], self.indD[0], self.indD[1], self.indD[2], self.ACCUMVV.stage2_out_loc, self.ACCUMVV.coe1, self.ACCUMVV.coe2, self.ACCUMVV.coe3, self.Np_loc, self.Nel, self.p, self.T[0], self.T[1], self.T[2], KIN.particles_loc, self.DOMAIN.kind_map, self.DOMAIN.params_map, self.DOMAIN.T[0], self.DOMAIN.T[1], self.DOMAIN.T[2], self.DOMAIN.p, self.DOMAIN.Nel, self.DOMAIN.NbaseN, self.DOMAIN.cx, self.DOMAIN.cy, self.DOMAIN.cz)
        #if control == True:
            #cv.vv_right(2, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        MPI_COMM.Barrier()
        pusher.rkfinal2(KIN.particles_loc, self.ACCUMVV.stage2_out_loc, self.Np_loc, dt)
        MPI_COMM.Barrier()

        # ============ energy calsulations ==================
        EN.cal_total(KIN, MPI_COMM)
        if self.mpi_rank == 0:
            print('total_energy_step2', EN.total[0])



    def scatter_gather_RK4(self, Np_loc, b1, b2, b3, Nel, p, domain, temp_particle, tensor_space_FEM, indN, indD, dt, Np, acc, M2, M2_PRE, particles_loc, mpi_comm, LO_w1, LO_w2, LO_w3, LO_r1, LO_r2, LO_r3, LO_inv, LO_b1, LO_b2, LO_b3, control):
        # scatter-gather method 
        mpi_rank = mpi_comm.Get_rank()
        #energies_loc['K'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
        #mpi_comm.Reduce(energies_loc['K'], kinetic_old, op=MPI.SUM, root=0)
        #mpi_comm.Bcast(kinetic_old, root=0)
        mpi_comm.Barrier()
        #==============RK4======================1-stage====================
        self.LO.gather(1, dt, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np)
        mpi_comm.Barrier()
        mpi_comm.Reduce(1./Np*acc.gather1_loc, acc.gather1, op=MPI.SUM, root=0)
        mpi_comm.Reduce(1./Np*acc.gather2_loc, acc.gather2, op=MPI.SUM, root=0)
        mpi_comm.Reduce(1./Np*acc.gather3_loc, acc.gather3, op=MPI.SUM, root=0)
        mpi_comm.Barrier()
        if mpi_rank == 0:    
            self.LO.scatter_gather_weight(acc, tensor_space_FEM, LO_b1, LO_b2, LO_b3)

        mpi_comm.Bcast(acc.weight1, root=0) 
        mpi_comm.Bcast(acc.weight2, root=0)
        mpi_comm.Bcast(acc.weight3, root=0) 

        self.LO.scatter(1, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np)

        #if control == True:
        #    cv.vv_right(1, Np_loc, u, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        mpi_comm.Barrier()
        #=======================================2-stage====================
        self.LO.gather(2, dt, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np)
        mpi_comm.Barrier()
        mpi_comm.Reduce(1./Np*acc.gather1_loc, acc.gather1, op=MPI.SUM, root=0)
        mpi_comm.Reduce(1./Np*acc.gather2_loc, acc.gather2, op=MPI.SUM, root=0)
        mpi_comm.Reduce(1./Np*acc.gather3_loc, acc.gather3, op=MPI.SUM, root=0)
        mpi_comm.Barrier()
        if mpi_rank == 0:    
            self.LO.scatter_gather_weight(acc, tensor_space_FEM, LO_b1, LO_b2, LO_b3)

        mpi_comm.Bcast(acc.weight1, root=0) 
        mpi_comm.Bcast(acc.weight2, root=0)
        mpi_comm.Bcast(acc.weight3, root=0) 

        self.LO.scatter(2, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np)
        #if control == True:
        #    cv.vv_right(2, Np_loc, u, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        mpi_comm.Barrier()
        #======================================3-stage=====================
        self.LO.gather(3, dt, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np)
        mpi_comm.Barrier()
        mpi_comm.Reduce(1./Np*acc.gather1_loc, acc.gather1, op=MPI.SUM, root=0)
        mpi_comm.Reduce(1./Np*acc.gather2_loc, acc.gather2, op=MPI.SUM, root=0)
        mpi_comm.Reduce(1./Np*acc.gather3_loc, acc.gather3, op=MPI.SUM, root=0)
        mpi_comm.Barrier()
        if mpi_rank == 0:    
            self.LO.scatter_gather_weight(acc, tensor_space_FEM, LO_b1, LO_b2, LO_b3)

        mpi_comm.Bcast(acc.weight1, root=0) 
        mpi_comm.Bcast(acc.weight2, root=0)
        mpi_comm.Bcast(acc.weight3, root=0) 

        self.LO.scatter(3, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np)
        #if control == True:
        #    cv.vv_right(3, Np_loc, u, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        mpi_comm.Barrier()
        #=====================================4-stage======================
        self.LO.gather(4, dt, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np)

        mpi_comm.Barrier()
        mpi_comm.Reduce(1./Np*acc.gather1_loc, acc.gather1, op=MPI.SUM, root=0)
        mpi_comm.Reduce(1./Np*acc.gather2_loc, acc.gather2, op=MPI.SUM, root=0)
        mpi_comm.Reduce(1./Np*acc.gather3_loc, acc.gather3, op=MPI.SUM, root=0)
        mpi_comm.Barrier()
        if mpi_rank == 0:    
            self.LO.scatter_gather_weight(acc, tensor_space_FEM, LO_b1, LO_b2, LO_b3)

        mpi_comm.Bcast(acc.weight1, root=0) 
        mpi_comm.Bcast(acc.weight2, root=0)
        mpi_comm.Bcast(acc.weight3, root=0) 

        self.LO.scatter(4, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np)

        #if control == True:
        #    cv.vv_right(4, Np_loc, u, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        mpi_comm.Barrier()
        #==================================================================
        pusher.rkfinal(particles_loc, acc.stage1_out_loc, acc.stage2_out_loc, acc.stage3_out_loc, acc.stage4_out_loc, Np_loc, dt)
    
        #====================projection====================================
        #energies_loc['K'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
        #mpi_comm.Reduce(energies_loc['K'], kinetic_new, op=MPI.SUM, root=0)
        #mpi_comm.Barrier()
        #mpi_comm.Bcast(kinetic_new, root=0)
        #particles_loc[3,:] = particles_loc[3,:] * (kinetic_old / kinetic_new) ** 0.5
        #particles_loc[4,:] = particles_loc[4,:] * (kinetic_old / kinetic_new) ** 0.5
        #particles_loc[5,:] = particles_loc[5,:] * (kinetic_old / kinetic_new) ** 0.5  


    # ============================fourth order Rounge-Kutta of using Local projector ====================================
    def local_projection_RK4(self, tol, local_shape, gather_grid, Np_loc, Nel, p, dt, Np, acc, M2, M2_PRE, particles_loc, mpi_comm, control):
        mpi_rank = mpi_comm.Get_rank()

        # ============ first stage ===================================
        acc.mid_particles[:,:] = 0.0
        local_shape.vv_S1(particles_loc, Np, self.DOMAIN, 1, acc, dt, mpi_comm)

        mpi_comm.Barrier()
        if mpi_rank == 0:
            self.LO.local_linearoperator_step_vv(self.indN, self.indD, acc, M2_PRE, M2, Np, self.TEMP)

        mpi_comm.Bcast(acc.coe1, root=0) 
        mpi_comm.Bcast(acc.coe2, root=0) 
        mpi_comm.Bcast(acc.coe3, root=0) 
        mpi_comm.Barrier()
    
        loc_proj_ker.vv_push(acc.stage1_out_loc, dt, acc.coe1, acc.coe2, acc.coe3, local_shape.pts[0][0], local_shape.pts[1][0], local_shape.pts[2][0], local_shape.wts[0][0], local_shape.wts[1][0], local_shape.wts[2][0], Np, local_shape.n_quad, p, local_shape.Nel, local_shape.p_shape, local_shape.p_size, particles_loc, local_shape.lambdas_1_11, local_shape.lambdas_1_12, local_shape.lambdas_1_13, local_shape.lambdas_1_21, local_shape.lambdas_1_22, local_shape.lambdas_1_23, local_shape.lambdas_1_31, local_shape.lambdas_1_32, local_shape.lambdas_1_33, local_shape.num_cell, local_shape.coeff_i[0], local_shape.coeff_i[1], local_shape.coeff_i[2], local_shape.coeff_h[0], local_shape.coeff_h[1], local_shape.coeff_h[2], local_shape.NbaseN, local_shape.NbaseD, local_shape.related, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)

        #if control == True:
        #    cv.vv_right(1, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        mpi_comm.Barrier()
    
        # ============= second stage =================================
        local_shape.vv_S1(particles_loc, Np, self.DOMAIN, 2, acc, dt, mpi_comm)
        mpi_comm.Barrier()
        
        if mpi_rank == 0:
            self.LO.local_linearoperator_step_vv(self.indN, self.indD, acc, M2_PRE, M2, Np, self.TEMP)

        mpi_comm.Bcast(acc.coe1, root=0) 
        mpi_comm.Bcast(acc.coe2, root=0) 
        mpi_comm.Bcast(acc.coe3, root=0) 
        mpi_comm.Barrier()

        loc_proj_ker.vv_push(acc.stage2_out_loc, dt, acc.coe1, acc.coe2, acc.coe3, local_shape.pts[0][0], local_shape.pts[1][0], local_shape.pts[2][0], local_shape.wts[0][0], local_shape.wts[1][0], local_shape.wts[2][0], Np, local_shape.n_quad, p, local_shape.Nel, local_shape.p_shape, local_shape.p_size, particles_loc, local_shape.lambdas_1_11, local_shape.lambdas_1_12, local_shape.lambdas_1_13, local_shape.lambdas_1_21, local_shape.lambdas_1_22, local_shape.lambdas_1_23, local_shape.lambdas_1_31, local_shape.lambdas_1_32, local_shape.lambdas_1_33, local_shape.num_cell, local_shape.coeff_i[0], local_shape.coeff_i[1], local_shape.coeff_i[2], local_shape.coeff_h[0], local_shape.coeff_h[1], local_shape.coeff_h[2], local_shape.NbaseN, local_shape.NbaseD, local_shape.related, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)

        #pusher.local_pushvv(indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], acc.basis_p, acc.stage2_out_loc, acc.coe1, acc.coe2, acc.coe3, particles_loc.shape[1], acc.Nel, acc.p, acc.T[0], acc.T[1], acc.T[2], particles_loc, acc.domain.kind_map, acc.domain.params_map, acc.domain.T[0], acc.domain.T[1], acc.domain.T[2], acc.domain.p, acc.domain.Nel, acc.domain.NbaseN, acc.domain.cx, acc.domain.cy, acc.domain.cz)
        #if control == True:
        #    cv.vv_right(2, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        mpi_comm.Barrier()

        # ==============third stage======================
        local_shape.vv_S1(particles_loc, Np, self.DOMAIN, 3, acc, dt, mpi_comm)
        if mpi_rank == 0:
            self.LO.local_linearoperator_step_vv(self.indN, self.indD, acc, M2_PRE, M2, Np, self.TEMP)

        mpi_comm.Bcast(acc.coe1, root=0) 
        mpi_comm.Bcast(acc.coe2, root=0) 
        mpi_comm.Bcast(acc.coe3, root=0) 
        mpi_comm.Barrier()

        loc_proj_ker.vv_push(acc.stage3_out_loc, dt, acc.coe1, acc.coe2, acc.coe3, local_shape.pts[0][0], local_shape.pts[1][0], local_shape.pts[2][0], local_shape.wts[0][0], local_shape.wts[1][0], local_shape.wts[2][0], Np, local_shape.n_quad, p, local_shape.Nel, local_shape.p_shape, local_shape.p_size, particles_loc, local_shape.lambdas_1_11, local_shape.lambdas_1_12, local_shape.lambdas_1_13, local_shape.lambdas_1_21, local_shape.lambdas_1_22, local_shape.lambdas_1_23, local_shape.lambdas_1_31, local_shape.lambdas_1_32, local_shape.lambdas_1_33, local_shape.num_cell, local_shape.coeff_i[0], local_shape.coeff_i[1], local_shape.coeff_i[2], local_shape.coeff_h[0], local_shape.coeff_h[1], local_shape.coeff_h[2], local_shape.NbaseN, local_shape.NbaseD, local_shape.related, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)

        #pusher.local_pushvv(indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], acc.basis_p, acc.stage3_out_loc, acc.coe1, acc.coe2, acc.coe3, particles_loc.shape[1], acc.Nel, acc.p, acc.T[0], acc.T[1], acc.T[2], particles_loc, acc.domain.kind_map, acc.domain.params_map, acc.domain.T[0], acc.domain.T[1], acc.domain.T[2], acc.domain.p, acc.domain.Nel, acc.domain.NbaseN, acc.domain.cx, acc.domain.cy, acc.domain.cz)

        mpi_comm.Barrier()
        #if control == True:
        #    cv.vv_right(3, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        # =============fourth stage====================
        local_shape.vv_S1(particles_loc, Np, self.DOMAIN, 4, acc, dt, mpi_comm)
        mpi_comm.Barrier()
        if mpi_rank == 0:
            self.LO.local_linearoperator_step_vv(self.indN, self.indD, acc, M2_PRE, M2, Np, self.TEMP)

        mpi_comm.Barrier()
        mpi_comm.Bcast(acc.coe1, root=0) 
        mpi_comm.Bcast(acc.coe2, root=0) 
        mpi_comm.Bcast(acc.coe3, root=0) 
        mpi_comm.Barrier()

        loc_proj_ker.vv_push(acc.stage4_out_loc, dt, acc.coe1, acc.coe2, acc.coe3, local_shape.pts[0][0], local_shape.pts[1][0], local_shape.pts[2][0], local_shape.wts[0][0], local_shape.wts[1][0], local_shape.wts[2][0], Np, local_shape.n_quad, p, local_shape.Nel, local_shape.p_shape, local_shape.p_size, particles_loc, local_shape.lambdas_1_11, local_shape.lambdas_1_12, local_shape.lambdas_1_13, local_shape.lambdas_1_21, local_shape.lambdas_1_22, local_shape.lambdas_1_23, local_shape.lambdas_1_31, local_shape.lambdas_1_32, local_shape.lambdas_1_33, local_shape.num_cell, local_shape.coeff_i[0], local_shape.coeff_i[1], local_shape.coeff_i[2], local_shape.coeff_h[0], local_shape.coeff_h[1], local_shape.coeff_h[2], local_shape.NbaseN, local_shape.NbaseD, local_shape.related, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)

        #pusher.local_pushvv(indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], acc.basis_p, acc.stage4_out_loc, acc.coe1, acc.coe2, acc.coe3, particles_loc.shape[1], acc.Nel, acc.p, acc.T[0], acc.T[1], acc.T[2], particles_loc, acc.domain.kind_map, acc.domain.params_map, acc.domain.T[0], acc.domain.T[1], acc.domain.T[2], acc.domain.p, acc.domain.Nel, acc.domain.NbaseN, acc.domain.cx, acc.domain.cy, acc.domain.cz)
        #if control == True:
        #    cv.vv_right(4, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        mpi_comm.Barrier()
        # ===============================================
        pusher.rkfinal(particles_loc, acc.stage1_out_loc, acc.stage2_out_loc, acc.stage3_out_loc, acc.stage4_out_loc, Np_loc, dt)



    # ============================second order Rounge-Kutta of using Local projector ====================================
    def local_projection_RK2(self, tol, local_shape, gather_grid, Np_loc, b1, b2, b3, Nel, p, domain, temp_particle, tensor_space_FEM, indN, indD, dt, Np, acc, M2, M2_PRE, particles_loc, mpi_comm, control):
    
        #mpi_comm = MPI.COMM_WORLD
        mpi_size = mpi_comm.Get_size()
        mpi_rank = mpi_comm.Get_rank()

        # ============ first stage ===================================
        acc.mid_particles[:,:] = 0.0
        local_shape.vv_S1(particles_loc, Np, domain, 1, acc, dt, mpi_comm)
        mpi_comm.Barrier()
        if mpi_rank == 0:
            self.LO.local_linearoperator_step_vv(indN, indD, acc, M2_PRE, M2, Np, self.TEMP)

        mpi_comm.Bcast(acc.coe1, root=0) 
        mpi_comm.Bcast(acc.coe2, root=0) 
        mpi_comm.Bcast(acc.coe3, root=0) 
        mpi_comm.Barrier()
    
        loc_proj_ker.vv_push(acc.stage1_out_loc, dt, acc.coe1, acc.coe2, acc.coe3, local_shape.pts[0][0], local_shape.pts[1][0], local_shape.pts[2][0], local_shape.wts[0][0], local_shape.wts[1][0], local_shape.wts[2][0], Np, local_shape.n_quad, p, local_shape.Nel, local_shape.p_shape, local_shape.p_size, particles_loc, local_shape.lambdas_1_11, local_shape.lambdas_1_12, local_shape.lambdas_1_13, local_shape.lambdas_1_21, local_shape.lambdas_1_22, local_shape.lambdas_1_23, local_shape.lambdas_1_31, local_shape.lambdas_1_32, local_shape.lambdas_1_33, local_shape.num_cell, local_shape.coeff_i[0], local_shape.coeff_i[1], local_shape.coeff_i[2], local_shape.coeff_h[0], local_shape.coeff_h[1], local_shape.coeff_h[2], local_shape.NbaseN, local_shape.NbaseD, local_shape.related, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)

        #pusher.local_pushvv(indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], acc.basis_p, acc.stage1_out_loc, acc.coe1, acc.coe2, acc.coe3, particles_loc.shape[1], acc.Nel, acc.p, acc.T[0], acc.T[1], acc.T[2], particles_loc, acc.domain.kind_map, acc.domain.params_map, acc.domain.T[0], acc.domain.T[1], acc.domain.T[2], acc.domain.p, acc.domain.Nel, acc.domain.NbaseN, acc.domain.cx, acc.domain.cy, acc.domain.cz)
        #if control == True:
        #    cv.vv_right(1, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        mpi_comm.Barrier()
    
        # ============= second stage =================================
        local_shape.vv_S1(particles_loc, Np, domain, 2, acc, dt, mpi_comm)
        mpi_comm.Barrier()
        if mpi_rank == 0:
            self.LO.local_linearoperator_step_vv(indN, indD, acc, M2_PRE, M2, Np, self.TEMP)

        mpi_comm.Bcast(acc.coe1, root=0) 
        mpi_comm.Bcast(acc.coe2, root=0) 
        mpi_comm.Bcast(acc.coe3, root=0) 
        mpi_comm.Barrier()

        loc_proj_ker.vv_push(acc.stage2_out_loc, dt, acc.coe1, acc.coe2, acc.coe3, local_shape.pts[0][0], local_shape.pts[1][0], local_shape.pts[2][0], local_shape.wts[0][0], local_shape.wts[1][0], local_shape.wts[2][0], Np, local_shape.n_quad, p, local_shape.Nel, local_shape.p_shape, local_shape.p_size, particles_loc, local_shape.lambdas_1_11, local_shape.lambdas_1_12, local_shape.lambdas_1_13, local_shape.lambdas_1_21, local_shape.lambdas_1_22, local_shape.lambdas_1_23, local_shape.lambdas_1_31, local_shape.lambdas_1_32, local_shape.lambdas_1_33, local_shape.num_cell, local_shape.coeff_i[0], local_shape.coeff_i[1], local_shape.coeff_i[2], local_shape.coeff_h[0], local_shape.coeff_h[1], local_shape.coeff_h[2], local_shape.NbaseN, local_shape.NbaseD, local_shape.related, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)

        #pusher.local_pushvv(indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], acc.basis_p, acc.stage2_out_loc, acc.coe1, acc.coe2, acc.coe3, particles_loc.shape[1], acc.Nel, acc.p, acc.T[0], acc.T[1], acc.T[2], particles_loc, acc.domain.kind_map, acc.domain.params_map, acc.domain.T[0], acc.domain.T[1], acc.domain.T[2], acc.domain.p, acc.domain.Nel, acc.domain.NbaseN, acc.domain.cx, acc.domain.cy, acc.domain.cz)
        #if control == True:
        #    cv.vv_right(2, tol, Np_loc, gather_grid, domain, acc, tensor_space_FEM.NbaseN, tensor_space_FEM.NbaseD, temp_particle, p, Nel, tensor_space_FEM, b1, b2, b3, particles_loc)
        mpi_comm.Barrier()
        # ===============================================
        pusher.rkfinal2(particles_loc, acc.stage2_out_loc, Np_loc, dt)

