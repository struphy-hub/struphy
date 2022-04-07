# import pyccel decorators
from pyccel.decorators import types

import struphy.feec.bsplines_kernels as bsp

import struphy.feec.basics.spline_evaluation_3d as eva

import numpy as np


# =========================================================
@types('double[:]','double[:,:]','int')
def kinetic(kinetic_loc, particles_loc, np):

    kinetic_loc[:] = 0.0

    #$ omp parallel
    #$ omp do reduction ( + : kinetic_loc) private (ip)
    for ip in range(np):
        kinetic_loc[0] += particles_loc[6, ip] * (particles_loc[3, ip]**2 + particles_loc[4, ip]**2 + particles_loc[5, ip]**2)/2.0 
    #$ omp end do
    #$ omp end parallel

    ierr = 0




# ============================================================
@types('int[:]','int','int','int','double[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]')
def thermal(Nel, nq1, nq2, nq3, thermal, wts1, wts2, wts3, n_quadrature, quadrature_log, df_det):

    from numpy import log

    thermal[:] = 0.0

    #$ omp parallel
    #$ omp do reduction ( + : thermal) private (il1, il2, il3, q1, q2, q3, value)
    for il1 in range(Nel[0]):
        for il2 in range(Nel[1]):
            for il3 in range(Nel[2]):
                value = 0.0
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value += n_quadrature[il1, il2, il3, q1, q2, q3] * quadrature_log[il1, il2, il3, q1, q2, q3] * df_det[il1, il2, il3, q1, q2, q3]

                thermal[0] += value

    #$ omp end do
    #$ omp end parallel
    ierr = 0
