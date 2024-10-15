# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

from pyccel.decorators import pure


@pure
def kernel_mass(nel: 'int[:]', p: 'int[:]', nq: 'int[:]', ni: 'int[:]', nj: 'int[:]', w1: 'float[:,:]', w2: 'float[:,:]', w3: 'float[:,:]', bi1: 'float[:,:,:,:]', bi2: 'float[:,:,:,:]', bi3: 'float[:,:,:,:]', bj1: 'float[:,:,:,:]', bj2: 'float[:,:,:,:]', bj3: 'float[:,:,:,:]', ind_base1: 'int[:,:]', ind_base2: 'int[:,:]', ind_base3: 'int[:,:]', mat: 'float[:,:,:,:,:,:]', mat_fun: 'float[:,:,:,:,:,:]'):

    mat[:, :, :, :, :, :] = 0.

    #$ omp parallel private(ie1, ie2, ie3, il1, il2, il3, jl1, jl2, jl3, value, q1, q2, q3, wvol, bi, bj)
    #$ omp for reduction ( + : mat)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):

                for il1 in range(p[0] + 1 - ni[0]):
                    for il2 in range(p[1] + 1 - ni[1]):
                        for il3 in range(p[2] + 1 - ni[2]):
                            for jl1 in range(p[0] + 1 - nj[0]):
                                for jl2 in range(p[1] + 1 - nj[1]):
                                    for jl3 in range(p[2] + 1 - nj[2]):

                                        value = 0.

                                        for q1 in range(nq[0]):
                                            for q2 in range(nq[1]):
                                                for q3 in range(nq[2]):

                                                    wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3,
                                                                                          q3] * mat_fun[ie1, q1, ie2, q2, ie3, q3]
                                                    bi = bi1[ie1, il1, 0, q1] * bi2[ie2,
                                                                                    il2, 0, q2] * bi3[ie3, il3, 0, q3]
                                                    bj = bj1[ie1, jl1, 0, q1] * bj2[ie2,
                                                                                    jl2, 0, q2] * bj3[ie3, jl3, 0, q3]

                                                    value += wvol * bi * bj

                                        mat[ind_base1[ie1, il1], ind_base2[ie2, il2], ind_base3[ie3, il3],
                                            p[0] + jl1 - il1, p[1] + jl2 - il2, p[2] + jl3 - il3] += value
    #$ omp end parallel

    ierr = 0


@pure
def kernel_inner(nel: 'int[:]', p: 'int[:]', nq: 'int[:]', ni: 'int[:]', w1: 'float[:,:]', w2: 'float[:,:]', w3: 'float[:,:]', bi1: 'float[:,:,:,:]', bi2: 'float[:,:,:,:]', bi3: 'float[:,:,:,:]', ind_base1: 'int[:,:]', ind_base2: 'int[:,:]', ind_base3: 'int[:,:]', mat: 'float[:,:,:]', mat_fun: 'float[:,:,:,:,:,:]'):

    mat[:, :, :] = 0.

    #$ omp parallel private(ie1, ie2, ie3, il1, il2, il3, value, q1, q2, q3, wvol, bi)
    #$ omp for reduction ( + : mat)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):

                for il1 in range(p[0] + 1 - ni[0]):
                    for il2 in range(p[1] + 1 - ni[1]):
                        for il3 in range(p[2] + 1 - ni[2]):

                            value = 0.

                            for q1 in range(nq[0]):
                                for q2 in range(nq[1]):
                                    for q3 in range(nq[2]):

                                        wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3,
                                                                              q3] * mat_fun[ie1, q1, ie2, q2, ie3, q3]
                                        bi = bi1[ie1, il1, 0, q1] * bi2[ie2,
                                                                        il2, 0, q2] * bi3[ie3, il3, 0, q3]

                                        value += wvol * bi

                            mat[ind_base1[ie1, il1], ind_base2[ie2, il2],
                                ind_base3[ie3, il3]] += value
    #$ omp end parallel

    ierr = 0


@pure
def kernel_l2error(nel: 'int[:]', p: 'int[:]', nq: 'int[:]', w1: 'float[:,:]', w2: 'float[:,:]', w3: 'float[:,:]', ni: 'int[:]', nj: 'int[:]', bi1: 'float[:,:,:,:]', bi2: 'float[:,:,:,:]', bi3: 'float[:,:,:,:]', bj1: 'float[:,:,:,:]', bj2: 'float[:,:,:,:]', bj3: 'float[:,:,:,:]', ind_basei1: 'int[:,:]', ind_basei2: 'int[:,:]', ind_basei3: 'int[:,:]', ind_basej1: 'int[:,:]', ind_basej2: 'int[:,:]', ind_basej3: 'int[:,:]', error: 'float[:,:,:]', mat_f1: 'float[:,:,:,:,:,:]', mat_f2: 'float[:,:,:,:,:,:]', c1: 'float[:,:,:]', c2: 'float[:,:,:]', mat_map: 'float[:,:,:,:,:,:]'):

    #$ omp parallel private(ie1, ie2, ie3, q1, q2, q3, wvol, bi, bj, il1, il2, il3, jl1, jl2, jl3)
    #$ omp for

    # loop over all elements
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):

                # loop over quadrature points in element
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):

                            wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3,
                                                                  q3] * mat_map[ie1, q1, ie2, q2, ie3, q3]

                            # evaluate discrete fields at quadrature point
                            bi = 0.
                            bj = 0.

                            for il1 in range(p[0] + 1 - ni[0]):
                                for il2 in range(p[1] + 1 - ni[1]):
                                    for il3 in range(p[2] + 1 - ni[2]):

                                        bi += c1[ind_basei1[ie1, il1], ind_basei2[ie2, il2], ind_basei3[ie3, il3]
                                                 ] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]

                            for jl1 in range(p[0] + 1 - nj[0]):
                                for jl2 in range(p[1] + 1 - nj[1]):
                                    for jl3 in range(p[2] + 1 - nj[2]):

                                        bj += c2[ind_basej1[ie1, il1], ind_basej2[ie2, il2], ind_basej3[ie3, il3]
                                                 ] * bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2] * bj3[ie3, jl3, 0, q3]

                            # compare this value to exact one and add contribution to error in element
                            error[ie1, ie2, ie3] += wvol * (bi - mat_f1[ie1, q1, ie2, q2, ie3, q3]) * (
                                bj - mat_f2[ie1, q1, ie2, q2, ie3, q3])

    #$ omp end parallel

    ierr = 0


@pure
def kernel_evaluate_2form(nel: 'int[:]', p: 'int[:]', ns: 'int[:]', nq: 'int[:]', b_coeff: 'float[:,:,:]', ind_base1: 'int[:,:]', ind_base2: 'int[:,:]', ind_base3: 'int[:,:]', bi1: 'float[:,:,:,:]', bi2: 'float[:,:,:,:]', bi3: 'float[:,:,:,:]', b_eva: 'float[:,:,:,:,:,:]'):

    b_eva[:, :, :, :, :, :] = 0.

    #$ omp parallel private(ie1, ie2, ie3, q1, q2, q3, il1, il2, il3)
    #$ omp for
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):

                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):

                            for il1 in range(p[0] + 1 - ns[0]):
                                for il2 in range(p[1] + 1 - ns[1]):
                                    for il3 in range(p[2] + 1 - ns[2]):

                                        b_eva[ie1, q1, ie2, q2, ie3, q3] += b_coeff[ind_base1[ie1, il1], ind_base2[ie2, il2],
                                                                                    ind_base3[ie3, il3]] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]
    #$ omp end parallel

    ierr = 0
