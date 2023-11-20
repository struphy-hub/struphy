# coding: utf-8

from pyccel.decorators import pure


@pure
def kernel_mass(nel: 'int[:]', p: 'int[:]', nq: 'int[:]', ni: 'int[:]', nj: 'int[:]', w1: 'float[:,:]', w2: 'float[:,:]', bi1: 'float[:,:,:,:]', bi2: 'float[:,:,:,:]', bj1: 'float[:,:,:,:]', bj2: 'float[:,:,:,:]', ind_base1: 'int[:,:]', ind_base2: 'int[:,:]', mat: 'float[:,:,:,:]', mat_fun: 'float[:,:,:,:]'):

    mat[:, :, :, :] = 0.

    #$ omp parallel private(ie1, ie2, il1, il2, jl1, jl2, value, q1, q2, wvol, bi, bj) shared(mat)
    #$ omp for reduction ( + : mat)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):

            for il1 in range(p[0] + 1 - ni[0]):
                for il2 in range(p[1] + 1 - ni[1]):
                    for jl1 in range(p[0] + 1 - nj[0]):
                        for jl2 in range(p[1] + 1 - nj[1]):

                            value = 0.

                            for q1 in range(nq[0]):
                                for q2 in range(nq[1]):

                                    wvol = w1[ie1, q1] * w2[ie2, q2] * mat_fun[ie1, q1, ie2, q2]
                                    bi = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]
                                    bj = bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2]

                                    value += wvol * bi * bj

                            mat[ind_base1[ie1, il1], ind_base2[ie2, il2], p[0] + jl1 - il1, p[1] + jl2 - il2] += value
    #$ omp end parallel

    ierr = 0


@pure
def kernel_inner(nel: 'int[:]', n3: 'int', p: 'int[:]', nq: 'int[:]', ni: 'int[:]', w1: 'float[:,:]', w2: 'float[:,:]', bi1: 'float[:,:,:,:]', bi2: 'float[:,:,:,:]', ind_base1: 'int[:,:]', ind_base2: 'int[:,:]', mat: 'float[:,:,:]', mat_fun: 'float[:,:,:,:,:]'):

    mat[:, :, :] = 0.

    #$ omp parallel private(ie1, ie2, ie3, il1, il2, value, q1, q2, wvol, bi) shared(mat)
    #$ omp for reduction ( + : mat)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(n3):

                for il1 in range(p[0] + 1 - ni[0]):
                    for il2 in range(p[1] + 1 - ni[1]):

                        value = 0.

                        for q1 in range(nq[0]):
                            for q2 in range(nq[1]):

                                wvol = w1[ie1, q1] * w2[ie2, q2] * mat_fun[ie1, q1, ie2, q2, ie3]
                                bi = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                                value += wvol * bi

                        mat[ind_base1[ie1, il1], ind_base2[ie2, il2], ie3] += value
    #$ omp end parallel

    ierr = 0


@pure
def kernel_l2error(nel: 'int[:]', p: 'int[:]', nq: 'int[:]', w1: 'float[:,:]', w2: 'float[:,:]', ni: 'int[:]', nj: 'int[:]', bi1: 'float[:,:,:,:]', bi2: 'float[:,:,:,:]', bj1: 'float[:,:,:,:]', bj2: 'float[:,:,:,:]', ind_basei1: 'int[:,:]', ind_basei2: 'int[:,:]', ind_basej1: 'int[:,:]', ind_basej2: 'int[:,:]', error: 'float[:,:]', mat_f1: 'float[:,:,:,:]', mat_f2: 'float[:,:,:,:]', c1: 'float[:,:,:]', c2: 'float[:,:,:]', mat_map: 'float[:,:,:,:]'):

    #$ omp parallel private(ie1, ie2, q1, q2, wvol, bi, bj, il1, il2, jl1, jl2)
    #$ omp for
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):

            # loop over quadrature points in element
            for q1 in range(nq[0]):
                for q2 in range(nq[1]):

                    wvol = w1[ie1, q1] * w2[ie2, q2] * mat_map[ie1, q1, ie2, q2]

                    # evaluate discrete fields at quadrature point
                    bi = 0.
                    bj = 0.

                    for il1 in range(p[0] + 1 - ni[0]):
                        for il2 in range(p[1] + 1 - ni[1]):

                            bi += c1[ind_basei1[ie1, il1], ind_basei2[ie2, il2], 0] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                    for jl1 in range(p[0] + 1 - nj[0]):
                        for jl2 in range(p[1] + 1 - nj[1]):

                            bj += c2[ind_basej1[ie1, jl1], ind_basej2[ie2, jl2], 0] * bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2]

                    # compare this value to exact one and add contribution to error in element
                    error[ie1, ie2] += wvol * (bi - mat_f1[ie1, q1, ie2, q2]) * (bj - mat_f2[ie1, q1, ie2, q2])
    #$ omp end parallel

    ierr = 0


@pure
def kernel_evaluate_2form(nel: 'int[:]', n3: 'int', p: 'int[:]', ns: 'int[:]', nq: 'int[:]', b_coeff: 'float[:,:,:]', ind_base1: 'int[:,:]', ind_base2: 'int[:,:]', bi1: 'float[:,:,:,:]', bi2: 'float[:,:,:,:]', b_eva: 'float[:,:,:,:,:]'):

    b_eva[:, :, :, :, :] = 0.

    #$ omp parallel private(ie1, ie2, ie3, q1, q2, il1, il2)
    #$ omp for
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(n3):

                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):

                        for il1 in range(p[0] + 1 - ns[0]):
                            for il2 in range(p[1] + 1 - ns[1]):

                                b_eva[ie1, q1, ie2, q2, ie3] += b_coeff[ind_base1[ie1, il1], ind_base2[ie2, il2], ie3] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]
    #$ omp end parallel

    ierr = 0
