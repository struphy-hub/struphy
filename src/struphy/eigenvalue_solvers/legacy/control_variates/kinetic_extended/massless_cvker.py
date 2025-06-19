import hylife.geometry.mappings_3d as map3d
import hylife.geometry.mappings_3d_fast as mapping_fast
import hylife.linear_algebra.core as linalg
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva
import hylife.utilitis_FEEC.bsplines_kernels as bsp
import input_run.equilibrium_PIC as equ_PIC
from pyccel.decorators import types


# ==========================================================================================
def bvright1(
    G_inv_11: "float[:,:,:,:,:,:]",
    G_inv_12: "float[:,:,:,:,:,:]",
    G_inv_13: "float[:,:,:,:,:,:]",
    G_inv_22: "float[:,:,:,:,:,:]",
    G_inv_23: "float[:,:,:,:,:,:]",
    G_inv_33: "float[:,:,:,:,:,:]",
    idnx: "int[:,:]",
    idny: "int[:,:]",
    idnz: "int[:,:]",
    iddx: "int[:,:]",
    iddy: "int[:,:]",
    iddz: "int[:,:]",
    nel1: "int",
    nel2: "int",
    nel3: "int",
    nq1: "int",
    nq2: "int",
    nq3: "int",
    p1: "int",
    p2: "int",
    p3: "int",
    d1: "int",
    d2: "int",
    d3: "int",
    b1value: "float[:,:,:,:,:,:]",
    b2value: "float[:,:,:,:,:,:]",
    b3value: "float[:,:,:,:,:,:]",
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    dft: "float[:,:]",
    generate_weight1: "float[:]",
    generate_weight3: "float[:]",
    Jeq: "float[:]",
    bn1: "float[:,:,:,:]",
    bn2: "float[:,:,:,:]",
    bn3: "float[:,:,:,:]",
    bd1: "float[:,:,:,:]",
    bd2: "float[:,:,:,:]",
    bd3: "float[:,:,:,:]",
):
    # ======================================================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)

    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value = 0.0
                            for il1 in range(d1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(p3 + 1):
                                        value += (
                                            bd1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                            * b1[iddx[ie1, il1], idny[ie2, il2], idnz[ie3, il3]]
                                        )

                            b1value[ie1, ie2, ie3, q1, q2, q3] = value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)

    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value = 0.0
                            for il1 in range(p1 + 1):
                                for il2 in range(d2 + 1):
                                    for il3 in range(p3 + 1):
                                        value += (
                                            bn1[ie1, il1, 0, q1]
                                            * bd2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                            * b2[idnx[ie1, il1], iddy[ie2, il2], idnz[ie3, il3]]
                                        )

                            b2value[ie1, ie2, ie3, q1, q2, q3] = value
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)

    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value = 0.0
                            for il1 in range(p1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(d3 + 1):
                                        value += (
                                            bn1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bd3[ie3, il3, 0, q3]
                                            * b3[idnx[ie1, il1], idny[ie2, il2], iddz[ie3, il3]]
                                        )

                            b3value[ie1, ie2, ie3, q1, q2, q3] = value
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, dft, generate_weight1, generate_weight3)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            dft[0, 0] = G_inv_11[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 1] = G_inv_12[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 2] = G_inv_13[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 0] = G_inv_12[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 1] = G_inv_22[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 2] = G_inv_23[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 0] = G_inv_13[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 1] = G_inv_23[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 2] = G_inv_33[ie1, ie2, ie3, q1, q2, q3]
                            generate_weight3[0] = b1value[ie1, ie2, ie3, q1, q2, q3]
                            generate_weight3[1] = b2value[ie1, ie2, ie3, q1, q2, q3]
                            generate_weight3[2] = b3value[ie1, ie2, ie3, q1, q2, q3]
                            linalg.matrix_vector(dft, generate_weight3, generate_weight1)
                            b1value[ie1, ie2, ie3, q1, q2, q3] = generate_weight1[0]
                            b2value[ie1, ie2, ie3, q1, q2, q3] = generate_weight1[1]
                            b3value[ie1, ie2, ie3, q1, q2, q3] = generate_weight1[2]

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================
def bvright2(
    DFI_11: "float[:,:,:,:,:,:]",
    DFI_12: "float[:,:,:,:,:,:]",
    DFI_13: "float[:,:,:,:,:,:]",
    DFI_21: "float[:,:,:,:,:,:]",
    DFI_22: "float[:,:,:,:,:,:]",
    DFI_23: "float[:,:,:,:,:,:]",
    DFI_31: "float[:,:,:,:,:,:]",
    DFI_32: "float[:,:,:,:,:,:]",
    DFI_33: "float[:,:,:,:,:,:]",
    df_det: "float[:,:,:,:,:,:]",
    Jeqx: "float[:,:,:,:,:,:]",
    Jeqy: "float[:,:,:,:,:,:]",
    Jeqz: "float[:,:,:,:,:,:]",
    nel1: "int",
    nel2: "int",
    nel3: "int",
    nq1: "int",
    nq2: "int",
    nq3: "int",
    p1: "int",
    p2: "int",
    p3: "int",
    d1: "int",
    d2: "int",
    d3: "int",
    b1value: "float[:,:,:,:,:,:]",
    b2value: "float[:,:,:,:,:,:]",
    b3value: "float[:,:,:,:,:,:]",
    uvalue: "float[:,:,:,:,:,:]",
    dft: "float[:,:]",
    generate_weight1: "float[:]",
    generate_weight3: "float[:]",
    Jeq: "float[:]",
    pts1: "float[:,:]",
    pts2: "float[:,:]",
    pts3: "float[:,:]",
    wts1: "float[:,:]",
    wts2: "float[:,:]",
    wts3: "float[:,:]",
):
    # ======================================================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, dft, detdet, generate_weight1, generate_weight3, Jeq)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            generate_weight1[0] = b1value[ie1, ie2, ie3, q1, q2, q3]
                            generate_weight1[1] = b2value[ie1, ie2, ie3, q1, q2, q3]
                            generate_weight1[2] = b3value[ie1, ie2, ie3, q1, q2, q3]

                            dft[0, 0] = DFI_11[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 1] = DFI_12[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 2] = DFI_13[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 0] = DFI_21[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 1] = DFI_22[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 2] = DFI_23[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 0] = DFI_31[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 1] = DFI_32[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 2] = DFI_33[ie1, ie2, ie3, q1, q2, q3]
                            detdet = df_det[ie1, ie2, ie3, q1, q2, q3] * wts1[ie1, q1] * wts2[ie2, q2] * wts3[ie3, q3]
                            Jeq[0] = Jeqx[ie1, ie2, ie3, q1, q2, q3]
                            Jeq[1] = Jeqy[ie1, ie2, ie3, q1, q2, q3]
                            Jeq[2] = Jeqz[ie1, ie2, ie3, q1, q2, q3]
                            linalg.matrix_vector(dft, Jeq, generate_weight3)

                            b1value[ie1, ie2, ie3, q1, q2, q3] = (
                                detdet
                                * uvalue[ie1, ie2, ie3, q1, q2, q3]
                                * (
                                    generate_weight3[1] * generate_weight1[2]
                                    - generate_weight3[2] * generate_weight1[1]
                                )
                            )
                            b2value[ie1, ie2, ie3, q1, q2, q3] = (
                                detdet
                                * uvalue[ie1, ie2, ie3, q1, q2, q3]
                                * (
                                    generate_weight3[2] * generate_weight1[0]
                                    - generate_weight3[0] * generate_weight1[2]
                                )
                            )
                            b3value[ie1, ie2, ie3, q1, q2, q3] = (
                                detdet
                                * uvalue[ie1, ie2, ie3, q1, q2, q3]
                                * (
                                    generate_weight3[0] * generate_weight1[1]
                                    - generate_weight3[1] * generate_weight1[0]
                                )
                            )

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0
