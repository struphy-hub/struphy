import hylife.geometry.mappings_3d as map3d
import input_run.equilibrium_PIC as equ_PIC


# =============== xvn substep ============================
def quadrature_density(
    Nel: "int[:]",
    pts1: "float[:,:]",
    pts2: "float[:,:]",
    pts3: "float[:,:]",
    n_quad: "int[:]",
    gather: "float[:,:,:,:,:,:]",
    kind_map: "int",
    params_map: "float[:]",
    tf1: "float[:]",
    tf2: "float[:]",
    tf3: "float[:]",
    pf: "int[:]",
    nbasef: "int[:]",
    cx: "float[:,:,:]",
    cy: "float[:,:,:]",
    cz: "float[:,:,:]",
):
    # =======================================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, il1, il2, il3, x1, x2, x3)
    for ie1 in range(Nel[0]):
        for ie2 in range(Nel[1]):
            for ie3 in range(Nel[2]):
                for il1 in range(n_quad[0]):
                    for il2 in range(n_quad[1]):
                        for il3 in range(n_quad[2]):
                            # ========= physical domain =============
                            x1 = map3d.f(
                                pts1[ie1, il1],
                                pts2[ie2, il2],
                                pts3[ie3, il3],
                                1,
                                kind_map,
                                params_map,
                                tf1,
                                tf2,
                                tf3,
                                pf,
                                nbasef,
                                cx,
                                cy,
                                cz,
                            )
                            x2 = map3d.f(
                                pts1[ie1, il1],
                                pts2[ie2, il2],
                                pts3[ie3, il3],
                                2,
                                kind_map,
                                params_map,
                                tf1,
                                tf2,
                                tf3,
                                pf,
                                nbasef,
                                cx,
                                cy,
                                cz,
                            )
                            x3 = map3d.f(
                                pts1[ie1, il1],
                                pts2[ie2, il2],
                                pts3[ie3, il3],
                                3,
                                kind_map,
                                params_map,
                                tf1,
                                tf2,
                                tf3,
                                pf,
                                nbasef,
                                cx,
                                cy,
                                cz,
                            )

                            gather[ie1, ie2, ie3, il1, il2, il3] += equ_PIC.nh_eq_phys(x1, x2, x3)
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0
