from pyccel.decorators import types

import hylife.geometry.mappings_3d as map3d

import input_run.equilibrium_PIC as equ_PIC


# =============== xvn substep ============================
@types('int[:]','double[:,:]','double[:,:]','double[:,:]','int[:]','double[:,:,:,:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def quadrature_density(Nel, pts1, pts2, pts3, n_quad, gather, kind_map, params_map, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):

    # =======================================================================

    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, il1, il2, il3, x1, x2, x3)
    for ie1 in range(Nel[0]):
        for ie2 in range(Nel[1]):
            for ie3 in range(Nel[2]):
                for il1 in range(n_quad[0]):
                    for il2 in range(n_quad[1]):
                        for il3 in range(n_quad[2]):
                            # ========= physical domain =============
                            x1 = map3d.f(pts1[ie1, il1], pts2[ie2, il2], pts3[ie3, il3], 1, kind_map, params_map, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                            x2 = map3d.f(pts1[ie1, il1], pts2[ie2, il2], pts3[ie3, il3], 2, kind_map, params_map, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                            x3 = map3d.f(pts1[ie1, il1], pts2[ie2, il2], pts3[ie3, il3], 3, kind_map, params_map, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

                            gather[ie1, ie2, ie3, il1, il2, il3] += equ_PIC.nh_eq_phys(x1, x2, x3)
    #$ omp end do
    #$ omp end parallel
    ierr = 0