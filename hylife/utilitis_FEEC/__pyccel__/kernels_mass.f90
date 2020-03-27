module kernels_mass

use mappings_analytical, only: mapping_fn2dfu_det_df => det_df
use mappings_analytical, only: mapping_fn2dfu_g_inv => g_inv
use mappings_analytical, only: mapping_fn2dfu_g => g
implicit none




contains

!........................................
function fun_3d(xi1, xi2, xi3, kind_fun, kind_map, params) result(value)

  implicit none
  real(kind=8)  :: value  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind_fun
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params (0:)



  value = 0.0d0


  !quantities for mass matrix V0
  if (kind_fun == 1_8 ) then
    value = mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 11_8 ) then
    value = mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_fn2dfu_g_inv(xi1, xi2, xi3, kind_map, params, 11_8)
  else if (kind_fun == 12_8 ) then
    value = mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_fn2dfu_g_inv(xi1, xi2, xi3, kind_map, params, 21_8)
  else if (kind_fun == 13_8 ) then
    value = mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_fn2dfu_g_inv(xi1, xi2, xi3, kind_map, params, 22_8)
  else if (kind_fun == 14_8 ) then
    value = mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_fn2dfu_g_inv(xi1, xi2, xi3, kind_map, params, 31_8)
  else if (kind_fun == 15_8 ) then
    value = mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_fn2dfu_g_inv(xi1, xi2, xi3, kind_map, params, 32_8)
  else if (kind_fun == 16_8 ) then
    value = mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_fn2dfu_g_inv(xi1, xi2, xi3, kind_map, params, 33_8)
  else if (kind_fun == 21_8 ) then
    value = mapping_fn2dfu_g(xi1, xi2, xi3, kind_map, params, 11_8)/ &
      mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 22_8 ) then
    value = mapping_fn2dfu_g(xi1, xi2, xi3, kind_map, params, 21_8)/ &
      mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 23_8 ) then
    value = mapping_fn2dfu_g(xi1, xi2, xi3, kind_map, params, 22_8)/ &
      mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 24_8 ) then
    value = mapping_fn2dfu_g(xi1, xi2, xi3, kind_map, params, 31_8)/ &
      mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 25_8 ) then
    value = mapping_fn2dfu_g(xi1, xi2, xi3, kind_map, params, 32_8)/ &
      mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 26_8 ) then
    value = mapping_fn2dfu_g(xi1, xi2, xi3, kind_map, params, 33_8)/ &
      mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 2_8 ) then
    value = 1.0d0/mapping_fn2dfu_det_df(xi1, xi2, xi3, kind_map, params)
  end if




  !quantities for mass matrix V3


  !quantities for mass matrix V2
  !quantities for mass matrix V1
  return
end function
!........................................

!........................................
subroutine kernel_eva_3d(n, xi1, xi2, xi3, mat_f, kind_fun, kind_map, &
      params)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  real(kind=8), intent(in)  :: xi1 (0:)
  real(kind=8), intent(in)  :: xi2 (0:)
  real(kind=8), intent(in)  :: xi3 (0:)
  real(kind=8), intent(inout)  :: mat_f (0:,0:,0:)
  integer(kind=8), value  :: kind_fun
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params (0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  



  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        mat_f(i1, i2, i3) = fun_3d(xi1(i1), xi2(i2), xi3(i3), kind_fun, &
      kind_map, params)










      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_mass_2d(Nel1, Nel2, p1, p2, nq1, nq2, ni1, ni2, nj1, &
      nj2, w1, w2, bi1, bi2, bj1, bj2, Nbase1, Nbase2, M, mat_map)

  implicit none
  integer(kind=8), value  :: Nel1
  integer(kind=8), value  :: Nel2
  integer(kind=8), value  :: p1
  integer(kind=8), value  :: p2
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq2
  integer(kind=8), value  :: ni1
  integer(kind=8), value  :: ni2
  integer(kind=8), value  :: nj1
  integer(kind=8), value  :: nj2
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: bi1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bj1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bj2 (0:,0:,0:,0:)
  integer(kind=8), value  :: Nbase1
  integer(kind=8), value  :: Nbase2
  real(kind=8), intent(inout)  :: M (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_map (0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: jl1  
  integer(kind=8)  :: jl2  
  real(kind=8)  :: value  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  real(kind=8)  :: wvol  
  real(kind=8)  :: bi  
  real(kind=8)  :: bj  



  do ie1 = 0, Nel1 - 1_8, 1
    do ie2 = 0, Nel2 - 1_8, 1


      do il1 = 0, -ni1 + p1, 1
        do il2 = 0, -ni2 + p2, 1
          do jl1 = 0, -nj1 + p1, 1
            do jl2 = 0, -nj2 + p2, 1


              value = 0.0d0


              do q1 = 0, nq1 - 1_8, 1
                do q2 = 0, nq2 - 1_8, 1


                  wvol = (mat_map(ie1*nq1 + q1, ie2*nq2 + q2)*w2(ie2, q2 &
      ))*w1(ie1, q1)
                  bi = bi1(ie1, il1, 0_8, q1)*bi2(ie2, il2, 0_8, q2)
                  bj = bj1(ie1, jl1, 0_8, q1)*bj2(ie2, jl2, 0_8, q2)


                  value = value + wvol*(bi*bj)


                end do

              end do

              M(modulo(ie1 + il1,Nbase1), modulo(ie2 + il2,Nbase2), p1 - &
      il1 + jl1, p2 - il2 + jl2) = value + M(modulo(ie1 + il1,Nbase1), &
      modulo(ie2 + il2,Nbase2), p1 - il1 + jl1, p2 - il2 + jl2)










            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_mass_3d(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, &
      ni1, ni2, ni3, nj1, nj2, nj3, w1, w2, w3, bi1, bi2, bi3, bj1, bj2 &
      , bj3, Nbase1, Nbase2, Nbase3, M, mat_map)

  implicit none
  integer(kind=8), value  :: Nel1
  integer(kind=8), value  :: Nel2
  integer(kind=8), value  :: Nel3
  integer(kind=8), value  :: p1
  integer(kind=8), value  :: p2
  integer(kind=8), value  :: p3
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq2
  integer(kind=8), value  :: nq3
  integer(kind=8), value  :: ni1
  integer(kind=8), value  :: ni2
  integer(kind=8), value  :: ni3
  integer(kind=8), value  :: nj1
  integer(kind=8), value  :: nj2
  integer(kind=8), value  :: nj3
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: w3 (0:,0:)
  real(kind=8), intent(in)  :: bi1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi3 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bj1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bj2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bj3 (0:,0:,0:,0:)
  integer(kind=8), value  :: Nbase1
  integer(kind=8), value  :: Nbase2
  integer(kind=8), value  :: Nbase3
  real(kind=8), intent(inout)  :: M (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_map (0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: jl1  
  integer(kind=8)  :: jl2  
  integer(kind=8)  :: jl3  
  real(kind=8)  :: value  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  
  real(kind=8)  :: bi  
  real(kind=8)  :: bj  



  do ie1 = 0, Nel1 - 1_8, 1
    do ie2 = 0, Nel2 - 1_8, 1
      do ie3 = 0, Nel3 - 1_8, 1


        do il1 = 0, -ni1 + p1, 1
          do il2 = 0, -ni2 + p2, 1
            do il3 = 0, -ni3 + p3, 1
              do jl1 = 0, -nj1 + p1, 1
                do jl2 = 0, -nj2 + p2, 1
                  do jl3 = 0, -nj3 + p3, 1


                    value = 0.0d0


                    do q1 = 0, nq1 - 1_8, 1
                      do q2 = 0, nq2 - 1_8, 1
                        do q3 = 0, nq3 - 1_8, 1


                          wvol = ((mat_map(ie1*nq1 + q1, ie2*nq2 + q2, &
      ie3*nq3 + q3)*w3(ie3, q3))*w2(ie2, q2))*w1(ie1, q1)
                          bi = (bi2(ie2, il2, 0_8, q2)*bi3(ie3, il3, 0_8 &
      , q3))*bi1(ie1, il1, 0_8, q1)
                          bj = (bj2(ie2, jl2, 0_8, q2)*bj3(ie3, jl3, 0_8 &
      , q3))*bj1(ie1, jl1, 0_8, q1)


                          value = value + wvol*(bi*bj)


                        end do

                      end do

                    end do

                    M(modulo(ie1 + il1,Nbase1), modulo(ie2 + il2,Nbase2) &
      , modulo(ie3 + il3,Nbase3), p1 - il1 + jl1, p2 - il2 + jl2, p3 - &
      il3 + jl3) = value + M(modulo(ie1 + il1,Nbase1), modulo(ie2 + il2 &
      ,Nbase2), modulo(ie3 + il3,Nbase3), p1 - il1 + jl1, p2 - il2 + &
      jl2, p3 - il3 + jl3)








                  end do

                end do

              end do

            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_inner_2d(Nel1, Nel2, p1, p2, nq1, nq2, ni1, ni2, w1, &
      w2, bi1, bi2, Nbase1, Nbase2, L, mat_f, mat_map)

  implicit none
  integer(kind=8), value  :: Nel1
  integer(kind=8), value  :: Nel2
  integer(kind=8), value  :: p1
  integer(kind=8), value  :: p2
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq2
  integer(kind=8), value  :: ni1
  integer(kind=8), value  :: ni2
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: bi1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi2 (0:,0:,0:,0:)
  integer(kind=8), value  :: Nbase1
  integer(kind=8), value  :: Nbase2
  real(kind=8), intent(inout)  :: L (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:)
  real(kind=8), intent(in)  :: mat_map (0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  real(kind=8)  :: value  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  real(kind=8)  :: wvol  
  real(kind=8)  :: bi  



  do ie1 = 0, Nel1 - 1_8, 1
    do ie2 = 0, Nel2 - 1_8, 1


      do il1 = 0, -ni1 + p1, 1
        do il2 = 0, -ni2 + p2, 1


          value = 0.0d0


          do q1 = 0, nq1 - 1_8, 1
            do q2 = 0, nq2 - 1_8, 1


              wvol = (mat_map(ie1*nq1 + q1, ie2*nq2 + q2)*w2(ie2, q2))* &
      w1(ie1, q1)
              bi = bi1(ie1, il1, 0_8, q1)*bi2(ie2, il2, 0_8, q2)


              value = value + wvol*(bi*mat_f(ie1*nq1 + q1, ie2*nq2 + q2 &
      ))


            end do

          end do

          L(modulo(ie1 + il1,Nbase1), modulo(ie2 + il2,Nbase2)) = value &
      + L(modulo(ie1 + il1,Nbase1), modulo(ie2 + il2,Nbase2))








        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_inner_3d(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, &
      ni1, ni2, ni3, w1, w2, w3, bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, &
      L, mat_f, mat_map)

  implicit none
  integer(kind=8), value  :: Nel1
  integer(kind=8), value  :: Nel2
  integer(kind=8), value  :: Nel3
  integer(kind=8), value  :: p1
  integer(kind=8), value  :: p2
  integer(kind=8), value  :: p3
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq2
  integer(kind=8), value  :: nq3
  integer(kind=8), value  :: ni1
  integer(kind=8), value  :: ni2
  integer(kind=8), value  :: ni3
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: w3 (0:,0:)
  real(kind=8), intent(in)  :: bi1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi3 (0:,0:,0:,0:)
  integer(kind=8), value  :: Nbase1
  integer(kind=8), value  :: Nbase2
  integer(kind=8), value  :: Nbase3
  real(kind=8), intent(inout)  :: L (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_map (0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  real(kind=8)  :: value  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  
  real(kind=8)  :: bi  



  do ie1 = 0, Nel1 - 1_8, 1
    do ie2 = 0, Nel2 - 1_8, 1
      do ie3 = 0, Nel3 - 1_8, 1


        do il1 = 0, -ni1 + p1, 1
          do il2 = 0, -ni2 + p2, 1
            do il3 = 0, -ni3 + p3, 1


              value = 0.0d0


              do q1 = 0, nq1 - 1_8, 1
                do q2 = 0, nq2 - 1_8, 1
                  do q3 = 0, nq3 - 1_8, 1


                    wvol = ((mat_map(ie1*nq1 + q1, ie2*nq2 + q2, ie3*nq3 &
      + q3)*w3(ie3, q3))*w2(ie2, q2))*w1(ie1, q1)
                    bi = (bi2(ie2, il2, 0_8, q2)*bi3(ie3, il3, 0_8, q3)) &
      *bi1(ie1, il1, 0_8, q1)


                    value = value + wvol*(bi*mat_f(ie1*nq1 + q1, ie2*nq2 &
      + q2, ie3*nq3 + q3))


                  end do

                end do

              end do

              L(modulo(ie1 + il1,Nbase1), modulo(ie2 + il2,Nbase2), &
      modulo(ie3 + il3,Nbase3)) = value + L(modulo(ie1 + il1,Nbase1), &
      modulo(ie2 + il2,Nbase2), modulo(ie3 + il3,Nbase3))








            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_l2error_v0_2d(Nel1, Nel2, p1, p2, nq1, nq2, w1, w2, &
      bi1, bi2, Nbase1, Nbase2, error, mat_f, mat_c, mat_g)

  implicit none
  integer(kind=8), value  :: Nel1
  integer(kind=8), value  :: Nel2
  integer(kind=8), value  :: p1
  integer(kind=8), value  :: p2
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq2
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: bi1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi2 (0:,0:,0:,0:)
  integer(kind=8), value  :: Nbase1
  integer(kind=8), value  :: Nbase2
  real(kind=8), intent(inout)  :: error (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:)
  real(kind=8), intent(in)  :: mat_c (0:,0:)
  real(kind=8), intent(in)  :: mat_g (0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  real(kind=8)  :: wvol  
  real(kind=8)  :: bi  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  



  do ie1 = 0, Nel1 - 1_8, 1
    do ie2 = 0, Nel2 - 1_8, 1


      do q1 = 0, nq1 - 1_8, 1
        do q2 = 0, nq2 - 1_8, 1


          wvol = (mat_g(ie1*nq1 + q1, ie2*nq2 + q2)*w2(ie2, q2))*w1(ie1, &
      q1)


          bi = 0.0d0


          do il1 = 0, p1, 1
            do il2 = 0, p2, 1


              bi = bi + (bi1(ie1, il1, 0_8, q1)*bi2(ie2, il2, 0_8, q2))* &
      mat_c(modulo(ie1 + il1,Nbase1), modulo(ie2 + il2,Nbase2))


            end do

          end do

          error(ie1, ie2) = wvol*(bi - mat_f(ie1*nq1 + q1, ie2*nq2 + q2 &
      ))**2_8 + error(ie1, ie2)








        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_l2error_v0_3d(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, &
      nq3, w1, w2, w3, bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, error, &
      mat_f, mat_c, mat_g)

  implicit none
  integer(kind=8), value  :: Nel1
  integer(kind=8), value  :: Nel2
  integer(kind=8), value  :: Nel3
  integer(kind=8), value  :: p1
  integer(kind=8), value  :: p2
  integer(kind=8), value  :: p3
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq2
  integer(kind=8), value  :: nq3
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: w3 (0:,0:)
  real(kind=8), intent(in)  :: bi1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi3 (0:,0:,0:,0:)
  integer(kind=8), value  :: Nbase1
  integer(kind=8), value  :: Nbase2
  integer(kind=8), value  :: Nbase3
  real(kind=8), intent(inout)  :: error (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_c (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_g (0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  
  real(kind=8)  :: bi  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  



  do ie1 = 0, Nel1 - 1_8, 1
    do ie2 = 0, Nel2 - 1_8, 1
      do ie3 = 0, Nel3 - 1_8, 1


        do q1 = 0, nq1 - 1_8, 1
          do q2 = 0, nq2 - 1_8, 1
            do q3 = 0, nq3 - 1_8, 1


              wvol = ((mat_g(ie1*nq1 + q1, ie2*nq2 + q2, ie3*nq3 + q3)* &
      w3(ie3, q3))*w2(ie2, q2))*w1(ie1, q1)


              bi = 0.0d0


              do il1 = 0, p1, 1
                do il2 = 0, p2, 1
                  do il3 = 0, p3, 1


                    bi = bi + ((bi2(ie2, il2, 0_8, q2)*bi3(ie3, il3, 0_8 &
      , q3))*bi1(ie1, il1, 0_8, q1))*mat_c(modulo(ie1 + il1,Nbase1), &
      modulo(ie2 + il2,Nbase2), modulo(ie3 + il3,Nbase3))


                  end do

                end do

              end do

              error(ie1, ie2, ie3) = wvol*(bi - mat_f(ie1*nq1 + q1, ie2* &
      nq2 + q2, ie3*nq3 + q3))**2_8 + error(ie1, ie2, ie3)
            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

end module