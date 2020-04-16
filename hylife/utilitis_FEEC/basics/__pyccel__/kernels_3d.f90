module kernels_3d

use mappings_analytical, only: mapping_0psq4d_g_inv => g_inv
use mappings_analytical, only: mapping_0psq4d_g => g
use mappings_analytical, only: mapping_0psq4d_det_df => det_df
implicit none




contains

!........................................
function fun(xi1, xi2, xi3, kind_fun, kind_map, params) result(value)

  implicit none
  real(kind=8)  :: value  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind_fun
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params (0:)



  value = 0.0d0


  !quantities for 0-form mass matrix (H1)
  if (kind_fun == 1_8 ) then
    value = mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 11_8 ) then
    value = mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_0psq4d_g_inv(xi1, xi2, xi3, kind_map, params, 11_8)
  else if (kind_fun == 12_8 ) then
    value = mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_0psq4d_g_inv(xi1, xi2, xi3, kind_map, params, 21_8)
  else if (kind_fun == 13_8 ) then
    value = mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_0psq4d_g_inv(xi1, xi2, xi3, kind_map, params, 22_8)
  else if (kind_fun == 14_8 ) then
    value = mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_0psq4d_g_inv(xi1, xi2, xi3, kind_map, params, 31_8)
  else if (kind_fun == 15_8 ) then
    value = mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_0psq4d_g_inv(xi1, xi2, xi3, kind_map, params, 32_8)
  else if (kind_fun == 16_8 ) then
    value = mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)* &
      mapping_0psq4d_g_inv(xi1, xi2, xi3, kind_map, params, 33_8)
  else if (kind_fun == 21_8 ) then
    value = mapping_0psq4d_g(xi1, xi2, xi3, kind_map, params, 11_8)/ &
      mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 22_8 ) then
    value = mapping_0psq4d_g(xi1, xi2, xi3, kind_map, params, 21_8)/ &
      mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 23_8 ) then
    value = mapping_0psq4d_g(xi1, xi2, xi3, kind_map, params, 22_8)/ &
      mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 24_8 ) then
    value = mapping_0psq4d_g(xi1, xi2, xi3, kind_map, params, 31_8)/ &
      mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 25_8 ) then
    value = mapping_0psq4d_g(xi1, xi2, xi3, kind_map, params, 32_8)/ &
      mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 26_8 ) then
    value = mapping_0psq4d_g(xi1, xi2, xi3, kind_map, params, 33_8)/ &
      mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 2_8 ) then
    value = 1.0d0/mapping_0psq4d_det_df(xi1, xi2, xi3, kind_map, params)
  end if




  !quantities for 3-form mass matrix (L2)


  !quantities for 2-form mass matrix (H div)
  !quantities for 1-form mass matrix (H curl)
  return
end function
!........................................

!........................................
subroutine kernel_evaluation(nel, nq, xi1, xi2, xi3, mat_f, kind_fun, &
      kind_map, params)

  implicit none
  integer(kind=8), intent(in)  :: nel (0:)
  integer(kind=8), intent(in)  :: nq (0:)
  real(kind=8), intent(in)  :: xi1 (0:,0:)
  real(kind=8), intent(in)  :: xi2 (0:,0:)
  real(kind=8), intent(in)  :: xi3 (0:,0:)
  real(kind=8), intent(inout)  :: mat_f (0:,0:,0:,0:,0:,0:)
  integer(kind=8), value  :: kind_fun
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params (0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  



  do ie1 = 0, nel(0_8) - 1_8, 1
    do ie2 = 0, nel(1_8) - 1_8, 1
      do ie3 = 0, nel(2_8) - 1_8, 1


        do q1 = 0, nq(0_8) - 1_8, 1
          do q2 = 0, nq(1_8) - 1_8, 1
            do q3 = 0, nq(2_8) - 1_8, 1
              mat_f(ie1, ie2, ie3, q1, q2, q3) = fun(xi1(ie1, q1), xi2( &
      ie2, q2), xi3(ie3, q3), kind_fun, kind_map, params)




            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_mass(nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, ni1, &
      ni2, ni3, nj1, nj2, nj3, w1, w2, w3, bi1, bi2, bi3, bj1, bj2, bj3 &
      , nbase1, nbase2, nbase3, M, mat_map)

  implicit none
  integer(kind=8), value  :: nel1
  integer(kind=8), value  :: nel2
  integer(kind=8), value  :: nel3
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
  integer(kind=8), value  :: nbase1
  integer(kind=8), value  :: nbase2
  integer(kind=8), value  :: nbase3
  real(kind=8), intent(inout)  :: M (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_map (0:,0:,0:,0:,0:,0:)
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



  do ie1 = 0, nel1 - 1_8, 1
    do ie2 = 0, nel2 - 1_8, 1
      do ie3 = 0, nel3 - 1_8, 1


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


                          wvol = ((mat_map(ie1, ie2, ie3, q1, q2, q3)*w3 &
      (ie3, q3))*w2(ie2, q2))*w1(ie1, q1)
                          bi = (bi2(ie2, il2, 0_8, q2)*bi3(ie3, il3, 0_8 &
      , q3))*bi1(ie1, il1, 0_8, q1)
                          bj = (bj2(ie2, jl2, 0_8, q2)*bj3(ie3, jl3, 0_8 &
      , q3))*bj1(ie1, jl1, 0_8, q1)


                          value = value + wvol*(bi*bj)


                        end do

                      end do

                    end do

                    M(modulo(ie1 + il1,nbase1), modulo(ie2 + il2,nbase2) &
      , modulo(ie3 + il3,nbase3), p1 - il1 + jl1, p2 - il2 + jl2, p3 - &
      il3 + jl3) = value + M(modulo(ie1 + il1,nbase1), modulo(ie2 + il2 &
      ,nbase2), modulo(ie3 + il3,nbase3), p1 - il1 + jl1, p2 - il2 + &
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
subroutine kernel_inner(nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, ni1 &
      , ni2, ni3, w1, w2, w3, bi1, bi2, bi3, nbase1, nbase2, nbase3, &
      mat, mat_f, mat_map)

  implicit none
  integer(kind=8), value  :: nel1
  integer(kind=8), value  :: nel2
  integer(kind=8), value  :: nel3
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
  integer(kind=8), value  :: nbase1
  integer(kind=8), value  :: nbase2
  integer(kind=8), value  :: nbase3
  real(kind=8), intent(inout)  :: mat (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_map (0:,0:,0:,0:,0:,0:)
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



  do ie1 = 0, nel1 - 1_8, 1
    do ie2 = 0, nel2 - 1_8, 1
      do ie3 = 0, nel3 - 1_8, 1


        do il1 = 0, -ni1 + p1, 1
          do il2 = 0, -ni2 + p2, 1
            do il3 = 0, -ni3 + p3, 1


              value = 0.0d0


              do q1 = 0, nq1 - 1_8, 1
                do q2 = 0, nq2 - 1_8, 1
                  do q3 = 0, nq3 - 1_8, 1


                    wvol = ((mat_map(ie1, ie2, ie3, q1, q2, q3)*w3(ie3, &
      q3))*w2(ie2, q2))*w1(ie1, q1)
                    bi = (bi2(ie2, il2, 0_8, q2)*bi3(ie3, il3, 0_8, q3)) &
      *bi1(ie1, il1, 0_8, q1)


                    value = value + wvol*(bi*mat_f(ie1*nq1 + q1, ie2*nq2 &
      + q2, ie3*nq3 + q3))


                  end do

                end do

              end do

              mat(modulo(ie1 + il1,nbase1), modulo(ie2 + il2,nbase2), &
      modulo(ie3 + il3,nbase3)) = value + mat(modulo(ie1 + il1,nbase1), &
      modulo(ie2 + il2,nbase2), modulo(ie3 + il3,nbase3))




            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_l2error(nel, p, nq, w1, w2, w3, ni, nj, bi1, bi2, bi3, &
      bj1, bj2, bj3, nbi, nbj, error, mat_f1, mat_f2, mat_c1, mat_c2, &
      mat_map)

  implicit none
  integer(kind=8), intent(in)  :: nel (0:)
  integer(kind=8), intent(in)  :: p (0:)
  integer(kind=8), intent(in)  :: nq (0:)
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: w3 (0:,0:)
  integer(kind=8), intent(in)  :: ni (0:)
  integer(kind=8), intent(in)  :: nj (0:)
  real(kind=8), intent(in)  :: bi1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bi3 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bj1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bj2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: bj3 (0:,0:,0:,0:)
  integer(kind=8), intent(in)  :: nbi (0:)
  integer(kind=8), intent(in)  :: nbj (0:)
  real(kind=8), intent(inout)  :: error (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_f1 (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_f2 (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_c1 (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_c2 (0:,0:,0:)
  real(kind=8), intent(in)  :: mat_map (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  
  real(kind=8)  :: bi  
  real(kind=8)  :: bj  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: jl1  
  integer(kind=8)  :: jl2  
  integer(kind=8)  :: jl3  



  !loop over all elements
  do ie1 = 0, nel(0_8) - 1_8, 1
    do ie2 = 0, nel(1_8) - 1_8, 1
      do ie3 = 0, nel(2_8) - 1_8, 1


        do q1 = 0, nq(0_8) - 1_8, 1
          do q2 = 0, nq(1_8) - 1_8, 1
            do q3 = 0, nq(2_8) - 1_8, 1


              wvol = ((mat_map(ie1, ie2, ie3, q1, q2, q3)*w3(ie3, q3))* &
      w2(ie2, q2))*w1(ie1, q1)


              bi = 0.0d0
              bj = 0.0d0


              do il1 = 0, -ni(0_8) + p(0_8), 1
                do il2 = 0, -ni(1_8) + p(1_8), 1
                  do il3 = 0, -ni(2_8) + p(2_8), 1


                    bi = bi + ((bi2(ie2, il2, 0_8, q2)*bi3(ie3, il3, 0_8 &
      , q3))*bi1(ie1, il1, 0_8, q1))*mat_c1(modulo(ie1 + il1,nbi(0_8)), &
      modulo(ie2 + il2,nbi(1_8)), modulo(ie3 + il3,nbi(2_8)))


                  end do

                end do

              end do

              do jl1 = 0, -nj(0_8) + p(0_8), 1
                do jl2 = 0, -nj(1_8) + p(1_8), 1
                  do jl3 = 0, -nj(2_8) + p(2_8), 1


                    bj = bj + ((bj2(ie2, jl2, 0_8, q2)*bj3(ie3, jl3, 0_8 &
      , q3))*bj1(ie1, jl1, 0_8, q1))*mat_c2(modulo(ie1 + jl1,nbj(0_8)), &
      modulo(ie2 + jl2,nbj(1_8)), modulo(ie3 + jl3,nbj(2_8)))




                  end do

                end do

              end do

              error(ie1, ie2, ie3) = wvol*((bi - mat_f1(ie1*nq(0_8) + q1 &
      , ie2*nq(1_8) + q2, ie3*nq(2_8) + q3))*(bj - mat_f2(ie1*nq(0_8) + &
      q1, ie2*nq(1_8) + q2, ie3*nq(2_8) + q3))) + error(ie1, ie2, ie3)
            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

end module