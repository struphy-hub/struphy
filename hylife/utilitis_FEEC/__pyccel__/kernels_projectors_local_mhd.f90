module kernels_projectors_local_mhd

implicit none




contains

!........................................
subroutine kernel_pi0(n, n_int, n_nvbf, i_glo1, i_glo2, i_glo3, c_loc1, &
      c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, coeff_ind2, &
      coeff_ind3, bs1, bs2, bs3, x_int_ind1, x_int_ind2, x_int_ind3, &
      tau, mat_eq)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: n_int (0:)
  integer(kind=8), intent(in)  :: n_nvbf (0:)
  integer(kind=8), intent(in)  :: i_glo1 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo2 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo3 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc1 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc2 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc3 (0:,0:)
  real(kind=8), intent(in)  :: coeff1 (0:,0:)
  real(kind=8), intent(in)  :: coeff2 (0:,0:)
  real(kind=8), intent(in)  :: coeff3 (0:,0:)
  integer(kind=8), intent(in)  :: coeff_ind1 (0:)
  integer(kind=8), intent(in)  :: coeff_ind2 (0:)
  integer(kind=8), intent(in)  :: coeff_ind3 (0:)
  real(kind=8), intent(in)  :: bs1 (0:,0:)
  real(kind=8), intent(in)  :: bs2 (0:,0:)
  real(kind=8), intent(in)  :: bs3 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind3 (0:,0:)
  real(kind=8), intent(inout)  :: tau (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_eq (0:,0:,0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: j1  
  integer(kind=8)  :: j2  
  integer(kind=8)  :: j3  
  real(kind=8)  :: coeff  
  integer(kind=8)  :: kl1  
  integer(kind=8)  :: k1  
  integer(kind=8)  :: c1  
  integer(kind=8)  :: kl2  
  integer(kind=8)  :: k2  
  integer(kind=8)  :: c2  
  integer(kind=8)  :: kl3  
  integer(kind=8)  :: k3  
  integer(kind=8)  :: c3  
  real(kind=8)  :: basis  



  tau(:, :, :, :, :, :) = 0.0d0


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1


        do j1 = 0, n_int(0_8) - 1_8, 1
          do j2 = 0, n_int(1_8) - 1_8, 1
            do j3 = 0, n_int(2_8) - 1_8, 1


              coeff = (coeff2(coeff_ind2(i2), j2)*coeff3(coeff_ind3(i3), &
      j3))*coeff1(coeff_ind1(i1), j1)


              do kl1 = 0, n_nvbf(0_8) - 1_8, 1


                k1 = i_glo1(i1, kl1)
                c1 = c_loc1(i1, kl1)


                do kl2 = 0, n_nvbf(1_8) - 1_8, 1


                  k2 = i_glo2(i2, kl2)
                  c2 = c_loc2(i2, kl2)


                  do kl3 = 0, n_nvbf(2_8) - 1_8, 1


                    k3 = i_glo3(i3, kl3)
                    c3 = c_loc3(i3, kl3)


                    basis = (bs2(x_int_ind2(i2, j2), k2)*bs3(x_int_ind3( &
      i3, j3), k3))*bs1(x_int_ind1(i1, j1), k1)


                    tau(k1, k2, k3, c1, c2, c3) = coeff*(basis*mat_eq( &
      x_int_ind1(i1, j1), x_int_ind2(i2, j2), x_int_ind3(i3, j3))) + &
      tau(k1, k2, k3, c1, c2, c3)




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
subroutine kernel_pi1_1(n, n_quad1, n_inthis, n_nvbf, i_glo1, i_glo2, &
      i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, &
      coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_his_ind1, &
      x_int_ind2, x_int_ind3, wts1, tau, mat_eq)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), value  :: n_quad1
  integer(kind=8), intent(in)  :: n_inthis (0:)
  integer(kind=8), intent(in)  :: n_nvbf (0:)
  integer(kind=8), intent(in)  :: i_glo1 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo2 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo3 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc1 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc2 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc3 (0:,0:)
  real(kind=8), intent(in)  :: coeff1 (0:,0:)
  real(kind=8), intent(in)  :: coeff2 (0:,0:)
  real(kind=8), intent(in)  :: coeff3 (0:,0:)
  integer(kind=8), intent(in)  :: coeff_ind1 (0:)
  integer(kind=8), intent(in)  :: coeff_ind2 (0:)
  integer(kind=8), intent(in)  :: coeff_ind3 (0:)
  real(kind=8), intent(in)  :: bs1 (0:,0:,0:)
  real(kind=8), intent(in)  :: bs2 (0:,0:)
  real(kind=8), intent(in)  :: bs3 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts1 (0:,0:)
  real(kind=8), intent(inout)  :: tau (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_eq (0:,0:,0:,0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: j1  
  integer(kind=8)  :: j2  
  integer(kind=8)  :: j3  
  real(kind=8)  :: coeff  
  integer(kind=8)  :: kl1  
  integer(kind=8)  :: k1  
  integer(kind=8)  :: c1  
  integer(kind=8)  :: kl2  
  integer(kind=8)  :: k2  
  integer(kind=8)  :: c2  
  integer(kind=8)  :: kl3  
  integer(kind=8)  :: k3  
  integer(kind=8)  :: c3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q1  



  tau(:, :, :, :, :, :) = 0.0d0


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1


        do j1 = 0, n_inthis(0_8) - 1_8, 1
          do j2 = 0, n_inthis(1_8) - 1_8, 1
            do j3 = 0, n_inthis(2_8) - 1_8, 1


              coeff = (coeff2(coeff_ind2(i2), j2)*coeff3(coeff_ind3(i3), &
      j3))*coeff1(coeff_ind1(i1), j1)


              do kl1 = 0, n_nvbf(0_8) - 1_8, 1


                k1 = i_glo1(i1, kl1)
                c1 = c_loc1(i1, kl1)


                do kl2 = 0, n_nvbf(1_8) - 1_8, 1


                  k2 = i_glo2(i2, kl2)
                  c2 = c_loc2(i2, kl2)


                  do kl3 = 0, n_nvbf(2_8) - 1_8, 1


                    k3 = i_glo3(i3, kl3)
                    c3 = c_loc3(i3, kl3)


                    f_int = 0.0d0


                    do q1 = 0, n_quad1 - 1_8, 1
                      f_int = f_int + (((bs3(x_int_ind3(i3, j3), k3)* &
      mat_eq(x_his_ind1(i1, j1), q1, x_int_ind2(i2, j2), x_int_ind3(i3, &
      j3)))*bs2(x_int_ind2(i2, j2), k2))*bs1(x_his_ind1(i1, j1), q1, k1 &
      ))*wts1(x_his_ind1(i1, j1), q1)


                    end do

                    tau(k1, k2, k3, c1, c2, c3) = coeff*f_int + tau(k1, &
      k2, k3, c1, c2, c3)




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
subroutine kernel_pi1_2(n, n_quad2, n_inthis, n_nvbf, i_glo1, i_glo2, &
      i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, &
      coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_int_ind1, &
      x_his_ind2, x_int_ind3, wts2, tau, mat_eq)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), value  :: n_quad2
  integer(kind=8), intent(in)  :: n_inthis (0:)
  integer(kind=8), intent(in)  :: n_nvbf (0:)
  integer(kind=8), intent(in)  :: i_glo1 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo2 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo3 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc1 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc2 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc3 (0:,0:)
  real(kind=8), intent(in)  :: coeff1 (0:,0:)
  real(kind=8), intent(in)  :: coeff2 (0:,0:)
  real(kind=8), intent(in)  :: coeff3 (0:,0:)
  integer(kind=8), intent(in)  :: coeff_ind1 (0:)
  integer(kind=8), intent(in)  :: coeff_ind2 (0:)
  integer(kind=8), intent(in)  :: coeff_ind3 (0:)
  real(kind=8), intent(in)  :: bs1 (0:,0:)
  real(kind=8), intent(in)  :: bs2 (0:,0:,0:)
  real(kind=8), intent(in)  :: bs3 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts2 (0:,0:)
  real(kind=8), intent(inout)  :: tau (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_eq (0:,0:,0:,0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: j1  
  integer(kind=8)  :: j2  
  integer(kind=8)  :: j3  
  real(kind=8)  :: coeff  
  integer(kind=8)  :: kl1  
  integer(kind=8)  :: k1  
  integer(kind=8)  :: c1  
  integer(kind=8)  :: kl2  
  integer(kind=8)  :: k2  
  integer(kind=8)  :: c2  
  integer(kind=8)  :: kl3  
  integer(kind=8)  :: k3  
  integer(kind=8)  :: c3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q2  



  tau(:, :, :, :, :, :) = 0.0d0


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1


        do j1 = 0, n_inthis(0_8) - 1_8, 1
          do j2 = 0, n_inthis(1_8) - 1_8, 1
            do j3 = 0, n_inthis(2_8) - 1_8, 1


              coeff = (coeff2(coeff_ind2(i2), j2)*coeff3(coeff_ind3(i3), &
      j3))*coeff1(coeff_ind1(i1), j1)


              do kl1 = 0, n_nvbf(0_8) - 1_8, 1


                k1 = i_glo1(i1, kl1)
                c1 = c_loc1(i1, kl1)


                do kl2 = 0, n_nvbf(1_8) - 1_8, 1


                  k2 = i_glo2(i2, kl2)
                  c2 = c_loc2(i2, kl2)


                  do kl3 = 0, n_nvbf(2_8) - 1_8, 1


                    k3 = i_glo3(i3, kl3)
                    c3 = c_loc3(i3, kl3)


                    f_int = 0.0d0


                    do q2 = 0, n_quad2 - 1_8, 1
                      f_int = f_int + (((bs3(x_int_ind3(i3, j3), k3)* &
      mat_eq(x_int_ind1(i1, j1), x_his_ind2(i2, j2), q2, x_int_ind3(i3, &
      j3)))*bs2(x_his_ind2(i2, j2), q2, k2))*bs1(x_int_ind1(i1, j1), k1 &
      ))*wts2(x_his_ind2(i2, j2), q2)


                    end do

                    tau(k1, k2, k3, c1, c2, c3) = coeff*f_int + tau(k1, &
      k2, k3, c1, c2, c3)




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
subroutine kernel_pi1_3(n, n_quad3, n_inthis, n_nvbf, i_glo1, i_glo2, &
      i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, &
      coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_int_ind1, &
      x_int_ind2, x_his_ind3, wts3, tau, mat_eq)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), value  :: n_quad3
  integer(kind=8), intent(in)  :: n_inthis (0:)
  integer(kind=8), intent(in)  :: n_nvbf (0:)
  integer(kind=8), intent(in)  :: i_glo1 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo2 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo3 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc1 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc2 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc3 (0:,0:)
  real(kind=8), intent(in)  :: coeff1 (0:,0:)
  real(kind=8), intent(in)  :: coeff2 (0:,0:)
  real(kind=8), intent(in)  :: coeff3 (0:,0:)
  integer(kind=8), intent(in)  :: coeff_ind1 (0:)
  integer(kind=8), intent(in)  :: coeff_ind2 (0:)
  integer(kind=8), intent(in)  :: coeff_ind3 (0:)
  real(kind=8), intent(in)  :: bs1 (0:,0:)
  real(kind=8), intent(in)  :: bs2 (0:,0:)
  real(kind=8), intent(in)  :: bs3 (0:,0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts3 (0:,0:)
  real(kind=8), intent(inout)  :: tau (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_eq (0:,0:,0:,0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: j1  
  integer(kind=8)  :: j2  
  integer(kind=8)  :: j3  
  real(kind=8)  :: coeff  
  integer(kind=8)  :: kl1  
  integer(kind=8)  :: k1  
  integer(kind=8)  :: c1  
  integer(kind=8)  :: kl2  
  integer(kind=8)  :: k2  
  integer(kind=8)  :: c2  
  integer(kind=8)  :: kl3  
  integer(kind=8)  :: k3  
  integer(kind=8)  :: c3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q3  



  tau(:, :, :, :, :, :) = 0.0d0


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1


        do j1 = 0, n_inthis(0_8) - 1_8, 1
          do j2 = 0, n_inthis(1_8) - 1_8, 1
            do j3 = 0, n_inthis(2_8) - 1_8, 1


              coeff = (coeff2(coeff_ind2(i2), j2)*coeff3(coeff_ind3(i3), &
      j3))*coeff1(coeff_ind1(i1), j1)


              do kl1 = 0, n_nvbf(0_8) - 1_8, 1


                k1 = i_glo1(i1, kl1)
                c1 = c_loc1(i1, kl1)


                do kl2 = 0, n_nvbf(1_8) - 1_8, 1


                  k2 = i_glo2(i2, kl2)
                  c2 = c_loc2(i2, kl2)


                  do kl3 = 0, n_nvbf(2_8) - 1_8, 1


                    k3 = i_glo3(i3, kl3)
                    c3 = c_loc3(i3, kl3)


                    f_int = 0.0d0


                    do q3 = 0, n_quad3 - 1_8, 1
                      f_int = f_int + (((bs3(x_his_ind3(i3, j3), q3, k3) &
      *mat_eq(x_int_ind1(i1, j1), x_int_ind2(i2, j2), x_his_ind3(i3, j3 &
      ), q3))*bs2(x_int_ind2(i2, j2), k2))*bs1(x_int_ind1(i1, j1), k1)) &
      *wts3(x_his_ind3(i3, j3), q3)


                    end do

                    tau(k1, k2, k3, c1, c2, c3) = coeff*f_int + tau(k1, &
      k2, k3, c1, c2, c3)




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
subroutine kernel_pi2_1(n, n_quad, n_inthis, n_nvbf, i_glo1, i_glo2, &
      i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, &
      coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_int_ind1, &
      x_his_ind2, x_his_ind3, wts2, wts3, tau, mat_eq)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: n_quad (0:)
  integer(kind=8), intent(in)  :: n_inthis (0:)
  integer(kind=8), intent(in)  :: n_nvbf (0:)
  integer(kind=8), intent(in)  :: i_glo1 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo2 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo3 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc1 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc2 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc3 (0:,0:)
  real(kind=8), intent(in)  :: coeff1 (0:,0:)
  real(kind=8), intent(in)  :: coeff2 (0:,0:)
  real(kind=8), intent(in)  :: coeff3 (0:,0:)
  integer(kind=8), intent(in)  :: coeff_ind1 (0:)
  integer(kind=8), intent(in)  :: coeff_ind2 (0:)
  integer(kind=8), intent(in)  :: coeff_ind3 (0:)
  real(kind=8), intent(in)  :: bs1 (0:,0:)
  real(kind=8), intent(in)  :: bs2 (0:,0:,0:)
  real(kind=8), intent(in)  :: bs3 (0:,0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts2 (0:,0:)
  real(kind=8), intent(in)  :: wts3 (0:,0:)
  real(kind=8), intent(inout)  :: tau (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_eq (0:,0:,0:,0:,0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: j1  
  integer(kind=8)  :: j2  
  integer(kind=8)  :: j3  
  real(kind=8)  :: coeff  
  integer(kind=8)  :: kl1  
  integer(kind=8)  :: k1  
  integer(kind=8)  :: c1  
  integer(kind=8)  :: kl2  
  integer(kind=8)  :: k2  
  integer(kind=8)  :: c2  
  integer(kind=8)  :: kl3  
  integer(kind=8)  :: k3  
  integer(kind=8)  :: c3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  



  tau(:, :, :, :, :, :) = 0.0d0


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1


        do j1 = 0, n_inthis(0_8) - 1_8, 1
          do j2 = 0, n_inthis(1_8) - 1_8, 1
            do j3 = 0, n_inthis(2_8) - 1_8, 1


              coeff = (coeff2(coeff_ind2(i2), j2)*coeff3(coeff_ind3(i3), &
      j3))*coeff1(coeff_ind1(i1), j1)


              do kl1 = 0, n_nvbf(0_8) - 1_8, 1


                k1 = i_glo1(i1, kl1)
                c1 = c_loc1(i1, kl1)


                do kl2 = 0, n_nvbf(1_8) - 1_8, 1


                  k2 = i_glo2(i2, kl2)
                  c2 = c_loc2(i2, kl2)


                  do kl3 = 0, n_nvbf(2_8) - 1_8, 1


                    k3 = i_glo3(i3, kl3)
                    c3 = c_loc3(i3, kl3)


                    f_int = 0.0d0


                    do q2 = 0, n_quad(0_8) - 1_8, 1
                      do q3 = 0, n_quad(1_8) - 1_8, 1
                        wvol = wts2(x_his_ind2(i2, j2), q2)*wts3( &
      x_his_ind3(i3, j3), q3)
                        f_int = f_int + wvol*(((bs3(x_his_ind3(i3, j3), &
      q3, k3)*mat_eq(x_int_ind1(i1, j1), x_his_ind2(i2, j2), q2, &
      x_his_ind3(i3, j3), q3))*bs2(x_his_ind2(i2, j2), q2, k2))*bs1( &
      x_int_ind1(i1, j1), k1))


                      end do

                    end do

                    tau(k1, k2, k3, c1, c2, c3) = coeff*f_int + tau(k1, &
      k2, k3, c1, c2, c3)






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
subroutine kernel_pi2_2(n, n_quad, n_inthis, n_nvbf, i_glo1, i_glo2, &
      i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, &
      coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_his_ind1, &
      x_int_ind2, x_his_ind3, wts1, wts3, tau, mat_eq)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: n_quad (0:)
  integer(kind=8), intent(in)  :: n_inthis (0:)
  integer(kind=8), intent(in)  :: n_nvbf (0:)
  integer(kind=8), intent(in)  :: i_glo1 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo2 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo3 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc1 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc2 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc3 (0:,0:)
  real(kind=8), intent(in)  :: coeff1 (0:,0:)
  real(kind=8), intent(in)  :: coeff2 (0:,0:)
  real(kind=8), intent(in)  :: coeff3 (0:,0:)
  integer(kind=8), intent(in)  :: coeff_ind1 (0:)
  integer(kind=8), intent(in)  :: coeff_ind2 (0:)
  integer(kind=8), intent(in)  :: coeff_ind3 (0:)
  real(kind=8), intent(in)  :: bs1 (0:,0:,0:)
  real(kind=8), intent(in)  :: bs2 (0:,0:)
  real(kind=8), intent(in)  :: bs3 (0:,0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts1 (0:,0:)
  real(kind=8), intent(in)  :: wts3 (0:,0:)
  real(kind=8), intent(inout)  :: tau (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_eq (0:,0:,0:,0:,0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: j1  
  integer(kind=8)  :: j2  
  integer(kind=8)  :: j3  
  real(kind=8)  :: coeff  
  integer(kind=8)  :: kl1  
  integer(kind=8)  :: k1  
  integer(kind=8)  :: c1  
  integer(kind=8)  :: kl2  
  integer(kind=8)  :: k2  
  integer(kind=8)  :: c2  
  integer(kind=8)  :: kl3  
  integer(kind=8)  :: k3  
  integer(kind=8)  :: c3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  



  tau(:, :, :, :, :, :) = 0.0d0


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1


        do j1 = 0, n_inthis(0_8) - 1_8, 1
          do j2 = 0, n_inthis(1_8) - 1_8, 1
            do j3 = 0, n_inthis(2_8) - 1_8, 1


              coeff = (coeff2(coeff_ind2(i2), j2)*coeff3(coeff_ind3(i3), &
      j3))*coeff1(coeff_ind1(i1), j1)


              do kl1 = 0, n_nvbf(0_8) - 1_8, 1


                k1 = i_glo1(i1, kl1)
                c1 = c_loc1(i1, kl1)


                do kl2 = 0, n_nvbf(1_8) - 1_8, 1


                  k2 = i_glo2(i2, kl2)
                  c2 = c_loc2(i2, kl2)


                  do kl3 = 0, n_nvbf(2_8) - 1_8, 1


                    k3 = i_glo3(i3, kl3)
                    c3 = c_loc3(i3, kl3)


                    f_int = 0.0d0


                    do q1 = 0, n_quad(0_8) - 1_8, 1
                      do q3 = 0, n_quad(1_8) - 1_8, 1
                        wvol = wts1(x_his_ind1(i1, j1), q1)*wts3( &
      x_his_ind3(i3, j3), q3)
                        f_int = f_int + wvol*(((bs3(x_his_ind3(i3, j3), &
      q3, k3)*mat_eq(x_his_ind1(i1, j1), q1, x_int_ind2(i2, j2), &
      x_his_ind3(i3, j3), q3))*bs2(x_int_ind2(i2, j2), k2))*bs1( &
      x_his_ind1(i1, j1), q1, k1))


                      end do

                    end do

                    tau(k1, k2, k3, c1, c2, c3) = coeff*f_int + tau(k1, &
      k2, k3, c1, c2, c3)






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
subroutine kernel_pi2_3(n, n_quad, n_inthis, n_nvbf, i_glo1, i_glo2, &
      i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, &
      coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_his_ind1, &
      x_his_ind2, x_int_ind3, wts1, wts2, tau, mat_eq)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: n_quad (0:)
  integer(kind=8), intent(in)  :: n_inthis (0:)
  integer(kind=8), intent(in)  :: n_nvbf (0:)
  integer(kind=8), intent(in)  :: i_glo1 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo2 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo3 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc1 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc2 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc3 (0:,0:)
  real(kind=8), intent(in)  :: coeff1 (0:,0:)
  real(kind=8), intent(in)  :: coeff2 (0:,0:)
  real(kind=8), intent(in)  :: coeff3 (0:,0:)
  integer(kind=8), intent(in)  :: coeff_ind1 (0:)
  integer(kind=8), intent(in)  :: coeff_ind2 (0:)
  integer(kind=8), intent(in)  :: coeff_ind3 (0:)
  real(kind=8), intent(in)  :: bs1 (0:,0:,0:)
  real(kind=8), intent(in)  :: bs2 (0:,0:,0:)
  real(kind=8), intent(in)  :: bs3 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts1 (0:,0:)
  real(kind=8), intent(in)  :: wts2 (0:,0:)
  real(kind=8), intent(inout)  :: tau (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_eq (0:,0:,0:,0:,0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: j1  
  integer(kind=8)  :: j2  
  integer(kind=8)  :: j3  
  real(kind=8)  :: coeff  
  integer(kind=8)  :: kl1  
  integer(kind=8)  :: k1  
  integer(kind=8)  :: c1  
  integer(kind=8)  :: kl2  
  integer(kind=8)  :: k2  
  integer(kind=8)  :: c2  
  integer(kind=8)  :: kl3  
  integer(kind=8)  :: k3  
  integer(kind=8)  :: c3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  real(kind=8)  :: wvol  



  tau(:, :, :, :, :, :) = 0.0d0


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1


        do j1 = 0, n_inthis(0_8) - 1_8, 1
          do j2 = 0, n_inthis(1_8) - 1_8, 1
            do j3 = 0, n_inthis(2_8) - 1_8, 1


              coeff = (coeff2(coeff_ind2(i2), j2)*coeff3(coeff_ind3(i3), &
      j3))*coeff1(coeff_ind1(i1), j1)


              do kl1 = 0, n_nvbf(0_8) - 1_8, 1


                k1 = i_glo1(i1, kl1)
                c1 = c_loc1(i1, kl1)


                do kl2 = 0, n_nvbf(1_8) - 1_8, 1


                  k2 = i_glo2(i2, kl2)
                  c2 = c_loc2(i2, kl2)


                  do kl3 = 0, n_nvbf(2_8) - 1_8, 1


                    k3 = i_glo3(i3, kl3)
                    c3 = c_loc3(i3, kl3)


                    f_int = 0.0d0


                    do q1 = 0, n_quad(0_8) - 1_8, 1
                      do q2 = 0, n_quad(1_8) - 1_8, 1
                        wvol = wts1(x_his_ind1(i1, j1), q1)*wts2( &
      x_his_ind2(i2, j2), q2)
                        f_int = f_int + wvol*(((bs3(x_int_ind3(i3, j3), &
      k3)*mat_eq(x_his_ind1(i1, j1), q1, x_his_ind2(i2, j2), q2, &
      x_int_ind3(i3, j3)))*bs2(x_his_ind2(i2, j2), q2, k2))*bs1( &
      x_his_ind1(i1, j1), q1, k1))


                      end do

                    end do

                    tau(k1, k2, k3, c1, c2, c3) = coeff*f_int + tau(k1, &
      k2, k3, c1, c2, c3)




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
subroutine kernel_pi3(n, n_quad, n_his, n_nvbf, i_glo1, i_glo2, i_glo3, &
      c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, &
      coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_his_ind1, x_his_ind2, &
      x_his_ind3, wts1, wts2, wts3, tau, mat_eq)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: n_quad (0:)
  integer(kind=8), intent(in)  :: n_his (0:)
  integer(kind=8), intent(in)  :: n_nvbf (0:)
  integer(kind=8), intent(in)  :: i_glo1 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo2 (0:,0:)
  integer(kind=8), intent(in)  :: i_glo3 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc1 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc2 (0:,0:)
  integer(kind=8), intent(in)  :: c_loc3 (0:,0:)
  real(kind=8), intent(in)  :: coeff1 (0:,0:)
  real(kind=8), intent(in)  :: coeff2 (0:,0:)
  real(kind=8), intent(in)  :: coeff3 (0:,0:)
  integer(kind=8), intent(in)  :: coeff_ind1 (0:)
  integer(kind=8), intent(in)  :: coeff_ind2 (0:)
  integer(kind=8), intent(in)  :: coeff_ind3 (0:)
  real(kind=8), intent(in)  :: bs1 (0:,0:,0:)
  real(kind=8), intent(in)  :: bs2 (0:,0:,0:)
  real(kind=8), intent(in)  :: bs3 (0:,0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts1 (0:,0:)
  real(kind=8), intent(in)  :: wts2 (0:,0:)
  real(kind=8), intent(in)  :: wts3 (0:,0:)
  real(kind=8), intent(inout)  :: tau (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat_eq (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: j1  
  integer(kind=8)  :: j2  
  integer(kind=8)  :: j3  
  real(kind=8)  :: coeff  
  integer(kind=8)  :: kl1  
  integer(kind=8)  :: k1  
  integer(kind=8)  :: c1  
  integer(kind=8)  :: kl2  
  integer(kind=8)  :: k2  
  integer(kind=8)  :: c2  
  integer(kind=8)  :: kl3  
  integer(kind=8)  :: k3  
  integer(kind=8)  :: c3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  



  tau(:, :, :, :, :, :) = 0.0d0


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1


        do j1 = 0, n_his(0_8) - 1_8, 1
          do j2 = 0, n_his(1_8) - 1_8, 1
            do j3 = 0, n_his(2_8) - 1_8, 1


              coeff = (coeff2(coeff_ind2(i2), j2)*coeff3(coeff_ind3(i3), &
      j3))*coeff1(coeff_ind1(i1), j1)


              do kl1 = 0, n_nvbf(0_8) - 1_8, 1


                k1 = i_glo1(i1, kl1)
                c1 = c_loc1(i1, kl1)


                do kl2 = 0, n_nvbf(1_8) - 1_8, 1


                  k2 = i_glo2(i2, kl2)
                  c2 = c_loc2(i2, kl2)


                  do kl3 = 0, n_nvbf(2_8) - 1_8, 1


                    k3 = i_glo3(i3, kl3)
                    c3 = c_loc3(i3, kl3)


                    f_int = 0.0d0


                    do q1 = 0, n_quad(0_8) - 1_8, 1
                      do q2 = 0, n_quad(1_8) - 1_8, 1
                        do q3 = 0, n_quad(2_8) - 1_8, 1
                          wvol = (wts2(x_his_ind2(i2, j2), q2)*wts3( &
      x_his_ind2(i3, j3), q3))*wts1(x_his_ind1(i1, j1), q1)
                          f_int = f_int + wvol*(((bs3(x_his_ind3(i3, j3) &
      , q3, k3)*mat_eq(x_his_ind1(i1, j1), q1, x_his_ind2(i2, j2), q2, &
      x_his_ind3(i3, j3), q3))*bs2(x_his_ind2(i2, j2), q2, k2))*bs1( &
      x_his_ind1(i1, j1), q1, k1))


                        end do

                      end do

                    end do

                    tau(k1, k2, k3, c1, c2, c3) = coeff*f_int + tau(k1, &
      k2, k3, c1, c2, c3)
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

end module