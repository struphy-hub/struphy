module kernels_projectors_local

implicit none




contains

!........................................
subroutine kernel_pi0_3d(n, p, coeff_i1, coeff_i2, coeff_i3, coeffi_ind1 &
      , coeffi_ind2, coeffi_ind3, x_int_ind1, x_int_ind2, x_int_ind3, &
      mat_f, lambdas)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: p (0:)
  real(kind=8), intent(in)  :: coeff_i1 (0:,0:)
  real(kind=8), intent(in)  :: coeff_i2 (0:,0:)
  real(kind=8), intent(in)  :: coeff_i3 (0:,0:)
  integer(kind=8), intent(in)  :: coeffi_ind1 (0:)
  integer(kind=8), intent(in)  :: coeffi_ind2 (0:)
  integer(kind=8), intent(in)  :: coeffi_ind3 (0:)
  integer(kind=8), intent(in)  :: x_int_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind3 (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:)
  real(kind=8), intent(inout)  :: lambdas (0:,0:,0:)
  integer(kind=8)  :: n_pts1  
  integer(kind=8)  :: n_pts2  
  integer(kind=8)  :: n_pts3  
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  



  n_pts1 = 2_8*(p(0_8) - 1_8) + 1_8
  n_pts2 = 2_8*(p(1_8) - 1_8) + 1_8
  n_pts3 = 2_8*(p(2_8) - 1_8) + 1_8


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        do il1 = 0, n_pts1 - 1_8, 1
          do il2 = 0, n_pts2 - 1_8, 1
            do il3 = 0, n_pts3 - 1_8, 1


              lambdas(i1, i2, i3) = ((coeff_i3(coeffi_ind3(i3), il3)* &
      mat_f(x_int_ind1(i1, il1), x_int_ind2(i2, il2), x_int_ind3(i3, &
      il3)))*coeff_i2(coeffi_ind2(i2), il2))*coeff_i1(coeffi_ind1(i1), &
      il1) + lambdas(i1, i2, i3)




            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi11_3d(n, p, coeff_h1, coeff_i2, coeff_i3, &
      coeffh_ind1, coeffi_ind2, coeffi_ind3, x_his_ind1, x_int_ind2, &
      x_int_ind3, wts1, mat_f, lambdas)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: p (0:)
  real(kind=8), intent(in)  :: coeff_h1 (0:,0:)
  real(kind=8), intent(in)  :: coeff_i2 (0:,0:)
  real(kind=8), intent(in)  :: coeff_i3 (0:,0:)
  integer(kind=8), intent(in)  :: coeffh_ind1 (0:)
  integer(kind=8), intent(in)  :: coeffi_ind2 (0:)
  integer(kind=8), intent(in)  :: coeffi_ind3 (0:)
  integer(kind=8), intent(in)  :: x_his_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts1 (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: lambdas (0:,0:,0:)
  integer(kind=8)  :: n_pts2  
  integer(kind=8)  :: n_pts3  
  integer(kind=8)  :: n_his1  
  integer(kind=8)  :: nq1  
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q1  



  n_pts2 = 2_8*(p(1_8) - 1_8) + 1_8
  n_pts3 = 2_8*(p(2_8) - 1_8) + 1_8


  n_his1 = 2_8*p(0_8)


  nq1 = p(0_8) + 1_8


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        do il1 = 0, n_his1 - 1_8, 1
          do il2 = 0, n_pts2 - 1_8, 1
            do il3 = 0, n_pts3 - 1_8, 1


              f_int = 0.0d0


              do q1 = 0, nq1 - 1_8, 1
                f_int = f_int + mat_f(x_his_ind1(i1, il1), q1, &
      x_int_ind2(i2, il2), x_int_ind3(i3, il3))*wts1(x_his_ind1(i1, il1 &
      ), q1)


              end do

              lambdas(i1, i2, i3) = ((f_int*coeff_i3(coeffi_ind3(i3), &
      il3))*coeff_i2(coeffi_ind2(i2), il2))*coeff_h1(coeffh_ind1(i1), &
      il1) + lambdas(i1, i2, i3)




            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi12_3d(n, p, coeff_i1, coeff_h2, coeff_i3, &
      coeffi_ind1, coeffh_ind2, coeffi_ind3, x_int_ind1, x_his_ind2, &
      x_int_ind3, wts2, mat_f, lambdas)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: p (0:)
  real(kind=8), intent(in)  :: coeff_i1 (0:,0:)
  real(kind=8), intent(in)  :: coeff_h2 (0:,0:)
  real(kind=8), intent(in)  :: coeff_i3 (0:,0:)
  integer(kind=8), intent(in)  :: coeffi_ind1 (0:)
  integer(kind=8), intent(in)  :: coeffh_ind2 (0:)
  integer(kind=8), intent(in)  :: coeffi_ind3 (0:)
  integer(kind=8), intent(in)  :: x_int_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts2 (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: lambdas (0:,0:,0:)
  integer(kind=8)  :: n_pts1  
  integer(kind=8)  :: n_pts3  
  integer(kind=8)  :: n_his2  
  integer(kind=8)  :: nq2  
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q2  



  n_pts1 = 2_8*(p(0_8) - 1_8) + 1_8
  n_pts3 = 2_8*(p(2_8) - 1_8) + 1_8


  n_his2 = 2_8*p(1_8)


  nq2 = p(1_8) + 1_8


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        do il1 = 0, n_pts1 - 1_8, 1
          do il2 = 0, n_his2 - 1_8, 1
            do il3 = 0, n_pts3 - 1_8, 1


              f_int = 0.0d0


              do q2 = 0, nq2 - 1_8, 1
                f_int = f_int + mat_f(x_int_ind1(i1, il1), x_his_ind2(i2 &
      , il2), q2, x_int_ind3(i3, il3))*wts2(x_his_ind2(i2, il2), q2)


              end do

              lambdas(i1, i2, i3) = ((f_int*coeff_i3(coeffi_ind3(i3), &
      il3))*coeff_h2(coeffh_ind2(i2), il2))*coeff_i1(coeffi_ind1(i1), &
      il1) + lambdas(i1, i2, i3)




            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi13_3d(n, p, coeff_i1, coeff_i2, coeff_h3, &
      coeffi_ind1, coeffi_ind2, coeffh_ind3, x_int_ind1, x_int_ind2, &
      x_his_ind3, wts3, mat_f, lambdas)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: p (0:)
  real(kind=8), intent(in)  :: coeff_i1 (0:,0:)
  real(kind=8), intent(in)  :: coeff_i2 (0:,0:)
  real(kind=8), intent(in)  :: coeff_h3 (0:,0:)
  integer(kind=8), intent(in)  :: coeffi_ind1 (0:)
  integer(kind=8), intent(in)  :: coeffi_ind2 (0:)
  integer(kind=8), intent(in)  :: coeffh_ind3 (0:)
  integer(kind=8), intent(in)  :: x_int_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts3 (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: lambdas (0:,0:,0:)
  integer(kind=8)  :: n_pts1  
  integer(kind=8)  :: n_pts2  
  integer(kind=8)  :: n_his3  
  integer(kind=8)  :: nq3  
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q3  



  n_pts1 = 2_8*(p(0_8) - 1_8) + 1_8
  n_pts2 = 2_8*(p(1_8) - 1_8) + 1_8


  n_his3 = 2_8*p(2_8)


  nq3 = p(2_8) + 1_8


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        do il1 = 0, n_pts1 - 1_8, 1
          do il2 = 0, n_pts2 - 1_8, 1
            do il3 = 0, n_his3 - 1_8, 1


              f_int = 0.0d0


              do q3 = 0, nq3 - 1_8, 1
                f_int = f_int + mat_f(x_int_ind1(i1, il1), x_int_ind2(i2 &
      , il2), x_his_ind3(i3, il3), q3)*wts3(x_his_ind3(i3, il3), q3)


              end do

              lambdas(i1, i2, i3) = ((f_int*coeff_h3(coeffh_ind3(i3), &
      il3))*coeff_i2(coeffi_ind2(i2), il2))*coeff_i1(coeffi_ind1(i1), &
      il1) + lambdas(i1, i2, i3)




            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi21_3d(n, p, coeff_i1, coeff_h2, coeff_h3, &
      coeffi_ind1, coeffh_ind2, coeffh_ind3, x_int_ind1, x_his_ind2, &
      x_his_ind3, wts2, wts3, mat_f, lambdas)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: p (0:)
  real(kind=8), intent(in)  :: coeff_i1 (0:,0:)
  real(kind=8), intent(in)  :: coeff_h2 (0:,0:)
  real(kind=8), intent(in)  :: coeff_h3 (0:,0:)
  integer(kind=8), intent(in)  :: coeffi_ind1 (0:)
  integer(kind=8), intent(in)  :: coeffh_ind2 (0:)
  integer(kind=8), intent(in)  :: coeffh_ind3 (0:)
  integer(kind=8), intent(in)  :: x_int_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts2 (0:,0:)
  real(kind=8), intent(in)  :: wts3 (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: lambdas (0:,0:,0:)
  integer(kind=8)  :: n_pts1  
  integer(kind=8)  :: n_his2  
  integer(kind=8)  :: n_his3  
  integer(kind=8)  :: nq2  
  integer(kind=8)  :: nq3  
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  



  n_pts1 = 2_8*(p(0_8) - 1_8) + 1_8


  n_his2 = 2_8*p(1_8)
  n_his3 = 2_8*p(2_8)


  nq2 = p(1_8) + 1_8
  nq3 = p(2_8) + 1_8


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        do il1 = 0, n_pts1 - 1_8, 1
          do il2 = 0, n_his2 - 1_8, 1
            do il3 = 0, n_his3 - 1_8, 1


              f_int = 0.0d0


              do q2 = 0, nq2 - 1_8, 1
                do q3 = 0, nq3 - 1_8, 1
                  wvol = wts2(x_his_ind2(i2, il2), q2)*wts3(x_his_ind3( &
      i3, il3), q3)
                  f_int = f_int + wvol*mat_f(x_int_ind1(i1, il1), &
      x_his_ind2(i2, il2), q2, x_his_ind3(i3, il3), q3)


                end do

              end do

              lambdas(i1, i2, i3) = ((f_int*coeff_h3(coeffh_ind3(i3), &
      il3))*coeff_h2(coeffh_ind2(i2), il2))*coeff_i1(coeffi_ind1(i1), &
      il1) + lambdas(i1, i2, i3)




            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi22_3d(n, p, coeff_h1, coeff_i2, coeff_h3, &
      coeffh_ind1, coeffi_ind2, coeffh_ind3, x_his_ind1, x_int_ind2, &
      x_his_ind3, wts1, wts3, mat_f, lambdas)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: p (0:)
  real(kind=8), intent(in)  :: coeff_h1 (0:,0:)
  real(kind=8), intent(in)  :: coeff_i2 (0:,0:)
  real(kind=8), intent(in)  :: coeff_h3 (0:,0:)
  integer(kind=8), intent(in)  :: coeffh_ind1 (0:)
  integer(kind=8), intent(in)  :: coeffi_ind2 (0:)
  integer(kind=8), intent(in)  :: coeffh_ind3 (0:)
  integer(kind=8), intent(in)  :: x_his_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts1 (0:,0:)
  real(kind=8), intent(in)  :: wts3 (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: lambdas (0:,0:,0:)
  integer(kind=8)  :: n_pts2  
  integer(kind=8)  :: n_his1  
  integer(kind=8)  :: n_his3  
  integer(kind=8)  :: nq1  
  integer(kind=8)  :: nq3  
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  



  n_pts2 = 2_8*(p(1_8) - 1_8) + 1_8


  n_his1 = 2_8*p(0_8)
  n_his3 = 2_8*p(2_8)


  nq1 = p(0_8) + 1_8
  nq3 = p(2_8) + 1_8


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        do il1 = 0, n_his1 - 1_8, 1
          do il2 = 0, n_pts2 - 1_8, 1
            do il3 = 0, n_his3 - 1_8, 1


              f_int = 0.0d0


              do q1 = 0, nq1 - 1_8, 1
                do q3 = 0, nq3 - 1_8, 1
                  wvol = wts1(x_his_ind1(i1, il1), q1)*wts3(x_his_ind3( &
      i3, il3), q3)
                  f_int = f_int + wvol*mat_f(x_his_ind1(i1, il1), q1, &
      x_int_ind2(i2, il2), x_his_ind3(i3, il3), q3)


                end do

              end do

              lambdas(i1, i2, i3) = ((f_int*coeff_h3(coeffh_ind3(i3), &
      il3))*coeff_i2(coeffi_ind2(i2), il2))*coeff_h1(coeffh_ind1(i1), &
      il1) + lambdas(i1, i2, i3)




            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi23_3d(n, p, coeff_h1, coeff_h2, coeff_i3, &
      coeffh_ind1, coeffh_ind2, coeffi_ind3, x_his_ind1, x_his_ind2, &
      x_int_ind3, wts1, wts2, mat_f, lambdas)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: p (0:)
  real(kind=8), intent(in)  :: coeff_h1 (0:,0:)
  real(kind=8), intent(in)  :: coeff_h2 (0:,0:)
  real(kind=8), intent(in)  :: coeff_i3 (0:,0:)
  integer(kind=8), intent(in)  :: coeffh_ind1 (0:)
  integer(kind=8), intent(in)  :: coeffh_ind2 (0:)
  integer(kind=8), intent(in)  :: coeffi_ind3 (0:)
  integer(kind=8), intent(in)  :: x_his_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_int_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts1 (0:,0:)
  real(kind=8), intent(in)  :: wts2 (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: lambdas (0:,0:,0:)
  integer(kind=8)  :: n_pts3  
  integer(kind=8)  :: n_his1  
  integer(kind=8)  :: n_his2  
  integer(kind=8)  :: nq1  
  integer(kind=8)  :: nq2  
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  real(kind=8)  :: wvol  



  n_pts3 = 2_8*(p(2_8) - 1_8) + 1_8


  n_his1 = 2_8*p(0_8)
  n_his2 = 2_8*p(1_8)


  nq1 = p(0_8) + 1_8
  nq2 = p(1_8) + 1_8


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        do il1 = 0, n_his1 - 1_8, 1
          do il2 = 0, n_his2 - 1_8, 1
            do il3 = 0, n_pts3 - 1_8, 1


              f_int = 0.0d0


              do q1 = 0, nq1 - 1_8, 1
                do q2 = 0, nq2 - 1_8, 1
                  wvol = wts1(x_his_ind1(i1, il1), q1)*wts2(x_his_ind2( &
      i2, il2), q2)
                  f_int = f_int + wvol*mat_f(x_his_ind1(i1, il1), q1, &
      x_his_ind2(i2, il2), q2, x_int_ind3(i3, il3))


                end do

              end do

              lambdas(i1, i2, i3) = ((f_int*coeff_i3(coeffi_ind3(i3), &
      il3))*coeff_h2(coeffh_ind2(i2), il2))*coeff_h1(coeffh_ind1(i1), &
      il1) + lambdas(i1, i2, i3)




            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi3_3d(n, p, coeff_h1, coeff_h2, coeff_h3, coeffh_ind1 &
      , coeffh_ind2, coeffh_ind3, x_his_ind1, x_his_ind2, x_his_ind3, &
      wts1, wts2, wts3, mat_f, lambdas)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  integer(kind=8), intent(in)  :: p (0:)
  real(kind=8), intent(in)  :: coeff_h1 (0:,0:)
  real(kind=8), intent(in)  :: coeff_h2 (0:,0:)
  real(kind=8), intent(in)  :: coeff_h3 (0:,0:)
  integer(kind=8), intent(in)  :: coeffh_ind1 (0:)
  integer(kind=8), intent(in)  :: coeffh_ind2 (0:)
  integer(kind=8), intent(in)  :: coeffh_ind3 (0:)
  integer(kind=8), intent(in)  :: x_his_ind1 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind2 (0:,0:)
  integer(kind=8), intent(in)  :: x_his_ind3 (0:,0:)
  real(kind=8), intent(in)  :: wts1 (0:,0:)
  real(kind=8), intent(in)  :: wts2 (0:,0:)
  real(kind=8), intent(in)  :: wts3 (0:,0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: lambdas (0:,0:,0:)
  integer(kind=8)  :: n_his1  
  integer(kind=8)  :: n_his2  
  integer(kind=8)  :: n_his3  
  integer(kind=8)  :: nq1  
  integer(kind=8)  :: nq2  
  integer(kind=8)  :: nq3  
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  real(kind=8)  :: f_int  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  
  real(kind=8)  :: wvol  



  n_his1 = 2_8*p(0_8)
  n_his2 = 2_8*p(1_8)
  n_his3 = 2_8*p(2_8)


  nq1 = p(0_8) + 1_8
  nq2 = p(1_8) + 1_8
  nq3 = p(2_8) + 1_8


  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        do il1 = 0, n_his1 - 1_8, 1
          do il2 = 0, n_his2 - 1_8, 1
            do il3 = 0, n_his3 - 1_8, 1


              f_int = 0.0d0


              do q1 = 0, nq1 - 1_8, 1
                do q2 = 0, nq2 - 1_8, 1
                  do q3 = 0, nq3 - 1_8, 1
                    wvol = (wts2(x_his_ind2(i2, il2), q2)*wts3( &
      x_his_ind3(i3, il3), q3))*wts1(x_his_ind1(i1, il1), q1)
                    f_int = f_int + wvol*mat_f(x_his_ind1(i1, il1), q1, &
      x_his_ind2(i2, il2), q2, x_his_ind3(i3, il3), q3)


                  end do

                end do

              end do

              lambdas(i1, i2, i3) = ((f_int*coeff_h3(coeffh_ind3(i3), &
      il3))*coeff_h2(coeffh_ind2(i2), il2))*coeff_h1(coeffh_ind1(i1), &
      il1) + lambdas(i1, i2, i3)
            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

end module