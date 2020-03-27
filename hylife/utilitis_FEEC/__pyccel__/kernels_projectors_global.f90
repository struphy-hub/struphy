module kernels_projectors_global

implicit none




contains

!........................................
subroutine kernel_int_1d(nq1, w1, mat_f, f_loc) 

  implicit none
  integer(kind=8), value  :: nq1
  real(kind=8), intent(in)  :: w1 (0:)
  real(kind=8), intent(in)  :: mat_f (0:)
  real(kind=8), intent(inout)  :: f_loc (0:)
  integer(kind=8)  :: q1  



  f_loc(:) = 0.0d0


  do q1 = 0, nq1 - 1_8, 1
    f_loc(:) = f_loc(:) + mat_f(q1)*w1(q1)








  end do

end subroutine
!........................................

!........................................
subroutine kernel_int_2d(nq1, nq2, w1, w2, mat_f, f_loc) 

  implicit none
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq2
  real(kind=8), intent(in)  :: w1 (0:)
  real(kind=8), intent(in)  :: w2 (0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:)
  real(kind=8), intent(inout)  :: f_loc (0:)
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  



  f_loc(:) = 0.0d0


  do q1 = 0, nq1 - 1_8, 1
    do q2 = 0, nq2 - 1_8, 1
      f_loc(:) = (mat_f(q1, q2)*w2(q2))*w1(q1) + f_loc(:)






    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_int_3d(nq1, nq2, nq3, w1, w2, w3, mat_f, f_loc) 

  implicit none
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq2
  integer(kind=8), value  :: nq3
  real(kind=8), intent(in)  :: w1 (0:)
  real(kind=8), intent(in)  :: w2 (0:)
  real(kind=8), intent(in)  :: w3 (0:)
  real(kind=8), intent(in)  :: mat_f (0:,0:,0:)
  real(kind=8), intent(inout)  :: f_loc (0:)
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  



  f_loc(:) = 0.0d0


  do q1 = 0, nq1 - 1_8, 1
    do q2 = 0, nq2 - 1_8, 1
      do q3 = 0, nq3 - 1_8, 1
        f_loc(:) = ((mat_f(q1, q2, q3)*w3(q3))*w2(q2))*w1(q1) + f_loc(:)
      end do

    end do

  end do

end subroutine
!........................................

end module