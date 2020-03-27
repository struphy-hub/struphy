subroutine kernel_int_1d (nq1, n0_w1, w1, n0_mat_f, mat_f, n0_f_loc, &
      f_loc)

  use kernels_projectors_global, only: mod_kernel_int_1d => &
      kernel_int_1d
  implicit none
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=4), intent(in)  :: n0_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1)
  integer(kind=4), intent(in)  :: n0_f_loc 
  real(kind=8), intent(inout)  :: f_loc (0:n0_f_loc-1)

  call mod_kernel_int_1d(nq1,w1,mat_f,f_loc)
end subroutine

subroutine kernel_int_2d (nq1, nq2, n0_w1, w1, n0_w2, w2, n0_mat_f, &
      n1_mat_f, mat_f, n0_f_loc, f_loc)

  use kernels_projectors_global, only: mod_kernel_int_2d => &
      kernel_int_2d
  implicit none
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=4), intent(in)  :: n0_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1)
  integer(kind=4), intent(in)  :: n0_f_loc 
  real(kind=8), intent(inout)  :: f_loc (0:n0_f_loc-1)

  call mod_kernel_int_2d(nq1,nq2,w1,w2,mat_f,f_loc)
end subroutine

subroutine kernel_int_3d (nq1, nq2, nq3, n0_w1, w1, n0_w2, w2, n0_w3, w3 &
      , n0_mat_f, n1_mat_f, n2_mat_f, mat_f, n0_f_loc, f_loc)

  use kernels_projectors_global, only: mod_kernel_int_3d => &
      kernel_int_3d
  implicit none
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=8), intent(in)  :: nq3 
  integer(kind=4), intent(in)  :: n0_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1)
  integer(kind=4), intent(in)  :: n0_w3 
  real(kind=8), intent(in)  :: w3 (0:n0_w3-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1)
  integer(kind=4), intent(in)  :: n0_f_loc 
  real(kind=8), intent(inout)  :: f_loc (0:n0_f_loc-1)

  call mod_kernel_int_3d(nq1,nq2,nq3,w1,w2,w3,mat_f,f_loc)
end subroutine