subroutine kernel_pi0 (n1, n2, n3, pl1, pl2, pl3, n0_b1, n1_b1, n2_b1, &
      n3_b1, b1, n0_b2, n1_b2, n2_b2, n3_b2, b2, n0_b3, n1_b3, n2_b3, &
      n3_b3, b3, n0_mat, n1_mat, n2_mat, mat, n0_rhs, n1_rhs, n2_rhs, &
      n3_rhs, n4_rhs, n5_rhs, rhs)

  use kernels_projectors_global_mhd, only: mod_kernel_pi0 => kernel_pi0
  implicit none
  integer(kind=8), intent(in)  :: n1 
  integer(kind=8), intent(in)  :: n2 
  integer(kind=8), intent(in)  :: n3 
  integer(kind=8), intent(in)  :: pl1 
  integer(kind=8), intent(in)  :: pl2 
  integer(kind=8), intent(in)  :: pl3 
  integer(kind=4), intent(in)  :: n0_b1 
  integer(kind=4), intent(in)  :: n1_b1 
  integer(kind=4), intent(in)  :: n2_b1 
  integer(kind=4), intent(in)  :: n3_b1 
  real(kind=8), intent(in)  :: b1 (0:n0_b1-1,0:n1_b1-1,0:n2_b1-1,0: &
      n3_b1-1)
  integer(kind=4), intent(in)  :: n0_b2 
  integer(kind=4), intent(in)  :: n1_b2 
  integer(kind=4), intent(in)  :: n2_b2 
  integer(kind=4), intent(in)  :: n3_b2 
  real(kind=8), intent(in)  :: b2 (0:n0_b2-1,0:n1_b2-1,0:n2_b2-1,0: &
      n3_b2-1)
  integer(kind=4), intent(in)  :: n0_b3 
  integer(kind=4), intent(in)  :: n1_b3 
  integer(kind=4), intent(in)  :: n2_b3 
  integer(kind=4), intent(in)  :: n3_b3 
  real(kind=8), intent(in)  :: b3 (0:n0_b3-1,0:n1_b3-1,0:n2_b3-1,0: &
      n3_b3-1)
  integer(kind=4), intent(in)  :: n0_mat 
  integer(kind=4), intent(in)  :: n1_mat 
  integer(kind=4), intent(in)  :: n2_mat 
  real(kind=8), intent(in)  :: mat (0:n0_mat-1,0:n1_mat-1,0:n2_mat-1)
  integer(kind=4), intent(in)  :: n0_rhs 
  integer(kind=4), intent(in)  :: n1_rhs 
  integer(kind=4), intent(in)  :: n2_rhs 
  integer(kind=4), intent(in)  :: n3_rhs 
  integer(kind=4), intent(in)  :: n4_rhs 
  integer(kind=4), intent(in)  :: n5_rhs 
  real(kind=8), intent(inout)  :: rhs (0:n0_rhs-1,0:n1_rhs-1,0:n2_rhs-1, &
      0:n3_rhs-1,0:n4_rhs-1,0:n5_rhs-1)

  call mod_kernel_pi0(n1,n2,n3,pl1,pl2,pl3,b1,b2,b3,mat,rhs)
end subroutine

subroutine kernel_pi1_1 (n1, n2, n3, pl1, pl2, pl3, n0_ies_1, ies_1, &
      n0_il_add_1, il_add_1, nq1, n0_w1, n1_w1, w1, n0_b1, n1_b1, n2_b1 &
      , n3_b1, b1, n0_b2, n1_b2, n2_b2, n3_b2, b2, n0_b3, n1_b3, n2_b3, &
      n3_b3, b3, n0_mat, n1_mat, n2_mat, mat, n0_rhs_1, n1_rhs_1, &
      n2_rhs_1, n3_rhs_1, n4_rhs_1, n5_rhs_1, rhs_1)

  use kernels_projectors_global_mhd, only: mod_kernel_pi1_1 => &
      kernel_pi1_1
  implicit none
  integer(kind=8), intent(in)  :: n1 
  integer(kind=8), intent(in)  :: n2 
  integer(kind=8), intent(in)  :: n3 
  integer(kind=8), intent(in)  :: pl1 
  integer(kind=8), intent(in)  :: pl2 
  integer(kind=8), intent(in)  :: pl3 
  integer(kind=4), intent(in)  :: n0_ies_1 
  integer(kind=8), intent(in)  :: ies_1 (0:n0_ies_1-1)
  integer(kind=4), intent(in)  :: n0_il_add_1 
  integer(kind=8), intent(in)  :: il_add_1 (0:n0_il_add_1-1)
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_b1 
  integer(kind=4), intent(in)  :: n1_b1 
  integer(kind=4), intent(in)  :: n2_b1 
  integer(kind=4), intent(in)  :: n3_b1 
  real(kind=8), intent(in)  :: b1 (0:n0_b1-1,0:n1_b1-1,0:n2_b1-1,0: &
      n3_b1-1)
  integer(kind=4), intent(in)  :: n0_b2 
  integer(kind=4), intent(in)  :: n1_b2 
  integer(kind=4), intent(in)  :: n2_b2 
  integer(kind=4), intent(in)  :: n3_b2 
  real(kind=8), intent(in)  :: b2 (0:n0_b2-1,0:n1_b2-1,0:n2_b2-1,0: &
      n3_b2-1)
  integer(kind=4), intent(in)  :: n0_b3 
  integer(kind=4), intent(in)  :: n1_b3 
  integer(kind=4), intent(in)  :: n2_b3 
  integer(kind=4), intent(in)  :: n3_b3 
  real(kind=8), intent(in)  :: b3 (0:n0_b3-1,0:n1_b3-1,0:n2_b3-1,0: &
      n3_b3-1)
  integer(kind=4), intent(in)  :: n0_mat 
  integer(kind=4), intent(in)  :: n1_mat 
  integer(kind=4), intent(in)  :: n2_mat 
  real(kind=8), intent(in)  :: mat (0:n0_mat-1,0:n1_mat-1,0:n2_mat-1)
  integer(kind=4), intent(in)  :: n0_rhs_1 
  integer(kind=4), intent(in)  :: n1_rhs_1 
  integer(kind=4), intent(in)  :: n2_rhs_1 
  integer(kind=4), intent(in)  :: n3_rhs_1 
  integer(kind=4), intent(in)  :: n4_rhs_1 
  integer(kind=4), intent(in)  :: n5_rhs_1 
  real(kind=8), intent(inout)  :: rhs_1 (0:n0_rhs_1-1,0:n1_rhs_1-1,0: &
      n2_rhs_1-1,0:n3_rhs_1-1,0:n4_rhs_1-1,0:n5_rhs_1-1)

  call mod_kernel_pi1_1(n1,n2,n3,pl1,pl2,pl3,ies_1,il_add_1,nq1,w1,b1,b2 &
      ,b3,mat,rhs_1)
end subroutine

subroutine kernel_pi1_2 (n1, n2, n3, pl1, pl2, pl3, n0_ies_2, ies_2, &
      n0_il_add_2, il_add_2, nq2, n0_w2, n1_w2, w2, n0_b1, n1_b1, n2_b1 &
      , n3_b1, b1, n0_b2, n1_b2, n2_b2, n3_b2, b2, n0_b3, n1_b3, n2_b3, &
      n3_b3, b3, n0_mat, n1_mat, n2_mat, mat, n0_rhs_2, n1_rhs_2, &
      n2_rhs_2, n3_rhs_2, n4_rhs_2, n5_rhs_2, rhs_2)

  use kernels_projectors_global_mhd, only: mod_kernel_pi1_2 => &
      kernel_pi1_2
  implicit none
  integer(kind=8), intent(in)  :: n1 
  integer(kind=8), intent(in)  :: n2 
  integer(kind=8), intent(in)  :: n3 
  integer(kind=8), intent(in)  :: pl1 
  integer(kind=8), intent(in)  :: pl2 
  integer(kind=8), intent(in)  :: pl3 
  integer(kind=4), intent(in)  :: n0_ies_2 
  integer(kind=8), intent(in)  :: ies_2 (0:n0_ies_2-1)
  integer(kind=4), intent(in)  :: n0_il_add_2 
  integer(kind=8), intent(in)  :: il_add_2 (0:n0_il_add_2-1)
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_b1 
  integer(kind=4), intent(in)  :: n1_b1 
  integer(kind=4), intent(in)  :: n2_b1 
  integer(kind=4), intent(in)  :: n3_b1 
  real(kind=8), intent(in)  :: b1 (0:n0_b1-1,0:n1_b1-1,0:n2_b1-1,0: &
      n3_b1-1)
  integer(kind=4), intent(in)  :: n0_b2 
  integer(kind=4), intent(in)  :: n1_b2 
  integer(kind=4), intent(in)  :: n2_b2 
  integer(kind=4), intent(in)  :: n3_b2 
  real(kind=8), intent(in)  :: b2 (0:n0_b2-1,0:n1_b2-1,0:n2_b2-1,0: &
      n3_b2-1)
  integer(kind=4), intent(in)  :: n0_b3 
  integer(kind=4), intent(in)  :: n1_b3 
  integer(kind=4), intent(in)  :: n2_b3 
  integer(kind=4), intent(in)  :: n3_b3 
  real(kind=8), intent(in)  :: b3 (0:n0_b3-1,0:n1_b3-1,0:n2_b3-1,0: &
      n3_b3-1)
  integer(kind=4), intent(in)  :: n0_mat 
  integer(kind=4), intent(in)  :: n1_mat 
  integer(kind=4), intent(in)  :: n2_mat 
  real(kind=8), intent(in)  :: mat (0:n0_mat-1,0:n1_mat-1,0:n2_mat-1)
  integer(kind=4), intent(in)  :: n0_rhs_2 
  integer(kind=4), intent(in)  :: n1_rhs_2 
  integer(kind=4), intent(in)  :: n2_rhs_2 
  integer(kind=4), intent(in)  :: n3_rhs_2 
  integer(kind=4), intent(in)  :: n4_rhs_2 
  integer(kind=4), intent(in)  :: n5_rhs_2 
  real(kind=8), intent(inout)  :: rhs_2 (0:n0_rhs_2-1,0:n1_rhs_2-1,0: &
      n2_rhs_2-1,0:n3_rhs_2-1,0:n4_rhs_2-1,0:n5_rhs_2-1)

  call mod_kernel_pi1_2(n1,n2,n3,pl1,pl2,pl3,ies_2,il_add_2,nq2,w2,b1,b2 &
      ,b3,mat,rhs_2)
end subroutine

subroutine kernel_pi1_3 (n1, n2, n3, pl1, pl2, pl3, n0_ies_3, ies_3, &
      n0_il_add_3, il_add_3, nq3, n0_w3, n1_w3, w3, n0_b1, n1_b1, n2_b1 &
      , n3_b1, b1, n0_b2, n1_b2, n2_b2, n3_b2, b2, n0_b3, n1_b3, n2_b3, &
      n3_b3, b3, n0_mat, n1_mat, n2_mat, mat, n0_rhs_3, n1_rhs_3, &
      n2_rhs_3, n3_rhs_3, n4_rhs_3, n5_rhs_3, rhs_3)

  use kernels_projectors_global_mhd, only: mod_kernel_pi1_3 => &
      kernel_pi1_3
  implicit none
  integer(kind=8), intent(in)  :: n1 
  integer(kind=8), intent(in)  :: n2 
  integer(kind=8), intent(in)  :: n3 
  integer(kind=8), intent(in)  :: pl1 
  integer(kind=8), intent(in)  :: pl2 
  integer(kind=8), intent(in)  :: pl3 
  integer(kind=4), intent(in)  :: n0_ies_3 
  integer(kind=8), intent(in)  :: ies_3 (0:n0_ies_3-1)
  integer(kind=4), intent(in)  :: n0_il_add_3 
  integer(kind=8), intent(in)  :: il_add_3 (0:n0_il_add_3-1)
  integer(kind=8), intent(in)  :: nq3 
  integer(kind=4), intent(in)  :: n0_w3 
  integer(kind=4), intent(in)  :: n1_w3 
  real(kind=8), intent(in)  :: w3 (0:n0_w3-1,0:n1_w3-1)
  integer(kind=4), intent(in)  :: n0_b1 
  integer(kind=4), intent(in)  :: n1_b1 
  integer(kind=4), intent(in)  :: n2_b1 
  integer(kind=4), intent(in)  :: n3_b1 
  real(kind=8), intent(in)  :: b1 (0:n0_b1-1,0:n1_b1-1,0:n2_b1-1,0: &
      n3_b1-1)
  integer(kind=4), intent(in)  :: n0_b2 
  integer(kind=4), intent(in)  :: n1_b2 
  integer(kind=4), intent(in)  :: n2_b2 
  integer(kind=4), intent(in)  :: n3_b2 
  real(kind=8), intent(in)  :: b2 (0:n0_b2-1,0:n1_b2-1,0:n2_b2-1,0: &
      n3_b2-1)
  integer(kind=4), intent(in)  :: n0_b3 
  integer(kind=4), intent(in)  :: n1_b3 
  integer(kind=4), intent(in)  :: n2_b3 
  integer(kind=4), intent(in)  :: n3_b3 
  real(kind=8), intent(in)  :: b3 (0:n0_b3-1,0:n1_b3-1,0:n2_b3-1,0: &
      n3_b3-1)
  integer(kind=4), intent(in)  :: n0_mat 
  integer(kind=4), intent(in)  :: n1_mat 
  integer(kind=4), intent(in)  :: n2_mat 
  real(kind=8), intent(in)  :: mat (0:n0_mat-1,0:n1_mat-1,0:n2_mat-1)
  integer(kind=4), intent(in)  :: n0_rhs_3 
  integer(kind=4), intent(in)  :: n1_rhs_3 
  integer(kind=4), intent(in)  :: n2_rhs_3 
  integer(kind=4), intent(in)  :: n3_rhs_3 
  integer(kind=4), intent(in)  :: n4_rhs_3 
  integer(kind=4), intent(in)  :: n5_rhs_3 
  real(kind=8), intent(inout)  :: rhs_3 (0:n0_rhs_3-1,0:n1_rhs_3-1,0: &
      n2_rhs_3-1,0:n3_rhs_3-1,0:n4_rhs_3-1,0:n5_rhs_3-1)

  call mod_kernel_pi1_3(n1,n2,n3,pl1,pl2,pl3,ies_3,il_add_3,nq3,w3,b1,b2 &
      ,b3,mat,rhs_3)
end subroutine

subroutine kernel_pi2_1 (n1, n2, n3, pl1, pl2, pl3, n0_ies_2, ies_2, &
      n0_ies_3, ies_3, n0_il_add_2, il_add_2, n0_il_add_3, il_add_3, &
      nq2, nq3, n0_w2, n1_w2, w2, n0_w3, n1_w3, w3, n0_b1, n1_b1, n2_b1 &
      , n3_b1, b1, n0_b2, n1_b2, n2_b2, n3_b2, b2, n0_b3, n1_b3, n2_b3, &
      n3_b3, b3, n0_mat, n1_mat, n2_mat, mat, n0_rhs_1, n1_rhs_1, &
      n2_rhs_1, n3_rhs_1, n4_rhs_1, n5_rhs_1, rhs_1)

  use kernels_projectors_global_mhd, only: mod_kernel_pi2_1 => &
      kernel_pi2_1
  implicit none
  integer(kind=8), intent(in)  :: n1 
  integer(kind=8), intent(in)  :: n2 
  integer(kind=8), intent(in)  :: n3 
  integer(kind=8), intent(in)  :: pl1 
  integer(kind=8), intent(in)  :: pl2 
  integer(kind=8), intent(in)  :: pl3 
  integer(kind=4), intent(in)  :: n0_ies_2 
  integer(kind=8), intent(in)  :: ies_2 (0:n0_ies_2-1)
  integer(kind=4), intent(in)  :: n0_ies_3 
  integer(kind=8), intent(in)  :: ies_3 (0:n0_ies_3-1)
  integer(kind=4), intent(in)  :: n0_il_add_2 
  integer(kind=8), intent(in)  :: il_add_2 (0:n0_il_add_2-1)
  integer(kind=4), intent(in)  :: n0_il_add_3 
  integer(kind=8), intent(in)  :: il_add_3 (0:n0_il_add_3-1)
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=8), intent(in)  :: nq3 
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_w3 
  integer(kind=4), intent(in)  :: n1_w3 
  real(kind=8), intent(in)  :: w3 (0:n0_w3-1,0:n1_w3-1)
  integer(kind=4), intent(in)  :: n0_b1 
  integer(kind=4), intent(in)  :: n1_b1 
  integer(kind=4), intent(in)  :: n2_b1 
  integer(kind=4), intent(in)  :: n3_b1 
  real(kind=8), intent(in)  :: b1 (0:n0_b1-1,0:n1_b1-1,0:n2_b1-1,0: &
      n3_b1-1)
  integer(kind=4), intent(in)  :: n0_b2 
  integer(kind=4), intent(in)  :: n1_b2 
  integer(kind=4), intent(in)  :: n2_b2 
  integer(kind=4), intent(in)  :: n3_b2 
  real(kind=8), intent(in)  :: b2 (0:n0_b2-1,0:n1_b2-1,0:n2_b2-1,0: &
      n3_b2-1)
  integer(kind=4), intent(in)  :: n0_b3 
  integer(kind=4), intent(in)  :: n1_b3 
  integer(kind=4), intent(in)  :: n2_b3 
  integer(kind=4), intent(in)  :: n3_b3 
  real(kind=8), intent(in)  :: b3 (0:n0_b3-1,0:n1_b3-1,0:n2_b3-1,0: &
      n3_b3-1)
  integer(kind=4), intent(in)  :: n0_mat 
  integer(kind=4), intent(in)  :: n1_mat 
  integer(kind=4), intent(in)  :: n2_mat 
  real(kind=8), intent(in)  :: mat (0:n0_mat-1,0:n1_mat-1,0:n2_mat-1)
  integer(kind=4), intent(in)  :: n0_rhs_1 
  integer(kind=4), intent(in)  :: n1_rhs_1 
  integer(kind=4), intent(in)  :: n2_rhs_1 
  integer(kind=4), intent(in)  :: n3_rhs_1 
  integer(kind=4), intent(in)  :: n4_rhs_1 
  integer(kind=4), intent(in)  :: n5_rhs_1 
  real(kind=8), intent(inout)  :: rhs_1 (0:n0_rhs_1-1,0:n1_rhs_1-1,0: &
      n2_rhs_1-1,0:n3_rhs_1-1,0:n4_rhs_1-1,0:n5_rhs_1-1)

  call mod_kernel_pi2_1(n1,n2,n3,pl1,pl2,pl3,ies_2,ies_3,il_add_2, &
      il_add_3,nq2,nq3,w2,w3,b1,b2,b3,mat,rhs_1)
end subroutine

subroutine kernel_pi2_2 (n1, n2, n3, pl1, pl2, pl3, n0_ies_1, ies_1, &
      n0_ies_3, ies_3, n0_il_add_1, il_add_1, n0_il_add_3, il_add_3, &
      nq1, nq3, n0_w1, n1_w1, w1, n0_w3, n1_w3, w3, n0_b1, n1_b1, n2_b1 &
      , n3_b1, b1, n0_b2, n1_b2, n2_b2, n3_b2, b2, n0_b3, n1_b3, n2_b3, &
      n3_b3, b3, n0_mat, n1_mat, n2_mat, mat, n0_rhs_2, n1_rhs_2, &
      n2_rhs_2, n3_rhs_2, n4_rhs_2, n5_rhs_2, rhs_2)

  use kernels_projectors_global_mhd, only: mod_kernel_pi2_2 => &
      kernel_pi2_2
  implicit none
  integer(kind=8), intent(in)  :: n1 
  integer(kind=8), intent(in)  :: n2 
  integer(kind=8), intent(in)  :: n3 
  integer(kind=8), intent(in)  :: pl1 
  integer(kind=8), intent(in)  :: pl2 
  integer(kind=8), intent(in)  :: pl3 
  integer(kind=4), intent(in)  :: n0_ies_1 
  integer(kind=8), intent(in)  :: ies_1 (0:n0_ies_1-1)
  integer(kind=4), intent(in)  :: n0_ies_3 
  integer(kind=8), intent(in)  :: ies_3 (0:n0_ies_3-1)
  integer(kind=4), intent(in)  :: n0_il_add_1 
  integer(kind=8), intent(in)  :: il_add_1 (0:n0_il_add_1-1)
  integer(kind=4), intent(in)  :: n0_il_add_3 
  integer(kind=8), intent(in)  :: il_add_3 (0:n0_il_add_3-1)
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq3 
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_w3 
  integer(kind=4), intent(in)  :: n1_w3 
  real(kind=8), intent(in)  :: w3 (0:n0_w3-1,0:n1_w3-1)
  integer(kind=4), intent(in)  :: n0_b1 
  integer(kind=4), intent(in)  :: n1_b1 
  integer(kind=4), intent(in)  :: n2_b1 
  integer(kind=4), intent(in)  :: n3_b1 
  real(kind=8), intent(in)  :: b1 (0:n0_b1-1,0:n1_b1-1,0:n2_b1-1,0: &
      n3_b1-1)
  integer(kind=4), intent(in)  :: n0_b2 
  integer(kind=4), intent(in)  :: n1_b2 
  integer(kind=4), intent(in)  :: n2_b2 
  integer(kind=4), intent(in)  :: n3_b2 
  real(kind=8), intent(in)  :: b2 (0:n0_b2-1,0:n1_b2-1,0:n2_b2-1,0: &
      n3_b2-1)
  integer(kind=4), intent(in)  :: n0_b3 
  integer(kind=4), intent(in)  :: n1_b3 
  integer(kind=4), intent(in)  :: n2_b3 
  integer(kind=4), intent(in)  :: n3_b3 
  real(kind=8), intent(in)  :: b3 (0:n0_b3-1,0:n1_b3-1,0:n2_b3-1,0: &
      n3_b3-1)
  integer(kind=4), intent(in)  :: n0_mat 
  integer(kind=4), intent(in)  :: n1_mat 
  integer(kind=4), intent(in)  :: n2_mat 
  real(kind=8), intent(in)  :: mat (0:n0_mat-1,0:n1_mat-1,0:n2_mat-1)
  integer(kind=4), intent(in)  :: n0_rhs_2 
  integer(kind=4), intent(in)  :: n1_rhs_2 
  integer(kind=4), intent(in)  :: n2_rhs_2 
  integer(kind=4), intent(in)  :: n3_rhs_2 
  integer(kind=4), intent(in)  :: n4_rhs_2 
  integer(kind=4), intent(in)  :: n5_rhs_2 
  real(kind=8), intent(inout)  :: rhs_2 (0:n0_rhs_2-1,0:n1_rhs_2-1,0: &
      n2_rhs_2-1,0:n3_rhs_2-1,0:n4_rhs_2-1,0:n5_rhs_2-1)

  call mod_kernel_pi2_2(n1,n2,n3,pl1,pl2,pl3,ies_1,ies_3,il_add_1, &
      il_add_3,nq1,nq3,w1,w3,b1,b2,b3,mat,rhs_2)
end subroutine

subroutine kernel_pi2_3 (n1, n2, n3, pl1, pl2, pl3, n0_ies_1, ies_1, &
      n0_ies_2, ies_2, n0_il_add_1, il_add_1, n0_il_add_2, il_add_2, &
      nq1, nq2, n0_w1, n1_w1, w1, n0_w2, n1_w2, w2, n0_b1, n1_b1, n2_b1 &
      , n3_b1, b1, n0_b2, n1_b2, n2_b2, n3_b2, b2, n0_b3, n1_b3, n2_b3, &
      n3_b3, b3, n0_mat, n1_mat, n2_mat, mat, n0_rhs_3, n1_rhs_3, &
      n2_rhs_3, n3_rhs_3, n4_rhs_3, n5_rhs_3, rhs_3)

  use kernels_projectors_global_mhd, only: mod_kernel_pi2_3 => &
      kernel_pi2_3
  implicit none
  integer(kind=8), intent(in)  :: n1 
  integer(kind=8), intent(in)  :: n2 
  integer(kind=8), intent(in)  :: n3 
  integer(kind=8), intent(in)  :: pl1 
  integer(kind=8), intent(in)  :: pl2 
  integer(kind=8), intent(in)  :: pl3 
  integer(kind=4), intent(in)  :: n0_ies_1 
  integer(kind=8), intent(in)  :: ies_1 (0:n0_ies_1-1)
  integer(kind=4), intent(in)  :: n0_ies_2 
  integer(kind=8), intent(in)  :: ies_2 (0:n0_ies_2-1)
  integer(kind=4), intent(in)  :: n0_il_add_1 
  integer(kind=8), intent(in)  :: il_add_1 (0:n0_il_add_1-1)
  integer(kind=4), intent(in)  :: n0_il_add_2 
  integer(kind=8), intent(in)  :: il_add_2 (0:n0_il_add_2-1)
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_b1 
  integer(kind=4), intent(in)  :: n1_b1 
  integer(kind=4), intent(in)  :: n2_b1 
  integer(kind=4), intent(in)  :: n3_b1 
  real(kind=8), intent(in)  :: b1 (0:n0_b1-1,0:n1_b1-1,0:n2_b1-1,0: &
      n3_b1-1)
  integer(kind=4), intent(in)  :: n0_b2 
  integer(kind=4), intent(in)  :: n1_b2 
  integer(kind=4), intent(in)  :: n2_b2 
  integer(kind=4), intent(in)  :: n3_b2 
  real(kind=8), intent(in)  :: b2 (0:n0_b2-1,0:n1_b2-1,0:n2_b2-1,0: &
      n3_b2-1)
  integer(kind=4), intent(in)  :: n0_b3 
  integer(kind=4), intent(in)  :: n1_b3 
  integer(kind=4), intent(in)  :: n2_b3 
  integer(kind=4), intent(in)  :: n3_b3 
  real(kind=8), intent(in)  :: b3 (0:n0_b3-1,0:n1_b3-1,0:n2_b3-1,0: &
      n3_b3-1)
  integer(kind=4), intent(in)  :: n0_mat 
  integer(kind=4), intent(in)  :: n1_mat 
  integer(kind=4), intent(in)  :: n2_mat 
  real(kind=8), intent(in)  :: mat (0:n0_mat-1,0:n1_mat-1,0:n2_mat-1)
  integer(kind=4), intent(in)  :: n0_rhs_3 
  integer(kind=4), intent(in)  :: n1_rhs_3 
  integer(kind=4), intent(in)  :: n2_rhs_3 
  integer(kind=4), intent(in)  :: n3_rhs_3 
  integer(kind=4), intent(in)  :: n4_rhs_3 
  integer(kind=4), intent(in)  :: n5_rhs_3 
  real(kind=8), intent(inout)  :: rhs_3 (0:n0_rhs_3-1,0:n1_rhs_3-1,0: &
      n2_rhs_3-1,0:n3_rhs_3-1,0:n4_rhs_3-1,0:n5_rhs_3-1)

  call mod_kernel_pi2_3(n1,n2,n3,pl1,pl2,pl3,ies_1,ies_2,il_add_1, &
      il_add_2,nq1,nq2,w1,w2,b1,b2,b3,mat,rhs_3)
end subroutine