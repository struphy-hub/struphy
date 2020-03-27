subroutine kernel_pi0 (n0_n, n, n0_n_int, n_int, n0_n_nvbf, n_nvbf, &
      n0_i_glo1, n1_i_glo1, i_glo1, n0_i_glo2, n1_i_glo2, i_glo2, &
      n0_i_glo3, n1_i_glo3, i_glo3, n0_c_loc1, n1_c_loc1, c_loc1, &
      n0_c_loc2, n1_c_loc2, c_loc2, n0_c_loc3, n1_c_loc3, c_loc3, &
      n0_coeff1, n1_coeff1, coeff1, n0_coeff2, n1_coeff2, coeff2, &
      n0_coeff3, n1_coeff3, coeff3, n0_coeff_ind1, coeff_ind1, &
      n0_coeff_ind2, coeff_ind2, n0_coeff_ind3, coeff_ind3, n0_bs1, &
      n1_bs1, bs1, n0_bs2, n1_bs2, bs2, n0_bs3, n1_bs3, bs3, &
      n0_x_int_ind1, n1_x_int_ind1, x_int_ind1, n0_x_int_ind2, &
      n1_x_int_ind2, x_int_ind2, n0_x_int_ind3, n1_x_int_ind3, &
      x_int_ind3, n0_tau, n1_tau, n2_tau, n3_tau, n4_tau, n5_tau, tau, &
      n0_mat_eq, n1_mat_eq, n2_mat_eq, mat_eq)

  use kernels_projectors_local_mhd, only: mod_kernel_pi0 => kernel_pi0
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_n_int 
  integer(kind=8), intent(in)  :: n_int (0:n0_n_int-1)
  integer(kind=4), intent(in)  :: n0_n_nvbf 
  integer(kind=8), intent(in)  :: n_nvbf (0:n0_n_nvbf-1)
  integer(kind=4), intent(in)  :: n0_i_glo1 
  integer(kind=4), intent(in)  :: n1_i_glo1 
  integer(kind=8), intent(in)  :: i_glo1 (0:n0_i_glo1-1,0:n1_i_glo1-1)
  integer(kind=4), intent(in)  :: n0_i_glo2 
  integer(kind=4), intent(in)  :: n1_i_glo2 
  integer(kind=8), intent(in)  :: i_glo2 (0:n0_i_glo2-1,0:n1_i_glo2-1)
  integer(kind=4), intent(in)  :: n0_i_glo3 
  integer(kind=4), intent(in)  :: n1_i_glo3 
  integer(kind=8), intent(in)  :: i_glo3 (0:n0_i_glo3-1,0:n1_i_glo3-1)
  integer(kind=4), intent(in)  :: n0_c_loc1 
  integer(kind=4), intent(in)  :: n1_c_loc1 
  integer(kind=8), intent(in)  :: c_loc1 (0:n0_c_loc1-1,0:n1_c_loc1-1)
  integer(kind=4), intent(in)  :: n0_c_loc2 
  integer(kind=4), intent(in)  :: n1_c_loc2 
  integer(kind=8), intent(in)  :: c_loc2 (0:n0_c_loc2-1,0:n1_c_loc2-1)
  integer(kind=4), intent(in)  :: n0_c_loc3 
  integer(kind=4), intent(in)  :: n1_c_loc3 
  integer(kind=8), intent(in)  :: c_loc3 (0:n0_c_loc3-1,0:n1_c_loc3-1)
  integer(kind=4), intent(in)  :: n0_coeff1 
  integer(kind=4), intent(in)  :: n1_coeff1 
  real(kind=8), intent(in)  :: coeff1 (0:n0_coeff1-1,0:n1_coeff1-1)
  integer(kind=4), intent(in)  :: n0_coeff2 
  integer(kind=4), intent(in)  :: n1_coeff2 
  real(kind=8), intent(in)  :: coeff2 (0:n0_coeff2-1,0:n1_coeff2-1)
  integer(kind=4), intent(in)  :: n0_coeff3 
  integer(kind=4), intent(in)  :: n1_coeff3 
  real(kind=8), intent(in)  :: coeff3 (0:n0_coeff3-1,0:n1_coeff3-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind1 
  integer(kind=8), intent(in)  :: coeff_ind1 (0:n0_coeff_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind2 
  integer(kind=8), intent(in)  :: coeff_ind2 (0:n0_coeff_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind3 
  integer(kind=8), intent(in)  :: coeff_ind3 (0:n0_coeff_ind3-1)
  integer(kind=4), intent(in)  :: n0_bs1 
  integer(kind=4), intent(in)  :: n1_bs1 
  real(kind=8), intent(in)  :: bs1 (0:n0_bs1-1,0:n1_bs1-1)
  integer(kind=4), intent(in)  :: n0_bs2 
  integer(kind=4), intent(in)  :: n1_bs2 
  real(kind=8), intent(in)  :: bs2 (0:n0_bs2-1,0:n1_bs2-1)
  integer(kind=4), intent(in)  :: n0_bs3 
  integer(kind=4), intent(in)  :: n1_bs3 
  real(kind=8), intent(in)  :: bs3 (0:n0_bs3-1,0:n1_bs3-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind1 
  integer(kind=4), intent(in)  :: n1_x_int_ind1 
  integer(kind=8), intent(in)  :: x_int_ind1 (0:n0_x_int_ind1-1,0: &
      n1_x_int_ind1-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind2 
  integer(kind=4), intent(in)  :: n1_x_int_ind2 
  integer(kind=8), intent(in)  :: x_int_ind2 (0:n0_x_int_ind2-1,0: &
      n1_x_int_ind2-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind3 
  integer(kind=4), intent(in)  :: n1_x_int_ind3 
  integer(kind=8), intent(in)  :: x_int_ind3 (0:n0_x_int_ind3-1,0: &
      n1_x_int_ind3-1)
  integer(kind=4), intent(in)  :: n0_tau 
  integer(kind=4), intent(in)  :: n1_tau 
  integer(kind=4), intent(in)  :: n2_tau 
  integer(kind=4), intent(in)  :: n3_tau 
  integer(kind=4), intent(in)  :: n4_tau 
  integer(kind=4), intent(in)  :: n5_tau 
  real(kind=8), intent(inout)  :: tau (0:n0_tau-1,0:n1_tau-1,0:n2_tau-1, &
      0:n3_tau-1,0:n4_tau-1,0:n5_tau-1)
  integer(kind=4), intent(in)  :: n0_mat_eq 
  integer(kind=4), intent(in)  :: n1_mat_eq 
  integer(kind=4), intent(in)  :: n2_mat_eq 
  real(kind=8), intent(in)  :: mat_eq (0:n0_mat_eq-1,0:n1_mat_eq-1,0: &
      n2_mat_eq-1)

  call mod_kernel_pi0(n,n_int,n_nvbf,i_glo1,i_glo2,i_glo3,c_loc1,c_loc2, &
      c_loc3,coeff1,coeff2,coeff3,coeff_ind1,coeff_ind2,coeff_ind3,bs1, &
      bs2,bs3,x_int_ind1,x_int_ind2,x_int_ind3,tau,mat_eq)
end subroutine

subroutine kernel_pi1_1 (n0_n, n, n_quad1, n0_n_inthis, n_inthis, &
      n0_n_nvbf, n_nvbf, n0_i_glo1, n1_i_glo1, i_glo1, n0_i_glo2, &
      n1_i_glo2, i_glo2, n0_i_glo3, n1_i_glo3, i_glo3, n0_c_loc1, &
      n1_c_loc1, c_loc1, n0_c_loc2, n1_c_loc2, c_loc2, n0_c_loc3, &
      n1_c_loc3, c_loc3, n0_coeff1, n1_coeff1, coeff1, n0_coeff2, &
      n1_coeff2, coeff2, n0_coeff3, n1_coeff3, coeff3, n0_coeff_ind1, &
      coeff_ind1, n0_coeff_ind2, coeff_ind2, n0_coeff_ind3, coeff_ind3, &
      n0_bs1, n1_bs1, n2_bs1, bs1, n0_bs2, n1_bs2, bs2, n0_bs3, n1_bs3, &
      bs3, n0_x_his_ind1, n1_x_his_ind1, x_his_ind1, n0_x_int_ind2, &
      n1_x_int_ind2, x_int_ind2, n0_x_int_ind3, n1_x_int_ind3, &
      x_int_ind3, n0_wts1, n1_wts1, wts1, n0_tau, n1_tau, n2_tau, &
      n3_tau, n4_tau, n5_tau, tau, n0_mat_eq, n1_mat_eq, n2_mat_eq, &
      n3_mat_eq, mat_eq)

  use kernels_projectors_local_mhd, only: mod_kernel_pi1_1 => &
      kernel_pi1_1
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=8), intent(in)  :: n_quad1 
  integer(kind=4), intent(in)  :: n0_n_inthis 
  integer(kind=8), intent(in)  :: n_inthis (0:n0_n_inthis-1)
  integer(kind=4), intent(in)  :: n0_n_nvbf 
  integer(kind=8), intent(in)  :: n_nvbf (0:n0_n_nvbf-1)
  integer(kind=4), intent(in)  :: n0_i_glo1 
  integer(kind=4), intent(in)  :: n1_i_glo1 
  integer(kind=8), intent(in)  :: i_glo1 (0:n0_i_glo1-1,0:n1_i_glo1-1)
  integer(kind=4), intent(in)  :: n0_i_glo2 
  integer(kind=4), intent(in)  :: n1_i_glo2 
  integer(kind=8), intent(in)  :: i_glo2 (0:n0_i_glo2-1,0:n1_i_glo2-1)
  integer(kind=4), intent(in)  :: n0_i_glo3 
  integer(kind=4), intent(in)  :: n1_i_glo3 
  integer(kind=8), intent(in)  :: i_glo3 (0:n0_i_glo3-1,0:n1_i_glo3-1)
  integer(kind=4), intent(in)  :: n0_c_loc1 
  integer(kind=4), intent(in)  :: n1_c_loc1 
  integer(kind=8), intent(in)  :: c_loc1 (0:n0_c_loc1-1,0:n1_c_loc1-1)
  integer(kind=4), intent(in)  :: n0_c_loc2 
  integer(kind=4), intent(in)  :: n1_c_loc2 
  integer(kind=8), intent(in)  :: c_loc2 (0:n0_c_loc2-1,0:n1_c_loc2-1)
  integer(kind=4), intent(in)  :: n0_c_loc3 
  integer(kind=4), intent(in)  :: n1_c_loc3 
  integer(kind=8), intent(in)  :: c_loc3 (0:n0_c_loc3-1,0:n1_c_loc3-1)
  integer(kind=4), intent(in)  :: n0_coeff1 
  integer(kind=4), intent(in)  :: n1_coeff1 
  real(kind=8), intent(in)  :: coeff1 (0:n0_coeff1-1,0:n1_coeff1-1)
  integer(kind=4), intent(in)  :: n0_coeff2 
  integer(kind=4), intent(in)  :: n1_coeff2 
  real(kind=8), intent(in)  :: coeff2 (0:n0_coeff2-1,0:n1_coeff2-1)
  integer(kind=4), intent(in)  :: n0_coeff3 
  integer(kind=4), intent(in)  :: n1_coeff3 
  real(kind=8), intent(in)  :: coeff3 (0:n0_coeff3-1,0:n1_coeff3-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind1 
  integer(kind=8), intent(in)  :: coeff_ind1 (0:n0_coeff_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind2 
  integer(kind=8), intent(in)  :: coeff_ind2 (0:n0_coeff_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind3 
  integer(kind=8), intent(in)  :: coeff_ind3 (0:n0_coeff_ind3-1)
  integer(kind=4), intent(in)  :: n0_bs1 
  integer(kind=4), intent(in)  :: n1_bs1 
  integer(kind=4), intent(in)  :: n2_bs1 
  real(kind=8), intent(in)  :: bs1 (0:n0_bs1-1,0:n1_bs1-1,0:n2_bs1-1)
  integer(kind=4), intent(in)  :: n0_bs2 
  integer(kind=4), intent(in)  :: n1_bs2 
  real(kind=8), intent(in)  :: bs2 (0:n0_bs2-1,0:n1_bs2-1)
  integer(kind=4), intent(in)  :: n0_bs3 
  integer(kind=4), intent(in)  :: n1_bs3 
  real(kind=8), intent(in)  :: bs3 (0:n0_bs3-1,0:n1_bs3-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind1 
  integer(kind=4), intent(in)  :: n1_x_his_ind1 
  integer(kind=8), intent(in)  :: x_his_ind1 (0:n0_x_his_ind1-1,0: &
      n1_x_his_ind1-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind2 
  integer(kind=4), intent(in)  :: n1_x_int_ind2 
  integer(kind=8), intent(in)  :: x_int_ind2 (0:n0_x_int_ind2-1,0: &
      n1_x_int_ind2-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind3 
  integer(kind=4), intent(in)  :: n1_x_int_ind3 
  integer(kind=8), intent(in)  :: x_int_ind3 (0:n0_x_int_ind3-1,0: &
      n1_x_int_ind3-1)
  integer(kind=4), intent(in)  :: n0_wts1 
  integer(kind=4), intent(in)  :: n1_wts1 
  real(kind=8), intent(in)  :: wts1 (0:n0_wts1-1,0:n1_wts1-1)
  integer(kind=4), intent(in)  :: n0_tau 
  integer(kind=4), intent(in)  :: n1_tau 
  integer(kind=4), intent(in)  :: n2_tau 
  integer(kind=4), intent(in)  :: n3_tau 
  integer(kind=4), intent(in)  :: n4_tau 
  integer(kind=4), intent(in)  :: n5_tau 
  real(kind=8), intent(inout)  :: tau (0:n0_tau-1,0:n1_tau-1,0:n2_tau-1, &
      0:n3_tau-1,0:n4_tau-1,0:n5_tau-1)
  integer(kind=4), intent(in)  :: n0_mat_eq 
  integer(kind=4), intent(in)  :: n1_mat_eq 
  integer(kind=4), intent(in)  :: n2_mat_eq 
  integer(kind=4), intent(in)  :: n3_mat_eq 
  real(kind=8), intent(in)  :: mat_eq (0:n0_mat_eq-1,0:n1_mat_eq-1,0: &
      n2_mat_eq-1,0:n3_mat_eq-1)

  call mod_kernel_pi1_1(n,n_quad1,n_inthis,n_nvbf,i_glo1,i_glo2,i_glo3, &
      c_loc1,c_loc2,c_loc3,coeff1,coeff2,coeff3,coeff_ind1,coeff_ind2, &
      coeff_ind3,bs1,bs2,bs3,x_his_ind1,x_int_ind2,x_int_ind3,wts1,tau, &
      mat_eq)
end subroutine

subroutine kernel_pi1_2 (n0_n, n, n_quad2, n0_n_inthis, n_inthis, &
      n0_n_nvbf, n_nvbf, n0_i_glo1, n1_i_glo1, i_glo1, n0_i_glo2, &
      n1_i_glo2, i_glo2, n0_i_glo3, n1_i_glo3, i_glo3, n0_c_loc1, &
      n1_c_loc1, c_loc1, n0_c_loc2, n1_c_loc2, c_loc2, n0_c_loc3, &
      n1_c_loc3, c_loc3, n0_coeff1, n1_coeff1, coeff1, n0_coeff2, &
      n1_coeff2, coeff2, n0_coeff3, n1_coeff3, coeff3, n0_coeff_ind1, &
      coeff_ind1, n0_coeff_ind2, coeff_ind2, n0_coeff_ind3, coeff_ind3, &
      n0_bs1, n1_bs1, bs1, n0_bs2, n1_bs2, n2_bs2, bs2, n0_bs3, n1_bs3, &
      bs3, n0_x_int_ind1, n1_x_int_ind1, x_int_ind1, n0_x_his_ind2, &
      n1_x_his_ind2, x_his_ind2, n0_x_int_ind3, n1_x_int_ind3, &
      x_int_ind3, n0_wts2, n1_wts2, wts2, n0_tau, n1_tau, n2_tau, &
      n3_tau, n4_tau, n5_tau, tau, n0_mat_eq, n1_mat_eq, n2_mat_eq, &
      n3_mat_eq, mat_eq)

  use kernels_projectors_local_mhd, only: mod_kernel_pi1_2 => &
      kernel_pi1_2
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=8), intent(in)  :: n_quad2 
  integer(kind=4), intent(in)  :: n0_n_inthis 
  integer(kind=8), intent(in)  :: n_inthis (0:n0_n_inthis-1)
  integer(kind=4), intent(in)  :: n0_n_nvbf 
  integer(kind=8), intent(in)  :: n_nvbf (0:n0_n_nvbf-1)
  integer(kind=4), intent(in)  :: n0_i_glo1 
  integer(kind=4), intent(in)  :: n1_i_glo1 
  integer(kind=8), intent(in)  :: i_glo1 (0:n0_i_glo1-1,0:n1_i_glo1-1)
  integer(kind=4), intent(in)  :: n0_i_glo2 
  integer(kind=4), intent(in)  :: n1_i_glo2 
  integer(kind=8), intent(in)  :: i_glo2 (0:n0_i_glo2-1,0:n1_i_glo2-1)
  integer(kind=4), intent(in)  :: n0_i_glo3 
  integer(kind=4), intent(in)  :: n1_i_glo3 
  integer(kind=8), intent(in)  :: i_glo3 (0:n0_i_glo3-1,0:n1_i_glo3-1)
  integer(kind=4), intent(in)  :: n0_c_loc1 
  integer(kind=4), intent(in)  :: n1_c_loc1 
  integer(kind=8), intent(in)  :: c_loc1 (0:n0_c_loc1-1,0:n1_c_loc1-1)
  integer(kind=4), intent(in)  :: n0_c_loc2 
  integer(kind=4), intent(in)  :: n1_c_loc2 
  integer(kind=8), intent(in)  :: c_loc2 (0:n0_c_loc2-1,0:n1_c_loc2-1)
  integer(kind=4), intent(in)  :: n0_c_loc3 
  integer(kind=4), intent(in)  :: n1_c_loc3 
  integer(kind=8), intent(in)  :: c_loc3 (0:n0_c_loc3-1,0:n1_c_loc3-1)
  integer(kind=4), intent(in)  :: n0_coeff1 
  integer(kind=4), intent(in)  :: n1_coeff1 
  real(kind=8), intent(in)  :: coeff1 (0:n0_coeff1-1,0:n1_coeff1-1)
  integer(kind=4), intent(in)  :: n0_coeff2 
  integer(kind=4), intent(in)  :: n1_coeff2 
  real(kind=8), intent(in)  :: coeff2 (0:n0_coeff2-1,0:n1_coeff2-1)
  integer(kind=4), intent(in)  :: n0_coeff3 
  integer(kind=4), intent(in)  :: n1_coeff3 
  real(kind=8), intent(in)  :: coeff3 (0:n0_coeff3-1,0:n1_coeff3-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind1 
  integer(kind=8), intent(in)  :: coeff_ind1 (0:n0_coeff_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind2 
  integer(kind=8), intent(in)  :: coeff_ind2 (0:n0_coeff_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind3 
  integer(kind=8), intent(in)  :: coeff_ind3 (0:n0_coeff_ind3-1)
  integer(kind=4), intent(in)  :: n0_bs1 
  integer(kind=4), intent(in)  :: n1_bs1 
  real(kind=8), intent(in)  :: bs1 (0:n0_bs1-1,0:n1_bs1-1)
  integer(kind=4), intent(in)  :: n0_bs2 
  integer(kind=4), intent(in)  :: n1_bs2 
  integer(kind=4), intent(in)  :: n2_bs2 
  real(kind=8), intent(in)  :: bs2 (0:n0_bs2-1,0:n1_bs2-1,0:n2_bs2-1)
  integer(kind=4), intent(in)  :: n0_bs3 
  integer(kind=4), intent(in)  :: n1_bs3 
  real(kind=8), intent(in)  :: bs3 (0:n0_bs3-1,0:n1_bs3-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind1 
  integer(kind=4), intent(in)  :: n1_x_int_ind1 
  integer(kind=8), intent(in)  :: x_int_ind1 (0:n0_x_int_ind1-1,0: &
      n1_x_int_ind1-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind2 
  integer(kind=4), intent(in)  :: n1_x_his_ind2 
  integer(kind=8), intent(in)  :: x_his_ind2 (0:n0_x_his_ind2-1,0: &
      n1_x_his_ind2-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind3 
  integer(kind=4), intent(in)  :: n1_x_int_ind3 
  integer(kind=8), intent(in)  :: x_int_ind3 (0:n0_x_int_ind3-1,0: &
      n1_x_int_ind3-1)
  integer(kind=4), intent(in)  :: n0_wts2 
  integer(kind=4), intent(in)  :: n1_wts2 
  real(kind=8), intent(in)  :: wts2 (0:n0_wts2-1,0:n1_wts2-1)
  integer(kind=4), intent(in)  :: n0_tau 
  integer(kind=4), intent(in)  :: n1_tau 
  integer(kind=4), intent(in)  :: n2_tau 
  integer(kind=4), intent(in)  :: n3_tau 
  integer(kind=4), intent(in)  :: n4_tau 
  integer(kind=4), intent(in)  :: n5_tau 
  real(kind=8), intent(inout)  :: tau (0:n0_tau-1,0:n1_tau-1,0:n2_tau-1, &
      0:n3_tau-1,0:n4_tau-1,0:n5_tau-1)
  integer(kind=4), intent(in)  :: n0_mat_eq 
  integer(kind=4), intent(in)  :: n1_mat_eq 
  integer(kind=4), intent(in)  :: n2_mat_eq 
  integer(kind=4), intent(in)  :: n3_mat_eq 
  real(kind=8), intent(in)  :: mat_eq (0:n0_mat_eq-1,0:n1_mat_eq-1,0: &
      n2_mat_eq-1,0:n3_mat_eq-1)

  call mod_kernel_pi1_2(n,n_quad2,n_inthis,n_nvbf,i_glo1,i_glo2,i_glo3, &
      c_loc1,c_loc2,c_loc3,coeff1,coeff2,coeff3,coeff_ind1,coeff_ind2, &
      coeff_ind3,bs1,bs2,bs3,x_int_ind1,x_his_ind2,x_int_ind3,wts2,tau, &
      mat_eq)
end subroutine

subroutine kernel_pi1_3 (n0_n, n, n_quad3, n0_n_inthis, n_inthis, &
      n0_n_nvbf, n_nvbf, n0_i_glo1, n1_i_glo1, i_glo1, n0_i_glo2, &
      n1_i_glo2, i_glo2, n0_i_glo3, n1_i_glo3, i_glo3, n0_c_loc1, &
      n1_c_loc1, c_loc1, n0_c_loc2, n1_c_loc2, c_loc2, n0_c_loc3, &
      n1_c_loc3, c_loc3, n0_coeff1, n1_coeff1, coeff1, n0_coeff2, &
      n1_coeff2, coeff2, n0_coeff3, n1_coeff3, coeff3, n0_coeff_ind1, &
      coeff_ind1, n0_coeff_ind2, coeff_ind2, n0_coeff_ind3, coeff_ind3, &
      n0_bs1, n1_bs1, bs1, n0_bs2, n1_bs2, bs2, n0_bs3, n1_bs3, n2_bs3, &
      bs3, n0_x_int_ind1, n1_x_int_ind1, x_int_ind1, n0_x_int_ind2, &
      n1_x_int_ind2, x_int_ind2, n0_x_his_ind3, n1_x_his_ind3, &
      x_his_ind3, n0_wts3, n1_wts3, wts3, n0_tau, n1_tau, n2_tau, &
      n3_tau, n4_tau, n5_tau, tau, n0_mat_eq, n1_mat_eq, n2_mat_eq, &
      n3_mat_eq, mat_eq)

  use kernels_projectors_local_mhd, only: mod_kernel_pi1_3 => &
      kernel_pi1_3
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=8), intent(in)  :: n_quad3 
  integer(kind=4), intent(in)  :: n0_n_inthis 
  integer(kind=8), intent(in)  :: n_inthis (0:n0_n_inthis-1)
  integer(kind=4), intent(in)  :: n0_n_nvbf 
  integer(kind=8), intent(in)  :: n_nvbf (0:n0_n_nvbf-1)
  integer(kind=4), intent(in)  :: n0_i_glo1 
  integer(kind=4), intent(in)  :: n1_i_glo1 
  integer(kind=8), intent(in)  :: i_glo1 (0:n0_i_glo1-1,0:n1_i_glo1-1)
  integer(kind=4), intent(in)  :: n0_i_glo2 
  integer(kind=4), intent(in)  :: n1_i_glo2 
  integer(kind=8), intent(in)  :: i_glo2 (0:n0_i_glo2-1,0:n1_i_glo2-1)
  integer(kind=4), intent(in)  :: n0_i_glo3 
  integer(kind=4), intent(in)  :: n1_i_glo3 
  integer(kind=8), intent(in)  :: i_glo3 (0:n0_i_glo3-1,0:n1_i_glo3-1)
  integer(kind=4), intent(in)  :: n0_c_loc1 
  integer(kind=4), intent(in)  :: n1_c_loc1 
  integer(kind=8), intent(in)  :: c_loc1 (0:n0_c_loc1-1,0:n1_c_loc1-1)
  integer(kind=4), intent(in)  :: n0_c_loc2 
  integer(kind=4), intent(in)  :: n1_c_loc2 
  integer(kind=8), intent(in)  :: c_loc2 (0:n0_c_loc2-1,0:n1_c_loc2-1)
  integer(kind=4), intent(in)  :: n0_c_loc3 
  integer(kind=4), intent(in)  :: n1_c_loc3 
  integer(kind=8), intent(in)  :: c_loc3 (0:n0_c_loc3-1,0:n1_c_loc3-1)
  integer(kind=4), intent(in)  :: n0_coeff1 
  integer(kind=4), intent(in)  :: n1_coeff1 
  real(kind=8), intent(in)  :: coeff1 (0:n0_coeff1-1,0:n1_coeff1-1)
  integer(kind=4), intent(in)  :: n0_coeff2 
  integer(kind=4), intent(in)  :: n1_coeff2 
  real(kind=8), intent(in)  :: coeff2 (0:n0_coeff2-1,0:n1_coeff2-1)
  integer(kind=4), intent(in)  :: n0_coeff3 
  integer(kind=4), intent(in)  :: n1_coeff3 
  real(kind=8), intent(in)  :: coeff3 (0:n0_coeff3-1,0:n1_coeff3-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind1 
  integer(kind=8), intent(in)  :: coeff_ind1 (0:n0_coeff_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind2 
  integer(kind=8), intent(in)  :: coeff_ind2 (0:n0_coeff_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind3 
  integer(kind=8), intent(in)  :: coeff_ind3 (0:n0_coeff_ind3-1)
  integer(kind=4), intent(in)  :: n0_bs1 
  integer(kind=4), intent(in)  :: n1_bs1 
  real(kind=8), intent(in)  :: bs1 (0:n0_bs1-1,0:n1_bs1-1)
  integer(kind=4), intent(in)  :: n0_bs2 
  integer(kind=4), intent(in)  :: n1_bs2 
  real(kind=8), intent(in)  :: bs2 (0:n0_bs2-1,0:n1_bs2-1)
  integer(kind=4), intent(in)  :: n0_bs3 
  integer(kind=4), intent(in)  :: n1_bs3 
  integer(kind=4), intent(in)  :: n2_bs3 
  real(kind=8), intent(in)  :: bs3 (0:n0_bs3-1,0:n1_bs3-1,0:n2_bs3-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind1 
  integer(kind=4), intent(in)  :: n1_x_int_ind1 
  integer(kind=8), intent(in)  :: x_int_ind1 (0:n0_x_int_ind1-1,0: &
      n1_x_int_ind1-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind2 
  integer(kind=4), intent(in)  :: n1_x_int_ind2 
  integer(kind=8), intent(in)  :: x_int_ind2 (0:n0_x_int_ind2-1,0: &
      n1_x_int_ind2-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind3 
  integer(kind=4), intent(in)  :: n1_x_his_ind3 
  integer(kind=8), intent(in)  :: x_his_ind3 (0:n0_x_his_ind3-1,0: &
      n1_x_his_ind3-1)
  integer(kind=4), intent(in)  :: n0_wts3 
  integer(kind=4), intent(in)  :: n1_wts3 
  real(kind=8), intent(in)  :: wts3 (0:n0_wts3-1,0:n1_wts3-1)
  integer(kind=4), intent(in)  :: n0_tau 
  integer(kind=4), intent(in)  :: n1_tau 
  integer(kind=4), intent(in)  :: n2_tau 
  integer(kind=4), intent(in)  :: n3_tau 
  integer(kind=4), intent(in)  :: n4_tau 
  integer(kind=4), intent(in)  :: n5_tau 
  real(kind=8), intent(inout)  :: tau (0:n0_tau-1,0:n1_tau-1,0:n2_tau-1, &
      0:n3_tau-1,0:n4_tau-1,0:n5_tau-1)
  integer(kind=4), intent(in)  :: n0_mat_eq 
  integer(kind=4), intent(in)  :: n1_mat_eq 
  integer(kind=4), intent(in)  :: n2_mat_eq 
  integer(kind=4), intent(in)  :: n3_mat_eq 
  real(kind=8), intent(in)  :: mat_eq (0:n0_mat_eq-1,0:n1_mat_eq-1,0: &
      n2_mat_eq-1,0:n3_mat_eq-1)

  call mod_kernel_pi1_3(n,n_quad3,n_inthis,n_nvbf,i_glo1,i_glo2,i_glo3, &
      c_loc1,c_loc2,c_loc3,coeff1,coeff2,coeff3,coeff_ind1,coeff_ind2, &
      coeff_ind3,bs1,bs2,bs3,x_int_ind1,x_int_ind2,x_his_ind3,wts3,tau, &
      mat_eq)
end subroutine

subroutine kernel_pi2_1 (n0_n, n, n0_n_quad, n_quad, n0_n_inthis, &
      n_inthis, n0_n_nvbf, n_nvbf, n0_i_glo1, n1_i_glo1, i_glo1, &
      n0_i_glo2, n1_i_glo2, i_glo2, n0_i_glo3, n1_i_glo3, i_glo3, &
      n0_c_loc1, n1_c_loc1, c_loc1, n0_c_loc2, n1_c_loc2, c_loc2, &
      n0_c_loc3, n1_c_loc3, c_loc3, n0_coeff1, n1_coeff1, coeff1, &
      n0_coeff2, n1_coeff2, coeff2, n0_coeff3, n1_coeff3, coeff3, &
      n0_coeff_ind1, coeff_ind1, n0_coeff_ind2, coeff_ind2, &
      n0_coeff_ind3, coeff_ind3, n0_bs1, n1_bs1, bs1, n0_bs2, n1_bs2, &
      n2_bs2, bs2, n0_bs3, n1_bs3, n2_bs3, bs3, n0_x_int_ind1, &
      n1_x_int_ind1, x_int_ind1, n0_x_his_ind2, n1_x_his_ind2, &
      x_his_ind2, n0_x_his_ind3, n1_x_his_ind3, x_his_ind3, n0_wts2, &
      n1_wts2, wts2, n0_wts3, n1_wts3, wts3, n0_tau, n1_tau, n2_tau, &
      n3_tau, n4_tau, n5_tau, tau, n0_mat_eq, n1_mat_eq, n2_mat_eq, &
      n3_mat_eq, n4_mat_eq, mat_eq)

  use kernels_projectors_local_mhd, only: mod_kernel_pi2_1 => &
      kernel_pi2_1
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_n_quad 
  integer(kind=8), intent(in)  :: n_quad (0:n0_n_quad-1)
  integer(kind=4), intent(in)  :: n0_n_inthis 
  integer(kind=8), intent(in)  :: n_inthis (0:n0_n_inthis-1)
  integer(kind=4), intent(in)  :: n0_n_nvbf 
  integer(kind=8), intent(in)  :: n_nvbf (0:n0_n_nvbf-1)
  integer(kind=4), intent(in)  :: n0_i_glo1 
  integer(kind=4), intent(in)  :: n1_i_glo1 
  integer(kind=8), intent(in)  :: i_glo1 (0:n0_i_glo1-1,0:n1_i_glo1-1)
  integer(kind=4), intent(in)  :: n0_i_glo2 
  integer(kind=4), intent(in)  :: n1_i_glo2 
  integer(kind=8), intent(in)  :: i_glo2 (0:n0_i_glo2-1,0:n1_i_glo2-1)
  integer(kind=4), intent(in)  :: n0_i_glo3 
  integer(kind=4), intent(in)  :: n1_i_glo3 
  integer(kind=8), intent(in)  :: i_glo3 (0:n0_i_glo3-1,0:n1_i_glo3-1)
  integer(kind=4), intent(in)  :: n0_c_loc1 
  integer(kind=4), intent(in)  :: n1_c_loc1 
  integer(kind=8), intent(in)  :: c_loc1 (0:n0_c_loc1-1,0:n1_c_loc1-1)
  integer(kind=4), intent(in)  :: n0_c_loc2 
  integer(kind=4), intent(in)  :: n1_c_loc2 
  integer(kind=8), intent(in)  :: c_loc2 (0:n0_c_loc2-1,0:n1_c_loc2-1)
  integer(kind=4), intent(in)  :: n0_c_loc3 
  integer(kind=4), intent(in)  :: n1_c_loc3 
  integer(kind=8), intent(in)  :: c_loc3 (0:n0_c_loc3-1,0:n1_c_loc3-1)
  integer(kind=4), intent(in)  :: n0_coeff1 
  integer(kind=4), intent(in)  :: n1_coeff1 
  real(kind=8), intent(in)  :: coeff1 (0:n0_coeff1-1,0:n1_coeff1-1)
  integer(kind=4), intent(in)  :: n0_coeff2 
  integer(kind=4), intent(in)  :: n1_coeff2 
  real(kind=8), intent(in)  :: coeff2 (0:n0_coeff2-1,0:n1_coeff2-1)
  integer(kind=4), intent(in)  :: n0_coeff3 
  integer(kind=4), intent(in)  :: n1_coeff3 
  real(kind=8), intent(in)  :: coeff3 (0:n0_coeff3-1,0:n1_coeff3-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind1 
  integer(kind=8), intent(in)  :: coeff_ind1 (0:n0_coeff_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind2 
  integer(kind=8), intent(in)  :: coeff_ind2 (0:n0_coeff_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind3 
  integer(kind=8), intent(in)  :: coeff_ind3 (0:n0_coeff_ind3-1)
  integer(kind=4), intent(in)  :: n0_bs1 
  integer(kind=4), intent(in)  :: n1_bs1 
  real(kind=8), intent(in)  :: bs1 (0:n0_bs1-1,0:n1_bs1-1)
  integer(kind=4), intent(in)  :: n0_bs2 
  integer(kind=4), intent(in)  :: n1_bs2 
  integer(kind=4), intent(in)  :: n2_bs2 
  real(kind=8), intent(in)  :: bs2 (0:n0_bs2-1,0:n1_bs2-1,0:n2_bs2-1)
  integer(kind=4), intent(in)  :: n0_bs3 
  integer(kind=4), intent(in)  :: n1_bs3 
  integer(kind=4), intent(in)  :: n2_bs3 
  real(kind=8), intent(in)  :: bs3 (0:n0_bs3-1,0:n1_bs3-1,0:n2_bs3-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind1 
  integer(kind=4), intent(in)  :: n1_x_int_ind1 
  integer(kind=8), intent(in)  :: x_int_ind1 (0:n0_x_int_ind1-1,0: &
      n1_x_int_ind1-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind2 
  integer(kind=4), intent(in)  :: n1_x_his_ind2 
  integer(kind=8), intent(in)  :: x_his_ind2 (0:n0_x_his_ind2-1,0: &
      n1_x_his_ind2-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind3 
  integer(kind=4), intent(in)  :: n1_x_his_ind3 
  integer(kind=8), intent(in)  :: x_his_ind3 (0:n0_x_his_ind3-1,0: &
      n1_x_his_ind3-1)
  integer(kind=4), intent(in)  :: n0_wts2 
  integer(kind=4), intent(in)  :: n1_wts2 
  real(kind=8), intent(in)  :: wts2 (0:n0_wts2-1,0:n1_wts2-1)
  integer(kind=4), intent(in)  :: n0_wts3 
  integer(kind=4), intent(in)  :: n1_wts3 
  real(kind=8), intent(in)  :: wts3 (0:n0_wts3-1,0:n1_wts3-1)
  integer(kind=4), intent(in)  :: n0_tau 
  integer(kind=4), intent(in)  :: n1_tau 
  integer(kind=4), intent(in)  :: n2_tau 
  integer(kind=4), intent(in)  :: n3_tau 
  integer(kind=4), intent(in)  :: n4_tau 
  integer(kind=4), intent(in)  :: n5_tau 
  real(kind=8), intent(inout)  :: tau (0:n0_tau-1,0:n1_tau-1,0:n2_tau-1, &
      0:n3_tau-1,0:n4_tau-1,0:n5_tau-1)
  integer(kind=4), intent(in)  :: n0_mat_eq 
  integer(kind=4), intent(in)  :: n1_mat_eq 
  integer(kind=4), intent(in)  :: n2_mat_eq 
  integer(kind=4), intent(in)  :: n3_mat_eq 
  integer(kind=4), intent(in)  :: n4_mat_eq 
  real(kind=8), intent(in)  :: mat_eq (0:n0_mat_eq-1,0:n1_mat_eq-1,0: &
      n2_mat_eq-1,0:n3_mat_eq-1,0:n4_mat_eq-1)

  call mod_kernel_pi2_1(n,n_quad,n_inthis,n_nvbf,i_glo1,i_glo2,i_glo3, &
      c_loc1,c_loc2,c_loc3,coeff1,coeff2,coeff3,coeff_ind1,coeff_ind2, &
      coeff_ind3,bs1,bs2,bs3,x_int_ind1,x_his_ind2,x_his_ind3,wts2,wts3 &
      ,tau,mat_eq)
end subroutine

subroutine kernel_pi2_2 (n0_n, n, n0_n_quad, n_quad, n0_n_inthis, &
      n_inthis, n0_n_nvbf, n_nvbf, n0_i_glo1, n1_i_glo1, i_glo1, &
      n0_i_glo2, n1_i_glo2, i_glo2, n0_i_glo3, n1_i_glo3, i_glo3, &
      n0_c_loc1, n1_c_loc1, c_loc1, n0_c_loc2, n1_c_loc2, c_loc2, &
      n0_c_loc3, n1_c_loc3, c_loc3, n0_coeff1, n1_coeff1, coeff1, &
      n0_coeff2, n1_coeff2, coeff2, n0_coeff3, n1_coeff3, coeff3, &
      n0_coeff_ind1, coeff_ind1, n0_coeff_ind2, coeff_ind2, &
      n0_coeff_ind3, coeff_ind3, n0_bs1, n1_bs1, n2_bs1, bs1, n0_bs2, &
      n1_bs2, bs2, n0_bs3, n1_bs3, n2_bs3, bs3, n0_x_his_ind1, &
      n1_x_his_ind1, x_his_ind1, n0_x_int_ind2, n1_x_int_ind2, &
      x_int_ind2, n0_x_his_ind3, n1_x_his_ind3, x_his_ind3, n0_wts1, &
      n1_wts1, wts1, n0_wts3, n1_wts3, wts3, n0_tau, n1_tau, n2_tau, &
      n3_tau, n4_tau, n5_tau, tau, n0_mat_eq, n1_mat_eq, n2_mat_eq, &
      n3_mat_eq, n4_mat_eq, mat_eq)

  use kernels_projectors_local_mhd, only: mod_kernel_pi2_2 => &
      kernel_pi2_2
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_n_quad 
  integer(kind=8), intent(in)  :: n_quad (0:n0_n_quad-1)
  integer(kind=4), intent(in)  :: n0_n_inthis 
  integer(kind=8), intent(in)  :: n_inthis (0:n0_n_inthis-1)
  integer(kind=4), intent(in)  :: n0_n_nvbf 
  integer(kind=8), intent(in)  :: n_nvbf (0:n0_n_nvbf-1)
  integer(kind=4), intent(in)  :: n0_i_glo1 
  integer(kind=4), intent(in)  :: n1_i_glo1 
  integer(kind=8), intent(in)  :: i_glo1 (0:n0_i_glo1-1,0:n1_i_glo1-1)
  integer(kind=4), intent(in)  :: n0_i_glo2 
  integer(kind=4), intent(in)  :: n1_i_glo2 
  integer(kind=8), intent(in)  :: i_glo2 (0:n0_i_glo2-1,0:n1_i_glo2-1)
  integer(kind=4), intent(in)  :: n0_i_glo3 
  integer(kind=4), intent(in)  :: n1_i_glo3 
  integer(kind=8), intent(in)  :: i_glo3 (0:n0_i_glo3-1,0:n1_i_glo3-1)
  integer(kind=4), intent(in)  :: n0_c_loc1 
  integer(kind=4), intent(in)  :: n1_c_loc1 
  integer(kind=8), intent(in)  :: c_loc1 (0:n0_c_loc1-1,0:n1_c_loc1-1)
  integer(kind=4), intent(in)  :: n0_c_loc2 
  integer(kind=4), intent(in)  :: n1_c_loc2 
  integer(kind=8), intent(in)  :: c_loc2 (0:n0_c_loc2-1,0:n1_c_loc2-1)
  integer(kind=4), intent(in)  :: n0_c_loc3 
  integer(kind=4), intent(in)  :: n1_c_loc3 
  integer(kind=8), intent(in)  :: c_loc3 (0:n0_c_loc3-1,0:n1_c_loc3-1)
  integer(kind=4), intent(in)  :: n0_coeff1 
  integer(kind=4), intent(in)  :: n1_coeff1 
  real(kind=8), intent(in)  :: coeff1 (0:n0_coeff1-1,0:n1_coeff1-1)
  integer(kind=4), intent(in)  :: n0_coeff2 
  integer(kind=4), intent(in)  :: n1_coeff2 
  real(kind=8), intent(in)  :: coeff2 (0:n0_coeff2-1,0:n1_coeff2-1)
  integer(kind=4), intent(in)  :: n0_coeff3 
  integer(kind=4), intent(in)  :: n1_coeff3 
  real(kind=8), intent(in)  :: coeff3 (0:n0_coeff3-1,0:n1_coeff3-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind1 
  integer(kind=8), intent(in)  :: coeff_ind1 (0:n0_coeff_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind2 
  integer(kind=8), intent(in)  :: coeff_ind2 (0:n0_coeff_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind3 
  integer(kind=8), intent(in)  :: coeff_ind3 (0:n0_coeff_ind3-1)
  integer(kind=4), intent(in)  :: n0_bs1 
  integer(kind=4), intent(in)  :: n1_bs1 
  integer(kind=4), intent(in)  :: n2_bs1 
  real(kind=8), intent(in)  :: bs1 (0:n0_bs1-1,0:n1_bs1-1,0:n2_bs1-1)
  integer(kind=4), intent(in)  :: n0_bs2 
  integer(kind=4), intent(in)  :: n1_bs2 
  real(kind=8), intent(in)  :: bs2 (0:n0_bs2-1,0:n1_bs2-1)
  integer(kind=4), intent(in)  :: n0_bs3 
  integer(kind=4), intent(in)  :: n1_bs3 
  integer(kind=4), intent(in)  :: n2_bs3 
  real(kind=8), intent(in)  :: bs3 (0:n0_bs3-1,0:n1_bs3-1,0:n2_bs3-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind1 
  integer(kind=4), intent(in)  :: n1_x_his_ind1 
  integer(kind=8), intent(in)  :: x_his_ind1 (0:n0_x_his_ind1-1,0: &
      n1_x_his_ind1-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind2 
  integer(kind=4), intent(in)  :: n1_x_int_ind2 
  integer(kind=8), intent(in)  :: x_int_ind2 (0:n0_x_int_ind2-1,0: &
      n1_x_int_ind2-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind3 
  integer(kind=4), intent(in)  :: n1_x_his_ind3 
  integer(kind=8), intent(in)  :: x_his_ind3 (0:n0_x_his_ind3-1,0: &
      n1_x_his_ind3-1)
  integer(kind=4), intent(in)  :: n0_wts1 
  integer(kind=4), intent(in)  :: n1_wts1 
  real(kind=8), intent(in)  :: wts1 (0:n0_wts1-1,0:n1_wts1-1)
  integer(kind=4), intent(in)  :: n0_wts3 
  integer(kind=4), intent(in)  :: n1_wts3 
  real(kind=8), intent(in)  :: wts3 (0:n0_wts3-1,0:n1_wts3-1)
  integer(kind=4), intent(in)  :: n0_tau 
  integer(kind=4), intent(in)  :: n1_tau 
  integer(kind=4), intent(in)  :: n2_tau 
  integer(kind=4), intent(in)  :: n3_tau 
  integer(kind=4), intent(in)  :: n4_tau 
  integer(kind=4), intent(in)  :: n5_tau 
  real(kind=8), intent(inout)  :: tau (0:n0_tau-1,0:n1_tau-1,0:n2_tau-1, &
      0:n3_tau-1,0:n4_tau-1,0:n5_tau-1)
  integer(kind=4), intent(in)  :: n0_mat_eq 
  integer(kind=4), intent(in)  :: n1_mat_eq 
  integer(kind=4), intent(in)  :: n2_mat_eq 
  integer(kind=4), intent(in)  :: n3_mat_eq 
  integer(kind=4), intent(in)  :: n4_mat_eq 
  real(kind=8), intent(in)  :: mat_eq (0:n0_mat_eq-1,0:n1_mat_eq-1,0: &
      n2_mat_eq-1,0:n3_mat_eq-1,0:n4_mat_eq-1)

  call mod_kernel_pi2_2(n,n_quad,n_inthis,n_nvbf,i_glo1,i_glo2,i_glo3, &
      c_loc1,c_loc2,c_loc3,coeff1,coeff2,coeff3,coeff_ind1,coeff_ind2, &
      coeff_ind3,bs1,bs2,bs3,x_his_ind1,x_int_ind2,x_his_ind3,wts1,wts3 &
      ,tau,mat_eq)
end subroutine

subroutine kernel_pi2_3 (n0_n, n, n0_n_quad, n_quad, n0_n_inthis, &
      n_inthis, n0_n_nvbf, n_nvbf, n0_i_glo1, n1_i_glo1, i_glo1, &
      n0_i_glo2, n1_i_glo2, i_glo2, n0_i_glo3, n1_i_glo3, i_glo3, &
      n0_c_loc1, n1_c_loc1, c_loc1, n0_c_loc2, n1_c_loc2, c_loc2, &
      n0_c_loc3, n1_c_loc3, c_loc3, n0_coeff1, n1_coeff1, coeff1, &
      n0_coeff2, n1_coeff2, coeff2, n0_coeff3, n1_coeff3, coeff3, &
      n0_coeff_ind1, coeff_ind1, n0_coeff_ind2, coeff_ind2, &
      n0_coeff_ind3, coeff_ind3, n0_bs1, n1_bs1, n2_bs1, bs1, n0_bs2, &
      n1_bs2, n2_bs2, bs2, n0_bs3, n1_bs3, bs3, n0_x_his_ind1, &
      n1_x_his_ind1, x_his_ind1, n0_x_his_ind2, n1_x_his_ind2, &
      x_his_ind2, n0_x_int_ind3, n1_x_int_ind3, x_int_ind3, n0_wts1, &
      n1_wts1, wts1, n0_wts2, n1_wts2, wts2, n0_tau, n1_tau, n2_tau, &
      n3_tau, n4_tau, n5_tau, tau, n0_mat_eq, n1_mat_eq, n2_mat_eq, &
      n3_mat_eq, n4_mat_eq, mat_eq)

  use kernels_projectors_local_mhd, only: mod_kernel_pi2_3 => &
      kernel_pi2_3
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_n_quad 
  integer(kind=8), intent(in)  :: n_quad (0:n0_n_quad-1)
  integer(kind=4), intent(in)  :: n0_n_inthis 
  integer(kind=8), intent(in)  :: n_inthis (0:n0_n_inthis-1)
  integer(kind=4), intent(in)  :: n0_n_nvbf 
  integer(kind=8), intent(in)  :: n_nvbf (0:n0_n_nvbf-1)
  integer(kind=4), intent(in)  :: n0_i_glo1 
  integer(kind=4), intent(in)  :: n1_i_glo1 
  integer(kind=8), intent(in)  :: i_glo1 (0:n0_i_glo1-1,0:n1_i_glo1-1)
  integer(kind=4), intent(in)  :: n0_i_glo2 
  integer(kind=4), intent(in)  :: n1_i_glo2 
  integer(kind=8), intent(in)  :: i_glo2 (0:n0_i_glo2-1,0:n1_i_glo2-1)
  integer(kind=4), intent(in)  :: n0_i_glo3 
  integer(kind=4), intent(in)  :: n1_i_glo3 
  integer(kind=8), intent(in)  :: i_glo3 (0:n0_i_glo3-1,0:n1_i_glo3-1)
  integer(kind=4), intent(in)  :: n0_c_loc1 
  integer(kind=4), intent(in)  :: n1_c_loc1 
  integer(kind=8), intent(in)  :: c_loc1 (0:n0_c_loc1-1,0:n1_c_loc1-1)
  integer(kind=4), intent(in)  :: n0_c_loc2 
  integer(kind=4), intent(in)  :: n1_c_loc2 
  integer(kind=8), intent(in)  :: c_loc2 (0:n0_c_loc2-1,0:n1_c_loc2-1)
  integer(kind=4), intent(in)  :: n0_c_loc3 
  integer(kind=4), intent(in)  :: n1_c_loc3 
  integer(kind=8), intent(in)  :: c_loc3 (0:n0_c_loc3-1,0:n1_c_loc3-1)
  integer(kind=4), intent(in)  :: n0_coeff1 
  integer(kind=4), intent(in)  :: n1_coeff1 
  real(kind=8), intent(in)  :: coeff1 (0:n0_coeff1-1,0:n1_coeff1-1)
  integer(kind=4), intent(in)  :: n0_coeff2 
  integer(kind=4), intent(in)  :: n1_coeff2 
  real(kind=8), intent(in)  :: coeff2 (0:n0_coeff2-1,0:n1_coeff2-1)
  integer(kind=4), intent(in)  :: n0_coeff3 
  integer(kind=4), intent(in)  :: n1_coeff3 
  real(kind=8), intent(in)  :: coeff3 (0:n0_coeff3-1,0:n1_coeff3-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind1 
  integer(kind=8), intent(in)  :: coeff_ind1 (0:n0_coeff_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind2 
  integer(kind=8), intent(in)  :: coeff_ind2 (0:n0_coeff_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind3 
  integer(kind=8), intent(in)  :: coeff_ind3 (0:n0_coeff_ind3-1)
  integer(kind=4), intent(in)  :: n0_bs1 
  integer(kind=4), intent(in)  :: n1_bs1 
  integer(kind=4), intent(in)  :: n2_bs1 
  real(kind=8), intent(in)  :: bs1 (0:n0_bs1-1,0:n1_bs1-1,0:n2_bs1-1)
  integer(kind=4), intent(in)  :: n0_bs2 
  integer(kind=4), intent(in)  :: n1_bs2 
  integer(kind=4), intent(in)  :: n2_bs2 
  real(kind=8), intent(in)  :: bs2 (0:n0_bs2-1,0:n1_bs2-1,0:n2_bs2-1)
  integer(kind=4), intent(in)  :: n0_bs3 
  integer(kind=4), intent(in)  :: n1_bs3 
  real(kind=8), intent(in)  :: bs3 (0:n0_bs3-1,0:n1_bs3-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind1 
  integer(kind=4), intent(in)  :: n1_x_his_ind1 
  integer(kind=8), intent(in)  :: x_his_ind1 (0:n0_x_his_ind1-1,0: &
      n1_x_his_ind1-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind2 
  integer(kind=4), intent(in)  :: n1_x_his_ind2 
  integer(kind=8), intent(in)  :: x_his_ind2 (0:n0_x_his_ind2-1,0: &
      n1_x_his_ind2-1)
  integer(kind=4), intent(in)  :: n0_x_int_ind3 
  integer(kind=4), intent(in)  :: n1_x_int_ind3 
  integer(kind=8), intent(in)  :: x_int_ind3 (0:n0_x_int_ind3-1,0: &
      n1_x_int_ind3-1)
  integer(kind=4), intent(in)  :: n0_wts1 
  integer(kind=4), intent(in)  :: n1_wts1 
  real(kind=8), intent(in)  :: wts1 (0:n0_wts1-1,0:n1_wts1-1)
  integer(kind=4), intent(in)  :: n0_wts2 
  integer(kind=4), intent(in)  :: n1_wts2 
  real(kind=8), intent(in)  :: wts2 (0:n0_wts2-1,0:n1_wts2-1)
  integer(kind=4), intent(in)  :: n0_tau 
  integer(kind=4), intent(in)  :: n1_tau 
  integer(kind=4), intent(in)  :: n2_tau 
  integer(kind=4), intent(in)  :: n3_tau 
  integer(kind=4), intent(in)  :: n4_tau 
  integer(kind=4), intent(in)  :: n5_tau 
  real(kind=8), intent(inout)  :: tau (0:n0_tau-1,0:n1_tau-1,0:n2_tau-1, &
      0:n3_tau-1,0:n4_tau-1,0:n5_tau-1)
  integer(kind=4), intent(in)  :: n0_mat_eq 
  integer(kind=4), intent(in)  :: n1_mat_eq 
  integer(kind=4), intent(in)  :: n2_mat_eq 
  integer(kind=4), intent(in)  :: n3_mat_eq 
  integer(kind=4), intent(in)  :: n4_mat_eq 
  real(kind=8), intent(in)  :: mat_eq (0:n0_mat_eq-1,0:n1_mat_eq-1,0: &
      n2_mat_eq-1,0:n3_mat_eq-1,0:n4_mat_eq-1)

  call mod_kernel_pi2_3(n,n_quad,n_inthis,n_nvbf,i_glo1,i_glo2,i_glo3, &
      c_loc1,c_loc2,c_loc3,coeff1,coeff2,coeff3,coeff_ind1,coeff_ind2, &
      coeff_ind3,bs1,bs2,bs3,x_his_ind1,x_his_ind2,x_int_ind3,wts1,wts2 &
      ,tau,mat_eq)
end subroutine

subroutine kernel_pi3 (n0_n, n, n0_n_quad, n_quad, n0_n_his, n_his, &
      n0_n_nvbf, n_nvbf, n0_i_glo1, n1_i_glo1, i_glo1, n0_i_glo2, &
      n1_i_glo2, i_glo2, n0_i_glo3, n1_i_glo3, i_glo3, n0_c_loc1, &
      n1_c_loc1, c_loc1, n0_c_loc2, n1_c_loc2, c_loc2, n0_c_loc3, &
      n1_c_loc3, c_loc3, n0_coeff1, n1_coeff1, coeff1, n0_coeff2, &
      n1_coeff2, coeff2, n0_coeff3, n1_coeff3, coeff3, n0_coeff_ind1, &
      coeff_ind1, n0_coeff_ind2, coeff_ind2, n0_coeff_ind3, coeff_ind3, &
      n0_bs1, n1_bs1, n2_bs1, bs1, n0_bs2, n1_bs2, n2_bs2, bs2, n0_bs3, &
      n1_bs3, n2_bs3, bs3, n0_x_his_ind1, n1_x_his_ind1, x_his_ind1, &
      n0_x_his_ind2, n1_x_his_ind2, x_his_ind2, n0_x_his_ind3, &
      n1_x_his_ind3, x_his_ind3, n0_wts1, n1_wts1, wts1, n0_wts2, &
      n1_wts2, wts2, n0_wts3, n1_wts3, wts3, n0_tau, n1_tau, n2_tau, &
      n3_tau, n4_tau, n5_tau, tau, n0_mat_eq, n1_mat_eq, n2_mat_eq, &
      n3_mat_eq, n4_mat_eq, n5_mat_eq, mat_eq)

  use kernels_projectors_local_mhd, only: mod_kernel_pi3 => kernel_pi3
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_n_quad 
  integer(kind=8), intent(in)  :: n_quad (0:n0_n_quad-1)
  integer(kind=4), intent(in)  :: n0_n_his 
  integer(kind=8), intent(in)  :: n_his (0:n0_n_his-1)
  integer(kind=4), intent(in)  :: n0_n_nvbf 
  integer(kind=8), intent(in)  :: n_nvbf (0:n0_n_nvbf-1)
  integer(kind=4), intent(in)  :: n0_i_glo1 
  integer(kind=4), intent(in)  :: n1_i_glo1 
  integer(kind=8), intent(in)  :: i_glo1 (0:n0_i_glo1-1,0:n1_i_glo1-1)
  integer(kind=4), intent(in)  :: n0_i_glo2 
  integer(kind=4), intent(in)  :: n1_i_glo2 
  integer(kind=8), intent(in)  :: i_glo2 (0:n0_i_glo2-1,0:n1_i_glo2-1)
  integer(kind=4), intent(in)  :: n0_i_glo3 
  integer(kind=4), intent(in)  :: n1_i_glo3 
  integer(kind=8), intent(in)  :: i_glo3 (0:n0_i_glo3-1,0:n1_i_glo3-1)
  integer(kind=4), intent(in)  :: n0_c_loc1 
  integer(kind=4), intent(in)  :: n1_c_loc1 
  integer(kind=8), intent(in)  :: c_loc1 (0:n0_c_loc1-1,0:n1_c_loc1-1)
  integer(kind=4), intent(in)  :: n0_c_loc2 
  integer(kind=4), intent(in)  :: n1_c_loc2 
  integer(kind=8), intent(in)  :: c_loc2 (0:n0_c_loc2-1,0:n1_c_loc2-1)
  integer(kind=4), intent(in)  :: n0_c_loc3 
  integer(kind=4), intent(in)  :: n1_c_loc3 
  integer(kind=8), intent(in)  :: c_loc3 (0:n0_c_loc3-1,0:n1_c_loc3-1)
  integer(kind=4), intent(in)  :: n0_coeff1 
  integer(kind=4), intent(in)  :: n1_coeff1 
  real(kind=8), intent(in)  :: coeff1 (0:n0_coeff1-1,0:n1_coeff1-1)
  integer(kind=4), intent(in)  :: n0_coeff2 
  integer(kind=4), intent(in)  :: n1_coeff2 
  real(kind=8), intent(in)  :: coeff2 (0:n0_coeff2-1,0:n1_coeff2-1)
  integer(kind=4), intent(in)  :: n0_coeff3 
  integer(kind=4), intent(in)  :: n1_coeff3 
  real(kind=8), intent(in)  :: coeff3 (0:n0_coeff3-1,0:n1_coeff3-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind1 
  integer(kind=8), intent(in)  :: coeff_ind1 (0:n0_coeff_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind2 
  integer(kind=8), intent(in)  :: coeff_ind2 (0:n0_coeff_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeff_ind3 
  integer(kind=8), intent(in)  :: coeff_ind3 (0:n0_coeff_ind3-1)
  integer(kind=4), intent(in)  :: n0_bs1 
  integer(kind=4), intent(in)  :: n1_bs1 
  integer(kind=4), intent(in)  :: n2_bs1 
  real(kind=8), intent(in)  :: bs1 (0:n0_bs1-1,0:n1_bs1-1,0:n2_bs1-1)
  integer(kind=4), intent(in)  :: n0_bs2 
  integer(kind=4), intent(in)  :: n1_bs2 
  integer(kind=4), intent(in)  :: n2_bs2 
  real(kind=8), intent(in)  :: bs2 (0:n0_bs2-1,0:n1_bs2-1,0:n2_bs2-1)
  integer(kind=4), intent(in)  :: n0_bs3 
  integer(kind=4), intent(in)  :: n1_bs3 
  integer(kind=4), intent(in)  :: n2_bs3 
  real(kind=8), intent(in)  :: bs3 (0:n0_bs3-1,0:n1_bs3-1,0:n2_bs3-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind1 
  integer(kind=4), intent(in)  :: n1_x_his_ind1 
  integer(kind=8), intent(in)  :: x_his_ind1 (0:n0_x_his_ind1-1,0: &
      n1_x_his_ind1-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind2 
  integer(kind=4), intent(in)  :: n1_x_his_ind2 
  integer(kind=8), intent(in)  :: x_his_ind2 (0:n0_x_his_ind2-1,0: &
      n1_x_his_ind2-1)
  integer(kind=4), intent(in)  :: n0_x_his_ind3 
  integer(kind=4), intent(in)  :: n1_x_his_ind3 
  integer(kind=8), intent(in)  :: x_his_ind3 (0:n0_x_his_ind3-1,0: &
      n1_x_his_ind3-1)
  integer(kind=4), intent(in)  :: n0_wts1 
  integer(kind=4), intent(in)  :: n1_wts1 
  real(kind=8), intent(in)  :: wts1 (0:n0_wts1-1,0:n1_wts1-1)
  integer(kind=4), intent(in)  :: n0_wts2 
  integer(kind=4), intent(in)  :: n1_wts2 
  real(kind=8), intent(in)  :: wts2 (0:n0_wts2-1,0:n1_wts2-1)
  integer(kind=4), intent(in)  :: n0_wts3 
  integer(kind=4), intent(in)  :: n1_wts3 
  real(kind=8), intent(in)  :: wts3 (0:n0_wts3-1,0:n1_wts3-1)
  integer(kind=4), intent(in)  :: n0_tau 
  integer(kind=4), intent(in)  :: n1_tau 
  integer(kind=4), intent(in)  :: n2_tau 
  integer(kind=4), intent(in)  :: n3_tau 
  integer(kind=4), intent(in)  :: n4_tau 
  integer(kind=4), intent(in)  :: n5_tau 
  real(kind=8), intent(inout)  :: tau (0:n0_tau-1,0:n1_tau-1,0:n2_tau-1, &
      0:n3_tau-1,0:n4_tau-1,0:n5_tau-1)
  integer(kind=4), intent(in)  :: n0_mat_eq 
  integer(kind=4), intent(in)  :: n1_mat_eq 
  integer(kind=4), intent(in)  :: n2_mat_eq 
  integer(kind=4), intent(in)  :: n3_mat_eq 
  integer(kind=4), intent(in)  :: n4_mat_eq 
  integer(kind=4), intent(in)  :: n5_mat_eq 
  real(kind=8), intent(in)  :: mat_eq (0:n0_mat_eq-1,0:n1_mat_eq-1,0: &
      n2_mat_eq-1,0:n3_mat_eq-1,0:n4_mat_eq-1,0:n5_mat_eq-1)

  call mod_kernel_pi3(n,n_quad,n_his,n_nvbf,i_glo1,i_glo2,i_glo3,c_loc1, &
      c_loc2,c_loc3,coeff1,coeff2,coeff3,coeff_ind1,coeff_ind2, &
      coeff_ind3,bs1,bs2,bs3,x_his_ind1,x_his_ind2,x_his_ind3,wts1,wts2 &
      ,wts3,tau,mat_eq)
end subroutine