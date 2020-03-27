subroutine kernel_pi0_3d (n0_n, n, n0_p, p, n0_coeff_i1, n1_coeff_i1, &
      coeff_i1, n0_coeff_i2, n1_coeff_i2, coeff_i2, n0_coeff_i3, &
      n1_coeff_i3, coeff_i3, n0_coeffi_ind1, coeffi_ind1, &
      n0_coeffi_ind2, coeffi_ind2, n0_coeffi_ind3, coeffi_ind3, &
      n0_x_int_ind1, n1_x_int_ind1, x_int_ind1, n0_x_int_ind2, &
      n1_x_int_ind2, x_int_ind2, n0_x_int_ind3, n1_x_int_ind3, &
      x_int_ind3, n0_mat_f, n1_mat_f, n2_mat_f, mat_f, n0_lambdas, &
      n1_lambdas, n2_lambdas, lambdas)

  use kernels_projectors_local, only: mod_kernel_pi0_3d => kernel_pi0_3d
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_p 
  integer(kind=8), intent(in)  :: p (0:n0_p-1)
  integer(kind=4), intent(in)  :: n0_coeff_i1 
  integer(kind=4), intent(in)  :: n1_coeff_i1 
  real(kind=8), intent(in)  :: coeff_i1 (0:n0_coeff_i1-1,0:n1_coeff_i1-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_i2 
  integer(kind=4), intent(in)  :: n1_coeff_i2 
  real(kind=8), intent(in)  :: coeff_i2 (0:n0_coeff_i2-1,0:n1_coeff_i2-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_i3 
  integer(kind=4), intent(in)  :: n1_coeff_i3 
  real(kind=8), intent(in)  :: coeff_i3 (0:n0_coeff_i3-1,0:n1_coeff_i3-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeffi_ind1 
  integer(kind=8), intent(in)  :: coeffi_ind1 (0:n0_coeffi_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeffi_ind2 
  integer(kind=8), intent(in)  :: coeffi_ind2 (0:n0_coeffi_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeffi_ind3 
  integer(kind=8), intent(in)  :: coeffi_ind3 (0:n0_coeffi_ind3-1)
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
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1)
  integer(kind=4), intent(in)  :: n0_lambdas 
  integer(kind=4), intent(in)  :: n1_lambdas 
  integer(kind=4), intent(in)  :: n2_lambdas 
  real(kind=8), intent(inout)  :: lambdas (0:n0_lambdas-1,0:n1_lambdas-1 &
      ,0:n2_lambdas-1)

  call mod_kernel_pi0_3d(n,p,coeff_i1,coeff_i2,coeff_i3,coeffi_ind1, &
      coeffi_ind2,coeffi_ind3,x_int_ind1,x_int_ind2,x_int_ind3,mat_f, &
      lambdas)
end subroutine

subroutine kernel_pi11_3d (n0_n, n, n0_p, p, n0_coeff_h1, n1_coeff_h1, &
      coeff_h1, n0_coeff_i2, n1_coeff_i2, coeff_i2, n0_coeff_i3, &
      n1_coeff_i3, coeff_i3, n0_coeffh_ind1, coeffh_ind1, &
      n0_coeffi_ind2, coeffi_ind2, n0_coeffi_ind3, coeffi_ind3, &
      n0_x_his_ind1, n1_x_his_ind1, x_his_ind1, n0_x_int_ind2, &
      n1_x_int_ind2, x_int_ind2, n0_x_int_ind3, n1_x_int_ind3, &
      x_int_ind3, n0_wts1, n1_wts1, wts1, n0_mat_f, n1_mat_f, n2_mat_f, &
      n3_mat_f, mat_f, n0_lambdas, n1_lambdas, n2_lambdas, lambdas)

  use kernels_projectors_local, only: mod_kernel_pi11_3d => &
      kernel_pi11_3d
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_p 
  integer(kind=8), intent(in)  :: p (0:n0_p-1)
  integer(kind=4), intent(in)  :: n0_coeff_h1 
  integer(kind=4), intent(in)  :: n1_coeff_h1 
  real(kind=8), intent(in)  :: coeff_h1 (0:n0_coeff_h1-1,0:n1_coeff_h1-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_i2 
  integer(kind=4), intent(in)  :: n1_coeff_i2 
  real(kind=8), intent(in)  :: coeff_i2 (0:n0_coeff_i2-1,0:n1_coeff_i2-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_i3 
  integer(kind=4), intent(in)  :: n1_coeff_i3 
  real(kind=8), intent(in)  :: coeff_i3 (0:n0_coeff_i3-1,0:n1_coeff_i3-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeffh_ind1 
  integer(kind=8), intent(in)  :: coeffh_ind1 (0:n0_coeffh_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeffi_ind2 
  integer(kind=8), intent(in)  :: coeffi_ind2 (0:n0_coeffi_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeffi_ind3 
  integer(kind=8), intent(in)  :: coeffi_ind3 (0:n0_coeffi_ind3-1)
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
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  integer(kind=4), intent(in)  :: n3_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1,0:n3_mat_f-1)
  integer(kind=4), intent(in)  :: n0_lambdas 
  integer(kind=4), intent(in)  :: n1_lambdas 
  integer(kind=4), intent(in)  :: n2_lambdas 
  real(kind=8), intent(inout)  :: lambdas (0:n0_lambdas-1,0:n1_lambdas-1 &
      ,0:n2_lambdas-1)

  call mod_kernel_pi11_3d(n,p,coeff_h1,coeff_i2,coeff_i3,coeffh_ind1, &
      coeffi_ind2,coeffi_ind3,x_his_ind1,x_int_ind2,x_int_ind3,wts1, &
      mat_f,lambdas)
end subroutine

subroutine kernel_pi12_3d (n0_n, n, n0_p, p, n0_coeff_i1, n1_coeff_i1, &
      coeff_i1, n0_coeff_h2, n1_coeff_h2, coeff_h2, n0_coeff_i3, &
      n1_coeff_i3, coeff_i3, n0_coeffi_ind1, coeffi_ind1, &
      n0_coeffh_ind2, coeffh_ind2, n0_coeffi_ind3, coeffi_ind3, &
      n0_x_int_ind1, n1_x_int_ind1, x_int_ind1, n0_x_his_ind2, &
      n1_x_his_ind2, x_his_ind2, n0_x_int_ind3, n1_x_int_ind3, &
      x_int_ind3, n0_wts2, n1_wts2, wts2, n0_mat_f, n1_mat_f, n2_mat_f, &
      n3_mat_f, mat_f, n0_lambdas, n1_lambdas, n2_lambdas, lambdas)

  use kernels_projectors_local, only: mod_kernel_pi12_3d => &
      kernel_pi12_3d
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_p 
  integer(kind=8), intent(in)  :: p (0:n0_p-1)
  integer(kind=4), intent(in)  :: n0_coeff_i1 
  integer(kind=4), intent(in)  :: n1_coeff_i1 
  real(kind=8), intent(in)  :: coeff_i1 (0:n0_coeff_i1-1,0:n1_coeff_i1-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_h2 
  integer(kind=4), intent(in)  :: n1_coeff_h2 
  real(kind=8), intent(in)  :: coeff_h2 (0:n0_coeff_h2-1,0:n1_coeff_h2-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_i3 
  integer(kind=4), intent(in)  :: n1_coeff_i3 
  real(kind=8), intent(in)  :: coeff_i3 (0:n0_coeff_i3-1,0:n1_coeff_i3-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeffi_ind1 
  integer(kind=8), intent(in)  :: coeffi_ind1 (0:n0_coeffi_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeffh_ind2 
  integer(kind=8), intent(in)  :: coeffh_ind2 (0:n0_coeffh_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeffi_ind3 
  integer(kind=8), intent(in)  :: coeffi_ind3 (0:n0_coeffi_ind3-1)
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
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  integer(kind=4), intent(in)  :: n3_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1,0:n3_mat_f-1)
  integer(kind=4), intent(in)  :: n0_lambdas 
  integer(kind=4), intent(in)  :: n1_lambdas 
  integer(kind=4), intent(in)  :: n2_lambdas 
  real(kind=8), intent(inout)  :: lambdas (0:n0_lambdas-1,0:n1_lambdas-1 &
      ,0:n2_lambdas-1)

  call mod_kernel_pi12_3d(n,p,coeff_i1,coeff_h2,coeff_i3,coeffi_ind1, &
      coeffh_ind2,coeffi_ind3,x_int_ind1,x_his_ind2,x_int_ind3,wts2, &
      mat_f,lambdas)
end subroutine

subroutine kernel_pi13_3d (n0_n, n, n0_p, p, n0_coeff_i1, n1_coeff_i1, &
      coeff_i1, n0_coeff_i2, n1_coeff_i2, coeff_i2, n0_coeff_h3, &
      n1_coeff_h3, coeff_h3, n0_coeffi_ind1, coeffi_ind1, &
      n0_coeffi_ind2, coeffi_ind2, n0_coeffh_ind3, coeffh_ind3, &
      n0_x_int_ind1, n1_x_int_ind1, x_int_ind1, n0_x_int_ind2, &
      n1_x_int_ind2, x_int_ind2, n0_x_his_ind3, n1_x_his_ind3, &
      x_his_ind3, n0_wts3, n1_wts3, wts3, n0_mat_f, n1_mat_f, n2_mat_f, &
      n3_mat_f, mat_f, n0_lambdas, n1_lambdas, n2_lambdas, lambdas)

  use kernels_projectors_local, only: mod_kernel_pi13_3d => &
      kernel_pi13_3d
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_p 
  integer(kind=8), intent(in)  :: p (0:n0_p-1)
  integer(kind=4), intent(in)  :: n0_coeff_i1 
  integer(kind=4), intent(in)  :: n1_coeff_i1 
  real(kind=8), intent(in)  :: coeff_i1 (0:n0_coeff_i1-1,0:n1_coeff_i1-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_i2 
  integer(kind=4), intent(in)  :: n1_coeff_i2 
  real(kind=8), intent(in)  :: coeff_i2 (0:n0_coeff_i2-1,0:n1_coeff_i2-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_h3 
  integer(kind=4), intent(in)  :: n1_coeff_h3 
  real(kind=8), intent(in)  :: coeff_h3 (0:n0_coeff_h3-1,0:n1_coeff_h3-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeffi_ind1 
  integer(kind=8), intent(in)  :: coeffi_ind1 (0:n0_coeffi_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeffi_ind2 
  integer(kind=8), intent(in)  :: coeffi_ind2 (0:n0_coeffi_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeffh_ind3 
  integer(kind=8), intent(in)  :: coeffh_ind3 (0:n0_coeffh_ind3-1)
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
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  integer(kind=4), intent(in)  :: n3_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1,0:n3_mat_f-1)
  integer(kind=4), intent(in)  :: n0_lambdas 
  integer(kind=4), intent(in)  :: n1_lambdas 
  integer(kind=4), intent(in)  :: n2_lambdas 
  real(kind=8), intent(inout)  :: lambdas (0:n0_lambdas-1,0:n1_lambdas-1 &
      ,0:n2_lambdas-1)

  call mod_kernel_pi13_3d(n,p,coeff_i1,coeff_i2,coeff_h3,coeffi_ind1, &
      coeffi_ind2,coeffh_ind3,x_int_ind1,x_int_ind2,x_his_ind3,wts3, &
      mat_f,lambdas)
end subroutine

subroutine kernel_pi21_3d (n0_n, n, n0_p, p, n0_coeff_i1, n1_coeff_i1, &
      coeff_i1, n0_coeff_h2, n1_coeff_h2, coeff_h2, n0_coeff_h3, &
      n1_coeff_h3, coeff_h3, n0_coeffi_ind1, coeffi_ind1, &
      n0_coeffh_ind2, coeffh_ind2, n0_coeffh_ind3, coeffh_ind3, &
      n0_x_int_ind1, n1_x_int_ind1, x_int_ind1, n0_x_his_ind2, &
      n1_x_his_ind2, x_his_ind2, n0_x_his_ind3, n1_x_his_ind3, &
      x_his_ind3, n0_wts2, n1_wts2, wts2, n0_wts3, n1_wts3, wts3, &
      n0_mat_f, n1_mat_f, n2_mat_f, n3_mat_f, n4_mat_f, mat_f, &
      n0_lambdas, n1_lambdas, n2_lambdas, lambdas)

  use kernels_projectors_local, only: mod_kernel_pi21_3d => &
      kernel_pi21_3d
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_p 
  integer(kind=8), intent(in)  :: p (0:n0_p-1)
  integer(kind=4), intent(in)  :: n0_coeff_i1 
  integer(kind=4), intent(in)  :: n1_coeff_i1 
  real(kind=8), intent(in)  :: coeff_i1 (0:n0_coeff_i1-1,0:n1_coeff_i1-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_h2 
  integer(kind=4), intent(in)  :: n1_coeff_h2 
  real(kind=8), intent(in)  :: coeff_h2 (0:n0_coeff_h2-1,0:n1_coeff_h2-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_h3 
  integer(kind=4), intent(in)  :: n1_coeff_h3 
  real(kind=8), intent(in)  :: coeff_h3 (0:n0_coeff_h3-1,0:n1_coeff_h3-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeffi_ind1 
  integer(kind=8), intent(in)  :: coeffi_ind1 (0:n0_coeffi_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeffh_ind2 
  integer(kind=8), intent(in)  :: coeffh_ind2 (0:n0_coeffh_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeffh_ind3 
  integer(kind=8), intent(in)  :: coeffh_ind3 (0:n0_coeffh_ind3-1)
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
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  integer(kind=4), intent(in)  :: n3_mat_f 
  integer(kind=4), intent(in)  :: n4_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1,0:n3_mat_f-1,0:n4_mat_f-1)
  integer(kind=4), intent(in)  :: n0_lambdas 
  integer(kind=4), intent(in)  :: n1_lambdas 
  integer(kind=4), intent(in)  :: n2_lambdas 
  real(kind=8), intent(inout)  :: lambdas (0:n0_lambdas-1,0:n1_lambdas-1 &
      ,0:n2_lambdas-1)

  call mod_kernel_pi21_3d(n,p,coeff_i1,coeff_h2,coeff_h3,coeffi_ind1, &
      coeffh_ind2,coeffh_ind3,x_int_ind1,x_his_ind2,x_his_ind3,wts2, &
      wts3,mat_f,lambdas)
end subroutine

subroutine kernel_pi22_3d (n0_n, n, n0_p, p, n0_coeff_h1, n1_coeff_h1, &
      coeff_h1, n0_coeff_i2, n1_coeff_i2, coeff_i2, n0_coeff_h3, &
      n1_coeff_h3, coeff_h3, n0_coeffh_ind1, coeffh_ind1, &
      n0_coeffi_ind2, coeffi_ind2, n0_coeffh_ind3, coeffh_ind3, &
      n0_x_his_ind1, n1_x_his_ind1, x_his_ind1, n0_x_int_ind2, &
      n1_x_int_ind2, x_int_ind2, n0_x_his_ind3, n1_x_his_ind3, &
      x_his_ind3, n0_wts1, n1_wts1, wts1, n0_wts3, n1_wts3, wts3, &
      n0_mat_f, n1_mat_f, n2_mat_f, n3_mat_f, n4_mat_f, mat_f, &
      n0_lambdas, n1_lambdas, n2_lambdas, lambdas)

  use kernels_projectors_local, only: mod_kernel_pi22_3d => &
      kernel_pi22_3d
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_p 
  integer(kind=8), intent(in)  :: p (0:n0_p-1)
  integer(kind=4), intent(in)  :: n0_coeff_h1 
  integer(kind=4), intent(in)  :: n1_coeff_h1 
  real(kind=8), intent(in)  :: coeff_h1 (0:n0_coeff_h1-1,0:n1_coeff_h1-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_i2 
  integer(kind=4), intent(in)  :: n1_coeff_i2 
  real(kind=8), intent(in)  :: coeff_i2 (0:n0_coeff_i2-1,0:n1_coeff_i2-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_h3 
  integer(kind=4), intent(in)  :: n1_coeff_h3 
  real(kind=8), intent(in)  :: coeff_h3 (0:n0_coeff_h3-1,0:n1_coeff_h3-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeffh_ind1 
  integer(kind=8), intent(in)  :: coeffh_ind1 (0:n0_coeffh_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeffi_ind2 
  integer(kind=8), intent(in)  :: coeffi_ind2 (0:n0_coeffi_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeffh_ind3 
  integer(kind=8), intent(in)  :: coeffh_ind3 (0:n0_coeffh_ind3-1)
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
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  integer(kind=4), intent(in)  :: n3_mat_f 
  integer(kind=4), intent(in)  :: n4_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1,0:n3_mat_f-1,0:n4_mat_f-1)
  integer(kind=4), intent(in)  :: n0_lambdas 
  integer(kind=4), intent(in)  :: n1_lambdas 
  integer(kind=4), intent(in)  :: n2_lambdas 
  real(kind=8), intent(inout)  :: lambdas (0:n0_lambdas-1,0:n1_lambdas-1 &
      ,0:n2_lambdas-1)

  call mod_kernel_pi22_3d(n,p,coeff_h1,coeff_i2,coeff_h3,coeffh_ind1, &
      coeffi_ind2,coeffh_ind3,x_his_ind1,x_int_ind2,x_his_ind3,wts1, &
      wts3,mat_f,lambdas)
end subroutine

subroutine kernel_pi23_3d (n0_n, n, n0_p, p, n0_coeff_h1, n1_coeff_h1, &
      coeff_h1, n0_coeff_h2, n1_coeff_h2, coeff_h2, n0_coeff_i3, &
      n1_coeff_i3, coeff_i3, n0_coeffh_ind1, coeffh_ind1, &
      n0_coeffh_ind2, coeffh_ind2, n0_coeffi_ind3, coeffi_ind3, &
      n0_x_his_ind1, n1_x_his_ind1, x_his_ind1, n0_x_his_ind2, &
      n1_x_his_ind2, x_his_ind2, n0_x_int_ind3, n1_x_int_ind3, &
      x_int_ind3, n0_wts1, n1_wts1, wts1, n0_wts2, n1_wts2, wts2, &
      n0_mat_f, n1_mat_f, n2_mat_f, n3_mat_f, n4_mat_f, mat_f, &
      n0_lambdas, n1_lambdas, n2_lambdas, lambdas)

  use kernels_projectors_local, only: mod_kernel_pi23_3d => &
      kernel_pi23_3d
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_p 
  integer(kind=8), intent(in)  :: p (0:n0_p-1)
  integer(kind=4), intent(in)  :: n0_coeff_h1 
  integer(kind=4), intent(in)  :: n1_coeff_h1 
  real(kind=8), intent(in)  :: coeff_h1 (0:n0_coeff_h1-1,0:n1_coeff_h1-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_h2 
  integer(kind=4), intent(in)  :: n1_coeff_h2 
  real(kind=8), intent(in)  :: coeff_h2 (0:n0_coeff_h2-1,0:n1_coeff_h2-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_i3 
  integer(kind=4), intent(in)  :: n1_coeff_i3 
  real(kind=8), intent(in)  :: coeff_i3 (0:n0_coeff_i3-1,0:n1_coeff_i3-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeffh_ind1 
  integer(kind=8), intent(in)  :: coeffh_ind1 (0:n0_coeffh_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeffh_ind2 
  integer(kind=8), intent(in)  :: coeffh_ind2 (0:n0_coeffh_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeffi_ind3 
  integer(kind=8), intent(in)  :: coeffi_ind3 (0:n0_coeffi_ind3-1)
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
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  integer(kind=4), intent(in)  :: n3_mat_f 
  integer(kind=4), intent(in)  :: n4_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1,0:n3_mat_f-1,0:n4_mat_f-1)
  integer(kind=4), intent(in)  :: n0_lambdas 
  integer(kind=4), intent(in)  :: n1_lambdas 
  integer(kind=4), intent(in)  :: n2_lambdas 
  real(kind=8), intent(inout)  :: lambdas (0:n0_lambdas-1,0:n1_lambdas-1 &
      ,0:n2_lambdas-1)

  call mod_kernel_pi23_3d(n,p,coeff_h1,coeff_h2,coeff_i3,coeffh_ind1, &
      coeffh_ind2,coeffi_ind3,x_his_ind1,x_his_ind2,x_int_ind3,wts1, &
      wts2,mat_f,lambdas)
end subroutine

subroutine kernel_pi3_3d (n0_n, n, n0_p, p, n0_coeff_h1, n1_coeff_h1, &
      coeff_h1, n0_coeff_h2, n1_coeff_h2, coeff_h2, n0_coeff_h3, &
      n1_coeff_h3, coeff_h3, n0_coeffh_ind1, coeffh_ind1, &
      n0_coeffh_ind2, coeffh_ind2, n0_coeffh_ind3, coeffh_ind3, &
      n0_x_his_ind1, n1_x_his_ind1, x_his_ind1, n0_x_his_ind2, &
      n1_x_his_ind2, x_his_ind2, n0_x_his_ind3, n1_x_his_ind3, &
      x_his_ind3, n0_wts1, n1_wts1, wts1, n0_wts2, n1_wts2, wts2, &
      n0_wts3, n1_wts3, wts3, n0_mat_f, n1_mat_f, n2_mat_f, n3_mat_f, &
      n4_mat_f, n5_mat_f, mat_f, n0_lambdas, n1_lambdas, n2_lambdas, &
      lambdas)

  use kernels_projectors_local, only: mod_kernel_pi3_3d => kernel_pi3_3d
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_p 
  integer(kind=8), intent(in)  :: p (0:n0_p-1)
  integer(kind=4), intent(in)  :: n0_coeff_h1 
  integer(kind=4), intent(in)  :: n1_coeff_h1 
  real(kind=8), intent(in)  :: coeff_h1 (0:n0_coeff_h1-1,0:n1_coeff_h1-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_h2 
  integer(kind=4), intent(in)  :: n1_coeff_h2 
  real(kind=8), intent(in)  :: coeff_h2 (0:n0_coeff_h2-1,0:n1_coeff_h2-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeff_h3 
  integer(kind=4), intent(in)  :: n1_coeff_h3 
  real(kind=8), intent(in)  :: coeff_h3 (0:n0_coeff_h3-1,0:n1_coeff_h3-1 &
      )
  integer(kind=4), intent(in)  :: n0_coeffh_ind1 
  integer(kind=8), intent(in)  :: coeffh_ind1 (0:n0_coeffh_ind1-1)
  integer(kind=4), intent(in)  :: n0_coeffh_ind2 
  integer(kind=8), intent(in)  :: coeffh_ind2 (0:n0_coeffh_ind2-1)
  integer(kind=4), intent(in)  :: n0_coeffh_ind3 
  integer(kind=8), intent(in)  :: coeffh_ind3 (0:n0_coeffh_ind3-1)
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
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  integer(kind=4), intent(in)  :: n3_mat_f 
  integer(kind=4), intent(in)  :: n4_mat_f 
  integer(kind=4), intent(in)  :: n5_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1,0:n3_mat_f-1,0:n4_mat_f-1,0:n5_mat_f-1)
  integer(kind=4), intent(in)  :: n0_lambdas 
  integer(kind=4), intent(in)  :: n1_lambdas 
  integer(kind=4), intent(in)  :: n2_lambdas 
  real(kind=8), intent(inout)  :: lambdas (0:n0_lambdas-1,0:n1_lambdas-1 &
      ,0:n2_lambdas-1)

  call mod_kernel_pi3_3d(n,p,coeff_h1,coeff_h2,coeff_h3,coeffh_ind1, &
      coeffh_ind2,coeffh_ind3,x_his_ind1,x_his_ind2,x_his_ind3,wts1, &
      wts2,wts3,mat_f,lambdas)
end subroutine