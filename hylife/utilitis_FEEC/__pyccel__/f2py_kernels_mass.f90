function fun_3d (xi1, xi2, xi3, kind_fun, kind_map, n0_params, params) &
      result(value)

  use kernels_mass, only: mod_fun_3d => fun_3d
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind_fun 
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: value  

  value = mod_fun_3d(xi1,xi2,xi3,kind_fun,kind_map,params)
end function

subroutine kernel_eva_3d (n0_n, n, n0_xi1, xi1, n0_xi2, xi2, n0_xi3, xi3 &
      , n0_mat_f, n1_mat_f, n2_mat_f, mat_f, kind_fun, kind_map, &
      n0_params, params)

  use kernels_mass, only: mod_kernel_eva_3d => kernel_eva_3d
  implicit none
  integer(kind=4), intent(in)  :: n0_n 
  integer(kind=8), intent(in)  :: n (0:n0_n-1)
  integer(kind=4), intent(in)  :: n0_xi1 
  real(kind=8), intent(in)  :: xi1 (0:n0_xi1-1)
  integer(kind=4), intent(in)  :: n0_xi2 
  real(kind=8), intent(in)  :: xi2 (0:n0_xi2-1)
  integer(kind=4), intent(in)  :: n0_xi3 
  real(kind=8), intent(in)  :: xi3 (0:n0_xi3-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  real(kind=8), intent(inout)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1)
  integer(kind=8), intent(in)  :: kind_fun 
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)

  call mod_kernel_eva_3d(n,xi1,xi2,xi3,mat_f,kind_fun,kind_map,params)
end subroutine

subroutine kernel_mass_2d (Nel1, Nel2, p1, p2, nq1, nq2, ni1, ni2, nj1, &
      nj2, n0_w1, n1_w1, w1, n0_w2, n1_w2, w2, n0_bi1, n1_bi1, n2_bi1, &
      n3_bi1, bi1, n0_bi2, n1_bi2, n2_bi2, n3_bi2, bi2, n0_bj1, n1_bj1, &
      n2_bj1, n3_bj1, bj1, n0_bj2, n1_bj2, n2_bj2, n3_bj2, bj2, Nbase1, &
      Nbase2, n0_M, n1_M, n2_M, n3_M, M, n0_mat_map, n1_mat_map, &
      mat_map)

  use kernels_mass, only: mod_kernel_mass_2d => kernel_mass_2d
  implicit none
  integer(kind=8), intent(in)  :: Nel1 
  integer(kind=8), intent(in)  :: Nel2 
  integer(kind=8), intent(in)  :: p1 
  integer(kind=8), intent(in)  :: p2 
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=8), intent(in)  :: ni1 
  integer(kind=8), intent(in)  :: ni2 
  integer(kind=8), intent(in)  :: nj1 
  integer(kind=8), intent(in)  :: nj2 
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_bi1 
  integer(kind=4), intent(in)  :: n1_bi1 
  integer(kind=4), intent(in)  :: n2_bi1 
  integer(kind=4), intent(in)  :: n3_bi1 
  real(kind=8), intent(in)  :: bi1 (0:n0_bi1-1,0:n1_bi1-1,0:n2_bi1-1,0: &
      n3_bi1-1)
  integer(kind=4), intent(in)  :: n0_bi2 
  integer(kind=4), intent(in)  :: n1_bi2 
  integer(kind=4), intent(in)  :: n2_bi2 
  integer(kind=4), intent(in)  :: n3_bi2 
  real(kind=8), intent(in)  :: bi2 (0:n0_bi2-1,0:n1_bi2-1,0:n2_bi2-1,0: &
      n3_bi2-1)
  integer(kind=4), intent(in)  :: n0_bj1 
  integer(kind=4), intent(in)  :: n1_bj1 
  integer(kind=4), intent(in)  :: n2_bj1 
  integer(kind=4), intent(in)  :: n3_bj1 
  real(kind=8), intent(in)  :: bj1 (0:n0_bj1-1,0:n1_bj1-1,0:n2_bj1-1,0: &
      n3_bj1-1)
  integer(kind=4), intent(in)  :: n0_bj2 
  integer(kind=4), intent(in)  :: n1_bj2 
  integer(kind=4), intent(in)  :: n2_bj2 
  integer(kind=4), intent(in)  :: n3_bj2 
  real(kind=8), intent(in)  :: bj2 (0:n0_bj2-1,0:n1_bj2-1,0:n2_bj2-1,0: &
      n3_bj2-1)
  integer(kind=8), intent(in)  :: Nbase1 
  integer(kind=8), intent(in)  :: Nbase2 
  integer(kind=4), intent(in)  :: n0_M 
  integer(kind=4), intent(in)  :: n1_M 
  integer(kind=4), intent(in)  :: n2_M 
  integer(kind=4), intent(in)  :: n3_M 
  real(kind=8), intent(inout)  :: M (0:n0_M-1,0:n1_M-1,0:n2_M-1,0:n3_M-1 &
      )
  integer(kind=4), intent(in)  :: n0_mat_map 
  integer(kind=4), intent(in)  :: n1_mat_map 
  real(kind=8), intent(in)  :: mat_map (0:n0_mat_map-1,0:n1_mat_map-1)

  call mod_kernel_mass_2d(Nel1,Nel2,p1,p2,nq1,nq2,ni1,ni2,nj1,nj2,w1,w2, &
      bi1,bi2,bj1,bj2,Nbase1,Nbase2,M,mat_map)
end subroutine

subroutine kernel_mass_3d (Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, &
      ni1, ni2, ni3, nj1, nj2, nj3, n0_w1, n1_w1, w1, n0_w2, n1_w2, w2, &
      n0_w3, n1_w3, w3, n0_bi1, n1_bi1, n2_bi1, n3_bi1, bi1, n0_bi2, &
      n1_bi2, n2_bi2, n3_bi2, bi2, n0_bi3, n1_bi3, n2_bi3, n3_bi3, bi3, &
      n0_bj1, n1_bj1, n2_bj1, n3_bj1, bj1, n0_bj2, n1_bj2, n2_bj2, &
      n3_bj2, bj2, n0_bj3, n1_bj3, n2_bj3, n3_bj3, bj3, Nbase1, Nbase2, &
      Nbase3, n0_M, n1_M, n2_M, n3_M, n4_M, n5_M, M, n0_mat_map, &
      n1_mat_map, n2_mat_map, mat_map)

  use kernels_mass, only: mod_kernel_mass_3d => kernel_mass_3d
  implicit none
  integer(kind=8), intent(in)  :: Nel1 
  integer(kind=8), intent(in)  :: Nel2 
  integer(kind=8), intent(in)  :: Nel3 
  integer(kind=8), intent(in)  :: p1 
  integer(kind=8), intent(in)  :: p2 
  integer(kind=8), intent(in)  :: p3 
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=8), intent(in)  :: nq3 
  integer(kind=8), intent(in)  :: ni1 
  integer(kind=8), intent(in)  :: ni2 
  integer(kind=8), intent(in)  :: ni3 
  integer(kind=8), intent(in)  :: nj1 
  integer(kind=8), intent(in)  :: nj2 
  integer(kind=8), intent(in)  :: nj3 
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_w3 
  integer(kind=4), intent(in)  :: n1_w3 
  real(kind=8), intent(in)  :: w3 (0:n0_w3-1,0:n1_w3-1)
  integer(kind=4), intent(in)  :: n0_bi1 
  integer(kind=4), intent(in)  :: n1_bi1 
  integer(kind=4), intent(in)  :: n2_bi1 
  integer(kind=4), intent(in)  :: n3_bi1 
  real(kind=8), intent(in)  :: bi1 (0:n0_bi1-1,0:n1_bi1-1,0:n2_bi1-1,0: &
      n3_bi1-1)
  integer(kind=4), intent(in)  :: n0_bi2 
  integer(kind=4), intent(in)  :: n1_bi2 
  integer(kind=4), intent(in)  :: n2_bi2 
  integer(kind=4), intent(in)  :: n3_bi2 
  real(kind=8), intent(in)  :: bi2 (0:n0_bi2-1,0:n1_bi2-1,0:n2_bi2-1,0: &
      n3_bi2-1)
  integer(kind=4), intent(in)  :: n0_bi3 
  integer(kind=4), intent(in)  :: n1_bi3 
  integer(kind=4), intent(in)  :: n2_bi3 
  integer(kind=4), intent(in)  :: n3_bi3 
  real(kind=8), intent(in)  :: bi3 (0:n0_bi3-1,0:n1_bi3-1,0:n2_bi3-1,0: &
      n3_bi3-1)
  integer(kind=4), intent(in)  :: n0_bj1 
  integer(kind=4), intent(in)  :: n1_bj1 
  integer(kind=4), intent(in)  :: n2_bj1 
  integer(kind=4), intent(in)  :: n3_bj1 
  real(kind=8), intent(in)  :: bj1 (0:n0_bj1-1,0:n1_bj1-1,0:n2_bj1-1,0: &
      n3_bj1-1)
  integer(kind=4), intent(in)  :: n0_bj2 
  integer(kind=4), intent(in)  :: n1_bj2 
  integer(kind=4), intent(in)  :: n2_bj2 
  integer(kind=4), intent(in)  :: n3_bj2 
  real(kind=8), intent(in)  :: bj2 (0:n0_bj2-1,0:n1_bj2-1,0:n2_bj2-1,0: &
      n3_bj2-1)
  integer(kind=4), intent(in)  :: n0_bj3 
  integer(kind=4), intent(in)  :: n1_bj3 
  integer(kind=4), intent(in)  :: n2_bj3 
  integer(kind=4), intent(in)  :: n3_bj3 
  real(kind=8), intent(in)  :: bj3 (0:n0_bj3-1,0:n1_bj3-1,0:n2_bj3-1,0: &
      n3_bj3-1)
  integer(kind=8), intent(in)  :: Nbase1 
  integer(kind=8), intent(in)  :: Nbase2 
  integer(kind=8), intent(in)  :: Nbase3 
  integer(kind=4), intent(in)  :: n0_M 
  integer(kind=4), intent(in)  :: n1_M 
  integer(kind=4), intent(in)  :: n2_M 
  integer(kind=4), intent(in)  :: n3_M 
  integer(kind=4), intent(in)  :: n4_M 
  integer(kind=4), intent(in)  :: n5_M 
  real(kind=8), intent(inout)  :: M (0:n0_M-1,0:n1_M-1,0:n2_M-1,0:n3_M-1 &
      ,0:n4_M-1,0:n5_M-1)
  integer(kind=4), intent(in)  :: n0_mat_map 
  integer(kind=4), intent(in)  :: n1_mat_map 
  integer(kind=4), intent(in)  :: n2_mat_map 
  real(kind=8), intent(in)  :: mat_map (0:n0_mat_map-1,0:n1_mat_map-1,0: &
      n2_mat_map-1)

  call mod_kernel_mass_3d(Nel1,Nel2,Nel3,p1,p2,p3,nq1,nq2,nq3,ni1,ni2, &
      ni3,nj1,nj2,nj3,w1,w2,w3,bi1,bi2,bi3,bj1,bj2,bj3,Nbase1,Nbase2, &
      Nbase3,M,mat_map)
end subroutine

subroutine kernel_inner_2d (Nel1, Nel2, p1, p2, nq1, nq2, ni1, ni2, &
      n0_w1, n1_w1, w1, n0_w2, n1_w2, w2, n0_bi1, n1_bi1, n2_bi1, &
      n3_bi1, bi1, n0_bi2, n1_bi2, n2_bi2, n3_bi2, bi2, Nbase1, Nbase2, &
      n0_L, n1_L, L, n0_mat_f, n1_mat_f, mat_f, n0_mat_map, n1_mat_map, &
      mat_map)

  use kernels_mass, only: mod_kernel_inner_2d => kernel_inner_2d
  implicit none
  integer(kind=8), intent(in)  :: Nel1 
  integer(kind=8), intent(in)  :: Nel2 
  integer(kind=8), intent(in)  :: p1 
  integer(kind=8), intent(in)  :: p2 
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=8), intent(in)  :: ni1 
  integer(kind=8), intent(in)  :: ni2 
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_bi1 
  integer(kind=4), intent(in)  :: n1_bi1 
  integer(kind=4), intent(in)  :: n2_bi1 
  integer(kind=4), intent(in)  :: n3_bi1 
  real(kind=8), intent(in)  :: bi1 (0:n0_bi1-1,0:n1_bi1-1,0:n2_bi1-1,0: &
      n3_bi1-1)
  integer(kind=4), intent(in)  :: n0_bi2 
  integer(kind=4), intent(in)  :: n1_bi2 
  integer(kind=4), intent(in)  :: n2_bi2 
  integer(kind=4), intent(in)  :: n3_bi2 
  real(kind=8), intent(in)  :: bi2 (0:n0_bi2-1,0:n1_bi2-1,0:n2_bi2-1,0: &
      n3_bi2-1)
  integer(kind=8), intent(in)  :: Nbase1 
  integer(kind=8), intent(in)  :: Nbase2 
  integer(kind=4), intent(in)  :: n0_L 
  integer(kind=4), intent(in)  :: n1_L 
  real(kind=8), intent(inout)  :: L (0:n0_L-1,0:n1_L-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1)
  integer(kind=4), intent(in)  :: n0_mat_map 
  integer(kind=4), intent(in)  :: n1_mat_map 
  real(kind=8), intent(in)  :: mat_map (0:n0_mat_map-1,0:n1_mat_map-1)

  call mod_kernel_inner_2d(Nel1,Nel2,p1,p2,nq1,nq2,ni1,ni2,w1,w2,bi1,bi2 &
      ,Nbase1,Nbase2,L,mat_f,mat_map)
end subroutine

subroutine kernel_inner_3d (Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, &
      ni1, ni2, ni3, n0_w1, n1_w1, w1, n0_w2, n1_w2, w2, n0_w3, n1_w3, &
      w3, n0_bi1, n1_bi1, n2_bi1, n3_bi1, bi1, n0_bi2, n1_bi2, n2_bi2, &
      n3_bi2, bi2, n0_bi3, n1_bi3, n2_bi3, n3_bi3, bi3, Nbase1, Nbase2, &
      Nbase3, n0_L, n1_L, n2_L, L, n0_mat_f, n1_mat_f, n2_mat_f, mat_f, &
      n0_mat_map, n1_mat_map, n2_mat_map, mat_map)

  use kernels_mass, only: mod_kernel_inner_3d => kernel_inner_3d
  implicit none
  integer(kind=8), intent(in)  :: Nel1 
  integer(kind=8), intent(in)  :: Nel2 
  integer(kind=8), intent(in)  :: Nel3 
  integer(kind=8), intent(in)  :: p1 
  integer(kind=8), intent(in)  :: p2 
  integer(kind=8), intent(in)  :: p3 
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=8), intent(in)  :: nq3 
  integer(kind=8), intent(in)  :: ni1 
  integer(kind=8), intent(in)  :: ni2 
  integer(kind=8), intent(in)  :: ni3 
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_w3 
  integer(kind=4), intent(in)  :: n1_w3 
  real(kind=8), intent(in)  :: w3 (0:n0_w3-1,0:n1_w3-1)
  integer(kind=4), intent(in)  :: n0_bi1 
  integer(kind=4), intent(in)  :: n1_bi1 
  integer(kind=4), intent(in)  :: n2_bi1 
  integer(kind=4), intent(in)  :: n3_bi1 
  real(kind=8), intent(in)  :: bi1 (0:n0_bi1-1,0:n1_bi1-1,0:n2_bi1-1,0: &
      n3_bi1-1)
  integer(kind=4), intent(in)  :: n0_bi2 
  integer(kind=4), intent(in)  :: n1_bi2 
  integer(kind=4), intent(in)  :: n2_bi2 
  integer(kind=4), intent(in)  :: n3_bi2 
  real(kind=8), intent(in)  :: bi2 (0:n0_bi2-1,0:n1_bi2-1,0:n2_bi2-1,0: &
      n3_bi2-1)
  integer(kind=4), intent(in)  :: n0_bi3 
  integer(kind=4), intent(in)  :: n1_bi3 
  integer(kind=4), intent(in)  :: n2_bi3 
  integer(kind=4), intent(in)  :: n3_bi3 
  real(kind=8), intent(in)  :: bi3 (0:n0_bi3-1,0:n1_bi3-1,0:n2_bi3-1,0: &
      n3_bi3-1)
  integer(kind=8), intent(in)  :: Nbase1 
  integer(kind=8), intent(in)  :: Nbase2 
  integer(kind=8), intent(in)  :: Nbase3 
  integer(kind=4), intent(in)  :: n0_L 
  integer(kind=4), intent(in)  :: n1_L 
  integer(kind=4), intent(in)  :: n2_L 
  real(kind=8), intent(inout)  :: L (0:n0_L-1,0:n1_L-1,0:n2_L-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1)
  integer(kind=4), intent(in)  :: n0_mat_map 
  integer(kind=4), intent(in)  :: n1_mat_map 
  integer(kind=4), intent(in)  :: n2_mat_map 
  real(kind=8), intent(in)  :: mat_map (0:n0_mat_map-1,0:n1_mat_map-1,0: &
      n2_mat_map-1)

  call mod_kernel_inner_3d(Nel1,Nel2,Nel3,p1,p2,p3,nq1,nq2,nq3,ni1,ni2, &
      ni3,w1,w2,w3,bi1,bi2,bi3,Nbase1,Nbase2,Nbase3,L,mat_f,mat_map)
end subroutine

subroutine kernel_l2error_v0_2d (Nel1, Nel2, p1, p2, nq1, nq2, n0_w1, &
      n1_w1, w1, n0_w2, n1_w2, w2, n0_bi1, n1_bi1, n2_bi1, n3_bi1, bi1, &
      n0_bi2, n1_bi2, n2_bi2, n3_bi2, bi2, Nbase1, Nbase2, n0_error, &
      n1_error, error, n0_mat_f, n1_mat_f, mat_f, n0_mat_c, n1_mat_c, &
      mat_c, n0_mat_g, n1_mat_g, mat_g)

  use kernels_mass, only: mod_kernel_l2error_v0_2d => &
      kernel_l2error_v0_2d
  implicit none
  integer(kind=8), intent(in)  :: Nel1 
  integer(kind=8), intent(in)  :: Nel2 
  integer(kind=8), intent(in)  :: p1 
  integer(kind=8), intent(in)  :: p2 
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_bi1 
  integer(kind=4), intent(in)  :: n1_bi1 
  integer(kind=4), intent(in)  :: n2_bi1 
  integer(kind=4), intent(in)  :: n3_bi1 
  real(kind=8), intent(in)  :: bi1 (0:n0_bi1-1,0:n1_bi1-1,0:n2_bi1-1,0: &
      n3_bi1-1)
  integer(kind=4), intent(in)  :: n0_bi2 
  integer(kind=4), intent(in)  :: n1_bi2 
  integer(kind=4), intent(in)  :: n2_bi2 
  integer(kind=4), intent(in)  :: n3_bi2 
  real(kind=8), intent(in)  :: bi2 (0:n0_bi2-1,0:n1_bi2-1,0:n2_bi2-1,0: &
      n3_bi2-1)
  integer(kind=8), intent(in)  :: Nbase1 
  integer(kind=8), intent(in)  :: Nbase2 
  integer(kind=4), intent(in)  :: n0_error 
  integer(kind=4), intent(in)  :: n1_error 
  real(kind=8), intent(inout)  :: error (0:n0_error-1,0:n1_error-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1)
  integer(kind=4), intent(in)  :: n0_mat_c 
  integer(kind=4), intent(in)  :: n1_mat_c 
  real(kind=8), intent(in)  :: mat_c (0:n0_mat_c-1,0:n1_mat_c-1)
  integer(kind=4), intent(in)  :: n0_mat_g 
  integer(kind=4), intent(in)  :: n1_mat_g 
  real(kind=8), intent(in)  :: mat_g (0:n0_mat_g-1,0:n1_mat_g-1)

  call mod_kernel_l2error_v0_2d(Nel1,Nel2,p1,p2,nq1,nq2,w1,w2,bi1,bi2, &
      Nbase1,Nbase2,error,mat_f,mat_c,mat_g)
end subroutine

subroutine kernel_l2error_v0_3d (Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, &
      nq3, n0_w1, n1_w1, w1, n0_w2, n1_w2, w2, n0_w3, n1_w3, w3, n0_bi1 &
      , n1_bi1, n2_bi1, n3_bi1, bi1, n0_bi2, n1_bi2, n2_bi2, n3_bi2, &
      bi2, n0_bi3, n1_bi3, n2_bi3, n3_bi3, bi3, Nbase1, Nbase2, Nbase3, &
      n0_error, n1_error, n2_error, error, n0_mat_f, n1_mat_f, n2_mat_f &
      , mat_f, n0_mat_c, n1_mat_c, n2_mat_c, mat_c, n0_mat_g, n1_mat_g, &
      n2_mat_g, mat_g)

  use kernels_mass, only: mod_kernel_l2error_v0_3d => &
      kernel_l2error_v0_3d
  implicit none
  integer(kind=8), intent(in)  :: Nel1 
  integer(kind=8), intent(in)  :: Nel2 
  integer(kind=8), intent(in)  :: Nel3 
  integer(kind=8), intent(in)  :: p1 
  integer(kind=8), intent(in)  :: p2 
  integer(kind=8), intent(in)  :: p3 
  integer(kind=8), intent(in)  :: nq1 
  integer(kind=8), intent(in)  :: nq2 
  integer(kind=8), intent(in)  :: nq3 
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_w3 
  integer(kind=4), intent(in)  :: n1_w3 
  real(kind=8), intent(in)  :: w3 (0:n0_w3-1,0:n1_w3-1)
  integer(kind=4), intent(in)  :: n0_bi1 
  integer(kind=4), intent(in)  :: n1_bi1 
  integer(kind=4), intent(in)  :: n2_bi1 
  integer(kind=4), intent(in)  :: n3_bi1 
  real(kind=8), intent(in)  :: bi1 (0:n0_bi1-1,0:n1_bi1-1,0:n2_bi1-1,0: &
      n3_bi1-1)
  integer(kind=4), intent(in)  :: n0_bi2 
  integer(kind=4), intent(in)  :: n1_bi2 
  integer(kind=4), intent(in)  :: n2_bi2 
  integer(kind=4), intent(in)  :: n3_bi2 
  real(kind=8), intent(in)  :: bi2 (0:n0_bi2-1,0:n1_bi2-1,0:n2_bi2-1,0: &
      n3_bi2-1)
  integer(kind=4), intent(in)  :: n0_bi3 
  integer(kind=4), intent(in)  :: n1_bi3 
  integer(kind=4), intent(in)  :: n2_bi3 
  integer(kind=4), intent(in)  :: n3_bi3 
  real(kind=8), intent(in)  :: bi3 (0:n0_bi3-1,0:n1_bi3-1,0:n2_bi3-1,0: &
      n3_bi3-1)
  integer(kind=8), intent(in)  :: Nbase1 
  integer(kind=8), intent(in)  :: Nbase2 
  integer(kind=8), intent(in)  :: Nbase3 
  integer(kind=4), intent(in)  :: n0_error 
  integer(kind=4), intent(in)  :: n1_error 
  integer(kind=4), intent(in)  :: n2_error 
  real(kind=8), intent(inout)  :: error (0:n0_error-1,0:n1_error-1,0: &
      n2_error-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1)
  integer(kind=4), intent(in)  :: n0_mat_c 
  integer(kind=4), intent(in)  :: n1_mat_c 
  integer(kind=4), intent(in)  :: n2_mat_c 
  real(kind=8), intent(in)  :: mat_c (0:n0_mat_c-1,0:n1_mat_c-1,0: &
      n2_mat_c-1)
  integer(kind=4), intent(in)  :: n0_mat_g 
  integer(kind=4), intent(in)  :: n1_mat_g 
  integer(kind=4), intent(in)  :: n2_mat_g 
  real(kind=8), intent(in)  :: mat_g (0:n0_mat_g-1,0:n1_mat_g-1,0: &
      n2_mat_g-1)

  call mod_kernel_l2error_v0_3d(Nel1,Nel2,Nel3,p1,p2,p3,nq1,nq2,nq3,w1, &
      w2,w3,bi1,bi2,bi3,Nbase1,Nbase2,Nbase3,error,mat_f,mat_c,mat_g)
end subroutine