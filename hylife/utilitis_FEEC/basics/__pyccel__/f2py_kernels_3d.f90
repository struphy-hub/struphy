function fun (xi1, xi2, xi3, kind_fun, kind_map, n0_params, params) &
      result(value)

  use kernels_3d, only: mod_fun => fun
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind_fun 
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: value  

  value = mod_fun(xi1,xi2,xi3,kind_fun,kind_map,params)
end function

subroutine kernel_evaluation (n0_nel, nel, n0_nq, nq, n0_xi1, n1_xi1, &
      xi1, n0_xi2, n1_xi2, xi2, n0_xi3, n1_xi3, xi3, n0_mat_f, n1_mat_f &
      , n2_mat_f, n3_mat_f, n4_mat_f, n5_mat_f, mat_f, kind_fun, &
      kind_map, n0_params, params)

  use kernels_3d, only: mod_kernel_evaluation => kernel_evaluation
  implicit none
  integer(kind=4), intent(in)  :: n0_nel 
  integer(kind=8), intent(in)  :: nel (0:n0_nel-1)
  integer(kind=4), intent(in)  :: n0_nq 
  integer(kind=8), intent(in)  :: nq (0:n0_nq-1)
  integer(kind=4), intent(in)  :: n0_xi1 
  integer(kind=4), intent(in)  :: n1_xi1 
  real(kind=8), intent(in)  :: xi1 (0:n0_xi1-1,0:n1_xi1-1)
  integer(kind=4), intent(in)  :: n0_xi2 
  integer(kind=4), intent(in)  :: n1_xi2 
  real(kind=8), intent(in)  :: xi2 (0:n0_xi2-1,0:n1_xi2-1)
  integer(kind=4), intent(in)  :: n0_xi3 
  integer(kind=4), intent(in)  :: n1_xi3 
  real(kind=8), intent(in)  :: xi3 (0:n0_xi3-1,0:n1_xi3-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  integer(kind=4), intent(in)  :: n3_mat_f 
  integer(kind=4), intent(in)  :: n4_mat_f 
  integer(kind=4), intent(in)  :: n5_mat_f 
  real(kind=8), intent(inout)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1,0:n3_mat_f-1,0:n4_mat_f-1,0:n5_mat_f-1)
  integer(kind=8), intent(in)  :: kind_fun 
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)

  call mod_kernel_evaluation(nel,nq,xi1,xi2,xi3,mat_f,kind_fun,kind_map, &
      params)
end subroutine

subroutine kernel_mass (nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, ni1 &
      , ni2, ni3, nj1, nj2, nj3, n0_w1, n1_w1, w1, n0_w2, n1_w2, w2, &
      n0_w3, n1_w3, w3, n0_bi1, n1_bi1, n2_bi1, n3_bi1, bi1, n0_bi2, &
      n1_bi2, n2_bi2, n3_bi2, bi2, n0_bi3, n1_bi3, n2_bi3, n3_bi3, bi3, &
      n0_bj1, n1_bj1, n2_bj1, n3_bj1, bj1, n0_bj2, n1_bj2, n2_bj2, &
      n3_bj2, bj2, n0_bj3, n1_bj3, n2_bj3, n3_bj3, bj3, nbase1, nbase2, &
      nbase3, n0_M, n1_M, n2_M, n3_M, n4_M, n5_M, M, n0_mat_map, &
      n1_mat_map, n2_mat_map, n3_mat_map, n4_mat_map, n5_mat_map, &
      mat_map)

  use kernels_3d, only: mod_kernel_mass => kernel_mass
  implicit none
  integer(kind=8), intent(in)  :: nel1 
  integer(kind=8), intent(in)  :: nel2 
  integer(kind=8), intent(in)  :: nel3 
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
  integer(kind=8), intent(in)  :: nbase1 
  integer(kind=8), intent(in)  :: nbase2 
  integer(kind=8), intent(in)  :: nbase3 
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
  integer(kind=4), intent(in)  :: n3_mat_map 
  integer(kind=4), intent(in)  :: n4_mat_map 
  integer(kind=4), intent(in)  :: n5_mat_map 
  real(kind=8), intent(in)  :: mat_map (0:n0_mat_map-1,0:n1_mat_map-1,0: &
      n2_mat_map-1,0:n3_mat_map-1,0:n4_mat_map-1,0:n5_mat_map-1)

  call mod_kernel_mass(nel1,nel2,nel3,p1,p2,p3,nq1,nq2,nq3,ni1,ni2,ni3, &
      nj1,nj2,nj3,w1,w2,w3,bi1,bi2,bi3,bj1,bj2,bj3,nbase1,nbase2,nbase3 &
      ,M,mat_map)
end subroutine

subroutine kernel_inner (nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, &
      ni1, ni2, ni3, n0_w1, n1_w1, w1, n0_w2, n1_w2, w2, n0_w3, n1_w3, &
      w3, n0_bi1, n1_bi1, n2_bi1, n3_bi1, bi1, n0_bi2, n1_bi2, n2_bi2, &
      n3_bi2, bi2, n0_bi3, n1_bi3, n2_bi3, n3_bi3, bi3, nbase1, nbase2, &
      nbase3, n0_mat, n1_mat, n2_mat, mat, n0_mat_f, n1_mat_f, n2_mat_f &
      , mat_f, n0_mat_map, n1_mat_map, n2_mat_map, n3_mat_map, &
      n4_mat_map, n5_mat_map, mat_map)

  use kernels_3d, only: mod_kernel_inner => kernel_inner
  implicit none
  integer(kind=8), intent(in)  :: nel1 
  integer(kind=8), intent(in)  :: nel2 
  integer(kind=8), intent(in)  :: nel3 
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
  integer(kind=8), intent(in)  :: nbase1 
  integer(kind=8), intent(in)  :: nbase2 
  integer(kind=8), intent(in)  :: nbase3 
  integer(kind=4), intent(in)  :: n0_mat 
  integer(kind=4), intent(in)  :: n1_mat 
  integer(kind=4), intent(in)  :: n2_mat 
  real(kind=8), intent(inout)  :: mat (0:n0_mat-1,0:n1_mat-1,0:n2_mat-1)
  integer(kind=4), intent(in)  :: n0_mat_f 
  integer(kind=4), intent(in)  :: n1_mat_f 
  integer(kind=4), intent(in)  :: n2_mat_f 
  real(kind=8), intent(in)  :: mat_f (0:n0_mat_f-1,0:n1_mat_f-1,0: &
      n2_mat_f-1)
  integer(kind=4), intent(in)  :: n0_mat_map 
  integer(kind=4), intent(in)  :: n1_mat_map 
  integer(kind=4), intent(in)  :: n2_mat_map 
  integer(kind=4), intent(in)  :: n3_mat_map 
  integer(kind=4), intent(in)  :: n4_mat_map 
  integer(kind=4), intent(in)  :: n5_mat_map 
  real(kind=8), intent(in)  :: mat_map (0:n0_mat_map-1,0:n1_mat_map-1,0: &
      n2_mat_map-1,0:n3_mat_map-1,0:n4_mat_map-1,0:n5_mat_map-1)

  call mod_kernel_inner(nel1,nel2,nel3,p1,p2,p3,nq1,nq2,nq3,ni1,ni2,ni3, &
      w1,w2,w3,bi1,bi2,bi3,nbase1,nbase2,nbase3,mat,mat_f,mat_map)
end subroutine

subroutine kernel_l2error (n0_nel, nel, n0_p, p, n0_nq, nq, n0_w1, n1_w1 &
      , w1, n0_w2, n1_w2, w2, n0_w3, n1_w3, w3, n0_ni, ni, n0_nj, nj, &
      n0_bi1, n1_bi1, n2_bi1, n3_bi1, bi1, n0_bi2, n1_bi2, n2_bi2, &
      n3_bi2, bi2, n0_bi3, n1_bi3, n2_bi3, n3_bi3, bi3, n0_bj1, n1_bj1, &
      n2_bj1, n3_bj1, bj1, n0_bj2, n1_bj2, n2_bj2, n3_bj2, bj2, n0_bj3, &
      n1_bj3, n2_bj3, n3_bj3, bj3, n0_nbi, nbi, n0_nbj, nbj, n0_error, &
      n1_error, n2_error, error, n0_mat_f1, n1_mat_f1, n2_mat_f1, &
      mat_f1, n0_mat_f2, n1_mat_f2, n2_mat_f2, mat_f2, n0_mat_c1, &
      n1_mat_c1, n2_mat_c1, mat_c1, n0_mat_c2, n1_mat_c2, n2_mat_c2, &
      mat_c2, n0_mat_map, n1_mat_map, n2_mat_map, n3_mat_map, &
      n4_mat_map, n5_mat_map, mat_map)

  use kernels_3d, only: mod_kernel_l2error => kernel_l2error
  implicit none
  integer(kind=4), intent(in)  :: n0_nel 
  integer(kind=8), intent(in)  :: nel (0:n0_nel-1)
  integer(kind=4), intent(in)  :: n0_p 
  integer(kind=8), intent(in)  :: p (0:n0_p-1)
  integer(kind=4), intent(in)  :: n0_nq 
  integer(kind=8), intent(in)  :: nq (0:n0_nq-1)
  integer(kind=4), intent(in)  :: n0_w1 
  integer(kind=4), intent(in)  :: n1_w1 
  real(kind=8), intent(in)  :: w1 (0:n0_w1-1,0:n1_w1-1)
  integer(kind=4), intent(in)  :: n0_w2 
  integer(kind=4), intent(in)  :: n1_w2 
  real(kind=8), intent(in)  :: w2 (0:n0_w2-1,0:n1_w2-1)
  integer(kind=4), intent(in)  :: n0_w3 
  integer(kind=4), intent(in)  :: n1_w3 
  real(kind=8), intent(in)  :: w3 (0:n0_w3-1,0:n1_w3-1)
  integer(kind=4), intent(in)  :: n0_ni 
  integer(kind=8), intent(in)  :: ni (0:n0_ni-1)
  integer(kind=4), intent(in)  :: n0_nj 
  integer(kind=8), intent(in)  :: nj (0:n0_nj-1)
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
  integer(kind=4), intent(in)  :: n0_nbi 
  integer(kind=8), intent(in)  :: nbi (0:n0_nbi-1)
  integer(kind=4), intent(in)  :: n0_nbj 
  integer(kind=8), intent(in)  :: nbj (0:n0_nbj-1)
  integer(kind=4), intent(in)  :: n0_error 
  integer(kind=4), intent(in)  :: n1_error 
  integer(kind=4), intent(in)  :: n2_error 
  real(kind=8), intent(inout)  :: error (0:n0_error-1,0:n1_error-1,0: &
      n2_error-1)
  integer(kind=4), intent(in)  :: n0_mat_f1 
  integer(kind=4), intent(in)  :: n1_mat_f1 
  integer(kind=4), intent(in)  :: n2_mat_f1 
  real(kind=8), intent(in)  :: mat_f1 (0:n0_mat_f1-1,0:n1_mat_f1-1,0: &
      n2_mat_f1-1)
  integer(kind=4), intent(in)  :: n0_mat_f2 
  integer(kind=4), intent(in)  :: n1_mat_f2 
  integer(kind=4), intent(in)  :: n2_mat_f2 
  real(kind=8), intent(in)  :: mat_f2 (0:n0_mat_f2-1,0:n1_mat_f2-1,0: &
      n2_mat_f2-1)
  integer(kind=4), intent(in)  :: n0_mat_c1 
  integer(kind=4), intent(in)  :: n1_mat_c1 
  integer(kind=4), intent(in)  :: n2_mat_c1 
  real(kind=8), intent(in)  :: mat_c1 (0:n0_mat_c1-1,0:n1_mat_c1-1,0: &
      n2_mat_c1-1)
  integer(kind=4), intent(in)  :: n0_mat_c2 
  integer(kind=4), intent(in)  :: n1_mat_c2 
  integer(kind=4), intent(in)  :: n2_mat_c2 
  real(kind=8), intent(in)  :: mat_c2 (0:n0_mat_c2-1,0:n1_mat_c2-1,0: &
      n2_mat_c2-1)
  integer(kind=4), intent(in)  :: n0_mat_map 
  integer(kind=4), intent(in)  :: n1_mat_map 
  integer(kind=4), intent(in)  :: n2_mat_map 
  integer(kind=4), intent(in)  :: n3_mat_map 
  integer(kind=4), intent(in)  :: n4_mat_map 
  integer(kind=4), intent(in)  :: n5_mat_map 
  real(kind=8), intent(in)  :: mat_map (0:n0_mat_map-1,0:n1_mat_map-1,0: &
      n2_mat_map-1,0:n3_mat_map-1,0:n4_mat_map-1,0:n5_mat_map-1)

  call mod_kernel_l2error(nel,p,nq,w1,w2,w3,ni,nj,bi1,bi2,bi3,bj1,bj2, &
      bj3,nbi,nbj,error,mat_f1,mat_f2,mat_c1,mat_c2,mat_map)
end subroutine