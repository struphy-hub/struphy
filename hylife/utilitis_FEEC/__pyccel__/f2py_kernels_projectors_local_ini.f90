function fun (xi1, xi2, xi3, kind_fun, kind_map, n0_params, params) &
      result(value)

  use kernels_projectors_local_ini, only: mod_fun => fun
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

subroutine kernel_eva (n0_n, n, n0_xi1, xi1, n0_xi2, xi2, n0_xi3, xi3, &
      n0_mat_f, n1_mat_f, n2_mat_f, mat_f, kind_fun, kind_map, &
      n0_params, params)

  use kernels_projectors_local_ini, only: mod_kernel_eva => kernel_eva
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

  call mod_kernel_eva(n,xi1,xi2,xi3,mat_f,kind_fun,kind_map,params)
end subroutine