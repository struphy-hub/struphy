function fun (xi1, xi2, xi3, kind_map, n0_params, params) result(value)

  use test, only: mod_fun => fun
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: value  

  value = mod_fun(xi1,xi2,xi3,kind_map,params)
end function