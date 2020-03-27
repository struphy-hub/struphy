function f (xi1, xi2, xi3, kind, n0_params, params, component) result( &
      value)

  use mappings_analytical, only: mod_f => f
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  integer(kind=8), intent(in)  :: component 
  real(kind=8)  :: value  

  value = mod_f(xi1,xi2,xi3,kind,params,component)
end function

function df (xi1, xi2, xi3, kind, n0_params, params, component) result( &
      value)

  use mappings_analytical, only: mod_df => df
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  integer(kind=8), intent(in)  :: component 
  real(kind=8)  :: value  

  value = mod_df(xi1,xi2,xi3,kind,params,component)
end function

function det_df (xi1, xi2, xi3, kind, n0_params, params) result(value)

  use mappings_analytical, only: mod_det_df => det_df
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: value  

  value = mod_det_df(xi1,xi2,xi3,kind,params)
end function

function df_inv (xi1, xi2, xi3, kind, n0_params, params, component) &
      result(Dummy_4403)

  use mappings_analytical, only: mod_df_inv => df_inv
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  integer(kind=8), intent(in)  :: component 
  real(kind=8)  :: Dummy_4403  

  Dummy_4403 = mod_df_inv(xi1,xi2,xi3,kind,params,component)
end function

function g (xi1, xi2, xi3, kind, n0_params, params, component) result( &
      value)

  use mappings_analytical, only: mod_g => g
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  integer(kind=8), intent(in)  :: component 
  real(kind=8)  :: value  

  value = mod_g(xi1,xi2,xi3,kind,params,component)
end function

function g_inv (xi1, xi2, xi3, kind, n0_params, params, component) &
      result(Dummy_7961)

  use mappings_analytical, only: mod_g_inv => g_inv
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  integer(kind=8), intent(in)  :: component 
  real(kind=8)  :: Dummy_7961  

  Dummy_7961 = mod_g_inv(xi1,xi2,xi3,kind,params,component)
end function