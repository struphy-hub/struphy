function p_eq (xi1, xi2, xi3, kind, n0_params, params) result(Dummy_7696 &
      )

  use equilibrium, only: mod_p_eq => p_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_7696  

  Dummy_7696 = mod_p_eq(xi1,xi2,xi3,kind,params)
end function

function u1_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_0019)

  use equilibrium, only: mod_u1_eq => u1_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_0019  

  Dummy_0019 = mod_u1_eq(xi1,xi2,xi3,kind,params)
end function

function u2_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_0187)

  use equilibrium, only: mod_u2_eq => u2_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_0187  

  Dummy_0187 = mod_u2_eq(xi1,xi2,xi3,kind,params)
end function

function u3_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_1888)

  use equilibrium, only: mod_u3_eq => u3_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_1888  

  Dummy_1888 = mod_u3_eq(xi1,xi2,xi3,kind,params)
end function

function b1_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_3634)

  use equilibrium, only: mod_b1_eq => b1_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_3634  

  Dummy_3634 = mod_b1_eq(xi1,xi2,xi3,kind,params)
end function

function b2_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_3683)

  use equilibrium, only: mod_b2_eq => b2_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_3683  

  Dummy_3683 = mod_b2_eq(xi1,xi2,xi3,kind,params)
end function

function b3_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_6987)

  use equilibrium, only: mod_b3_eq => b3_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_6987  

  Dummy_6987 = mod_b3_eq(xi1,xi2,xi3,kind,params)
end function

function rho_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_9934)

  use equilibrium, only: mod_rho_eq => rho_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_9934  

  Dummy_9934 = mod_rho_eq(xi1,xi2,xi3,kind,params)
end function

function curlb1_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_3106)

  use equilibrium, only: mod_curlb1_eq => curlb1_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_3106  

  Dummy_3106 = mod_curlb1_eq(xi1,xi2,xi3,kind,params)
end function

function curlb2_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_3291)

  use equilibrium, only: mod_curlb2_eq => curlb2_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_3291  

  Dummy_3291 = mod_curlb2_eq(xi1,xi2,xi3,kind,params)
end function

function curlb3_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_2938)

  use equilibrium, only: mod_curlb3_eq => curlb3_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_2938  

  Dummy_2938 = mod_curlb3_eq(xi1,xi2,xi3,kind,params)
end function