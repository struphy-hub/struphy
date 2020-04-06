function p_eq_phys (x, y, z) result(p_phys)

  use equilibrium_MHD, only: mod_p_eq_phys => p_eq_phys
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: p_phys  

  p_phys = mod_p_eq_phys(x,y,z)
end function

function ux_eq (x, y, z) result(ux)

  use equilibrium_MHD, only: mod_ux_eq => ux_eq
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: ux  

  ux = mod_ux_eq(x,y,z)
end function

function uy_eq (x, y, z) result(uy)

  use equilibrium_MHD, only: mod_uy_eq => uy_eq
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: uy  

  uy = mod_uy_eq(x,y,z)
end function

function uz_eq (x, y, z) result(uz)

  use equilibrium_MHD, only: mod_uz_eq => uz_eq
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: uz  

  uz = mod_uz_eq(x,y,z)
end function

function bx_eq (x, y, z) result(bx)

  use equilibrium_MHD, only: mod_bx_eq => bx_eq
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: bx  

  bx = mod_bx_eq(x,y,z)
end function

function by_eq (x, y, z) result(by)

  use equilibrium_MHD, only: mod_by_eq => by_eq
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: by  

  by = mod_by_eq(x,y,z)
end function

function bz_eq (x, y, z) result(bz)

  use equilibrium_MHD, only: mod_bz_eq => bz_eq
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: bz  

  bz = mod_bz_eq(x,y,z)
end function

function rho_eq_phys (x, y, z) result(rho_phys)

  use equilibrium_MHD, only: mod_rho_eq_phys => rho_eq_phys
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: rho_phys  

  rho_phys = mod_rho_eq_phys(x,y,z)
end function

function p_eq (xi1, xi2, xi3, kind, n0_params, params) result(p_phys)

  use equilibrium_MHD, only: mod_p_eq => p_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: p_phys  

  p_phys = mod_p_eq(xi1,xi2,xi3,kind,params)
end function

function u1_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_9178)

  use equilibrium_MHD, only: mod_u1_eq => u1_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_9178  

  Dummy_9178 = mod_u1_eq(xi1,xi2,xi3,kind,params)
end function

function u2_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_7113)

  use equilibrium_MHD, only: mod_u2_eq => u2_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_7113  

  Dummy_7113 = mod_u2_eq(xi1,xi2,xi3,kind,params)
end function

function u3_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_8397)

  use equilibrium_MHD, only: mod_u3_eq => u3_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_8397  

  Dummy_8397 = mod_u3_eq(xi1,xi2,xi3,kind,params)
end function

function b1_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_2611)

  use equilibrium_MHD, only: mod_b1_eq => b1_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_2611  

  Dummy_2611 = mod_b1_eq(xi1,xi2,xi3,kind,params)
end function

function b2_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_9609)

  use equilibrium_MHD, only: mod_b2_eq => b2_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_9609  

  Dummy_9609 = mod_b2_eq(xi1,xi2,xi3,kind,params)
end function

function b3_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_9260)

  use equilibrium_MHD, only: mod_b3_eq => b3_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_9260  

  Dummy_9260 = mod_b3_eq(xi1,xi2,xi3,kind,params)
end function

function rho_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_2636)

  use equilibrium_MHD, only: mod_rho_eq => rho_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_2636  

  Dummy_2636 = mod_rho_eq(xi1,xi2,xi3,kind,params)
end function

function curlb1_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_5079)

  use equilibrium_MHD, only: mod_curlb1_eq => curlb1_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_5079  

  Dummy_5079 = mod_curlb1_eq(xi1,xi2,xi3,kind,params)
end function

function curlb2_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_5230)

  use equilibrium_MHD, only: mod_curlb2_eq => curlb2_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_5230  

  Dummy_5230 = mod_curlb2_eq(xi1,xi2,xi3,kind,params)
end function

function curlb3_eq (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_5277)

  use equilibrium_MHD, only: mod_curlb3_eq => curlb3_eq
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_5277  

  Dummy_5277 = mod_curlb3_eq(xi1,xi2,xi3,kind,params)
end function