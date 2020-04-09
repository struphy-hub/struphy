function p_ini_phys (x, y, z) result(p_phys)

  use initial_conditions_MHD, only: mod_p_ini_phys => p_ini_phys
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: p_phys  

  p_phys = mod_p_ini_phys(x,y,z)
end function

function ux_ini (x, y, z) result(ux)

  use initial_conditions_MHD, only: mod_ux_ini => ux_ini
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: ux  

  ux = mod_ux_ini(x,y,z)
end function

function uy_ini (x, y, z) result(uy)

  use initial_conditions_MHD, only: mod_uy_ini => uy_ini
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: uy  

  uy = mod_uy_ini(x,y,z)
end function

function uz_ini (x, y, z) result(uz)

  use initial_conditions_MHD, only: mod_uz_ini => uz_ini
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: uz  

  uz = mod_uz_ini(x,y,z)
end function

function bx_ini (x, y, z) result(bx)

  use initial_conditions_MHD, only: mod_bx_ini => bx_ini
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: bx  

  bx = mod_bx_ini(x,y,z)
end function

function by_ini (x, y, z) result(by)

  use initial_conditions_MHD, only: mod_by_ini => by_ini
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: by  

  by = mod_by_ini(x,y,z)
end function

function bz_ini (x, y, z) result(bz)

  use initial_conditions_MHD, only: mod_bz_ini => bz_ini
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: bz  

  bz = mod_bz_ini(x,y,z)
end function

function rho_ini_phys (x, y, z) result(rho_phys)

  use initial_conditions_MHD, only: mod_rho_ini_phys => rho_ini_phys
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: rho_phys  

  rho_phys = mod_rho_ini_phys(x,y,z)
end function

function p_ini (xi1, xi2, xi3, kind, n0_params, params) result(p_phys)

  use initial_conditions_MHD, only: mod_p_ini => p_ini
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: p_phys  

  p_phys = mod_p_ini(xi1,xi2,xi3,kind,params)
end function

function u1_ini (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_2797)

  use initial_conditions_MHD, only: mod_u1_ini => u1_ini
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_2797  

  Dummy_2797 = mod_u1_ini(xi1,xi2,xi3,kind,params)
end function

function u2_ini (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_8514)

  use initial_conditions_MHD, only: mod_u2_ini => u2_ini
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_8514  

  Dummy_8514 = mod_u2_ini(xi1,xi2,xi3,kind,params)
end function

function u3_ini (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_0955)

  use initial_conditions_MHD, only: mod_u3_ini => u3_ini
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_0955  

  Dummy_0955 = mod_u3_ini(xi1,xi2,xi3,kind,params)
end function

function b1_ini (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_6457)

  use initial_conditions_MHD, only: mod_b1_ini => b1_ini
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_6457  

  Dummy_6457 = mod_b1_ini(xi1,xi2,xi3,kind,params)
end function

function b2_ini (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_6318)

  use initial_conditions_MHD, only: mod_b2_ini => b2_ini
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_6318  

  Dummy_6318 = mod_b2_ini(xi1,xi2,xi3,kind,params)
end function

function b3_ini (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_8355)

  use initial_conditions_MHD, only: mod_b3_ini => b3_ini
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_8355  

  Dummy_8355 = mod_b3_ini(xi1,xi2,xi3,kind,params)
end function

function rho_ini (xi1, xi2, xi3, kind, n0_params, params) result( &
      Dummy_9291)

  use initial_conditions_MHD, only: mod_rho_ini => rho_ini
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  integer(kind=8), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params-1)
  real(kind=8)  :: Dummy_9291  

  Dummy_9291 = mod_rho_ini(xi1,xi2,xi3,kind,params)
end function