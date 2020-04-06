function fh_ini_phys (x, y, z, vx, vy, vz) result(value)

  use initial_conditions_PIC, only: mod_fh_ini_phys => fh_ini_phys
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8), intent(in)  :: vx 
  real(kind=8), intent(in)  :: vy 
  real(kind=8), intent(in)  :: vz 
  real(kind=8)  :: value  

  value = mod_fh_ini_phys(x,y,z,vx,vy,vz)
end function

function fh_ini (xi1, xi2, xi3, vx, vy, vz, kind_map, n0_params_map, &
      params_map) result(Dummy_1343)

  use initial_conditions_PIC, only: mod_fh_ini => fh_ini
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  real(kind=8), intent(in)  :: vx 
  real(kind=8), intent(in)  :: vy 
  real(kind=8), intent(in)  :: vz 
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)
  real(kind=8)  :: Dummy_1343  

  Dummy_1343 = mod_fh_ini(xi1,xi2,xi3,vx,vy,vz,kind_map,params_map)
end function

function g_sampling (xi1, xi2, xi3, vx, vy, vz) result(Dummy_6819)

  use initial_conditions_PIC, only: mod_g_sampling => g_sampling
  implicit none
  real(kind=8), intent(in)  :: xi1 
  real(kind=8), intent(in)  :: xi2 
  real(kind=8), intent(in)  :: xi3 
  real(kind=8), intent(in)  :: vx 
  real(kind=8), intent(in)  :: vy 
  real(kind=8), intent(in)  :: vz 
  real(kind=8)  :: Dummy_6819  

  Dummy_6819 = mod_g_sampling(xi1,xi2,xi3,vx,vy,vz)
end function