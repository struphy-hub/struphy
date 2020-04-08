function fh_eq_phys (x, y, z, vx, vy, vz) result(value)

  use equilibrium_PIC, only: mod_fh_eq_phys => fh_eq_phys
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8), intent(in)  :: vx 
  real(kind=8), intent(in)  :: vy 
  real(kind=8), intent(in)  :: vz 
  real(kind=8)  :: value  

  value = mod_fh_eq_phys(x,y,z,vx,vy,vz)
end function

function nh_eq_phys (x, y, z) result(nh0)

  use equilibrium_PIC, only: mod_nh_eq_phys => nh_eq_phys
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: nh0  

  nh0 = mod_nh_eq_phys(x,y,z)
end function

function jhx_eq (x, y, z) result(Dummy_2188)

  use equilibrium_PIC, only: mod_jhx_eq => jhx_eq
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: Dummy_2188  

  Dummy_2188 = mod_jhx_eq(x,y,z)
end function

function jhy_eq (x, y, z) result(Dummy_7873)

  use equilibrium_PIC, only: mod_jhy_eq => jhy_eq
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: Dummy_7873  

  Dummy_7873 = mod_jhy_eq(x,y,z)
end function

function jhz_eq (x, y, z) result(Dummy_0658)

  use equilibrium_PIC, only: mod_jhz_eq => jhz_eq
  implicit none
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  real(kind=8), intent(in)  :: z 
  real(kind=8)  :: Dummy_0658  

  Dummy_0658 = mod_jhz_eq(x,y,z)
end function

function eh_eq (kind_map, n0_params_map, params_map) result(value)

  use equilibrium_PIC, only: mod_eh_eq => eh_eq
  implicit none
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)
  real(kind=8)  :: value  

  value = mod_eh_eq(kind_map,params_map)
end function

function fh_eq (xi1, xi2, xi3, vx, vy, vz, kind_map, n0_params_map, &
      params_map) result(Dummy_3477)

  use equilibrium_PIC, only: mod_fh_eq => fh_eq
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
  real(kind=8)  :: Dummy_3477  

  Dummy_3477 = mod_fh_eq(xi1,xi2,xi3,vx,vy,vz,kind_map,params_map)
end function