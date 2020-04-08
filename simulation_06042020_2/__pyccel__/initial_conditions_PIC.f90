module initial_conditions_PIC

use mappings_analytical, only: mapping_m95z42_f => f
use mappings_analytical, only: mapping_m95z42_det_df => det_df
implicit none




contains

!........................................
function fh_ini_phys(x, y, z, vx, vy, vz) result(value)

  implicit none
  real(kind=8)  :: value  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z
  real(kind=8), value  :: vx
  real(kind=8), value  :: vy
  real(kind=8), value  :: vz
  real(kind=8)  :: v0x  
  real(kind=8)  :: v0y  
  real(kind=8)  :: v0z  
  real(kind=8)  :: vth  
  real(kind=8)  :: nh0  



  v0x = 2.5d0
  v0y = 0.0d0
  v0z = 0.0d0


  vth = 1.0d0


  nh0 = 0.05d0


  value = nh0*(exp((-(-v0z + vz)**2_8)/vth**2_8 - (-v0y + vy)**2_8/vth** &
      2_8 + (-(-v0x + vx)**2_8)/vth**2_8)/(3.14159265358979d0**(3_8/ &
      Real(2_8, 8))*vth**3_8))


  return
end function
!........................................

!........................................
function fh_ini(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map) result( &
      Dummy_5092)

  implicit none
  real(kind=8)  :: Dummy_5092  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  real(kind=8), value  :: vx
  real(kind=8), value  :: vy
  real(kind=8), value  :: vz
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  



  x = mapping_m95z42_f(xi1, xi2, xi3, kind_map, params_map, 1_8)
  y = mapping_m95z42_f(xi1, xi2, xi3, kind_map, params_map, 2_8)
  z = mapping_m95z42_f(xi1, xi2, xi3, kind_map, params_map, 3_8)


  Dummy_5092 = fh_ini_phys(x, y, z, vx, vy, vz)*mapping_m95z42_det_df( &
      xi1, xi2, xi3, kind_map, params_map)
  return
end function
!........................................

!........................................
function g_sampling(xi1, xi2, xi3, vx, vy, vz) result(Dummy_5530)

  implicit none
  real(kind=8)  :: Dummy_5530  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  real(kind=8), value  :: vx
  real(kind=8), value  :: vy
  real(kind=8), value  :: vz
  real(kind=8)  :: v0x  
  real(kind=8)  :: v0y  
  real(kind=8)  :: v0z  
  real(kind=8)  :: vth  



  v0x = 2.5d0
  v0y = 0.0d0
  v0z = 0.0d0


  vth = 1.0d0


  Dummy_5530 = exp((-(-v0z + vz)**2_8)/vth**2_8 - (-v0y + vy)**2_8/vth** &
      2_8 + (-(-v0x + vx)**2_8)/vth**2_8)/(3.14159265358979d0**(3_8/ &
      Real(2_8, 8))*vth**3_8)
  return
end function
!........................................

end module