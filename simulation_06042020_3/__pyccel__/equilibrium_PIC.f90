module equilibrium_PIC

use mappings_analytical, only: mapping_ph8b2v_det_df => det_df
use mappings_analytical, only: mapping_ph8b2v_f => f
implicit none




contains

!........................................
function fh_eq_phys(x, y, z, vx, vy, vz) result(value)

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
function nh_eq_phys(x, y, z) result(nh0)

  implicit none
  real(kind=8)  :: nh0  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  nh0 = 0.05d0


  return
end function
!........................................

!........................................
function jhx_eq(x, y, z) result(Dummy_1820)

  implicit none
  real(kind=8)  :: Dummy_1820  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z
  real(kind=8)  :: nh0  
  real(kind=8)  :: v0x  



  nh0 = 0.05d0
  v0x = 2.5d0


  Dummy_1820 = nh0*v0x
  return
end function
!........................................

!........................................
function jhy_eq(x, y, z) result(Dummy_1320)

  implicit none
  real(kind=8)  :: Dummy_1320  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z
  real(kind=8)  :: nh0  
  real(kind=8)  :: v0y  



  nh0 = 0.05d0
  v0y = 0.0d0


  Dummy_1320 = nh0*v0y
  return
end function
!........................................

!........................................
function jhz_eq(x, y, z) result(Dummy_0311)

  implicit none
  real(kind=8)  :: Dummy_0311  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z
  real(kind=8)  :: nh0  
  real(kind=8)  :: v0z  



  nh0 = 0.05d0
  v0z = 0.0d0


  Dummy_0311 = nh0*v0z
  return
end function
!........................................

!........................................
function eh_eq(kind_map, params_map) result(value)

  implicit none
  real(kind=8)  :: value  
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
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


  value = nh0*(((((v0x**2_8 + v0y**2_8 + v0z**2_8 + 3_8*(vth**2_8/Real( &
      2_8, 8)))*params_map(2_8))*params_map(1_8))*params_map(0_8))/Real &
      (2_8, 8))


  return
end function
!........................................

!........................................
function fh_eq(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map) result( &
      Dummy_4945)

  implicit none
  real(kind=8)  :: Dummy_4945  
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



  x = mapping_ph8b2v_f(xi1, xi2, xi3, kind_map, params_map, 1_8)
  y = mapping_ph8b2v_f(xi1, xi2, xi3, kind_map, params_map, 2_8)
  z = mapping_ph8b2v_f(xi1, xi2, xi3, kind_map, params_map, 3_8)


  Dummy_4945 = fh_eq_phys(x, y, z, vx, vy, vz)*mapping_ph8b2v_det_df(xi1 &
      , xi2, xi3, kind_map, params_map)
  return
end function
!........................................

end module