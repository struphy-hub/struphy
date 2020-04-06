module initial_conditions_MHD

use mappings_analytical, only: mapping_vsoju2_f => f
use mappings_analytical, only: mapping_vsoju2_df_inv => df_inv
use mappings_analytical, only: mapping_vsoju2_det_df => det_df
use mappings_analytical, only: mapping_vsoju2_df => df
implicit none




contains

!........................................
function p_ini_phys(x, y, z) result(p_phys)

  implicit none
  real(kind=8)  :: p_phys  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  p_phys = 0.0d0
  !p_phys = sin(2*pi*xi1) * sin(2*pi*xi2) * sin(2*pi*xi3)


  return
end function
!........................................

!........................................
function ux_ini(x, y, z) result(ux)

  implicit none
  real(kind=8)  :: ux  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  ux = 0.0d0
  !ux = cos(2*pi*xi1) * sin(2*pi*xi2) * sin(2*pi*xi3) * 2*pi


  return
end function
!........................................

!........................................
function uy_ini(x, y, z) result(uy)

  implicit none
  real(kind=8)  :: uy  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  uy = 0.0d0
  !uy = sin(2*pi*xi1) * cos(2*pi*xi2) * sin(2*pi*xi3) * 2*pi


  return
end function
!........................................

!........................................
function uz_ini(x, y, z) result(uz)

  implicit none
  real(kind=8)  :: uz  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  uz = 0.0d0
  !uz = sin(2*pi*xi1) * sin(2*pi*xi2) * cos(2*pi*xi3) * 2*pi


  return
end function
!........................................

!........................................
function bx_ini(x, y, z) result(bx)

  implicit none
  real(kind=8)  :: bx  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  bx = 0.0d0
  !bx = sin(2*pi*xi1) * cos(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**2


  return
end function
!........................................

!........................................
function by_ini(x, y, z) result(by)

  implicit none
  real(kind=8)  :: by  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  by = 0.0d0
  !by = cos(2*pi*xi1) * sin(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**2


  return
end function
!........................................

!........................................
function bz_ini(x, y, z) result(bz)

  implicit none
  real(kind=8)  :: bz  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z
  real(kind=8)  :: amp  
  real(kind=8)  :: kx  
  real(kind=8)  :: ky  
  real(kind=8)  :: kz  



  amp = 0.001d0


  kx = 0.75d0
  ky = 0.0d0
  kz = 0.0d0


  bz = amp*sin(kx*x + ky*y)


  !bz = cos(2*pi*xi1) * cos(2*pi*xi2) * sin(2*pi*xi3) * (2*pi)**2


  return
end function
!........................................

!........................................
function rho_ini_phys(x, y, z) result(rho_phys)

  implicit none
  real(kind=8)  :: rho_phys  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  rho_phys = 0.0d0
  !rho_phys = cos(2*pi*xi1) * cos(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**3


  return
end function
!........................................

!........................................
function p_ini(xi1, xi2, xi3, kind, params) result(p_phys)

  implicit none
  real(kind=8)  :: p_phys  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  



  x = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 3_8)


  p_phys = p_ini_phys(x, y, z)


  return
end function
!........................................

!........................................
function u1_ini(xi1, xi2, xi3, kind, params) result(Dummy_5650)

  implicit none
  real(kind=8)  :: Dummy_5650  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  
  real(kind=8)  :: df_11  
  real(kind=8)  :: df_21  
  real(kind=8)  :: df_31  
  real(kind=8)  :: ux  
  real(kind=8)  :: uy  
  real(kind=8)  :: uz  



  x = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 3_8)


  df_11 = mapping_vsoju2_df(xi1, xi2, xi3, kind, params, 11_8)
  df_21 = mapping_vsoju2_df(xi1, xi2, xi3, kind, params, 21_8)
  df_31 = mapping_vsoju2_df(xi1, xi2, xi3, kind, params, 31_8)


  ux = ux_ini(x, y, z)
  uy = uy_ini(x, y, z)
  uz = uz_ini(x, y, z)


  Dummy_5650 = df_11*ux + df_21*uy + df_31*uz
  return
end function
!........................................

!........................................
function u2_ini(xi1, xi2, xi3, kind, params) result(Dummy_6846)

  implicit none
  real(kind=8)  :: Dummy_6846  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  
  real(kind=8)  :: df_12  
  real(kind=8)  :: df_22  
  real(kind=8)  :: df_32  
  real(kind=8)  :: ux  
  real(kind=8)  :: uy  
  real(kind=8)  :: uz  



  x = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 3_8)


  df_12 = mapping_vsoju2_df(xi1, xi2, xi3, kind, params, 12_8)
  df_22 = mapping_vsoju2_df(xi1, xi2, xi3, kind, params, 22_8)
  df_32 = mapping_vsoju2_df(xi1, xi2, xi3, kind, params, 32_8)


  ux = ux_ini(x, y, z)
  uy = uy_ini(x, y, z)
  uz = uz_ini(x, y, z)


  Dummy_6846 = df_12*ux + df_22*uy + df_32*uz
  return
end function
!........................................

!........................................
function u3_ini(xi1, xi2, xi3, kind, params) result(Dummy_6839)

  implicit none
  real(kind=8)  :: Dummy_6839  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  
  real(kind=8)  :: df_13  
  real(kind=8)  :: df_23  
  real(kind=8)  :: df_33  
  real(kind=8)  :: ux  
  real(kind=8)  :: uy  
  real(kind=8)  :: uz  



  x = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 3_8)


  df_13 = mapping_vsoju2_df(xi1, xi2, xi3, kind, params, 13_8)
  df_23 = mapping_vsoju2_df(xi1, xi2, xi3, kind, params, 23_8)
  df_33 = mapping_vsoju2_df(xi1, xi2, xi3, kind, params, 33_8)


  ux = ux_ini(x, y, z)
  uy = uy_ini(x, y, z)
  uz = uz_ini(x, y, z)


  Dummy_6839 = df_13*ux + df_23*uy + df_33*uz
  return
end function
!........................................

!........................................
function b1_ini(xi1, xi2, xi3, kind, params) result(Dummy_9721)

  implicit none
  real(kind=8)  :: Dummy_9721  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  
  real(kind=8)  :: dfinv_11  
  real(kind=8)  :: dfinv_12  
  real(kind=8)  :: dfinv_13  
  real(kind=8)  :: det_df  
  real(kind=8)  :: bx  
  real(kind=8)  :: by  
  real(kind=8)  :: bz  



  x = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 3_8)


  dfinv_11 = mapping_vsoju2_df_inv(xi1, xi2, xi3, kind, params, 11_8)
  dfinv_12 = mapping_vsoju2_df_inv(xi1, xi2, xi3, kind, params, 12_8)
  dfinv_13 = mapping_vsoju2_df_inv(xi1, xi2, xi3, kind, params, 13_8)


  det_df = mapping_vsoju2_det_df(xi1, xi2, xi3, kind, params)


  bx = bx_ini(x, y, z)
  by = by_ini(x, y, z)
  bz = bz_ini(x, y, z)


  Dummy_9721 = det_df*(bx*dfinv_11 + by*dfinv_12 + bz*dfinv_13)
  return
end function
!........................................

!........................................
function b2_ini(xi1, xi2, xi3, kind, params) result(Dummy_1000)

  implicit none
  real(kind=8)  :: Dummy_1000  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  
  real(kind=8)  :: dfinv_21  
  real(kind=8)  :: dfinv_22  
  real(kind=8)  :: dfinv_23  
  real(kind=8)  :: det_df  
  real(kind=8)  :: bx  
  real(kind=8)  :: by  
  real(kind=8)  :: bz  



  x = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 3_8)


  dfinv_21 = mapping_vsoju2_df_inv(xi1, xi2, xi3, kind, params, 21_8)
  dfinv_22 = mapping_vsoju2_df_inv(xi1, xi2, xi3, kind, params, 22_8)
  dfinv_23 = mapping_vsoju2_df_inv(xi1, xi2, xi3, kind, params, 23_8)


  det_df = mapping_vsoju2_det_df(xi1, xi2, xi3, kind, params)


  bx = bx_ini(x, y, z)
  by = by_ini(x, y, z)
  bz = bz_ini(x, y, z)


  Dummy_1000 = det_df*(bx*dfinv_21 + by*dfinv_22 + bz*dfinv_23)
  return
end function
!........................................

!........................................
function b3_ini(xi1, xi2, xi3, kind, params) result(Dummy_7899)

  implicit none
  real(kind=8)  :: Dummy_7899  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  
  real(kind=8)  :: dfinv_31  
  real(kind=8)  :: dfinv_32  
  real(kind=8)  :: dfinv_33  
  real(kind=8)  :: det_df  
  real(kind=8)  :: bx  
  real(kind=8)  :: by  
  real(kind=8)  :: bz  



  x = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 3_8)


  dfinv_31 = mapping_vsoju2_df_inv(xi1, xi2, xi3, kind, params, 31_8)
  dfinv_32 = mapping_vsoju2_df_inv(xi1, xi2, xi3, kind, params, 32_8)
  dfinv_33 = mapping_vsoju2_df_inv(xi1, xi2, xi3, kind, params, 33_8)


  det_df = mapping_vsoju2_det_df(xi1, xi2, xi3, kind, params)


  bx = bx_ini(x, y, z)
  by = by_ini(x, y, z)
  bz = bz_ini(x, y, z)


  Dummy_7899 = det_df*(bx*dfinv_31 + by*dfinv_32 + bz*dfinv_33)
  return
end function
!........................................

!........................................
function rho_ini(xi1, xi2, xi3, kind, params) result(Dummy_1431)

  implicit none
  real(kind=8)  :: Dummy_1431  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  
  real(kind=8)  :: rho_phys  



  x = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_vsoju2_f(xi1, xi2, xi3, kind, params, 3_8)


  rho_phys = rho_ini_phys(x, y, z)


  Dummy_1431 = rho_phys*mapping_vsoju2_det_df(xi1, xi2, xi3, kind, &
      params)
  return
end function
!........................................

end module