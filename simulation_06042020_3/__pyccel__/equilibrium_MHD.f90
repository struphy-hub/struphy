module equilibrium_MHD

use mappings_analytical, only: mapping_sq8y85_df => df
use mappings_analytical, only: mapping_sq8y85_det_df => det_df
use mappings_analytical, only: mapping_sq8y85_f => f
use mappings_analytical, only: mapping_sq8y85_df_inv => df_inv
implicit none




contains

!........................................
function p_eq_phys(x, y, z) result(p_phys)

  implicit none
  real(kind=8)  :: p_phys  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  p_phys = 1.0d0
  !p_phys = (1 - x) * (1 - y) * (1 - z) * x * y * z


  return
end function
!........................................

!........................................
function ux_eq(x, y, z) result(ux)

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
function uy_eq(x, y, z) result(uy)

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
function uz_eq(x, y, z) result(uz)

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
function bx_eq(x, y, z) result(bx)

  implicit none
  real(kind=8)  :: bx  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  bx = 1.0d0
  !bx = sin(2*pi*xi1) * cos(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**2


  return
end function
!........................................

!........................................
function by_eq(x, y, z) result(by)

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
function bz_eq(x, y, z) result(bz)

  implicit none
  real(kind=8)  :: bz  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  bz = 0.0d0
  !bz = cos(2*pi*xi1) * cos(2*pi*xi2) * sin(2*pi*xi3) * (2*pi)**2


  return
end function
!........................................

!........................................
function rho_eq_phys(x, y, z) result(rho_phys)

  implicit none
  real(kind=8)  :: rho_phys  
  real(kind=8), value  :: x
  real(kind=8), value  :: y
  real(kind=8), value  :: z



  rho_phys = 1.0d0
  !rho_phys = cos(2*pi*xi1) * cos(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**3


  return
end function
!........................................

!........................................
function p_eq(xi1, xi2, xi3, kind, params) result(p_phys)

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



  x = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 3_8)


  p_phys = p_eq_phys(x, y, z)


  return
end function
!........................................

!........................................
function u1_eq(xi1, xi2, xi3, kind, params) result(Dummy_5804)

  implicit none
  real(kind=8)  :: Dummy_5804  
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



  x = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 3_8)


  df_11 = mapping_sq8y85_df(xi1, xi2, xi3, kind, params, 11_8)
  df_21 = mapping_sq8y85_df(xi1, xi2, xi3, kind, params, 21_8)
  df_31 = mapping_sq8y85_df(xi1, xi2, xi3, kind, params, 31_8)


  ux = ux_eq(x, y, z)
  uy = uy_eq(x, y, z)
  uz = uz_eq(x, y, z)


  Dummy_5804 = df_11*ux + df_21*uy + df_31*uz
  return
end function
!........................................

!........................................
function u2_eq(xi1, xi2, xi3, kind, params) result(Dummy_4462)

  implicit none
  real(kind=8)  :: Dummy_4462  
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



  x = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 3_8)


  df_12 = mapping_sq8y85_df(xi1, xi2, xi3, kind, params, 12_8)
  df_22 = mapping_sq8y85_df(xi1, xi2, xi3, kind, params, 22_8)
  df_32 = mapping_sq8y85_df(xi1, xi2, xi3, kind, params, 32_8)


  ux = ux_eq(x, y, z)
  uy = uy_eq(x, y, z)
  uz = uz_eq(x, y, z)


  Dummy_4462 = df_12*ux + df_22*uy + df_32*uz
  return
end function
!........................................

!........................................
function u3_eq(xi1, xi2, xi3, kind, params) result(Dummy_8443)

  implicit none
  real(kind=8)  :: Dummy_8443  
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



  x = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 3_8)


  df_13 = mapping_sq8y85_df(xi1, xi2, xi3, kind, params, 13_8)
  df_23 = mapping_sq8y85_df(xi1, xi2, xi3, kind, params, 23_8)
  df_33 = mapping_sq8y85_df(xi1, xi2, xi3, kind, params, 33_8)


  ux = ux_eq(x, y, z)
  uy = uy_eq(x, y, z)
  uz = uz_eq(x, y, z)


  Dummy_8443 = df_13*ux + df_23*uy + df_33*uz
  return
end function
!........................................

!........................................
function b1_eq(xi1, xi2, xi3, kind, params) result(Dummy_1385)

  implicit none
  real(kind=8)  :: Dummy_1385  
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



  x = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 3_8)


  dfinv_11 = mapping_sq8y85_df_inv(xi1, xi2, xi3, kind, params, 11_8)
  dfinv_12 = mapping_sq8y85_df_inv(xi1, xi2, xi3, kind, params, 12_8)
  dfinv_13 = mapping_sq8y85_df_inv(xi1, xi2, xi3, kind, params, 13_8)


  det_df = mapping_sq8y85_det_df(xi1, xi2, xi3, kind, params)


  bx = bx_eq(x, y, z)
  by = by_eq(x, y, z)
  bz = bz_eq(x, y, z)


  Dummy_1385 = det_df*(bx*dfinv_11 + by*dfinv_12 + bz*dfinv_13)
  return
end function
!........................................

!........................................
function b2_eq(xi1, xi2, xi3, kind, params) result(Dummy_5342)

  implicit none
  real(kind=8)  :: Dummy_5342  
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



  x = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 3_8)


  dfinv_21 = mapping_sq8y85_df_inv(xi1, xi2, xi3, kind, params, 21_8)
  dfinv_22 = mapping_sq8y85_df_inv(xi1, xi2, xi3, kind, params, 22_8)
  dfinv_23 = mapping_sq8y85_df_inv(xi1, xi2, xi3, kind, params, 23_8)


  det_df = mapping_sq8y85_det_df(xi1, xi2, xi3, kind, params)


  bx = bx_eq(x, y, z)
  by = by_eq(x, y, z)
  bz = bz_eq(x, y, z)


  Dummy_5342 = det_df*(bx*dfinv_21 + by*dfinv_22 + bz*dfinv_23)
  return
end function
!........................................

!........................................
function b3_eq(xi1, xi2, xi3, kind, params) result(Dummy_7281)

  implicit none
  real(kind=8)  :: Dummy_7281  
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



  x = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 3_8)


  dfinv_31 = mapping_sq8y85_df_inv(xi1, xi2, xi3, kind, params, 31_8)
  dfinv_32 = mapping_sq8y85_df_inv(xi1, xi2, xi3, kind, params, 32_8)
  dfinv_33 = mapping_sq8y85_df_inv(xi1, xi2, xi3, kind, params, 33_8)


  det_df = mapping_sq8y85_det_df(xi1, xi2, xi3, kind, params)


  bx = bx_eq(x, y, z)
  by = by_eq(x, y, z)
  bz = bz_eq(x, y, z)


  Dummy_7281 = det_df*(bx*dfinv_31 + by*dfinv_32 + bz*dfinv_33)
  return
end function
!........................................

!........................................
function rho_eq(xi1, xi2, xi3, kind, params) result(Dummy_2220)

  implicit none
  real(kind=8)  :: Dummy_2220  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  
  real(kind=8)  :: rho_phys  



  x = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_sq8y85_f(xi1, xi2, xi3, kind, params, 3_8)


  rho_phys = rho_eq_phys(x, y, z)


  Dummy_2220 = rho_phys*mapping_sq8y85_det_df(xi1, xi2, xi3, kind, &
      params)
  return
end function
!........................................

!........................................
function curlb1_eq(xi1, xi2, xi3, kind, params) result(Dummy_3529)

  implicit none
  real(kind=8)  :: Dummy_3529  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_3529 = 0.0d0
  return
end function
!........................................

!........................................
function curlb2_eq(xi1, xi2, xi3, kind, params) result(Dummy_3544)

  implicit none
  real(kind=8)  :: Dummy_3544  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_3544 = 0.0d0
  return
end function
!........................................

!........................................
function curlb3_eq(xi1, xi2, xi3, kind, params) result(Dummy_3649)

  implicit none
  real(kind=8)  :: Dummy_3649  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_3649 = 0.0d0
  return
end function
!........................................

end module