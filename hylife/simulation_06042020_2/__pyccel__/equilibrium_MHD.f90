module equilibrium_MHD

use mappings_analytical, only: mapping_tvbypa_det_df => det_df
use mappings_analytical, only: mapping_tvbypa_f => f
use mappings_analytical, only: mapping_tvbypa_df => df
use mappings_analytical, only: mapping_tvbypa_df_inv => df_inv
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



  by = 1.0d0
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



  x = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 3_8)


  p_phys = p_eq_phys(x, y, z)


  return
end function
!........................................

!........................................
function u1_eq(xi1, xi2, xi3, kind, params) result(Dummy_8519)

  implicit none
  real(kind=8)  :: Dummy_8519  
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



  x = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 3_8)


  df_11 = mapping_tvbypa_df(xi1, xi2, xi3, kind, params, 11_8)
  df_21 = mapping_tvbypa_df(xi1, xi2, xi3, kind, params, 21_8)
  df_31 = mapping_tvbypa_df(xi1, xi2, xi3, kind, params, 31_8)


  ux = ux_eq(x, y, z)
  uy = uy_eq(x, y, z)
  uz = uz_eq(x, y, z)


  Dummy_8519 = df_11*ux + df_21*uy + df_31*uz
  return
end function
!........................................

!........................................
function u2_eq(xi1, xi2, xi3, kind, params) result(Dummy_9151)

  implicit none
  real(kind=8)  :: Dummy_9151  
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



  x = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 3_8)


  df_12 = mapping_tvbypa_df(xi1, xi2, xi3, kind, params, 12_8)
  df_22 = mapping_tvbypa_df(xi1, xi2, xi3, kind, params, 22_8)
  df_32 = mapping_tvbypa_df(xi1, xi2, xi3, kind, params, 32_8)


  ux = ux_eq(x, y, z)
  uy = uy_eq(x, y, z)
  uz = uz_eq(x, y, z)


  Dummy_9151 = df_12*ux + df_22*uy + df_32*uz
  return
end function
!........................................

!........................................
function u3_eq(xi1, xi2, xi3, kind, params) result(Dummy_3898)

  implicit none
  real(kind=8)  :: Dummy_3898  
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



  x = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 3_8)


  df_13 = mapping_tvbypa_df(xi1, xi2, xi3, kind, params, 13_8)
  df_23 = mapping_tvbypa_df(xi1, xi2, xi3, kind, params, 23_8)
  df_33 = mapping_tvbypa_df(xi1, xi2, xi3, kind, params, 33_8)


  ux = ux_eq(x, y, z)
  uy = uy_eq(x, y, z)
  uz = uz_eq(x, y, z)


  Dummy_3898 = df_13*ux + df_23*uy + df_33*uz
  return
end function
!........................................

!........................................
function b1_eq(xi1, xi2, xi3, kind, params) result(Dummy_9373)

  implicit none
  real(kind=8)  :: Dummy_9373  
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



  x = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 3_8)


  dfinv_11 = mapping_tvbypa_df_inv(xi1, xi2, xi3, kind, params, 11_8)
  dfinv_12 = mapping_tvbypa_df_inv(xi1, xi2, xi3, kind, params, 12_8)
  dfinv_13 = mapping_tvbypa_df_inv(xi1, xi2, xi3, kind, params, 13_8)


  det_df = mapping_tvbypa_det_df(xi1, xi2, xi3, kind, params)


  bx = bx_eq(x, y, z)
  by = by_eq(x, y, z)
  bz = bz_eq(x, y, z)


  Dummy_9373 = det_df*(bx*dfinv_11 + by*dfinv_12 + bz*dfinv_13)
  return
end function
!........................................

!........................................
function b2_eq(xi1, xi2, xi3, kind, params) result(Dummy_1603)

  implicit none
  real(kind=8)  :: Dummy_1603  
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



  x = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 3_8)


  dfinv_21 = mapping_tvbypa_df_inv(xi1, xi2, xi3, kind, params, 21_8)
  dfinv_22 = mapping_tvbypa_df_inv(xi1, xi2, xi3, kind, params, 22_8)
  dfinv_23 = mapping_tvbypa_df_inv(xi1, xi2, xi3, kind, params, 23_8)


  det_df = mapping_tvbypa_det_df(xi1, xi2, xi3, kind, params)


  bx = bx_eq(x, y, z)
  by = by_eq(x, y, z)
  bz = bz_eq(x, y, z)


  Dummy_1603 = det_df*(bx*dfinv_21 + by*dfinv_22 + bz*dfinv_23)
  return
end function
!........................................

!........................................
function b3_eq(xi1, xi2, xi3, kind, params) result(Dummy_1337)

  implicit none
  real(kind=8)  :: Dummy_1337  
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



  x = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 3_8)


  dfinv_31 = mapping_tvbypa_df_inv(xi1, xi2, xi3, kind, params, 31_8)
  dfinv_32 = mapping_tvbypa_df_inv(xi1, xi2, xi3, kind, params, 32_8)
  dfinv_33 = mapping_tvbypa_df_inv(xi1, xi2, xi3, kind, params, 33_8)


  det_df = mapping_tvbypa_det_df(xi1, xi2, xi3, kind, params)


  bx = bx_eq(x, y, z)
  by = by_eq(x, y, z)
  bz = bz_eq(x, y, z)


  Dummy_1337 = det_df*(bx*dfinv_31 + by*dfinv_32 + bz*dfinv_33)
  return
end function
!........................................

!........................................
function rho_eq(xi1, xi2, xi3, kind, params) result(Dummy_9385)

  implicit none
  real(kind=8)  :: Dummy_9385  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: x  
  real(kind=8)  :: y  
  real(kind=8)  :: z  
  real(kind=8)  :: rho_phys  



  x = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 1_8)
  y = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 2_8)
  z = mapping_tvbypa_f(xi1, xi2, xi3, kind, params, 3_8)


  rho_phys = rho_eq_phys(x, y, z)


  Dummy_9385 = rho_phys*mapping_tvbypa_det_df(xi1, xi2, xi3, kind, &
      params)
  return
end function
!........................................

!........................................
function curlb1_eq(xi1, xi2, xi3, kind, params) result(Dummy_7817)

  implicit none
  real(kind=8)  :: Dummy_7817  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_7817 = 0.0d0
  return
end function
!........................................

!........................................
function curlb2_eq(xi1, xi2, xi3, kind, params) result(Dummy_7656)

  implicit none
  real(kind=8)  :: Dummy_7656  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_7656 = 0.0d0
  return
end function
!........................................

!........................................
function curlb3_eq(xi1, xi2, xi3, kind, params) result(Dummy_7791)

  implicit none
  real(kind=8)  :: Dummy_7791  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_7791 = 0.0d0
  return
end function
!........................................

end module