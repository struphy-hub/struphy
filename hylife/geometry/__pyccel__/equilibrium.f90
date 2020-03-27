module equilibrium

use mappings_analytical, only: mapping_ix6gvt_df => df
use mappings_analytical, only: mapping_ix6gvt_df_inv => df_inv
use mappings_analytical, only: mapping_ix6gvt_det_df => det_df
implicit none




contains

!........................................
function p_eq(xi1, xi2, xi3, kind, params) result(Dummy_7696)

  implicit none
  real(kind=8)  :: Dummy_7696  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_7696 = (((xi1*(xi2*xi3))*(-xi3 + 1_8))*(-xi2 + 1_8))*(-xi1 + 1_8 &
      )
  return
end function
!........................................

!........................................
function u1_eq(xi1, xi2, xi3, kind, params) result(Dummy_0019)

  implicit none
  real(kind=8)  :: Dummy_0019  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: df_11  
  real(kind=8)  :: df_21  
  real(kind=8)  :: df_31  
  real(kind=8)  :: ux  
  real(kind=8)  :: uy  
  real(kind=8)  :: uz  



  df_11 = mapping_ix6gvt_df(xi1, xi2, xi3, kind, params, 11_8)
  df_21 = mapping_ix6gvt_df(xi1, xi2, xi3, kind, params, 21_8)
  df_31 = mapping_ix6gvt_df(xi1, xi2, xi3, kind, params, 31_8)


  ux = 0.0d0
  uy = 0.0d0
  uz = 0.0d0


  Dummy_0019 = df_11*ux + df_21*uy + df_31*uz
  return
end function
!........................................

!........................................
function u2_eq(xi1, xi2, xi3, kind, params) result(Dummy_0187)

  implicit none
  real(kind=8)  :: Dummy_0187  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: df_12  
  real(kind=8)  :: df_22  
  real(kind=8)  :: df_32  
  real(kind=8)  :: ux  
  real(kind=8)  :: uy  
  real(kind=8)  :: uz  



  df_12 = mapping_ix6gvt_df(xi1, xi2, xi3, kind, params, 12_8)
  df_22 = mapping_ix6gvt_df(xi1, xi2, xi3, kind, params, 22_8)
  df_32 = mapping_ix6gvt_df(xi1, xi2, xi3, kind, params, 32_8)


  ux = 0.0d0
  uy = 0.0d0
  uz = 0.0d0


  Dummy_0187 = df_12*ux + df_22*uy + df_32*uz
  return
end function
!........................................

!........................................
function u3_eq(xi1, xi2, xi3, kind, params) result(Dummy_1888)

  implicit none
  real(kind=8)  :: Dummy_1888  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: df_13  
  real(kind=8)  :: df_23  
  real(kind=8)  :: df_33  
  real(kind=8)  :: ux  
  real(kind=8)  :: uy  
  real(kind=8)  :: uz  



  df_13 = mapping_ix6gvt_df(xi1, xi2, xi3, kind, params, 13_8)
  df_23 = mapping_ix6gvt_df(xi1, xi2, xi3, kind, params, 23_8)
  df_33 = mapping_ix6gvt_df(xi1, xi2, xi3, kind, params, 33_8)


  ux = 0.0d0
  uy = 0.0d0
  uz = 0.0d0


  Dummy_1888 = df_13*ux + df_23*uy + df_33*uz
  return
end function
!........................................

!........................................
function b1_eq(xi1, xi2, xi3, kind, params) result(Dummy_3634)

  implicit none
  real(kind=8)  :: Dummy_3634  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: dfinv_11  
  real(kind=8)  :: dfinv_12  
  real(kind=8)  :: dfinv_13  
  real(kind=8)  :: det_df  
  real(kind=8)  :: bx  
  real(kind=8)  :: by  
  real(kind=8)  :: bz  



  dfinv_11 = mapping_ix6gvt_df_inv(xi1, xi2, xi3, kind, params, 11_8)
  dfinv_12 = mapping_ix6gvt_df_inv(xi1, xi2, xi3, kind, params, 12_8)
  dfinv_13 = mapping_ix6gvt_df_inv(xi1, xi2, xi3, kind, params, 13_8)


  det_df = mapping_ix6gvt_det_df(xi1, xi2, xi3, kind, params)


  bx = (((xi1*(xi2*xi3))*(-xi3 + 1_8))*(-xi2 + 1_8))*(-xi1 + 1_8)
  by = 0.0d0
  bz = 0.0d0


  Dummy_3634 = det_df*(bx*dfinv_11 + by*dfinv_12 + bz*dfinv_13)
  return
end function
!........................................

!........................................
function b2_eq(xi1, xi2, xi3, kind, params) result(Dummy_3683)

  implicit none
  real(kind=8)  :: Dummy_3683  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: dfinv_21  
  real(kind=8)  :: dfinv_22  
  real(kind=8)  :: dfinv_23  
  real(kind=8)  :: det_df  
  real(kind=8)  :: bx  
  real(kind=8)  :: by  
  real(kind=8)  :: bz  



  dfinv_21 = mapping_ix6gvt_df_inv(xi1, xi2, xi3, kind, params, 21_8)
  dfinv_22 = mapping_ix6gvt_df_inv(xi1, xi2, xi3, kind, params, 22_8)
  dfinv_23 = mapping_ix6gvt_df_inv(xi1, xi2, xi3, kind, params, 23_8)


  det_df = mapping_ix6gvt_det_df(xi1, xi2, xi3, kind, params)


  bx = 1.0d0
  by = 0.0d0
  bz = 0.0d0


  Dummy_3683 = det_df*(bx*dfinv_21 + by*dfinv_22 + bz*dfinv_23)
  return
end function
!........................................

!........................................
function b3_eq(xi1, xi2, xi3, kind, params) result(Dummy_6987)

  implicit none
  real(kind=8)  :: Dummy_6987  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: dfinv_31  
  real(kind=8)  :: dfinv_32  
  real(kind=8)  :: dfinv_33  
  real(kind=8)  :: det_df  
  real(kind=8)  :: bx  
  real(kind=8)  :: by  
  real(kind=8)  :: bz  



  dfinv_31 = mapping_ix6gvt_df_inv(xi1, xi2, xi3, kind, params, 31_8)
  dfinv_32 = mapping_ix6gvt_df_inv(xi1, xi2, xi3, kind, params, 32_8)
  dfinv_33 = mapping_ix6gvt_df_inv(xi1, xi2, xi3, kind, params, 33_8)


  det_df = mapping_ix6gvt_det_df(xi1, xi2, xi3, kind, params)


  bx = 1.0d0
  by = 0.0d0
  bz = 0.0d0


  Dummy_6987 = det_df*(bx*dfinv_31 + by*dfinv_32 + bz*dfinv_33)
  return
end function
!........................................

!........................................
function rho_eq(xi1, xi2, xi3, kind, params) result(Dummy_9934)

  implicit none
  real(kind=8)  :: Dummy_9934  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_9934 = (((xi1*(xi2*(xi3*mapping_ix6gvt_det_df(xi1, xi2, xi3, &
      kind, params))))*(-xi3 + 1_8))*(-xi2 + 1_8))*(-xi1 + 1_8)
  return
end function
!........................................

!........................................
function curlb1_eq(xi1, xi2, xi3, kind, params) result(Dummy_3106)

  implicit none
  real(kind=8)  :: Dummy_3106  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_3106 = 0.0d0
  return
end function
!........................................

!........................................
function curlb2_eq(xi1, xi2, xi3, kind, params) result(Dummy_3291)

  implicit none
  real(kind=8)  :: Dummy_3291  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_3291 = 0.0d0
  return
end function
!........................................

!........................................
function curlb3_eq(xi1, xi2, xi3, kind, params) result(Dummy_2938)

  implicit none
  real(kind=8)  :: Dummy_2938  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)

  Dummy_2938 = 0.0d0
  return
end function
!........................................

end module