module kernels_projectors_local_ini

use initial_conditions, only: ini_9llcuj_rho_ini => rho_ini
use initial_conditions, only: ini_9llcuj_b1_ini => b1_ini
use initial_conditions, only: ini_9llcuj_u2_ini => u2_ini
use initial_conditions, only: ini_9llcuj_p_ini => p_ini
use initial_conditions, only: ini_9llcuj_u1_ini => u1_ini
use initial_conditions, only: ini_9llcuj_u3_ini => u3_ini
use initial_conditions, only: ini_9llcuj_b3_ini => b3_ini
use initial_conditions, only: ini_9llcuj_b2_ini => b2_ini

use equilibrium, only: eq_ocmj64_curlb2_eq => curlb2_eq
use equilibrium, only: eq_ocmj64_b3_eq => b3_eq
use equilibrium, only: eq_ocmj64_p_eq => p_eq
use equilibrium, only: eq_ocmj64_b1_eq => b1_eq
use equilibrium, only: eq_ocmj64_b2_eq => b2_eq
use equilibrium, only: eq_ocmj64_rho_eq => rho_eq
use equilibrium, only: eq_ocmj64_curlb1_eq => curlb1_eq
use equilibrium, only: eq_ocmj64_curlb3_eq => curlb3_eq

use mappings_analytical, only: mapping_8lqdx7_det_df => det_df
use mappings_analytical, only: mapping_8lqdx7_g_inv => g_inv
implicit none




contains

!........................................
function fun(xi1, xi2, xi3, kind_fun, kind_map, params) result(value)

  implicit none
  real(kind=8)  :: value  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind_fun
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params (0:)



  value = 0.0d0


  !initial conditions
  if (kind_fun == 1_8 ) then
    value = ini_9llcuj_p_ini(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 2_8 ) then
    value = ini_9llcuj_u1_ini(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 3_8 ) then
    value = ini_9llcuj_u2_ini(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 4_8 ) then
    value = ini_9llcuj_u3_ini(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 5_8 ) then
    value = ini_9llcuj_b1_ini(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 6_8 ) then
    value = ini_9llcuj_b2_ini(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 7_8 ) then
    value = ini_9llcuj_b3_ini(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 8_8 ) then
    value = ini_9llcuj_rho_ini(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 11_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 11_8)
  else if (kind_fun == 12_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 12_8)
  else if (kind_fun == 13_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 13_8)
  else if (kind_fun == 14_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 21_8)
  else if (kind_fun == 15_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 22_8)
  else if (kind_fun == 16_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 23_8)
  else if (kind_fun == 17_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 31_8)
  else if (kind_fun == 18_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 32_8)
  else if (kind_fun == 19_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 33_8)
  else if (kind_fun == 21_8 ) then
    value = eq_ocmj64_b2_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 31_8) + ( &
      -eq_ocmj64_b3_eq(xi1, xi2, xi3, kind_map, params))* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 21_8)
  else if (kind_fun == 22_8 ) then
    value = eq_ocmj64_b2_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 32_8) + ( &
      -eq_ocmj64_b3_eq(xi1, xi2, xi3, kind_map, params))* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 22_8)
  else if (kind_fun == 23_8 ) then
    value = eq_ocmj64_b2_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 33_8) + ( &
      -eq_ocmj64_b3_eq(xi1, xi2, xi3, kind_map, params))* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 23_8)
  else if (kind_fun == 24_8 ) then
    value = (-eq_ocmj64_b1_eq(xi1, xi2, xi3, kind_map, params))* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 31_8) + &
      eq_ocmj64_b3_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 11_8)
  else if (kind_fun == 25_8 ) then
    value = (-eq_ocmj64_b1_eq(xi1, xi2, xi3, kind_map, params))* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 32_8) + &
      eq_ocmj64_b3_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 12_8)
  else if (kind_fun == 26_8 ) then
    value = (-eq_ocmj64_b1_eq(xi1, xi2, xi3, kind_map, params))* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 33_8) + &
      eq_ocmj64_b3_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 13_8)
  else if (kind_fun == 27_8 ) then
    value = eq_ocmj64_b1_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 21_8) + ( &
      -eq_ocmj64_b2_eq(xi1, xi2, xi3, kind_map, params))* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 11_8)
  else if (kind_fun == 28_8 ) then
    value = eq_ocmj64_b1_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 22_8) + ( &
      -eq_ocmj64_b2_eq(xi1, xi2, xi3, kind_map, params))* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 12_8)
  else if (kind_fun == 29_8 ) then
    value = eq_ocmj64_b1_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 23_8) + ( &
      -eq_ocmj64_b2_eq(xi1, xi2, xi3, kind_map, params))* &
      mapping_8lqdx7_g_inv(xi1, xi2, xi3, kind_map, params, 13_8)
  else if (kind_fun == 31_8 ) then
    value = eq_ocmj64_rho_eq(xi1, xi2, xi3, kind_map, params)/ &
      mapping_8lqdx7_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 41_8 ) then
    value = (-eq_ocmj64_curlb3_eq(xi1, xi2, xi3, kind_map, params))/ &
      mapping_8lqdx7_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 42_8 ) then
    value = eq_ocmj64_curlb2_eq(xi1, xi2, xi3, kind_map, params)/ &
      mapping_8lqdx7_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 43_8 ) then
    value = eq_ocmj64_curlb3_eq(xi1, xi2, xi3, kind_map, params)/ &
      mapping_8lqdx7_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 44_8 ) then
    value = (-eq_ocmj64_curlb1_eq(xi1, xi2, xi3, kind_map, params))/ &
      mapping_8lqdx7_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 45_8 ) then
    value = (-eq_ocmj64_curlb2_eq(xi1, xi2, xi3, kind_map, params))/ &
      mapping_8lqdx7_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 46_8 ) then
    value = eq_ocmj64_curlb1_eq(xi1, xi2, xi3, kind_map, params)/ &
      mapping_8lqdx7_det_df(xi1, xi2, xi3, kind_map, params)
  else if (kind_fun == 91_8 ) then
    value = eq_ocmj64_p_eq(xi1, xi2, xi3, kind_map, params)
  end if






  !quantities for projection matrices K and S


  !quantities for projection matrix P


  !quantities for projection matrix W


  !quantities for projection matrix T
  !quantities for projection matrix Q
  return
end function
!........................................

!........................................
subroutine kernel_eva(n, xi1, xi2, xi3, mat_f, kind_fun, kind_map, &
      params)

  implicit none
  integer(kind=8), intent(in)  :: n (0:)
  real(kind=8), intent(in)  :: xi1 (0:)
  real(kind=8), intent(in)  :: xi2 (0:)
  real(kind=8), intent(in)  :: xi3 (0:)
  real(kind=8), intent(inout)  :: mat_f (0:,0:,0:)
  integer(kind=8), value  :: kind_fun
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params (0:)
  integer(kind=8)  :: i1  
  integer(kind=8)  :: i2  
  integer(kind=8)  :: i3  



  do i1 = 0, n(0_8) - 1_8, 1
    do i2 = 0, n(1_8) - 1_8, 1
      do i3 = 0, n(2_8) - 1_8, 1
        mat_f(i1, i2, i3) = fun(xi1(i1), xi2(i2), xi3(i3), kind_fun, &
      kind_map, params)


      end do

    end do

  end do

end subroutine
!........................................

end module