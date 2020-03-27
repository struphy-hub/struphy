module test

use equilibrium, only: eq_4bcrlv_p_eq => p_eq

use mappings_analytical, only: mapping_1bcrhz_g_inv => g_inv
implicit none




contains

!........................................
function fun(xi1, xi2, xi3, kind_map, params) result(value)

  implicit none
  real(kind=8)  :: value  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params (0:)



  value = eq_4bcrlv_p_eq(xi1, xi2, xi3, kind_map, params)* &
      mapping_1bcrhz_g_inv(xi1, xi2, xi3, kind_map, params, 11_8)


  return
end function
!........................................

end module