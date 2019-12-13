module f2py_utilitis_pic

use utilitis_pic, only: find_span

use utilitis_pic, only: basis_funs

use utilitis_pic, only: cross
implicit none




contains

!........................................
subroutine f2py_cross(n0_a, a, n0_b, b, n0_r, r) 

  implicit none
  integer(kind=4), intent(in)  :: n0_a 
  real(kind=8), intent(in)  :: a (0:n0_a - 1)
  integer(kind=4), intent(in)  :: n0_b 
  real(kind=8), intent(in)  :: b (0:n0_b - 1)
  integer(kind=4), intent(in)  :: n0_r 
  real(kind=8), intent(inout)  :: r (0:n0_r - 1)

  call cross(a,b,r)
end subroutine
!........................................

end module