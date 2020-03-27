module STRUPHY_sampling

implicit none




contains

!........................................
subroutine set_particles_symmetric(numbers, particles) 

  implicit none
  real(kind=8), intent(in)  :: numbers (0:,0:)
  real(kind=8), intent(inout)  :: particles (0:,0:)
  real(kind=8), allocatable :: q (:) 
  real(kind=8), allocatable :: v (:) 
  integer(kind=4) :: np  
  integer(kind=4) :: ierr  
  integer(kind=4) :: i_part  
  integer(kind=4) :: ip  






  allocate(q(0:2))
  q = 0.0
  allocate(v(0:2))
  v = 0.0
  np = size(particles(:, 0),1)


  do i_part = 0, np - 1, 1
    ip = modulo(i_part,64)


    if (ip == 0 ) then
      q = numbers(0:2, Int(i_part/Real(64, 8), 4))
      v = numbers(3:5, Int(i_part/Real(64, 8), 4))
    else if (modulo(ip,32) == 0 ) then
      v(2) = -v(2) + 1
    else if (modulo(ip,16) == 0 ) then
      v(1) = -v(1) + 1
    else if (modulo(ip,8) == 0 ) then
      v(0) = -v(0) + 1
    else if (modulo(ip,4) == 0 ) then
      q(2) = -q(2) + 1
    else if (modulo(ip,2) == 0 ) then
      q(1) = -q(1) + 1
    else
      q(0) = -q(0) + 1
    end if












    particles(i_part, 0:2) = q
    particles(i_part, 3:5) = v




  end do

  ierr = 0
end subroutine
!........................................

end module