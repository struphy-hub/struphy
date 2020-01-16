module f2py_echo_pusher

use echo_pusher, only: pusher_step3

use echo_pusher, only: mapping_matrices

use echo_pusher, only: transpose

use echo_pusher, only: det

use echo_pusher, only: matrix_matrix

use echo_pusher, only: pusher_step4

use echo_pusher, only: pusher_step5

use echo_pusher, only: matrix_vector
implicit none




contains

!........................................
subroutine f2py_pusher_step3(n0_particles, n1_particles, particles, &
      n0_mapping, mapping, dt, n0_B_part, n1_B_part, B_part, n0_U_part, &
      n1_U_part, U_part)

  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles - 1,0: &
      n1_particles - 1)
  integer(kind=4), intent(in)  :: n0_mapping 
  real(kind=8), intent(in)  :: mapping (0:n0_mapping - 1)
  real(kind=8), intent(in)  :: dt 
  integer(kind=4), intent(in)  :: n0_B_part 
  integer(kind=4), intent(in)  :: n1_B_part 
  real(kind=8), intent(in)  :: B_part (0:n0_B_part - 1,0:n1_B_part - 1)
  integer(kind=4), intent(in)  :: n0_U_part 
  integer(kind=4), intent(in)  :: n1_U_part 
  real(kind=8), intent(in)  :: U_part (0:n0_U_part - 1,0:n1_U_part - 1)

  call pusher_step3(particles,mapping,dt,B_part,U_part)
end subroutine
!........................................

!........................................
subroutine f2py_pusher_step4(n0_particles, n1_particles, particles, &
      n0_mapping, mapping, dt)

  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles - 1,0: &
      n1_particles - 1)
  integer(kind=4), intent(in)  :: n0_mapping 
  real(kind=8), intent(in)  :: mapping (0:n0_mapping - 1)
  real(kind=8), intent(in)  :: dt 

  call pusher_step4(particles,mapping,dt)
end subroutine
!........................................

!........................................
subroutine f2py_pusher_step5(n0_particles, n1_particles, particles, &
      n0_mapping, mapping, dt, n0_B_part, n1_B_part, B_part)

  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles - 1,0: &
      n1_particles - 1)
  integer(kind=4), intent(in)  :: n0_mapping 
  real(kind=8), intent(in)  :: mapping (0:n0_mapping - 1)
  real(kind=8), intent(in)  :: dt 
  integer(kind=4), intent(in)  :: n0_B_part 
  integer(kind=4), intent(in)  :: n1_B_part 
  real(kind=8), intent(in)  :: B_part (0:n0_B_part - 1,0:n1_B_part - 1)

  call pusher_step5(particles,mapping,dt,B_part)
end subroutine
!........................................

end module