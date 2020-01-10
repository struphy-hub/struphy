module f2py_hylife_pic

use hylife_pic, only: mapping_matrices

use hylife_pic, only: pusher_step5

use hylife_pic, only: matrix_matrix

use hylife_pic, only: pusher_step4

use hylife_pic, only: matrix_vector

use hylife_pic, only: cross

use hylife_pic, only: det

use hylife_pic, only: transpose
implicit none




contains

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
      n0_p0, p0, n0_spans0, n1_spans0, spans0, n0_Nbase, Nbase, n0_b1, &
      n1_b1, n2_b1, b1, n0_b2, n1_b2, n2_b2, b2, n0_b3, n1_b3, n2_b3, &
      b3, n0_pp0_1, n1_pp0_1, pp0_1, n0_pp0_2, n1_pp0_2, pp0_2, &
      n0_pp0_3, n1_pp0_3, pp0_3, n0_pp1_1, n1_pp1_1, pp1_1, n0_pp1_2, &
      n1_pp1_2, pp1_2, n0_pp1_3, n1_pp1_3, pp1_3, n0_mapping, mapping, &
      dt, n0_Beq, Beq)

  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles - 1,0: &
      n1_particles - 1)
  integer(kind=4), intent(in)  :: n0_p0 
  integer(kind=4), intent(in)  :: p0 (0:n0_p0 - 1)
  integer(kind=4), intent(in)  :: n0_spans0 
  integer(kind=4), intent(in)  :: n1_spans0 
  integer(kind=4), intent(in)  :: spans0 (0:n0_spans0 - 1,0:n1_spans0 - &
      1)
  integer(kind=4), intent(in)  :: n0_Nbase 
  integer(kind=4), intent(in)  :: Nbase (0:n0_Nbase - 1)
  integer(kind=4), intent(in)  :: n0_b1 
  integer(kind=4), intent(in)  :: n1_b1 
  integer(kind=4), intent(in)  :: n2_b1 
  real(kind=8), intent(in)  :: b1 (0:n0_b1 - 1,0:n1_b1 - 1,0:n2_b1 - 1)
  integer(kind=4), intent(in)  :: n0_b2 
  integer(kind=4), intent(in)  :: n1_b2 
  integer(kind=4), intent(in)  :: n2_b2 
  real(kind=8), intent(in)  :: b2 (0:n0_b2 - 1,0:n1_b2 - 1,0:n2_b2 - 1)
  integer(kind=4), intent(in)  :: n0_b3 
  integer(kind=4), intent(in)  :: n1_b3 
  integer(kind=4), intent(in)  :: n2_b3 
  real(kind=8), intent(in)  :: b3 (0:n0_b3 - 1,0:n1_b3 - 1,0:n2_b3 - 1)
  integer(kind=4), intent(in)  :: n0_pp0_1 
  integer(kind=4), intent(in)  :: n1_pp0_1 
  real(kind=8), intent(in)  :: pp0_1 (0:n0_pp0_1 - 1,0:n1_pp0_1 - 1)
  integer(kind=4), intent(in)  :: n0_pp0_2 
  integer(kind=4), intent(in)  :: n1_pp0_2 
  real(kind=8), intent(in)  :: pp0_2 (0:n0_pp0_2 - 1,0:n1_pp0_2 - 1)
  integer(kind=4), intent(in)  :: n0_pp0_3 
  integer(kind=4), intent(in)  :: n1_pp0_3 
  real(kind=8), intent(in)  :: pp0_3 (0:n0_pp0_3 - 1,0:n1_pp0_3 - 1)
  integer(kind=4), intent(in)  :: n0_pp1_1 
  integer(kind=4), intent(in)  :: n1_pp1_1 
  real(kind=8), intent(in)  :: pp1_1 (0:n0_pp1_1 - 1,0:n1_pp1_1 - 1)
  integer(kind=4), intent(in)  :: n0_pp1_2 
  integer(kind=4), intent(in)  :: n1_pp1_2 
  real(kind=8), intent(in)  :: pp1_2 (0:n0_pp1_2 - 1,0:n1_pp1_2 - 1)
  integer(kind=4), intent(in)  :: n0_pp1_3 
  integer(kind=4), intent(in)  :: n1_pp1_3 
  real(kind=8), intent(in)  :: pp1_3 (0:n0_pp1_3 - 1,0:n1_pp1_3 - 1)
  integer(kind=4), intent(in)  :: n0_mapping 
  real(kind=8), intent(in)  :: mapping (0:n0_mapping - 1)
  real(kind=8), intent(in)  :: dt 
  integer(kind=4), intent(in)  :: n0_Beq 
  real(kind=8), intent(in)  :: Beq (0:n0_Beq - 1)

  call pusher_step5(particles,p0,spans0,Nbase,b1,b2,b3,pp0_1,pp0_2,pp0_3 &
      ,pp1_1,pp1_2,pp1_3,mapping,dt,Beq)
end subroutine
!........................................

end module