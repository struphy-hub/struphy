module f2py_echo_fields

use echo_fields, only: evaluate_1form

use echo_fields, only: evaluate_2form
implicit none




contains

!........................................
subroutine f2py_evaluate_1form(n0_particles_pos, n1_particles_pos, &
      particles_pos, n0_p0, p0, n0_spans0, n1_spans0, spans0, n0_Nbase, &
      Nbase, Np, n0_u1, n1_u1, n2_u1, u1, n0_u2, n1_u2, n2_u2, u2, &
      n0_u3, n1_u3, n2_u3, u3, n0_Ueq, Ueq, n0_pp0_1, n1_pp0_1, pp0_1, &
      n0_pp0_2, n1_pp0_2, pp0_2, n0_pp0_3, n1_pp0_3, pp0_3, n0_pp1_1, &
      n1_pp1_1, pp1_1, n0_pp1_2, n1_pp1_2, pp1_2, n0_pp1_3, n1_pp1_3, &
      pp1_3, n0_U_part, n1_U_part, U_part)

  implicit none
  integer(kind=4), intent(in)  :: n0_particles_pos 
  integer(kind=4), intent(in)  :: n1_particles_pos 
  real(kind=8), intent(in)  :: particles_pos (0:n0_particles_pos - 1,0: &
      n1_particles_pos - 1)
  integer(kind=4), intent(in)  :: n0_p0 
  integer(kind=4), intent(in)  :: p0 (0:n0_p0 - 1)
  integer(kind=4), intent(in)  :: n0_spans0 
  integer(kind=4), intent(in)  :: n1_spans0 
  integer(kind=4), intent(in)  :: spans0 (0:n0_spans0 - 1,0:n1_spans0 - &
      1)
  integer(kind=4), intent(in)  :: n0_Nbase 
  integer(kind=4), intent(in)  :: Nbase (0:n0_Nbase - 1)
  integer(kind=4), intent(in)  :: Np 
  integer(kind=4), intent(in)  :: n0_u1 
  integer(kind=4), intent(in)  :: n1_u1 
  integer(kind=4), intent(in)  :: n2_u1 
  real(kind=8), intent(in)  :: u1 (0:n0_u1 - 1,0:n1_u1 - 1,0:n2_u1 - 1)
  integer(kind=4), intent(in)  :: n0_u2 
  integer(kind=4), intent(in)  :: n1_u2 
  integer(kind=4), intent(in)  :: n2_u2 
  real(kind=8), intent(in)  :: u2 (0:n0_u2 - 1,0:n1_u2 - 1,0:n2_u2 - 1)
  integer(kind=4), intent(in)  :: n0_u3 
  integer(kind=4), intent(in)  :: n1_u3 
  integer(kind=4), intent(in)  :: n2_u3 
  real(kind=8), intent(in)  :: u3 (0:n0_u3 - 1,0:n1_u3 - 1,0:n2_u3 - 1)
  integer(kind=4), intent(in)  :: n0_Ueq 
  real(kind=8), intent(in)  :: Ueq (0:n0_Ueq - 1)
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
  integer(kind=4), intent(in)  :: n0_U_part 
  integer(kind=4), intent(in)  :: n1_U_part 
  real(kind=8), intent(inout)  :: U_part (0:n0_U_part - 1,0:n1_U_part - &
      1)

  call evaluate_1form(particles_pos,p0,spans0,Nbase,Np,u1,u2,u3,Ueq, &
      pp0_1,pp0_2,pp0_3,pp1_1,pp1_2,pp1_3,U_part)
end subroutine
!........................................

!........................................
subroutine f2py_evaluate_2form(n0_particles_pos, n1_particles_pos, &
      particles_pos, n0_p0, p0, n0_spans0, n1_spans0, spans0, n0_Nbase, &
      Nbase, Np, n0_b1, n1_b1, n2_b1, b1, n0_b2, n1_b2, n2_b2, b2, &
      n0_b3, n1_b3, n2_b3, b3, n0_Beq, Beq, n0_pp0_1, n1_pp0_1, pp0_1, &
      n0_pp0_2, n1_pp0_2, pp0_2, n0_pp0_3, n1_pp0_3, pp0_3, n0_pp1_1, &
      n1_pp1_1, pp1_1, n0_pp1_2, n1_pp1_2, pp1_2, n0_pp1_3, n1_pp1_3, &
      pp1_3, n0_B_part, n1_B_part, B_part)

  implicit none
  integer(kind=4), intent(in)  :: n0_particles_pos 
  integer(kind=4), intent(in)  :: n1_particles_pos 
  real(kind=8), intent(in)  :: particles_pos (0:n0_particles_pos - 1,0: &
      n1_particles_pos - 1)
  integer(kind=4), intent(in)  :: n0_p0 
  integer(kind=4), intent(in)  :: p0 (0:n0_p0 - 1)
  integer(kind=4), intent(in)  :: n0_spans0 
  integer(kind=4), intent(in)  :: n1_spans0 
  integer(kind=4), intent(in)  :: spans0 (0:n0_spans0 - 1,0:n1_spans0 - &
      1)
  integer(kind=4), intent(in)  :: n0_Nbase 
  integer(kind=4), intent(in)  :: Nbase (0:n0_Nbase - 1)
  integer(kind=4), intent(in)  :: Np 
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
  integer(kind=4), intent(in)  :: n0_Beq 
  real(kind=8), intent(in)  :: Beq (0:n0_Beq - 1)
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
  integer(kind=4), intent(in)  :: n0_B_part 
  integer(kind=4), intent(in)  :: n1_B_part 
  real(kind=8), intent(inout)  :: B_part (0:n0_B_part - 1,0:n1_B_part - &
      1)

  call evaluate_2form(particles_pos,p0,spans0,Nbase,Np,b1,b2,b3,Beq, &
      pp0_1,pp0_2,pp0_3,pp1_1,pp1_2,pp1_3,B_part)
end subroutine
!........................................

end module