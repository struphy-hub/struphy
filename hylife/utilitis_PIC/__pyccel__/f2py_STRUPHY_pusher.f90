subroutine mapping_matrices(n0_q, q, kind, n0_params, params, output, &
      n0_A, n1_A, A)

  use STRUPHY_pusher, only: mod_mapping_matrices => mapping_matrices
  implicit none
  integer(kind=4), intent(in)  :: n0_q 
  real(kind=8), intent(in)  :: q (0:n0_q - 1)
  integer(kind=4), intent(in)  :: kind 
  integer(kind=4), intent(in)  :: n0_params 
  real(kind=8), intent(in)  :: params (0:n0_params - 1)
  integer(kind=4), intent(in)  :: output 
  integer(kind=4), intent(in)  :: n0_A 
  integer(kind=4), intent(in)  :: n1_A 
  real(kind=8), intent(inout)  :: A (0:n0_A - 1,0:n1_A - 1)

  call mod_mapping_matrices(q,kind,params,output,A)
end subroutine

subroutine matrix_vector(n0_A, n1_A, A, n0_b, b, n0_c, c) 

  use STRUPHY_pusher, only: mod_matrix_vector => matrix_vector
  implicit none
  integer(kind=4), intent(in)  :: n0_A 
  integer(kind=4), intent(in)  :: n1_A 
  real(kind=8), intent(in)  :: A (0:n0_A - 1,0:n1_A - 1)
  integer(kind=4), intent(in)  :: n0_b 
  real(kind=8), intent(in)  :: b (0:n0_b - 1)
  integer(kind=4), intent(in)  :: n0_c 
  real(kind=8), intent(inout)  :: c (0:n0_c - 1)

  call mod_matrix_vector(A,b,c)
end subroutine

subroutine matrix_matrix(n0_A, n1_A, A, n0_B, n1_B, B, n0_C, n1_C, C) 

  use STRUPHY_pusher, only: mod_matrix_matrix => matrix_matrix
  implicit none
  integer(kind=4), intent(in)  :: n0_A 
  integer(kind=4), intent(in)  :: n1_A 
  real(kind=8), intent(in)  :: A (0:n0_A - 1,0:n1_A - 1)
  integer(kind=4), intent(in)  :: n0_B 
  integer(kind=4), intent(in)  :: n1_B 
  real(kind=8), intent(in)  :: B (0:n0_B - 1,0:n1_B - 1)
  integer(kind=4), intent(in)  :: n0_C 
  integer(kind=4), intent(in)  :: n1_C 
  real(kind=8), intent(inout)  :: C (0:n0_C - 1,0:n1_C - 1)

  call mod_matrix_matrix(A,B,C)
end subroutine

subroutine transpose(n0_A, n1_A, A, n0_B, n1_B, B) 

  use STRUPHY_pusher, only: mod_transpose => transpose
  implicit none
  integer(kind=4), intent(in)  :: n0_A 
  integer(kind=4), intent(in)  :: n1_A 
  real(kind=8), intent(in)  :: A (0:n0_A - 1,0:n1_A - 1)
  integer(kind=4), intent(in)  :: n0_B 
  integer(kind=4), intent(in)  :: n1_B 
  real(kind=8), intent(inout)  :: B (0:n0_B - 1,0:n1_B - 1)

  call mod_transpose(A,B)
end subroutine

function det(n0_A, n1_A, A) result(Dummy_4220)

  use STRUPHY_pusher, only: mod_det => det
  implicit none
  real(kind=8) :: Dummy_4220  
  integer(kind=4), intent(in)  :: n0_A 
  integer(kind=4), intent(in)  :: n1_A 
  real(kind=8), intent(in)  :: A (0:n0_A - 1,0:n1_A - 1)

  Dummy_4220 = mod_det(A)
end function

subroutine pusher_step3(n0_particles, n1_particles, particles, &
      n0_mapping, mapping, dt, n0_B_part, n1_B_part, B_part, n0_U_part, &
      n1_U_part, U_part)

  use STRUPHY_pusher, only: mod_pusher_step3 => pusher_step3
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

  call mod_pusher_step3(particles,mapping,dt,B_part,U_part)
end subroutine

subroutine pusher_step4(n0_particles, n1_particles, particles, &
      n0_mapping, mapping, dt)

  use STRUPHY_pusher, only: mod_pusher_step4 => pusher_step4
  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles - 1,0: &
      n1_particles - 1)
  integer(kind=4), intent(in)  :: n0_mapping 
  real(kind=8), intent(in)  :: mapping (0:n0_mapping - 1)
  real(kind=8), intent(in)  :: dt 

  call mod_pusher_step4(particles,mapping,dt)
end subroutine

subroutine pusher_step5(n0_particles, n1_particles, particles, &
      n0_mapping, mapping, dt, n0_B_part, n1_B_part, B_part)

  use STRUPHY_pusher, only: mod_pusher_step5 => pusher_step5
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

  call mod_pusher_step5(particles,mapping,dt,B_part)
end subroutine