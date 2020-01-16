module f2py_echo_accumulation

use echo_accumulation, only: det

use echo_accumulation, only: matrix_vector

use echo_accumulation, only: matrix_step3

use echo_accumulation, only: matrix_matrix

use echo_accumulation, only: matrix_step1

use echo_accumulation, only: transpose

use echo_accumulation, only: vector_step3

use echo_accumulation, only: basis_funs

use echo_accumulation, only: mapping_matrices
implicit none




contains

!........................................
subroutine f2py_matrix_step1(n0_particles, n1_particles, particles, &
      n0_p0, p0, n0_spans0, n1_spans0, spans0, n0_Nbase, Nbase, n0_b1, &
      n1_b1, n2_b1, b1, n0_b2, n1_b2, n2_b2, b2, n0_b3, n1_b3, n2_b3, &
      b3, n0_T1, T1, n0_T2, T2, n0_T3, T3, n0_tt1, tt1, n0_tt2, tt2, &
      n0_tt3, tt3, n0_mapping, mapping, dt, n0_Beq, Beq, n0_mat12, &
      n1_mat12, n2_mat12, n3_mat12, n4_mat12, n5_mat12, mat12, n0_mat13 &
      , n1_mat13, n2_mat13, n3_mat13, n4_mat13, n5_mat13, mat13, &
      n0_mat23, n1_mat23, n2_mat23, n3_mat23, n4_mat23, n5_mat23, mat23 &
      )

  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(in)  :: particles (0:n0_particles - 1,0: &
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
  integer(kind=4), intent(in)  :: n0_T1 
  real(kind=8), intent(in)  :: T1 (0:n0_T1 - 1)
  integer(kind=4), intent(in)  :: n0_T2 
  real(kind=8), intent(in)  :: T2 (0:n0_T2 - 1)
  integer(kind=4), intent(in)  :: n0_T3 
  real(kind=8), intent(in)  :: T3 (0:n0_T3 - 1)
  integer(kind=4), intent(in)  :: n0_tt1 
  real(kind=8), intent(in)  :: tt1 (0:n0_tt1 - 1)
  integer(kind=4), intent(in)  :: n0_tt2 
  real(kind=8), intent(in)  :: tt2 (0:n0_tt2 - 1)
  integer(kind=4), intent(in)  :: n0_tt3 
  real(kind=8), intent(in)  :: tt3 (0:n0_tt3 - 1)
  integer(kind=4), intent(in)  :: n0_mapping 
  real(kind=8), intent(in)  :: mapping (0:n0_mapping - 1)
  real(kind=8), intent(in)  :: dt 
  integer(kind=4), intent(in)  :: n0_Beq 
  real(kind=8), intent(in)  :: Beq (0:n0_Beq - 1)
  integer(kind=4), intent(in)  :: n0_mat12 
  integer(kind=4), intent(in)  :: n1_mat12 
  integer(kind=4), intent(in)  :: n2_mat12 
  integer(kind=4), intent(in)  :: n3_mat12 
  integer(kind=4), intent(in)  :: n4_mat12 
  integer(kind=4), intent(in)  :: n5_mat12 
  real(kind=8), intent(inout)  :: mat12 (0:n0_mat12 - 1,0:n1_mat12 - 1,0 &
      :n2_mat12 - 1,0:n3_mat12 - 1,0:n4_mat12 - 1,0:n5_mat12 - 1)
  integer(kind=4), intent(in)  :: n0_mat13 
  integer(kind=4), intent(in)  :: n1_mat13 
  integer(kind=4), intent(in)  :: n2_mat13 
  integer(kind=4), intent(in)  :: n3_mat13 
  integer(kind=4), intent(in)  :: n4_mat13 
  integer(kind=4), intent(in)  :: n5_mat13 
  real(kind=8), intent(inout)  :: mat13 (0:n0_mat13 - 1,0:n1_mat13 - 1,0 &
      :n2_mat13 - 1,0:n3_mat13 - 1,0:n4_mat13 - 1,0:n5_mat13 - 1)
  integer(kind=4), intent(in)  :: n0_mat23 
  integer(kind=4), intent(in)  :: n1_mat23 
  integer(kind=4), intent(in)  :: n2_mat23 
  integer(kind=4), intent(in)  :: n3_mat23 
  integer(kind=4), intent(in)  :: n4_mat23 
  integer(kind=4), intent(in)  :: n5_mat23 
  real(kind=8), intent(inout)  :: mat23 (0:n0_mat23 - 1,0:n1_mat23 - 1,0 &
      :n2_mat23 - 1,0:n3_mat23 - 1,0:n4_mat23 - 1,0:n5_mat23 - 1)

  call matrix_step1(particles,p0,spans0,Nbase,b1,b2,b3,T1,T2,T3,tt1,tt2, &
      tt3,mapping,dt,Beq,mat12,mat13,mat23)
end subroutine
!........................................

!........................................
subroutine f2py_matrix_step3(n0_particles, n1_particles, particles, &
      n0_p0, p0, n0_spans0, n1_spans0, spans0, n0_Nbase, Nbase, n0_b1, &
      n1_b1, n2_b1, b1, n0_b2, n1_b2, n2_b2, b2, n0_b3, n1_b3, n2_b3, &
      b3, n0_T1, T1, n0_T2, T2, n0_T3, T3, n0_tt1, tt1, n0_tt2, tt2, &
      n0_tt3, tt3, n0_mapping, mapping, dt, n0_Beq, Beq, n0_mat11, &
      n1_mat11, n2_mat11, n3_mat11, n4_mat11, n5_mat11, mat11, n0_mat12 &
      , n1_mat12, n2_mat12, n3_mat12, n4_mat12, n5_mat12, mat12, &
      n0_mat13, n1_mat13, n2_mat13, n3_mat13, n4_mat13, n5_mat13, mat13 &
      , n0_mat22, n1_mat22, n2_mat22, n3_mat22, n4_mat22, n5_mat22, &
      mat22, n0_mat23, n1_mat23, n2_mat23, n3_mat23, n4_mat23, n5_mat23 &
      , mat23, n0_mat33, n1_mat33, n2_mat33, n3_mat33, n4_mat33, &
      n5_mat33, mat33)

  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(in)  :: particles (0:n0_particles - 1,0: &
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
  integer(kind=4), intent(in)  :: n0_T1 
  real(kind=8), intent(in)  :: T1 (0:n0_T1 - 1)
  integer(kind=4), intent(in)  :: n0_T2 
  real(kind=8), intent(in)  :: T2 (0:n0_T2 - 1)
  integer(kind=4), intent(in)  :: n0_T3 
  real(kind=8), intent(in)  :: T3 (0:n0_T3 - 1)
  integer(kind=4), intent(in)  :: n0_tt1 
  real(kind=8), intent(in)  :: tt1 (0:n0_tt1 - 1)
  integer(kind=4), intent(in)  :: n0_tt2 
  real(kind=8), intent(in)  :: tt2 (0:n0_tt2 - 1)
  integer(kind=4), intent(in)  :: n0_tt3 
  real(kind=8), intent(in)  :: tt3 (0:n0_tt3 - 1)
  integer(kind=4), intent(in)  :: n0_mapping 
  real(kind=8), intent(in)  :: mapping (0:n0_mapping - 1)
  real(kind=8), intent(in)  :: dt 
  integer(kind=4), intent(in)  :: n0_Beq 
  real(kind=8), intent(in)  :: Beq (0:n0_Beq - 1)
  integer(kind=4), intent(in)  :: n0_mat11 
  integer(kind=4), intent(in)  :: n1_mat11 
  integer(kind=4), intent(in)  :: n2_mat11 
  integer(kind=4), intent(in)  :: n3_mat11 
  integer(kind=4), intent(in)  :: n4_mat11 
  integer(kind=4), intent(in)  :: n5_mat11 
  real(kind=8), intent(inout)  :: mat11 (0:n0_mat11 - 1,0:n1_mat11 - 1,0 &
      :n2_mat11 - 1,0:n3_mat11 - 1,0:n4_mat11 - 1,0:n5_mat11 - 1)
  integer(kind=4), intent(in)  :: n0_mat12 
  integer(kind=4), intent(in)  :: n1_mat12 
  integer(kind=4), intent(in)  :: n2_mat12 
  integer(kind=4), intent(in)  :: n3_mat12 
  integer(kind=4), intent(in)  :: n4_mat12 
  integer(kind=4), intent(in)  :: n5_mat12 
  real(kind=8), intent(inout)  :: mat12 (0:n0_mat12 - 1,0:n1_mat12 - 1,0 &
      :n2_mat12 - 1,0:n3_mat12 - 1,0:n4_mat12 - 1,0:n5_mat12 - 1)
  integer(kind=4), intent(in)  :: n0_mat13 
  integer(kind=4), intent(in)  :: n1_mat13 
  integer(kind=4), intent(in)  :: n2_mat13 
  integer(kind=4), intent(in)  :: n3_mat13 
  integer(kind=4), intent(in)  :: n4_mat13 
  integer(kind=4), intent(in)  :: n5_mat13 
  real(kind=8), intent(inout)  :: mat13 (0:n0_mat13 - 1,0:n1_mat13 - 1,0 &
      :n2_mat13 - 1,0:n3_mat13 - 1,0:n4_mat13 - 1,0:n5_mat13 - 1)
  integer(kind=4), intent(in)  :: n0_mat22 
  integer(kind=4), intent(in)  :: n1_mat22 
  integer(kind=4), intent(in)  :: n2_mat22 
  integer(kind=4), intent(in)  :: n3_mat22 
  integer(kind=4), intent(in)  :: n4_mat22 
  integer(kind=4), intent(in)  :: n5_mat22 
  real(kind=8), intent(inout)  :: mat22 (0:n0_mat22 - 1,0:n1_mat22 - 1,0 &
      :n2_mat22 - 1,0:n3_mat22 - 1,0:n4_mat22 - 1,0:n5_mat22 - 1)
  integer(kind=4), intent(in)  :: n0_mat23 
  integer(kind=4), intent(in)  :: n1_mat23 
  integer(kind=4), intent(in)  :: n2_mat23 
  integer(kind=4), intent(in)  :: n3_mat23 
  integer(kind=4), intent(in)  :: n4_mat23 
  integer(kind=4), intent(in)  :: n5_mat23 
  real(kind=8), intent(inout)  :: mat23 (0:n0_mat23 - 1,0:n1_mat23 - 1,0 &
      :n2_mat23 - 1,0:n3_mat23 - 1,0:n4_mat23 - 1,0:n5_mat23 - 1)
  integer(kind=4), intent(in)  :: n0_mat33 
  integer(kind=4), intent(in)  :: n1_mat33 
  integer(kind=4), intent(in)  :: n2_mat33 
  integer(kind=4), intent(in)  :: n3_mat33 
  integer(kind=4), intent(in)  :: n4_mat33 
  integer(kind=4), intent(in)  :: n5_mat33 
  real(kind=8), intent(inout)  :: mat33 (0:n0_mat33 - 1,0:n1_mat33 - 1,0 &
      :n2_mat33 - 1,0:n3_mat33 - 1,0:n4_mat33 - 1,0:n5_mat33 - 1)

  call matrix_step3(particles,p0,spans0,Nbase,b1,b2,b3,T1,T2,T3,tt1,tt2, &
      tt3,mapping,dt,Beq,mat11,mat12,mat13,mat22,mat23,mat33)
end subroutine
!........................................

!........................................
subroutine f2py_vector_step3(n0_particles, n1_particles, particles, &
      n0_p0, p0, n0_spans0, n1_spans0, spans0, n0_Nbase, Nbase, n0_b1, &
      n1_b1, n2_b1, b1, n0_b2, n1_b2, n2_b2, b2, n0_b3, n1_b3, n2_b3, &
      b3, n0_T1, T1, n0_T2, T2, n0_T3, T3, n0_tt1, tt1, n0_tt2, tt2, &
      n0_tt3, tt3, n0_mapping, mapping, n0_Beq, Beq, n0_vec1, n1_vec1, &
      n2_vec1, vec1, n0_vec2, n1_vec2, n2_vec2, vec2, n0_vec3, n1_vec3, &
      n2_vec3, vec3)

  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(in)  :: particles (0:n0_particles - 1,0: &
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
  integer(kind=4), intent(in)  :: n0_T1 
  real(kind=8), intent(in)  :: T1 (0:n0_T1 - 1)
  integer(kind=4), intent(in)  :: n0_T2 
  real(kind=8), intent(in)  :: T2 (0:n0_T2 - 1)
  integer(kind=4), intent(in)  :: n0_T3 
  real(kind=8), intent(in)  :: T3 (0:n0_T3 - 1)
  integer(kind=4), intent(in)  :: n0_tt1 
  real(kind=8), intent(in)  :: tt1 (0:n0_tt1 - 1)
  integer(kind=4), intent(in)  :: n0_tt2 
  real(kind=8), intent(in)  :: tt2 (0:n0_tt2 - 1)
  integer(kind=4), intent(in)  :: n0_tt3 
  real(kind=8), intent(in)  :: tt3 (0:n0_tt3 - 1)
  integer(kind=4), intent(in)  :: n0_mapping 
  real(kind=8), intent(in)  :: mapping (0:n0_mapping - 1)
  integer(kind=4), intent(in)  :: n0_Beq 
  real(kind=8), intent(in)  :: Beq (0:n0_Beq - 1)
  integer(kind=4), intent(in)  :: n0_vec1 
  integer(kind=4), intent(in)  :: n1_vec1 
  integer(kind=4), intent(in)  :: n2_vec1 
  real(kind=8), intent(inout)  :: vec1 (0:n0_vec1 - 1,0:n1_vec1 - 1,0: &
      n2_vec1 - 1)
  integer(kind=4), intent(in)  :: n0_vec2 
  integer(kind=4), intent(in)  :: n1_vec2 
  integer(kind=4), intent(in)  :: n2_vec2 
  real(kind=8), intent(inout)  :: vec2 (0:n0_vec2 - 1,0:n1_vec2 - 1,0: &
      n2_vec2 - 1)
  integer(kind=4), intent(in)  :: n0_vec3 
  integer(kind=4), intent(in)  :: n1_vec3 
  integer(kind=4), intent(in)  :: n2_vec3 
  real(kind=8), intent(inout)  :: vec3 (0:n0_vec3 - 1,0:n1_vec3 - 1,0: &
      n2_vec3 - 1)

  call vector_step3(particles,p0,spans0,Nbase,b1,b2,b3,T1,T2,T3,tt1,tt2, &
      tt3,mapping,Beq,vec1,vec2,vec3)
end subroutine
!........................................

end module