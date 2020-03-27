subroutine mapping_matrices(n0_q, q, kind, n0_params, params, output, &
      n0_A, n1_A, A)

  use STRUPHY_accumulation, only: mod_mapping_matrices => &
      mapping_matrices
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

  use STRUPHY_accumulation, only: mod_matrix_vector => matrix_vector
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

  use STRUPHY_accumulation, only: mod_matrix_matrix => matrix_matrix
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

  use STRUPHY_accumulation, only: mod_transpose => transpose
  implicit none
  integer(kind=4), intent(in)  :: n0_A 
  integer(kind=4), intent(in)  :: n1_A 
  real(kind=8), intent(in)  :: A (0:n0_A - 1,0:n1_A - 1)
  integer(kind=4), intent(in)  :: n0_B 
  integer(kind=4), intent(in)  :: n1_B 
  real(kind=8), intent(inout)  :: B (0:n0_B - 1,0:n1_B - 1)

  call mod_transpose(A,B)
end subroutine

function det(n0_A, n1_A, A) result(Dummy_4855)

  use STRUPHY_accumulation, only: mod_det => det
  implicit none
  real(kind=8) :: Dummy_4855  
  integer(kind=4), intent(in)  :: n0_A 
  integer(kind=4), intent(in)  :: n1_A 
  real(kind=8), intent(in)  :: A (0:n0_A - 1,0:n1_A - 1)

  Dummy_4855 = mod_det(A)
end function

subroutine basis_funs(n0_knots, knots, degree, x, span, n0_left, left, &
      n0_right, right, n0_values, values)

  use STRUPHY_accumulation, only: mod_basis_funs => basis_funs
  implicit none
  integer(kind=4), intent(in)  :: n0_knots 
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
  integer(kind=4), intent(in)  :: degree 
  real(kind=8), intent(in)  :: x 
  integer(kind=4), intent(in)  :: span 
  integer(kind=4), intent(in)  :: n0_left 
  real(kind=8), intent(inout)  :: left (0:n0_left - 1)
  integer(kind=4), intent(in)  :: n0_right 
  real(kind=8), intent(inout)  :: right (0:n0_right - 1)
  integer(kind=4), intent(in)  :: n0_values 
  real(kind=8), intent(inout)  :: values (0:n0_values - 1)

  call mod_basis_funs(knots,degree,x,span,left,right,values)
end subroutine

subroutine accumulation_step1(n0_particles, n1_particles, particles, &
      n0_p0, p0, n0_spans0, n1_spans0, spans0, n0_Nbase, Nbase, n0_T1, &
      T1, n0_T2, T2, n0_T3, T3, n0_tt1, tt1, n0_tt2, tt2, n0_tt3, tt3, &
      n0_mapping, mapping, n0_B_part, n1_B_part, B_part, n0_mat12, &
      n1_mat12, n2_mat12, n3_mat12, n4_mat12, n5_mat12, mat12, n0_mat13 &
      , n1_mat13, n2_mat13, n3_mat13, n4_mat13, n5_mat13, mat13, &
      n0_mat23, n1_mat23, n2_mat23, n3_mat23, n4_mat23, n5_mat23, mat23 &
      )

  use STRUPHY_accumulation, only: mod_accumulation_step1 => &
      accumulation_step1
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
  integer(kind=4), intent(in)  :: n0_B_part 
  integer(kind=4), intent(in)  :: n1_B_part 
  real(kind=8), intent(in)  :: B_part (0:n0_B_part - 1,0:n1_B_part - 1)
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

  call mod_accumulation_step1(particles,p0,spans0,Nbase,T1,T2,T3,tt1,tt2 &
      ,tt3,mapping,B_part,mat12,mat13,mat23)
end subroutine

subroutine accumulation_step3(n0_particles, n1_particles, particles, &
      n0_p0, p0, n0_spans0, n1_spans0, spans0, n0_Nbase, Nbase, n0_T1, &
      T1, n0_T2, T2, n0_T3, T3, n0_tt1, tt1, n0_tt2, tt2, n0_tt3, tt3, &
      n0_mapping, mapping, n0_B_part, n1_B_part, B_part, n0_mat11, &
      n1_mat11, n2_mat11, n3_mat11, n4_mat11, n5_mat11, mat11, n0_mat12 &
      , n1_mat12, n2_mat12, n3_mat12, n4_mat12, n5_mat12, mat12, &
      n0_mat13, n1_mat13, n2_mat13, n3_mat13, n4_mat13, n5_mat13, mat13 &
      , n0_mat22, n1_mat22, n2_mat22, n3_mat22, n4_mat22, n5_mat22, &
      mat22, n0_mat23, n1_mat23, n2_mat23, n3_mat23, n4_mat23, n5_mat23 &
      , mat23, n0_mat33, n1_mat33, n2_mat33, n3_mat33, n4_mat33, &
      n5_mat33, mat33, n0_vec1, n1_vec1, n2_vec1, vec1, n0_vec2, &
      n1_vec2, n2_vec2, vec2, n0_vec3, n1_vec3, n2_vec3, vec3)

  use STRUPHY_accumulation, only: mod_accumulation_step3 => &
      accumulation_step3
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
  integer(kind=4), intent(in)  :: n0_B_part 
  integer(kind=4), intent(in)  :: n1_B_part 
  real(kind=8), intent(in)  :: B_part (0:n0_B_part - 1,0:n1_B_part - 1)
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

  call mod_accumulation_step3(particles,p0,spans0,Nbase,T1,T2,T3,tt1,tt2 &
      ,tt3,mapping,B_part,mat11,mat12,mat13,mat22,mat23,mat33,vec1,vec2 &
      ,vec3)
end subroutine