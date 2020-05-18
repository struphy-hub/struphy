subroutine basis_funs (n0_knots, knots, degree, x, span, n0_left, left, &
      n0_right, right, n0_values, values)

  use STRUPHY_accumulation_kernels, only: mod_basis_funs => basis_funs
  implicit none
  integer(kind=4), intent(in)  :: n0_knots 
  real(kind=8), intent(in)  :: knots (0:n0_knots-1)
  integer(kind=8), intent(in)  :: degree 
  real(kind=8), intent(in)  :: x 
  integer(kind=8), intent(in)  :: span 
  integer(kind=4), intent(in)  :: n0_left 
  real(kind=8), intent(inout)  :: left (0:n0_left-1)
  integer(kind=4), intent(in)  :: n0_right 
  real(kind=8), intent(inout)  :: right (0:n0_right-1)
  integer(kind=4), intent(in)  :: n0_values 
  real(kind=8), intent(inout)  :: values (0:n0_values-1)

  call mod_basis_funs(knots,degree,x,span,left,right,values)
end subroutine

subroutine kernel_step1 (n0_particles, n1_particles, particles, n0_p0, &
      p0, n0_nel, nel, n0_nbase, nbase, n0_t0_1, t0_1, n0_t0_2, t0_2, &
      n0_t0_3, t0_3, n0_t1_1, t1_1, n0_t1_2, t1_2, n0_t1_3, t1_3, &
      n0_b_part, n1_b_part, b_part, kind_map, n0_params_map, params_map &
      , n0_mat12, n1_mat12, n2_mat12, n3_mat12, n4_mat12, n5_mat12, &
      mat12, n0_mat13, n1_mat13, n2_mat13, n3_mat13, n4_mat13, n5_mat13 &
      , mat13, n0_mat23, n1_mat23, n2_mat23, n3_mat23, n4_mat23, &
      n5_mat23, mat23)

  use STRUPHY_accumulation_kernels, only: mod_kernel_step1 => &
      kernel_step1
  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(in)  :: particles (0:n0_particles-1,0: &
      n1_particles-1)
  integer(kind=4), intent(in)  :: n0_p0 
  integer(kind=8), intent(in)  :: p0 (0:n0_p0-1)
  integer(kind=4), intent(in)  :: n0_nel 
  integer(kind=8), intent(in)  :: nel (0:n0_nel-1)
  integer(kind=4), intent(in)  :: n0_nbase 
  integer(kind=8), intent(in)  :: nbase (0:n0_nbase-1)
  integer(kind=4), intent(in)  :: n0_t0_1 
  real(kind=8), intent(in)  :: t0_1 (0:n0_t0_1-1)
  integer(kind=4), intent(in)  :: n0_t0_2 
  real(kind=8), intent(in)  :: t0_2 (0:n0_t0_2-1)
  integer(kind=4), intent(in)  :: n0_t0_3 
  real(kind=8), intent(in)  :: t0_3 (0:n0_t0_3-1)
  integer(kind=4), intent(in)  :: n0_t1_1 
  real(kind=8), intent(in)  :: t1_1 (0:n0_t1_1-1)
  integer(kind=4), intent(in)  :: n0_t1_2 
  real(kind=8), intent(in)  :: t1_2 (0:n0_t1_2-1)
  integer(kind=4), intent(in)  :: n0_t1_3 
  real(kind=8), intent(in)  :: t1_3 (0:n0_t1_3-1)
  integer(kind=4), intent(in)  :: n0_b_part 
  integer(kind=4), intent(in)  :: n1_b_part 
  real(kind=8), intent(in)  :: b_part (0:n0_b_part-1,0:n1_b_part-1)
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)
  integer(kind=4), intent(in)  :: n0_mat12 
  integer(kind=4), intent(in)  :: n1_mat12 
  integer(kind=4), intent(in)  :: n2_mat12 
  integer(kind=4), intent(in)  :: n3_mat12 
  integer(kind=4), intent(in)  :: n4_mat12 
  integer(kind=4), intent(in)  :: n5_mat12 
  real(kind=8), intent(inout)  :: mat12 (0:n0_mat12-1,0:n1_mat12-1,0: &
      n2_mat12-1,0:n3_mat12-1,0:n4_mat12-1,0:n5_mat12-1)
  integer(kind=4), intent(in)  :: n0_mat13 
  integer(kind=4), intent(in)  :: n1_mat13 
  integer(kind=4), intent(in)  :: n2_mat13 
  integer(kind=4), intent(in)  :: n3_mat13 
  integer(kind=4), intent(in)  :: n4_mat13 
  integer(kind=4), intent(in)  :: n5_mat13 
  real(kind=8), intent(inout)  :: mat13 (0:n0_mat13-1,0:n1_mat13-1,0: &
      n2_mat13-1,0:n3_mat13-1,0:n4_mat13-1,0:n5_mat13-1)
  integer(kind=4), intent(in)  :: n0_mat23 
  integer(kind=4), intent(in)  :: n1_mat23 
  integer(kind=4), intent(in)  :: n2_mat23 
  integer(kind=4), intent(in)  :: n3_mat23 
  integer(kind=4), intent(in)  :: n4_mat23 
  integer(kind=4), intent(in)  :: n5_mat23 
  real(kind=8), intent(inout)  :: mat23 (0:n0_mat23-1,0:n1_mat23-1,0: &
      n2_mat23-1,0:n3_mat23-1,0:n4_mat23-1,0:n5_mat23-1)

  call mod_kernel_step1(particles,p0,nel,nbase,t0_1,t0_2,t0_3,t1_1,t1_2, &
      t1_3,b_part,kind_map,params_map,mat12,mat13,mat23)
end subroutine

subroutine kernel_step3 (n0_particles, n1_particles, particles, n0_p0, &
      p0, n0_nel, nel, n0_nbase, nbase, n0_t0_1, t0_1, n0_t0_2, t0_2, &
      n0_t0_3, t0_3, n0_t1_1, t1_1, n0_t1_2, t1_2, n0_t1_3, t1_3, &
      n0_b_part, n1_b_part, b_part, kind_map, n0_params_map, params_map &
      , n0_mat11, n1_mat11, n2_mat11, n3_mat11, n4_mat11, n5_mat11, &
      mat11, n0_mat12, n1_mat12, n2_mat12, n3_mat12, n4_mat12, n5_mat12 &
      , mat12, n0_mat13, n1_mat13, n2_mat13, n3_mat13, n4_mat13, &
      n5_mat13, mat13, n0_mat22, n1_mat22, n2_mat22, n3_mat22, n4_mat22 &
      , n5_mat22, mat22, n0_mat23, n1_mat23, n2_mat23, n3_mat23, &
      n4_mat23, n5_mat23, mat23, n0_mat33, n1_mat33, n2_mat33, n3_mat33 &
      , n4_mat33, n5_mat33, mat33, n0_vec1, n1_vec1, n2_vec1, vec1, &
      n0_vec2, n1_vec2, n2_vec2, vec2, n0_vec3, n1_vec3, n2_vec3, vec3 &
      )

  use STRUPHY_accumulation_kernels, only: mod_kernel_step3 => &
      kernel_step3
  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(in)  :: particles (0:n0_particles-1,0: &
      n1_particles-1)
  integer(kind=4), intent(in)  :: n0_p0 
  integer(kind=8), intent(in)  :: p0 (0:n0_p0-1)
  integer(kind=4), intent(in)  :: n0_nel 
  integer(kind=8), intent(in)  :: nel (0:n0_nel-1)
  integer(kind=4), intent(in)  :: n0_nbase 
  integer(kind=8), intent(in)  :: nbase (0:n0_nbase-1)
  integer(kind=4), intent(in)  :: n0_t0_1 
  real(kind=8), intent(in)  :: t0_1 (0:n0_t0_1-1)
  integer(kind=4), intent(in)  :: n0_t0_2 
  real(kind=8), intent(in)  :: t0_2 (0:n0_t0_2-1)
  integer(kind=4), intent(in)  :: n0_t0_3 
  real(kind=8), intent(in)  :: t0_3 (0:n0_t0_3-1)
  integer(kind=4), intent(in)  :: n0_t1_1 
  real(kind=8), intent(in)  :: t1_1 (0:n0_t1_1-1)
  integer(kind=4), intent(in)  :: n0_t1_2 
  real(kind=8), intent(in)  :: t1_2 (0:n0_t1_2-1)
  integer(kind=4), intent(in)  :: n0_t1_3 
  real(kind=8), intent(in)  :: t1_3 (0:n0_t1_3-1)
  integer(kind=4), intent(in)  :: n0_b_part 
  integer(kind=4), intent(in)  :: n1_b_part 
  real(kind=8), intent(in)  :: b_part (0:n0_b_part-1,0:n1_b_part-1)
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)
  integer(kind=4), intent(in)  :: n0_mat11 
  integer(kind=4), intent(in)  :: n1_mat11 
  integer(kind=4), intent(in)  :: n2_mat11 
  integer(kind=4), intent(in)  :: n3_mat11 
  integer(kind=4), intent(in)  :: n4_mat11 
  integer(kind=4), intent(in)  :: n5_mat11 
  real(kind=8), intent(inout)  :: mat11 (0:n0_mat11-1,0:n1_mat11-1,0: &
      n2_mat11-1,0:n3_mat11-1,0:n4_mat11-1,0:n5_mat11-1)
  integer(kind=4), intent(in)  :: n0_mat12 
  integer(kind=4), intent(in)  :: n1_mat12 
  integer(kind=4), intent(in)  :: n2_mat12 
  integer(kind=4), intent(in)  :: n3_mat12 
  integer(kind=4), intent(in)  :: n4_mat12 
  integer(kind=4), intent(in)  :: n5_mat12 
  real(kind=8), intent(inout)  :: mat12 (0:n0_mat12-1,0:n1_mat12-1,0: &
      n2_mat12-1,0:n3_mat12-1,0:n4_mat12-1,0:n5_mat12-1)
  integer(kind=4), intent(in)  :: n0_mat13 
  integer(kind=4), intent(in)  :: n1_mat13 
  integer(kind=4), intent(in)  :: n2_mat13 
  integer(kind=4), intent(in)  :: n3_mat13 
  integer(kind=4), intent(in)  :: n4_mat13 
  integer(kind=4), intent(in)  :: n5_mat13 
  real(kind=8), intent(inout)  :: mat13 (0:n0_mat13-1,0:n1_mat13-1,0: &
      n2_mat13-1,0:n3_mat13-1,0:n4_mat13-1,0:n5_mat13-1)
  integer(kind=4), intent(in)  :: n0_mat22 
  integer(kind=4), intent(in)  :: n1_mat22 
  integer(kind=4), intent(in)  :: n2_mat22 
  integer(kind=4), intent(in)  :: n3_mat22 
  integer(kind=4), intent(in)  :: n4_mat22 
  integer(kind=4), intent(in)  :: n5_mat22 
  real(kind=8), intent(inout)  :: mat22 (0:n0_mat22-1,0:n1_mat22-1,0: &
      n2_mat22-1,0:n3_mat22-1,0:n4_mat22-1,0:n5_mat22-1)
  integer(kind=4), intent(in)  :: n0_mat23 
  integer(kind=4), intent(in)  :: n1_mat23 
  integer(kind=4), intent(in)  :: n2_mat23 
  integer(kind=4), intent(in)  :: n3_mat23 
  integer(kind=4), intent(in)  :: n4_mat23 
  integer(kind=4), intent(in)  :: n5_mat23 
  real(kind=8), intent(inout)  :: mat23 (0:n0_mat23-1,0:n1_mat23-1,0: &
      n2_mat23-1,0:n3_mat23-1,0:n4_mat23-1,0:n5_mat23-1)
  integer(kind=4), intent(in)  :: n0_mat33 
  integer(kind=4), intent(in)  :: n1_mat33 
  integer(kind=4), intent(in)  :: n2_mat33 
  integer(kind=4), intent(in)  :: n3_mat33 
  integer(kind=4), intent(in)  :: n4_mat33 
  integer(kind=4), intent(in)  :: n5_mat33 
  real(kind=8), intent(inout)  :: mat33 (0:n0_mat33-1,0:n1_mat33-1,0: &
      n2_mat33-1,0:n3_mat33-1,0:n4_mat33-1,0:n5_mat33-1)
  integer(kind=4), intent(in)  :: n0_vec1 
  integer(kind=4), intent(in)  :: n1_vec1 
  integer(kind=4), intent(in)  :: n2_vec1 
  real(kind=8), intent(inout)  :: vec1 (0:n0_vec1-1,0:n1_vec1-1,0: &
      n2_vec1-1)
  integer(kind=4), intent(in)  :: n0_vec2 
  integer(kind=4), intent(in)  :: n1_vec2 
  integer(kind=4), intent(in)  :: n2_vec2 
  real(kind=8), intent(inout)  :: vec2 (0:n0_vec2-1,0:n1_vec2-1,0: &
      n2_vec2-1)
  integer(kind=4), intent(in)  :: n0_vec3 
  integer(kind=4), intent(in)  :: n1_vec3 
  integer(kind=4), intent(in)  :: n2_vec3 
  real(kind=8), intent(inout)  :: vec3 (0:n0_vec3-1,0:n1_vec3-1,0: &
      n2_vec3-1)

  call mod_kernel_step3(particles,p0,nel,nbase,t0_1,t0_2,t0_3,t1_1,t1_2, &
      t1_3,b_part,kind_map,params_map,mat11,mat12,mat13,mat22,mat23, &
      mat33,vec1,vec2,vec3)
end subroutine