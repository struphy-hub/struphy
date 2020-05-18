subroutine basis_funs (n0_knots, knots, degree, x, span, n0_left, left, &
      n0_right, right, n0_values, values)

  use STRUPHY_fields, only: mod_basis_funs => basis_funs
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

subroutine evaluate_1form (n0_particles_pos, n1_particles_pos, &
      particles_pos, n0_t0_1, t0_1, n0_t0_2, t0_2, n0_t0_3, t0_3, &
      n0_t1_1, t1_1, n0_t1_2, t1_2, n0_t1_3, t1_3, n0_p0, p0, n0_nel, &
      nel, n0_nbase, n1_nbase, nbase, np, n0_u1, n1_u1, n2_u1, u1, &
      n0_u2, n1_u2, n2_u2, u2, n0_u3, n1_u3, n2_u3, u3, n0_pp0_1, &
      n1_pp0_1, pp0_1, n0_pp0_2, n1_pp0_2, pp0_2, n0_pp0_3, n1_pp0_3, &
      pp0_3, n0_pp1_1, n1_pp1_1, pp1_1, n0_pp1_2, n1_pp1_2, pp1_2, &
      n0_pp1_3, n1_pp1_3, pp1_3, n0_u_part, n1_u_part, u_part, kind_map &
      , n0_params_map, params_map)

  use STRUPHY_fields, only: mod_evaluate_1form => evaluate_1form
  implicit none
  integer(kind=4), intent(in)  :: n0_particles_pos 
  integer(kind=4), intent(in)  :: n1_particles_pos 
  real(kind=8), intent(in)  :: particles_pos (0:n0_particles_pos-1,0: &
      n1_particles_pos-1)
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
  integer(kind=4), intent(in)  :: n0_p0 
  integer(kind=8), intent(in)  :: p0 (0:n0_p0-1)
  integer(kind=4), intent(in)  :: n0_nel 
  integer(kind=8), intent(in)  :: nel (0:n0_nel-1)
  integer(kind=4), intent(in)  :: n0_nbase 
  integer(kind=4), intent(in)  :: n1_nbase 
  integer(kind=8), intent(in)  :: nbase (0:n0_nbase-1,0:n1_nbase-1)
  integer(kind=8), intent(in)  :: np 
  integer(kind=4), intent(in)  :: n0_u1 
  integer(kind=4), intent(in)  :: n1_u1 
  integer(kind=4), intent(in)  :: n2_u1 
  real(kind=8), intent(in)  :: u1 (0:n0_u1-1,0:n1_u1-1,0:n2_u1-1)
  integer(kind=4), intent(in)  :: n0_u2 
  integer(kind=4), intent(in)  :: n1_u2 
  integer(kind=4), intent(in)  :: n2_u2 
  real(kind=8), intent(in)  :: u2 (0:n0_u2-1,0:n1_u2-1,0:n2_u2-1)
  integer(kind=4), intent(in)  :: n0_u3 
  integer(kind=4), intent(in)  :: n1_u3 
  integer(kind=4), intent(in)  :: n2_u3 
  real(kind=8), intent(in)  :: u3 (0:n0_u3-1,0:n1_u3-1,0:n2_u3-1)
  integer(kind=4), intent(in)  :: n0_pp0_1 
  integer(kind=4), intent(in)  :: n1_pp0_1 
  real(kind=8), intent(in)  :: pp0_1 (0:n0_pp0_1-1,0:n1_pp0_1-1)
  integer(kind=4), intent(in)  :: n0_pp0_2 
  integer(kind=4), intent(in)  :: n1_pp0_2 
  real(kind=8), intent(in)  :: pp0_2 (0:n0_pp0_2-1,0:n1_pp0_2-1)
  integer(kind=4), intent(in)  :: n0_pp0_3 
  integer(kind=4), intent(in)  :: n1_pp0_3 
  real(kind=8), intent(in)  :: pp0_3 (0:n0_pp0_3-1,0:n1_pp0_3-1)
  integer(kind=4), intent(in)  :: n0_pp1_1 
  integer(kind=4), intent(in)  :: n1_pp1_1 
  real(kind=8), intent(in)  :: pp1_1 (0:n0_pp1_1-1,0:n1_pp1_1-1)
  integer(kind=4), intent(in)  :: n0_pp1_2 
  integer(kind=4), intent(in)  :: n1_pp1_2 
  real(kind=8), intent(in)  :: pp1_2 (0:n0_pp1_2-1,0:n1_pp1_2-1)
  integer(kind=4), intent(in)  :: n0_pp1_3 
  integer(kind=4), intent(in)  :: n1_pp1_3 
  real(kind=8), intent(in)  :: pp1_3 (0:n0_pp1_3-1,0:n1_pp1_3-1)
  integer(kind=4), intent(in)  :: n0_u_part 
  integer(kind=4), intent(in)  :: n1_u_part 
  real(kind=8), intent(inout)  :: u_part (0:n0_u_part-1,0:n1_u_part-1)
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)

  call mod_evaluate_1form(particles_pos,t0_1,t0_2,t0_3,t1_1,t1_2,t1_3,p0 &
      ,nel,nbase,np,u1,u2,u3,pp0_1,pp0_2,pp0_3,pp1_1,pp1_2,pp1_3,u_part &
      ,kind_map,params_map)
end subroutine

subroutine evaluate_2form (n0_particles_pos, n1_particles_pos, &
      particles_pos, n0_t0_1, t0_1, n0_t0_2, t0_2, n0_t0_3, t0_3, &
      n0_t1_1, t1_1, n0_t1_2, t1_2, n0_t1_3, t1_3, n0_p0, p0, n0_nel, &
      nel, n0_nbase, n1_nbase, nbase, np, n0_b1, n1_b1, n2_b1, b1, &
      n0_b2, n1_b2, n2_b2, b2, n0_b3, n1_b3, n2_b3, b3, n0_pp0_1, &
      n1_pp0_1, pp0_1, n0_pp0_2, n1_pp0_2, pp0_2, n0_pp0_3, n1_pp0_3, &
      pp0_3, n0_pp1_1, n1_pp1_1, pp1_1, n0_pp1_2, n1_pp1_2, pp1_2, &
      n0_pp1_3, n1_pp1_3, pp1_3, n0_b_part, n1_b_part, b_part, kind_map &
      , n0_params_map, params_map)

  use STRUPHY_fields, only: mod_evaluate_2form => evaluate_2form
  implicit none
  integer(kind=4), intent(in)  :: n0_particles_pos 
  integer(kind=4), intent(in)  :: n1_particles_pos 
  real(kind=8), intent(in)  :: particles_pos (0:n0_particles_pos-1,0: &
      n1_particles_pos-1)
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
  integer(kind=4), intent(in)  :: n0_p0 
  integer(kind=8), intent(in)  :: p0 (0:n0_p0-1)
  integer(kind=4), intent(in)  :: n0_nel 
  integer(kind=8), intent(in)  :: nel (0:n0_nel-1)
  integer(kind=4), intent(in)  :: n0_nbase 
  integer(kind=4), intent(in)  :: n1_nbase 
  integer(kind=8), intent(in)  :: nbase (0:n0_nbase-1,0:n1_nbase-1)
  integer(kind=8), intent(in)  :: np 
  integer(kind=4), intent(in)  :: n0_b1 
  integer(kind=4), intent(in)  :: n1_b1 
  integer(kind=4), intent(in)  :: n2_b1 
  real(kind=8), intent(in)  :: b1 (0:n0_b1-1,0:n1_b1-1,0:n2_b1-1)
  integer(kind=4), intent(in)  :: n0_b2 
  integer(kind=4), intent(in)  :: n1_b2 
  integer(kind=4), intent(in)  :: n2_b2 
  real(kind=8), intent(in)  :: b2 (0:n0_b2-1,0:n1_b2-1,0:n2_b2-1)
  integer(kind=4), intent(in)  :: n0_b3 
  integer(kind=4), intent(in)  :: n1_b3 
  integer(kind=4), intent(in)  :: n2_b3 
  real(kind=8), intent(in)  :: b3 (0:n0_b3-1,0:n1_b3-1,0:n2_b3-1)
  integer(kind=4), intent(in)  :: n0_pp0_1 
  integer(kind=4), intent(in)  :: n1_pp0_1 
  real(kind=8), intent(in)  :: pp0_1 (0:n0_pp0_1-1,0:n1_pp0_1-1)
  integer(kind=4), intent(in)  :: n0_pp0_2 
  integer(kind=4), intent(in)  :: n1_pp0_2 
  real(kind=8), intent(in)  :: pp0_2 (0:n0_pp0_2-1,0:n1_pp0_2-1)
  integer(kind=4), intent(in)  :: n0_pp0_3 
  integer(kind=4), intent(in)  :: n1_pp0_3 
  real(kind=8), intent(in)  :: pp0_3 (0:n0_pp0_3-1,0:n1_pp0_3-1)
  integer(kind=4), intent(in)  :: n0_pp1_1 
  integer(kind=4), intent(in)  :: n1_pp1_1 
  real(kind=8), intent(in)  :: pp1_1 (0:n0_pp1_1-1,0:n1_pp1_1-1)
  integer(kind=4), intent(in)  :: n0_pp1_2 
  integer(kind=4), intent(in)  :: n1_pp1_2 
  real(kind=8), intent(in)  :: pp1_2 (0:n0_pp1_2-1,0:n1_pp1_2-1)
  integer(kind=4), intent(in)  :: n0_pp1_3 
  integer(kind=4), intent(in)  :: n1_pp1_3 
  real(kind=8), intent(in)  :: pp1_3 (0:n0_pp1_3-1,0:n1_pp1_3-1)
  integer(kind=4), intent(in)  :: n0_b_part 
  integer(kind=4), intent(in)  :: n1_b_part 
  real(kind=8), intent(inout)  :: b_part (0:n0_b_part-1,0:n1_b_part-1)
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)

  call mod_evaluate_2form(particles_pos,t0_1,t0_2,t0_3,t1_1,t1_2,t1_3,p0 &
      ,nel,nbase,np,b1,b2,b3,pp0_1,pp0_2,pp0_3,pp1_1,pp1_2,pp1_3,b_part &
      ,kind_map,params_map)
end subroutine