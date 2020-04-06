subroutine pusher_step3 (n0_particles, n1_particles, particles, dt, &
      n0_b_part, n1_b_part, b_part, n0_u_part, n1_u_part, u_part, &
      kind_map, n0_params_map, params_map)

  use STRUPHY_pusher, only: mod_pusher_step3 => pusher_step3
  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles-1,0: &
      n1_particles-1)
  real(kind=8), intent(in)  :: dt 
  integer(kind=4), intent(in)  :: n0_b_part 
  integer(kind=4), intent(in)  :: n1_b_part 
  real(kind=8), intent(in)  :: b_part (0:n0_b_part-1,0:n1_b_part-1)
  integer(kind=4), intent(in)  :: n0_u_part 
  integer(kind=4), intent(in)  :: n1_u_part 
  real(kind=8), intent(in)  :: u_part (0:n0_u_part-1,0:n1_u_part-1)
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)

  call mod_pusher_step3(particles,dt,b_part,u_part,kind_map,params_map)
end subroutine

subroutine pusher_step4 (n0_particles, n1_particles, particles, dt, &
      kind_map, n0_params_map, params_map)

  use STRUPHY_pusher, only: mod_pusher_step4 => pusher_step4
  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles-1,0: &
      n1_particles-1)
  real(kind=8), intent(in)  :: dt 
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)

  call mod_pusher_step4(particles,dt,kind_map,params_map)
end subroutine

subroutine pusher_step5 (n0_particles, n1_particles, particles, dt, &
      n0_b_part, n1_b_part, b_part, kind_map, n0_params_map, params_map &
      )

  use STRUPHY_pusher, only: mod_pusher_step5 => pusher_step5
  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles-1,0: &
      n1_particles-1)
  real(kind=8), intent(in)  :: dt 
  integer(kind=4), intent(in)  :: n0_b_part 
  integer(kind=4), intent(in)  :: n1_b_part 
  real(kind=8), intent(in)  :: b_part (0:n0_b_part-1,0:n1_b_part-1)
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)

  call mod_pusher_step5(particles,dt,b_part,kind_map,params_map)
end subroutine