subroutine set_particles_symmetric (n0_numbers, n1_numbers, numbers, &
      n0_particles, n1_particles, particles)

  use STRUPHY_sampling, only: mod_set_particles_symmetric => &
      set_particles_symmetric
  implicit none
  integer(kind=4), intent(in)  :: n0_numbers 
  integer(kind=4), intent(in)  :: n1_numbers 
  real(kind=8), intent(in)  :: numbers (0:n0_numbers-1,0:n1_numbers-1)
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles-1,0: &
      n1_particles-1)

  call mod_set_particles_symmetric(numbers,particles)
end subroutine

subroutine compute_weights_ini (n0_particles, n1_particles, particles, &
      n0_w0, w0, n0_g0, g0, kind_map, n0_params_map, params_map)

  use STRUPHY_sampling, only: mod_compute_weights_ini => &
      compute_weights_ini
  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(in)  :: particles (0:n0_particles-1,0: &
      n1_particles-1)
  integer(kind=4), intent(in)  :: n0_w0 
  real(kind=8), intent(inout)  :: w0 (0:n0_w0-1)
  integer(kind=4), intent(in)  :: n0_g0 
  real(kind=8), intent(inout)  :: g0 (0:n0_g0-1)
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)

  call mod_compute_weights_ini(particles,w0,g0,kind_map,params_map)
end subroutine

subroutine update_weights (n0_particles, n1_particles, particles, n0_w0, &
      w0, n0_g0, g0, kind_map, n0_params_map, params_map)

  use STRUPHY_sampling, only: mod_update_weights => update_weights
  implicit none
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles-1,0: &
      n1_particles-1)
  integer(kind=4), intent(in)  :: n0_w0 
  real(kind=8), intent(in)  :: w0 (0:n0_w0-1)
  integer(kind=4), intent(in)  :: n0_g0 
  real(kind=8), intent(in)  :: g0 (0:n0_g0-1)
  integer(kind=8), intent(in)  :: kind_map 
  integer(kind=4), intent(in)  :: n0_params_map 
  real(kind=8), intent(in)  :: params_map (0:n0_params_map-1)

  call mod_update_weights(particles,w0,g0,kind_map,params_map)
end subroutine