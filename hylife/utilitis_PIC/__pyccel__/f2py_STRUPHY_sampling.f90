subroutine set_particles_symmetric(n0_numbers, n1_numbers, numbers, &
      n0_particles, n1_particles, particles)

  use STRUPHY_sampling, only: mod_set_particles_symmetric => &
      set_particles_symmetric
  implicit none
  integer(kind=4), intent(in)  :: n0_numbers 
  integer(kind=4), intent(in)  :: n1_numbers 
  real(kind=8), intent(in)  :: numbers (0:n0_numbers - 1,0:n1_numbers - &
      1)
  integer(kind=4), intent(in)  :: n0_particles 
  integer(kind=4), intent(in)  :: n1_particles 
  real(kind=8), intent(inout)  :: particles (0:n0_particles - 1,0: &
      n1_particles - 1)

  !f2py integer(kind=4) :: n0_numbers=shape(numbers,1)
  !f2py integer(kind=4) :: n1_numbers=shape(numbers,0)
  !f2py intent(c) numbers
  call mod_set_particles_symmetric(numbers,particles)
end subroutine