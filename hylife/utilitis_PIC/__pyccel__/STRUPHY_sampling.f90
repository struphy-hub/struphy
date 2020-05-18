module STRUPHY_sampling

use interface, only: inter_mlo7sl_fh_eq => fh_eq
use interface, only: inter_mlo7sl_g_sampling => g_sampling
use interface, only: inter_mlo7sl_fh_ini => fh_ini
implicit none




contains

!........................................
subroutine set_particles_symmetric(numbers, particles) 

  implicit none
  real(kind=8), intent(in)  :: numbers (0:,0:)
  real(kind=8), intent(inout)  :: particles (0:,0:)
  real(kind=8), allocatable  :: q (:) 
  real(kind=8), allocatable  :: v (:) 
  integer(kind=8)  :: np  
  integer(kind=8)  :: ierr  
  integer(kind=8)  :: i_part  
  integer(kind=8)  :: ip  






  allocate(q(0:2_8))
  q = 0.0
  allocate(v(0:2_8))
  v = 0.0
  np = size(particles(:, 0_8),1)


  do i_part = 0, np - 1_8, 1
    ip = modulo(i_part,64_8)


    if (ip == 0_8 ) then
      q = numbers(Int(i_part/Real(64_8, 8), 8), 0_8:2_8)
      v = numbers(Int(i_part/Real(64_8, 8), 8), 3_8:5_8)
    else if (modulo(ip,32_8) == 0_8 ) then
      v(2_8) = -v(2_8) + 1_8
    else if (modulo(ip,16_8) == 0_8 ) then
      v(1_8) = -v(1_8) + 1_8
    else if (modulo(ip,8_8) == 0_8 ) then
      v(0_8) = -v(0_8) + 1_8
    else if (modulo(ip,4_8) == 0_8 ) then
      q(2_8) = -q(2_8) + 1_8
    else if (modulo(ip,2_8) == 0_8 ) then
      q(1_8) = -q(1_8) + 1_8
    else
      q(0_8) = -q(0_8) + 1_8
    end if












    particles(i_part, 0_8:2_8) = q
    particles(i_part, 3_8:5_8) = v




  end do

  ierr = 0_8
end subroutine
!........................................

!........................................
subroutine compute_weights_ini(particles, w0, g0, kind_map, params_map) 

  implicit none
  real(kind=8), intent(in)  :: particles (0:,0:)
  real(kind=8), intent(inout)  :: w0 (0:)
  real(kind=8), intent(inout)  :: g0 (0:)
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  integer(kind=8)  :: np  
  integer(kind=8)  :: ip  
  real(kind=8)  :: xi1  
  real(kind=8)  :: xi2  
  real(kind=8)  :: xi3  
  real(kind=8)  :: vx  
  real(kind=8)  :: vy  
  real(kind=8)  :: vz  



  np = size(particles(:, 0_8),1)


  do ip = 0, np - 1_8, 1


    xi1 = particles(ip, 0_8)
    xi2 = particles(ip, 1_8)
    xi3 = particles(ip, 2_8)


    vx = particles(ip, 3_8)
    vy = particles(ip, 4_8)
    vz = particles(ip, 5_8)


    g0(ip) = inter_mlo7sl_g_sampling(xi1, xi2, xi3, vx, vy, vz)
    w0(ip) = inter_mlo7sl_fh_ini(xi1, xi2, xi3, vx, vy, vz, kind_map, &
      params_map)/g0(ip)




  end do

end subroutine
!........................................

!........................................
subroutine update_weights(particles, w0, g0, kind_map, params_map) 

  implicit none
  real(kind=8), intent(inout)  :: particles (0:,0:)
  real(kind=8), intent(in)  :: w0 (0:)
  real(kind=8), intent(in)  :: g0 (0:)
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  integer(kind=8)  :: np  
  integer(kind=8)  :: ip  
  real(kind=8)  :: xi1  
  real(kind=8)  :: xi2  
  real(kind=8)  :: xi3  
  real(kind=8)  :: vx  
  real(kind=8)  :: vy  
  real(kind=8)  :: vz  



  np = size(particles(:, 0_8),1)


  do ip = 0, np - 1_8, 1


    xi1 = particles(ip, 0_8)
    xi2 = particles(ip, 1_8)
    xi3 = particles(ip, 2_8)


    vx = particles(ip, 3_8)
    vy = particles(ip, 4_8)
    vz = particles(ip, 5_8)


    particles(ip, 6_8) = (-inter_mlo7sl_fh_eq(xi1, xi2, xi3, vx, vy, vz, &
      kind_map, params_map))/g0(ip) + w0(ip)
  end do

end subroutine
!........................................

end module