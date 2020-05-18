module STRUPHY_pusher

use mappings_analytical, only: mapping_8z3wk9_g_inv => g_inv
use mappings_analytical, only: mapping_8z3wk9_df_inv => df_inv

use core, only: linalg_f7anys_det => det
use core, only: linalg_f7anys_transpose => transpose
use core, only: linalg_f7anys_matrix_matrix => matrix_matrix
use core, only: linalg_f7anys_matrix_vector => matrix_vector
implicit none




contains

!........................................
subroutine pusher_step3(particles, dt, b_part, u_part, kind_map, &
      params_map)

  implicit none
  real(kind=8), intent(inout)  :: particles (0:,0:)
  real(kind=8), value  :: dt
  real(kind=8), intent(in)  :: b_part (0:,0:)
  real(kind=8), intent(in)  :: u_part (0:,0:)
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  real(kind=8), allocatable  :: b (:) 
  real(kind=8), allocatable  :: u (:) 
  real(kind=8), allocatable  :: b_prod (:,:) 
  real(kind=8), allocatable  :: xi (:) 
  real(kind=8), allocatable  :: v (:) 
  real(kind=8), allocatable  :: dfinv (:,:) 
  real(kind=8), allocatable  :: dfinv_t (:,:) 
  real(kind=8), allocatable  :: ginv (:,:) 
  real(kind=8), allocatable  :: temp_mat1 (:,:) 
  real(kind=8), allocatable  :: temp_mat2 (:,:) 
  real(kind=8), allocatable  :: temp_vec (:) 
  integer(kind=8)  :: np  
  integer(kind=8), allocatable  :: components (:,:) 
  integer(kind=8)  :: ierr  
  integer(kind=8)  :: ip  
  integer(kind=8)  :: i  
  integer(kind=8)  :: j  







  allocate(b(0:2_8))
  allocate(u(0:2_8))


  allocate(b_prod(0:2_8, 0:2_8))
  b_prod = 0.0


  allocate(xi(0:2_8))
  allocate(v(0:2_8))


  allocate(dfinv(0:2_8, 0:2_8))
  allocate(dfinv_t(0:2_8, 0:2_8))
  allocate(ginv(0:2_8, 0:2_8))


  allocate(temp_mat1(0:2_8, 0:2_8))
  allocate(temp_mat2(0:2_8, 0:2_8))


  allocate(temp_vec(0:2_8))


  np = size(particles(:, 0_8),1)


  allocate(components(0:2_8, 0:2_8))


  components(0_8, 0_8) = 11_8
  components(0_8, 1_8) = 12_8
  components(0_8, 2_8) = 13_8
  components(1_8, 0_8) = 21_8
  components(1_8, 1_8) = 22_8
  components(1_8, 2_8) = 23_8
  components(2_8, 0_8) = 31_8
  components(2_8, 1_8) = 32_8
  components(2_8, 2_8) = 33_8


  !$omp parallel
  !$omp do private(ip, b, u, xi, v, i, j, dfinv, dfinv_t, ginv, temp_mat1, temp_&
      !$omp &mat2, temp_vec) firstprivate(b_prod)
  do ip = 0, np - 1_8, 1


    b(0_8) = b_part(ip, 0_8)
    b(1_8) = b_part(ip, 1_8)
    b(2_8) = b_part(ip, 2_8)


    u(0_8) = u_part(ip, 0_8)
    u(1_8) = u_part(ip, 1_8)
    u(2_8) = u_part(ip, 2_8)


    b_prod(0_8, 1_8) = -b(2_8)
    b_prod(0_8, 2_8) = b(1_8)


    b_prod(1_8, 0_8) = b(2_8)
    b_prod(1_8, 2_8) = -b(0_8)


    b_prod(2_8, 0_8) = -b(1_8)
    b_prod(2_8, 1_8) = b(0_8)


    xi = particles(ip, 0_8:2_8)
    v = particles(ip, 3_8:5_8)


    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        dfinv(i, j) = mapping_8z3wk9_df_inv(xi(0_8), xi(1_8), xi(2_8), &
      kind_map, params_map, components(i, j))


      end do

    end do

    call linalg_f7anys_transpose(dfinv, dfinv_t)


    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        ginv(i, j) = mapping_8z3wk9_g_inv(xi(0_8), xi(1_8), xi(2_8), &
      kind_map, params_map, components(i, j))


      end do

    end do

    call linalg_f7anys_matrix_matrix(dfinv_t, b_prod, temp_mat1)
    call linalg_f7anys_matrix_matrix(temp_mat1, ginv, temp_mat2)
    call linalg_f7anys_matrix_vector(temp_mat2, u, temp_vec)


    particles(ip, 3_8) = dt*temp_vec(0_8) + particles(ip, 3_8)
    particles(ip, 4_8) = dt*temp_vec(1_8) + particles(ip, 4_8)
    particles(ip, 5_8) = dt*temp_vec(2_8) + particles(ip, 5_8)




  end do

  !evaluate inverse Jacobian matrix
  !transpose of inverse Jacobian matrix
  !evaluate inverse metric tensor
  !perform matrix-matrix and matrix-vector products
  !update particle velocities
  !$omp end do  
  !$omp end parallel  
  ierr = 0_8
end subroutine
!........................................

!........................................
subroutine pusher_step4(particles, dt, kind_map, params_map) 

  implicit none
  real(kind=8), intent(inout)  :: particles (0:,0:)
  real(kind=8), value  :: dt
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  real(kind=8), allocatable  :: xi (:) 
  real(kind=8), allocatable  :: v (:) 
  real(kind=8), allocatable  :: dfinv (:,:) 
  integer(kind=8)  :: np  
  integer(kind=8), allocatable  :: components (:,:) 
  real(kind=8), allocatable  :: k1 (:) 
  real(kind=8), allocatable  :: k2 (:) 
  real(kind=8), allocatable  :: k3 (:) 
  real(kind=8), allocatable  :: k4 (:) 
  integer(kind=8)  :: ierr  
  integer(kind=8)  :: ip  
  integer(kind=8)  :: i  
  integer(kind=8)  :: j  






  allocate(xi(0:2_8))
  allocate(v(0:2_8))


  allocate(dfinv(0:2_8, 0:2_8))


  np = size(particles(:, 0_8),1)


  allocate(components(0:2_8, 0:2_8))


  components(0_8, 0_8) = 11_8
  components(0_8, 1_8) = 12_8
  components(0_8, 2_8) = 13_8
  components(1_8, 0_8) = 21_8
  components(1_8, 1_8) = 22_8
  components(1_8, 2_8) = 23_8
  components(2_8, 0_8) = 31_8
  components(2_8, 1_8) = 32_8
  components(2_8, 2_8) = 33_8


  allocate(k1(0:2_8))
  allocate(k2(0:2_8))
  allocate(k3(0:2_8))
  allocate(k4(0:2_8))


  !$omp parallel
  !$omp do private(ip, xi, v, dfinv, k1, k2, k3, k4)
  do ip = 0, np - 1_8, 1


    xi = particles(ip, 0_8:2_8)
    v = particles(ip, 3_8:5_8)


    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        dfinv(i, j) = mapping_8z3wk9_df_inv(xi(0_8), xi(1_8), xi(2_8), &
      kind_map, params_map, components(i, j))


      end do

    end do

    call linalg_f7anys_matrix_vector(dfinv, v, k1)


    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        dfinv(i, j) = mapping_8z3wk9_df_inv(dt*(k1(0_8)/Real(2_8, 8)) + &
      xi(0_8), dt*(k1(1_8)/Real(2_8, 8)) + xi(1_8), dt*(k1(2_8)/Real( &
      2_8, 8)) + xi(2_8), kind_map, params_map, components(i, j))


      end do

    end do

    call linalg_f7anys_matrix_vector(dfinv, v, k2)


    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        dfinv(i, j) = mapping_8z3wk9_df_inv(dt*(k2(0_8)/Real(2_8, 8)) + &
      xi(0_8), dt*(k2(1_8)/Real(2_8, 8)) + xi(1_8), dt*(k2(2_8)/Real( &
      2_8, 8)) + xi(2_8), kind_map, params_map, components(i, j))


      end do

    end do

    call linalg_f7anys_matrix_vector(dfinv, v, k3)


    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        dfinv(i, j) = mapping_8z3wk9_df_inv(dt*k3(0_8) + xi(0_8), dt*k3( &
      1_8) + xi(1_8), dt*k3(2_8) + xi(2_8), kind_map, params_map, &
      components(i, j))


      end do

    end do

    call linalg_f7anys_matrix_vector(dfinv, v, k4)


    particles(ip, 0_8) = modulo(0.166666666666667d0*dt*(2_8*k3(0_8) + k4 &
      (0_8) + 2_8*k2(0_8) + k1(0_8)) + 1.0d0*xi(0_8),1.0d0)
    particles(ip, 1_8) = modulo(0.166666666666667d0*dt*(2_8*k3(1_8) + k4 &
      (1_8) + 2_8*k2(1_8) + k1(1_8)) + 1.0d0*xi(1_8),1.0d0)
    particles(ip, 2_8) = modulo(0.166666666666667d0*dt*(2_8*k3(2_8) + k4 &
      (2_8) + 2_8*k2(2_8) + k1(2_8)) + 1.0d0*xi(2_8),1.0d0)




  end do

  !step 1 in Runge-Kutta method
  !step 2 in Runge-Kutta method
  !step 3 in Runge-Kutta method
  !step 4 in Runge-Kutta method
  !update logical coordinates
  !$omp end do  
  !$omp end parallel  
  ierr = 0_8
end subroutine
!........................................

!........................................
subroutine pusher_step5(particles, dt, b_part, kind_map, params_map) 

  implicit none
  real(kind=8), intent(inout)  :: particles (0:,0:)
  real(kind=8), value  :: dt
  real(kind=8), intent(in)  :: b_part (0:,0:)
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  real(kind=8), allocatable  :: b (:) 
  real(kind=8), allocatable  :: b_prod (:,:) 
  real(kind=8), allocatable  :: v (:) 
  real(kind=8), allocatable  :: xi (:) 
  real(kind=8), allocatable  :: dfinv (:,:) 
  real(kind=8), allocatable  :: dfinv_t (:,:) 
  real(kind=8), allocatable  :: temp_mat1 (:,:) 
  real(kind=8), allocatable  :: temp_mat2 (:,:) 
  real(kind=8), allocatable  :: rhs (:) 
  real(kind=8), allocatable  :: identity (:,:) 
  real(kind=8), allocatable  :: lhs (:,:) 
  real(kind=8), allocatable  :: lhs1 (:,:) 
  real(kind=8), allocatable  :: lhs2 (:,:) 
  real(kind=8), allocatable  :: lhs3 (:,:) 
  integer(kind=8)  :: np  
  integer(kind=8), allocatable  :: components (:,:) 
  integer(kind=8)  :: ierr  
  integer(kind=8)  :: ip  
  real(kind=8)  :: det_lhs  
  real(kind=8)  :: det_lhs1  
  real(kind=8)  :: det_lhs2  
  real(kind=8)  :: det_lhs3  
  integer(kind=8)  :: i  
  integer(kind=8)  :: j  







  allocate(b(0:2_8))


  allocate(b_prod(0:2_8, 0:2_8))
  b_prod = 0.0


  allocate(v(0:2_8))
  allocate(xi(0:2_8))


  allocate(dfinv(0:2_8, 0:2_8))
  allocate(dfinv_t(0:2_8, 0:2_8))


  allocate(temp_mat1(0:2_8, 0:2_8))
  allocate(temp_mat2(0:2_8, 0:2_8))


  allocate(rhs(0:2_8))


  allocate(identity(0:2_8, 0:2_8))
  identity = 0.0
  identity(0_8, 0_8) = 1.0d0
  identity(1_8, 1_8) = 1.0d0
  identity(2_8, 2_8) = 1.0d0


  allocate(lhs(0:2_8, 0:2_8))


  allocate(lhs1(0:2_8, 0:2_8))
  allocate(lhs2(0:2_8, 0:2_8))
  allocate(lhs3(0:2_8, 0:2_8))


  np = size(particles(:, 0_8),1)


  allocate(components(0:2_8, 0:2_8))


  components(0_8, 0_8) = 11_8
  components(0_8, 1_8) = 12_8
  components(0_8, 2_8) = 13_8
  components(1_8, 0_8) = 21_8
  components(1_8, 1_8) = 22_8
  components(1_8, 2_8) = 23_8
  components(2_8, 0_8) = 31_8
  components(2_8, 1_8) = 32_8
  components(2_8, 2_8) = 33_8


  !$omp parallel
  !$omp do private(ip, b, xi, v, dfinv, dfinv_t, temp_mat1, temp_mat2, rhs, lhs,&
      !$omp & det_lhs, lhs1, lhs2, lhs3, det_lhs1, det_lhs2, det_lhs3) firstprivate(b&
      !$omp &_prod)
  do ip = 0, np - 1_8, 1


    b(0_8) = b_part(ip, 0_8)
    b(1_8) = b_part(ip, 1_8)
    b(2_8) = b_part(ip, 2_8)


    b_prod(0_8, 1_8) = -b(2_8)
    b_prod(0_8, 2_8) = b(1_8)


    b_prod(1_8, 0_8) = b(2_8)
    b_prod(1_8, 2_8) = -b(0_8)


    b_prod(2_8, 0_8) = -b(1_8)
    b_prod(2_8, 1_8) = b(0_8)


    xi = particles(ip, 0_8:2_8)
    v = particles(ip, 3_8:5_8)


    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        dfinv(i, j) = mapping_8z3wk9_df_inv(xi(0_8), xi(1_8), xi(2_8), &
      kind_map, params_map, components(i, j))


      end do

    end do

    call linalg_f7anys_transpose(dfinv, dfinv_t)


    call linalg_f7anys_matrix_matrix(b_prod, dfinv, temp_mat1)
    call linalg_f7anys_matrix_matrix(dfinv_t, temp_mat1, temp_mat2)


    call linalg_f7anys_matrix_vector((-dt)*(temp_mat2/Real(2_8, 8)) + &
      identity, v, rhs)


    lhs = dt*(temp_mat2/Real(2_8, 8)) + identity


    det_lhs = linalg_f7anys_det(lhs)


    lhs1(:, 0_8) = rhs
    lhs1(:, 1_8) = lhs(:, 1_8)
    lhs1(:, 2_8) = lhs(:, 2_8)


    lhs2(:, 0_8) = lhs(:, 0_8)
    lhs2(:, 1_8) = rhs
    lhs2(:, 2_8) = lhs(:, 2_8)


    lhs3(:, 0_8) = lhs(:, 0_8)
    lhs3(:, 1_8) = lhs(:, 1_8)
    lhs3(:, 2_8) = rhs


    det_lhs1 = linalg_f7anys_det(lhs1)
    det_lhs2 = linalg_f7anys_det(lhs2)
    det_lhs3 = linalg_f7anys_det(lhs3)


    particles(ip, 3_8) = det_lhs1/det_lhs
    particles(ip, 4_8) = det_lhs2/det_lhs
    particles(ip, 5_8) = det_lhs3/det_lhs




  end do

  !evaluate inverse Jacobian matrix
  !transpose of inverse Jacobian matrix
  !perform matrix-matrix and matrix-vector multiplications
  !explicit part of update rule
  !implicit part of update rule
  !solve 3 x 3 system with Cramer's rule
  !$omp end do  
  !$omp end parallel  
  ierr = 0_8
end subroutine
!........................................

end module