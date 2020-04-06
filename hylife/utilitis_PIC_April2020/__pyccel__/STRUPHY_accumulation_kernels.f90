module STRUPHY_accumulation_kernels

use core, only: linalg_o9wxuz_matrix_matrix => matrix_matrix
use core, only: linalg_o9wxuz_matrix_vector => matrix_vector
use core, only: linalg_o9wxuz_transpose => transpose

use mappings_analytical, only: mapping_otqb47_df_inv => df_inv
use mappings_analytical, only: mapping_otqb47_g_inv => g_inv
implicit none




contains

!........................................
subroutine basis_funs(knots, degree, x, span, left, right, values) 

  implicit none
  real(kind=8), intent(in)  :: knots (0:)
  integer(kind=8), value  :: degree
  real(kind=8), value  :: x
  integer(kind=8), value  :: span
  real(kind=8), intent(inout)  :: left (0:)
  real(kind=8), intent(inout)  :: right (0:)
  real(kind=8), intent(inout)  :: values (0:)
  integer(kind=8)  :: j  
  real(kind=8)  :: saved  
  integer(kind=8)  :: r  
  real(kind=8)  :: temp  



  left(:) = 0.0d0
  right(:) = 0.0d0


  values(0_8) = 1.0d0


  do j = 0, degree - 1_8, 1
    left(j) = x - knots(-j + span)
    right(j) = -x + knots(span + j + 1_8)
    saved = 0.0d0
    do r = 0, j, 1
      temp = values(r)/(left(j - r) + right(r))
      values(r) = saved + temp*right(r)
      saved = temp*left(j - r)


    end do

    values(j + 1_8) = saved






  end do

end subroutine
!........................................

!........................................
subroutine kernel_step1(particles, p0, nel, nbase, t0_1, t0_2, t0_3, &
      t1_1, t1_2, t1_3, b_part, kind_map, params_map, mat12, mat13, &
      mat23)

  implicit none
  real(kind=8), intent(in)  :: particles (0:,0:)
  integer(kind=8), intent(in)  :: p0 (0:)
  integer(kind=8), intent(in)  :: nel (0:)
  integer(kind=8), intent(in)  :: nbase (0:)
  real(kind=8), intent(in)  :: t0_1 (0:)
  real(kind=8), intent(in)  :: t0_2 (0:)
  real(kind=8), intent(in)  :: t0_3 (0:)
  real(kind=8), intent(in)  :: t1_1 (0:)
  real(kind=8), intent(in)  :: t1_2 (0:)
  real(kind=8), intent(in)  :: t1_3 (0:)
  real(kind=8), intent(in)  :: b_part (0:,0:)
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  real(kind=8), intent(inout)  :: mat12 (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: mat13 (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: mat23 (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: p0_1  
  integer(kind=8)  :: p0_2  
  integer(kind=8)  :: p0_3  
  integer(kind=8)  :: p1_1  
  integer(kind=8)  :: p1_2  
  integer(kind=8)  :: p1_3  
  real(kind=8), allocatable  :: nl1 (:) 
  real(kind=8), allocatable  :: nr1 (:) 
  real(kind=8), allocatable  :: nn1 (:) 
  real(kind=8), allocatable  :: nl2 (:) 
  real(kind=8), allocatable  :: nr2 (:) 
  real(kind=8), allocatable  :: nn2 (:) 
  real(kind=8), allocatable  :: nl3 (:) 
  real(kind=8), allocatable  :: nr3 (:) 
  real(kind=8), allocatable  :: nn3 (:) 
  real(kind=8), allocatable  :: dl1 (:) 
  real(kind=8), allocatable  :: dr1 (:) 
  real(kind=8), allocatable  :: dd1 (:) 
  real(kind=8), allocatable  :: dl2 (:) 
  real(kind=8), allocatable  :: dr2 (:) 
  real(kind=8), allocatable  :: dd2 (:) 
  real(kind=8), allocatable  :: dl3 (:) 
  real(kind=8), allocatable  :: dr3 (:) 
  real(kind=8), allocatable  :: dd3 (:) 
  real(kind=8), allocatable  :: b (:) 
  real(kind=8), allocatable  :: b_prod (:,:) 
  real(kind=8), allocatable  :: ginv (:,:) 
  real(kind=8), allocatable  :: temp_mat1 (:,:) 
  real(kind=8), allocatable  :: temp_mat2 (:,:) 
  integer(kind=8)  :: np  
  integer(kind=8), allocatable  :: components (:,:) 
  integer(kind=8)  :: ierr  
  integer(kind=8)  :: ip  
  real(kind=8)  :: pos1  
  real(kind=8)  :: pos2  
  real(kind=8)  :: pos3  
  integer(kind=8)  :: span0_1  
  integer(kind=8)  :: span0_2  
  integer(kind=8)  :: span0_3  
  integer(kind=8)  :: span1_1  
  integer(kind=8)  :: span1_2  
  integer(kind=8)  :: span1_3  
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  real(kind=8)  :: w  
  real(kind=8)  :: temp12  
  real(kind=8)  :: temp13  
  real(kind=8)  :: temp23  
  integer(kind=8)  :: i  
  integer(kind=8)  :: j  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: i1  
  real(kind=8)  :: bi1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: i2  
  real(kind=8)  :: bi2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: i3  
  real(kind=8)  :: bi3  
  integer(kind=8)  :: jl1  
  integer(kind=8)  :: j1  
  real(kind=8)  :: bj1  
  integer(kind=8)  :: jl2  
  integer(kind=8)  :: j2  
  real(kind=8)  :: bj2  
  integer(kind=8)  :: jl3  
  integer(kind=8)  :: j3  
  real(kind=8)  :: bj3  







  p0_1 = p0(0_8)
  p0_2 = p0(1_8)
  p0_3 = p0(2_8)


  p1_1 = p0_1 - 1_8
  p1_2 = p0_2 - 1_8
  p1_3 = p0_3 - 1_8


  allocate(nl1(0:p0_1 - 1_8))
  allocate(nr1(0:p0_1 - 1_8))
  allocate(nn1(0:p0_1))
  nn1 = 0.0


  allocate(nl2(0:p0_2 - 1_8))
  allocate(nr2(0:p0_2 - 1_8))
  allocate(nn2(0:p0_2))
  nn2 = 0.0


  allocate(nl3(0:p0_3 - 1_8))
  allocate(nr3(0:p0_3 - 1_8))
  allocate(nn3(0:p0_3))
  nn3 = 0.0


  allocate(dl1(0:p1_1 - 1_8))
  allocate(dr1(0:p1_1 - 1_8))
  allocate(dd1(0:p1_1))
  dd1 = 0.0


  allocate(dl2(0:p1_2 - 1_8))
  allocate(dr2(0:p1_2 - 1_8))
  allocate(dd2(0:p1_2))
  dd2 = 0.0


  allocate(dl3(0:p1_3 - 1_8))
  allocate(dr3(0:p1_3 - 1_8))
  allocate(dd3(0:p1_3))
  dd3 = 0.0


  allocate(b(0:2_8))


  allocate(b_prod(0:2_8, 0:2_8))
  b_prod = 0.0


  allocate(ginv(0:2_8, 0:2_8))


  allocate(temp_mat1(0:2_8, 0:2_8))
  allocate(temp_mat2(0:2_8, 0:2_8))


  np = size(particles(:, 0_8),1)


  mat12(:, :, :, :, :, :) = 0.0d0
  mat13(:, :, :, :, :, :) = 0.0d0
  mat23(:, :, :, :, :, :) = 0.0d0


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
  !$omp do reduction(+: mat12, mat13, mat23) private(ip, b, pos1, pos2, pos3, sp&
      !$omp &an0_1, span0_2, span0_3, span1_1, span1_2, span1_3, ie1, ie2, ie3, nl1, &
      !$omp &nr1, nn1, nl2, nr2, nn2, nl3, nr3, nn3, dl1, dr1, dd1, dl2, dr2, dd2, dl&
      !$omp &3, dr3, dd3, w, i, j, ginv, temp_mat1, temp_mat2, temp12, temp13, temp23&
      !$omp &, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, j1, j2, j3, bi1, bi2, bi3, b&
      !$omp &j1, bj2, bj3) firstprivate(b_prod)
  do ip = 0, np - 1_8, 1


    b(0_8) = b_part(ip, 0_8)
    b(1_8) = b_part(ip, 1_8)
    b(2_8) = b_part(ip, 2_8)


    pos1 = particles(ip, 0_8)
    pos2 = particles(ip, 1_8)
    pos3 = particles(ip, 2_8)


    span0_1 = p0_1 + Int(pos1*nel(0_8), 8)
    span0_2 = p0_2 + Int(pos2*nel(1_8), 8)
    span0_3 = p0_3 + Int(pos3*nel(2_8), 8)


    span1_1 = span0_1 - 1_8
    span1_2 = span0_2 - 1_8
    span1_3 = span0_3 - 1_8


    ie1 = -p0_1 + span0_1
    ie2 = -p0_2 + span0_2
    ie3 = -p0_3 + span0_3


    call basis_funs(t0_1, p0_1, pos1, span0_1, nl1, nr1, nn1)
    call basis_funs(t0_2, p0_2, pos2, span0_2, nl2, nr2, nn2)
    call basis_funs(t0_3, p0_3, pos3, span0_3, nl3, nr3, nn3)


    call basis_funs(t1_1, p1_1, pos1, span1_1, dl1, dr1, dd1)
    call basis_funs(t1_2, p1_2, pos2, span1_2, dl2, dr2, dd2)
    call basis_funs(t1_3, p1_3, pos3, span1_3, dl3, dr3, dd3)




    b_prod(0_8, 1_8) = -b(2_8)
    b_prod(0_8, 2_8) = b(1_8)


    b_prod(1_8, 0_8) = b(2_8)
    b_prod(1_8, 2_8) = -b(0_8)


    b_prod(2_8, 0_8) = -b(1_8)
    b_prod(2_8, 1_8) = b(0_8)


    w = particles(ip, 6_8)


    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        ginv(i, j) = mapping_otqb47_g_inv(pos1, pos2, pos3, kind_map, &
      params_map, components(i, j))




      end do

    end do

    call linalg_o9wxuz_matrix_matrix(ginv, b_prod, temp_mat1)
    call linalg_o9wxuz_matrix_matrix(temp_mat1, ginv, temp_mat2)




    temp12 = w*temp_mat2(0_8, 1_8)
    temp13 = w*temp_mat2(0_8, 2_8)
    temp23 = w*temp_mat2(1_8, 2_8)




    do il1 = 0, p1_1, 1
      i1 = modulo(ie1 + il1,nbase(3_8))
      bi1 = (p0_1/(t1_1(i1 + p0_1) - t1_1(i1)))*dd1(il1)
      do il2 = 0, p0_2, 1
        i2 = modulo(ie2 + il2,nbase(1_8))
        bi2 = bi1*nn2(il2)
        do il3 = 0, p0_3, 1
          i3 = modulo(ie3 + il3,nbase(2_8))
          bi3 = bi2*nn3(il3)
          do jl1 = 0, p0_1, 1
            j1 = modulo(ie1 + jl1,nbase(0_8))
            bj1 = bi3*nn1(jl1)


            do jl2 = 0, p1_2, 1
              j2 = modulo(ie2 + jl2,nbase(4_8))
              bj2 = bj1*((p0_2*(temp12/(t1_2(j2 + p0_2) - t1_2(j2))))* &
      dd2(jl2))
              do jl3 = 0, p0_3, 1
                j3 = modulo(ie3 + jl3,nbase(2_8))
                bj3 = bj2*nn3(jl3)


                mat12(i1, i2, i3, p0_1 - il1 + jl1, p0_2 - il2 + jl2, &
      p0_3 - il3 + jl3) = bj3 + mat12(i1, i2, i3, p0_1 - il1 + jl1, &
      p0_2 - il2 + jl2, p0_3 - il3 + jl3)


              end do

            end do

            do jl2 = 0, p0_2, 1
              j2 = modulo(ie2 + jl2,nbase(1_8))
              bj2 = bj1*(temp13*nn2(jl2))
              do jl3 = 0, p1_3, 1
                j3 = modulo(ie3 + jl3,nbase(5_8))
                bj3 = bj2*((p0_3/(t1_3(j3 + p0_3) - t1_3(j3)))*dd3(jl3))


                mat13(i1, i2, i3, p0_1 - il1 + jl1, p0_2 - il2 + jl2, &
      p0_3 - il3 + jl3) = bj3 + mat13(i1, i2, i3, p0_1 - il1 + jl1, &
      p0_2 - il2 + jl2, p0_3 - il3 + jl3)








              end do

            end do

          end do

        end do

      end do

    end do

    do il1 = 0, p0_1, 1
      i1 = modulo(ie1 + il1,nbase(0_8))
      bi1 = temp23*nn1(il1)
      do il2 = 0, p1_2, 1
        i2 = modulo(ie2 + il2,nbase(4_8))
        bi2 = bi1*((p0_2/(t1_2(i2 + p0_2) - t1_2(i2)))*dd2(il2))
        do il3 = 0, p0_3, 1
          i3 = modulo(ie3 + il3,nbase(2_8))
          bi3 = bi2*nn3(il3)
          do jl1 = 0, p0_1, 1
            j1 = modulo(ie1 + jl1,nbase(0_8))
            bj1 = bi3*nn1(jl1)
            do jl2 = 0, p0_2, 1
              j2 = modulo(ie2 + jl2,nbase(1_8))
              bj2 = bj1*nn2(jl2)
              do jl3 = 0, p1_3, 1
                j3 = modulo(ie3 + jl3,nbase(5_8))
                bj3 = bj2*((p0_3/(t1_3(j3 + p0_3) - t1_3(j3)))*dd3(jl3))


                mat23(i1, i2, i3, p0_1 - il1 + jl1, p0_2 - il2 + jl2, &
      p0_3 - il3 + jl3) = bj3 + mat23(i1, i2, i3, p0_1 - il1 + jl1, &
      p0_2 - il2 + jl2, p0_3 - il3 + jl3)






              end do

            end do

          end do

        end do

      end do

    end do

  end do

  !evaluate inverse metric tensor
  !perform matrix-matrix multiplications
  !add contribution to 12 component (DNN NDN) and 13 component (DNN NND)
  !add contribution to 23 component (NDN NND)
  !$omp end do  
  !$omp end parallel  
  ierr = 0_8
end subroutine
!........................................

!........................................
subroutine kernel_step3(particles, p0, nel, nbase, t0_1, t0_2, t0_3, &
      t1_1, t1_2, t1_3, b_part, kind_map, params_map, mat11, mat12, &
      mat13, mat22, mat23, mat33, vec1, vec2, vec3)

  implicit none
  real(kind=8), intent(in)  :: particles (0:,0:)
  integer(kind=8), intent(in)  :: p0 (0:)
  integer(kind=8), intent(in)  :: nel (0:)
  integer(kind=8), intent(in)  :: nbase (0:)
  real(kind=8), intent(in)  :: t0_1 (0:)
  real(kind=8), intent(in)  :: t0_2 (0:)
  real(kind=8), intent(in)  :: t0_3 (0:)
  real(kind=8), intent(in)  :: t1_1 (0:)
  real(kind=8), intent(in)  :: t1_2 (0:)
  real(kind=8), intent(in)  :: t1_3 (0:)
  real(kind=8), intent(in)  :: b_part (0:,0:)
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  real(kind=8), intent(inout)  :: mat11 (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: mat12 (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: mat13 (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: mat22 (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: mat23 (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: mat33 (0:,0:,0:,0:,0:,0:)
  real(kind=8), intent(inout)  :: vec1 (0:,0:,0:)
  real(kind=8), intent(inout)  :: vec2 (0:,0:,0:)
  real(kind=8), intent(inout)  :: vec3 (0:,0:,0:)
  integer(kind=8)  :: p0_1  
  integer(kind=8)  :: p0_2  
  integer(kind=8)  :: p0_3  
  integer(kind=8)  :: p1_1  
  integer(kind=8)  :: p1_2  
  integer(kind=8)  :: p1_3  
  real(kind=8), allocatable  :: nl1 (:) 
  real(kind=8), allocatable  :: nr1 (:) 
  real(kind=8), allocatable  :: nn1 (:) 
  real(kind=8), allocatable  :: nl2 (:) 
  real(kind=8), allocatable  :: nr2 (:) 
  real(kind=8), allocatable  :: nn2 (:) 
  real(kind=8), allocatable  :: nl3 (:) 
  real(kind=8), allocatable  :: nr3 (:) 
  real(kind=8), allocatable  :: nn3 (:) 
  real(kind=8), allocatable  :: dl1 (:) 
  real(kind=8), allocatable  :: dr1 (:) 
  real(kind=8), allocatable  :: dd1 (:) 
  real(kind=8), allocatable  :: dl2 (:) 
  real(kind=8), allocatable  :: dr2 (:) 
  real(kind=8), allocatable  :: dd2 (:) 
  real(kind=8), allocatable  :: dl3 (:) 
  real(kind=8), allocatable  :: dr3 (:) 
  real(kind=8), allocatable  :: dd3 (:) 
  real(kind=8), allocatable  :: b (:) 
  real(kind=8), allocatable  :: b_prod (:,:) 
  real(kind=8), allocatable  :: b_prod_t (:,:) 
  real(kind=8), allocatable  :: ginv (:,:) 
  real(kind=8), allocatable  :: dfinv (:,:) 
  real(kind=8), allocatable  :: temp_mat1 (:,:) 
  real(kind=8), allocatable  :: temp_mat2 (:,:) 
  real(kind=8), allocatable  :: temp_mat_vec (:,:) 
  real(kind=8), allocatable  :: temp_vec (:) 
  real(kind=8), allocatable  :: v (:) 
  integer(kind=8)  :: np  
  integer(kind=8), allocatable  :: components (:,:) 
  integer(kind=8)  :: ierr  
  integer(kind=8)  :: ip  
  real(kind=8)  :: pos1  
  real(kind=8)  :: pos2  
  real(kind=8)  :: pos3  
  integer(kind=8)  :: span0_1  
  integer(kind=8)  :: span0_2  
  integer(kind=8)  :: span0_3  
  integer(kind=8)  :: span1_1  
  integer(kind=8)  :: span1_2  
  integer(kind=8)  :: span1_3  
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  real(kind=8)  :: w  
  real(kind=8)  :: temp11  
  real(kind=8)  :: temp12  
  real(kind=8)  :: temp13  
  real(kind=8)  :: temp22  
  real(kind=8)  :: temp23  
  real(kind=8)  :: temp33  
  real(kind=8)  :: temp1  
  real(kind=8)  :: temp2  
  real(kind=8)  :: temp3  
  integer(kind=8)  :: i  
  integer(kind=8)  :: j  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: i1  
  real(kind=8)  :: bi1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: i2  
  real(kind=8)  :: bi2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: i3  
  real(kind=8)  :: bi3  
  integer(kind=8)  :: jl1  
  integer(kind=8)  :: j1  
  real(kind=8)  :: bj1  
  integer(kind=8)  :: jl2  
  integer(kind=8)  :: j2  
  real(kind=8)  :: bj2  
  integer(kind=8)  :: jl3  
  integer(kind=8)  :: j3  
  real(kind=8)  :: bj3  







  p0_1 = p0(0_8)
  p0_2 = p0(1_8)
  p0_3 = p0(2_8)


  p1_1 = p0_1 - 1_8
  p1_2 = p0_2 - 1_8
  p1_3 = p0_3 - 1_8


  allocate(nl1(0:p0_1 - 1_8))
  allocate(nr1(0:p0_1 - 1_8))
  allocate(nn1(0:p0_1))
  nn1 = 0.0


  allocate(nl2(0:p0_2 - 1_8))
  allocate(nr2(0:p0_2 - 1_8))
  allocate(nn2(0:p0_2))
  nn2 = 0.0


  allocate(nl3(0:p0_3 - 1_8))
  allocate(nr3(0:p0_3 - 1_8))
  allocate(nn3(0:p0_3))
  nn3 = 0.0


  allocate(dl1(0:p1_1 - 1_8))
  allocate(dr1(0:p1_1 - 1_8))
  allocate(dd1(0:p1_1))
  dd1 = 0.0


  allocate(dl2(0:p1_2 - 1_8))
  allocate(dr2(0:p1_2 - 1_8))
  allocate(dd2(0:p1_2))
  dd2 = 0.0


  allocate(dl3(0:p1_3 - 1_8))
  allocate(dr3(0:p1_3 - 1_8))
  allocate(dd3(0:p1_3))
  dd3 = 0.0


  allocate(b(0:2_8))


  allocate(b_prod(0:2_8, 0:2_8))
  b_prod = 0.0
  allocate(b_prod_t(0:2_8, 0:2_8))
  b_prod_t = 0.0


  allocate(ginv(0:2_8, 0:2_8))
  allocate(dfinv(0:2_8, 0:2_8))


  allocate(temp_mat1(0:2_8, 0:2_8))
  allocate(temp_mat2(0:2_8, 0:2_8))


  allocate(temp_mat_vec(0:2_8, 0:2_8))


  allocate(temp_vec(0:2_8))


  allocate(v(0:2_8))


  np = size(particles(:, 0_8),1)


  mat11(:, :, :, :, :, :) = 0.0d0
  mat12(:, :, :, :, :, :) = 0.0d0
  mat13(:, :, :, :, :, :) = 0.0d0
  mat22(:, :, :, :, :, :) = 0.0d0
  mat23(:, :, :, :, :, :) = 0.0d0
  mat33(:, :, :, :, :, :) = 0.0d0


  vec1(:, :, :) = 0.0d0
  vec2(:, :, :) = 0.0d0
  vec3(:, :, :) = 0.0d0


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
  !$omp do reduction(+: vec1, mat11, mat12, mat13, vec2, mat22, mat23, vec3, mat&
      !$omp &33) private(ip, b, pos1, pos2, pos3, span0_1, span0_2, span0_3, span1_1,&
      !$omp & span1_2, span1_3, ie1, ie2, ie3, nl1, nr1, nn1, nl2, nr2, nn2, nl3, nr3&
      !$omp &, nn3, dl1, dr1, dd1, dl2, dr2, dd2, dl3, dr3, dd3, v, w, i, j, ginv, df&
      !$omp &inv, temp_mat1, temp_mat2, temp_mat_vec, temp_vec, b_prod_t, temp11, tem&
      !$omp &p12, temp13, temp22, temp23, temp33, temp1, temp2, temp3, il1, il2, il3,&
      !$omp & jl1, jl2, jl3, i1, i2, i3, j1, j2, j3, bi1, bi2, bi3, bj1, bj2, bj3) fi&
      !$omp &rstprivate(b_prod)
  do ip = 0, np - 1_8, 1


    b(0_8) = b_part(ip, 0_8)
    b(1_8) = b_part(ip, 1_8)
    b(2_8) = b_part(ip, 2_8)


    pos1 = particles(ip, 0_8)
    pos2 = particles(ip, 1_8)
    pos3 = particles(ip, 2_8)


    span0_1 = p0_1 + Int(pos1*nel(0_8), 8)
    span0_2 = p0_2 + Int(pos2*nel(1_8), 8)
    span0_3 = p0_3 + Int(pos3*nel(2_8), 8)


    span1_1 = span0_1 - 1_8
    span1_2 = span0_2 - 1_8
    span1_3 = span0_3 - 1_8


    ie1 = -p0_1 + span0_1
    ie2 = -p0_2 + span0_2
    ie3 = -p0_3 + span0_3


    call basis_funs(t0_1, p0_1, pos1, span0_1, nl1, nr1, nn1)
    call basis_funs(t0_2, p0_2, pos2, span0_2, nl2, nr2, nn2)
    call basis_funs(t0_3, p0_3, pos3, span0_3, nl3, nr3, nn3)


    call basis_funs(t1_1, p1_1, pos1, span1_1, dl1, dr1, dd1)
    call basis_funs(t1_2, p1_2, pos2, span1_2, dl2, dr2, dd2)
    call basis_funs(t1_3, p1_3, pos3, span1_3, dl3, dr3, dd3)




    b_prod(0_8, 1_8) = -b(2_8)
    b_prod(0_8, 2_8) = b(1_8)


    b_prod(1_8, 0_8) = b(2_8)
    b_prod(1_8, 2_8) = -b(0_8)


    b_prod(2_8, 0_8) = -b(1_8)
    b_prod(2_8, 1_8) = b(0_8)


    v = particles(ip, 3_8:5_8)
    w = particles(ip, 6_8)


    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        ginv(i, j) = mapping_otqb47_g_inv(pos1, pos2, pos3, kind_map, &
      params_map, components(i, j))


      end do

    end do

    do i = 0, 2_8, 1
      do j = 0, 2_8, 1
        dfinv(i, j) = mapping_otqb47_df_inv(pos1, pos2, pos3, kind_map, &
      params_map, components(i, j))




      end do

    end do

    call linalg_o9wxuz_matrix_matrix(ginv, b_prod, temp_mat1)
    call linalg_o9wxuz_matrix_matrix(temp_mat1, dfinv, temp_mat_vec)
    call linalg_o9wxuz_matrix_vector(temp_mat_vec, v, temp_vec)


    call linalg_o9wxuz_matrix_matrix(temp_mat1, ginv, temp_mat2)
    call linalg_o9wxuz_transpose(b_prod, b_prod_t)
    call linalg_o9wxuz_matrix_matrix(temp_mat2, b_prod_t, temp_mat1)
    call linalg_o9wxuz_matrix_matrix(temp_mat1, ginv, temp_mat2)


    temp11 = w*temp_mat2(0_8, 0_8)
    temp12 = w*temp_mat2(0_8, 1_8)
    temp13 = w*temp_mat2(0_8, 2_8)
    temp22 = w*temp_mat2(1_8, 1_8)
    temp23 = w*temp_mat2(1_8, 2_8)
    temp33 = w*temp_mat2(2_8, 2_8)


    temp1 = w*temp_vec(0_8)
    temp2 = w*temp_vec(1_8)
    temp3 = w*temp_vec(2_8)




    do il1 = 0, p1_1, 1
      i1 = modulo(ie1 + il1,nbase(3_8))
      bi1 = (p0_1/(t1_1(i1 + p0_1) - t1_1(i1)))*dd1(il1)
      do il2 = 0, p0_2, 1
        i2 = modulo(ie2 + il2,nbase(1_8))
        bi2 = bi1*nn2(il2)
        do il3 = 0, p0_3, 1
          i3 = modulo(ie3 + il3,nbase(2_8))
          bi3 = bi2*nn3(il3)


          vec1(i1, i2, i3) = bi3*temp1 + vec1(i1, i2, i3)


          do jl1 = 0, p1_1, 1
            j1 = modulo(ie1 + jl1,nbase(3_8))
            bj1 = bi3*((p0_1*(temp11/(t1_1(j1 + p0_1) - t1_1(j1))))*dd1( &
      jl1))
            do jl2 = 0, p0_2, 1
              j2 = modulo(ie2 + jl2,nbase(1_8))
              bj2 = bj1*nn2(jl2)
              do jl3 = 0, p0_3, 1
                j3 = modulo(ie3 + jl3,nbase(2_8))
                bj3 = bj2*nn3(jl3)


                mat11(i1, i2, i3, p0_1 - il1 + jl1, p0_2 - il2 + jl2, &
      p0_3 - il3 + jl3) = bj3 + mat11(i1, i2, i3, p0_1 - il1 + jl1, &
      p0_2 - il2 + jl2, p0_3 - il3 + jl3)


              end do

            end do

          end do

          do jl1 = 0, p0_1, 1
            j1 = modulo(ie1 + jl1,nbase(0_8))
            bj1 = bi3*(temp12*nn1(jl1))
            do jl2 = 0, p1_2, 1
              j2 = modulo(ie2 + jl2,nbase(4_8))
              bj2 = bj1*((p0_2/(t1_2(j2 + p0_2) - t1_2(j2)))*dd2(jl2))
              do jl3 = 0, p0_3, 1
                j3 = modulo(ie3 + jl3,nbase(2_8))
                bj3 = bj2*nn3(jl3)


                mat12(i1, i2, i3, p0_1 - il1 + jl1, p0_2 - il2 + jl2, &
      p0_3 - il3 + jl3) = bj3 + mat12(i1, i2, i3, p0_1 - il1 + jl1, &
      p0_2 - il2 + jl2, p0_3 - il3 + jl3)


              end do

            end do

          end do

          do jl1 = 0, p0_1, 1
            j1 = modulo(ie1 + jl1,nbase(0_8))
            bj1 = bi3*(temp13*nn1(jl1))
            do jl2 = 0, p0_2, 1
              j2 = modulo(ie2 + jl2,nbase(1_8))
              bj2 = bj1*nn2(jl2)
              do jl3 = 0, p1_3, 1
                j3 = modulo(ie3 + jl3,nbase(5_8))
                bj3 = bj2*((p0_3/(t1_3(j3 + p0_3) - t1_3(j3)))*dd3(jl3))


                mat13(i1, i2, i3, p0_1 - il1 + jl1, p0_2 - il2 + jl2, &
      p0_3 - il3 + jl3) = bj3 + mat13(i1, i2, i3, p0_1 - il1 + jl1, &
      p0_2 - il2 + jl2, p0_3 - il3 + jl3)




              end do

            end do

          end do

        end do

      end do

    end do

    do il1 = 0, p0_1, 1
      i1 = modulo(ie1 + il1,nbase(0_8))
      bi1 = nn1(il1)
      do il2 = 0, p1_2, 1
        i2 = modulo(ie2 + il2,nbase(4_8))
        bi2 = bi1*((p0_2/(t1_2(i2 + p0_2) - t1_2(i2)))*dd2(il2))
        do il3 = 0, p0_3, 1
          i3 = modulo(ie3 + il3,nbase(2_8))
          bi3 = bi2*nn3(il3)
          vec2(i1, i2, i3) = bi3*temp2 + vec2(i1, i2, i3)
          do jl1 = 0, p0_1, 1
            j1 = modulo(ie1 + jl1,nbase(0_8))
            bj1 = bi3*nn1(jl1)


            do jl2 = 0, p1_2, 1
              j2 = modulo(ie2 + jl2,nbase(4_8))
              bj2 = bj1*((p0_2*(temp22/(t1_2(j2 + p0_2) - t1_2(j2))))* &
      dd2(jl2))
              do jl3 = 0, p0_3, 1
                j3 = modulo(ie3 + jl3,nbase(2_8))
                bj3 = bj2*nn3(jl3)


                mat22(i1, i2, i3, p0_1 - il1 + jl1, p0_2 - il2 + jl2, &
      p0_3 - il3 + jl3) = bj3 + mat22(i1, i2, i3, p0_1 - il1 + jl1, &
      p0_2 - il2 + jl2, p0_3 - il3 + jl3)




              end do

            end do

            do jl2 = 0, p0_2, 1
              j2 = modulo(ie2 + jl2,nbase(1_8))
              bj2 = bj1*(temp23*nn2(jl2))
              do jl3 = 0, p1_3, 1
                j3 = modulo(ie3 + jl3,nbase(5_8))
                bj3 = bj2*((p0_3/(t1_3(j3 + p0_3) - t1_3(j3)))*dd3(jl3))


                mat23(i1, i2, i3, p0_1 - il1 + jl1, p0_2 - il2 + jl2, &
      p0_3 - il3 + jl3) = bj3 + mat23(i1, i2, i3, p0_1 - il1 + jl1, &
      p0_2 - il2 + jl2, p0_3 - il3 + jl3)




              end do

            end do

          end do

        end do

      end do

    end do

    do il1 = 0, p0_1, 1
      i1 = modulo(ie1 + il1,nbase(0_8))
      bi1 = nn1(il1)
      do il2 = 0, p0_2, 1
        i2 = modulo(ie2 + il2,nbase(1_8))
        bi2 = bi1*nn2(il2)
        do il3 = 0, p1_3, 1
          i3 = modulo(ie3 + il3,nbase(5_8))
          bi3 = bi2*((p0_3/(t1_3(i3 + p0_3) - t1_3(i3)))*dd3(il3))
          vec3(i1, i2, i3) = bi3*temp3 + vec3(i1, i2, i3)
          do jl1 = 0, p0_1, 1
            j1 = modulo(ie1 + jl1,nbase(0_8))
            bj1 = bi3*(temp33*nn1(jl1))
            do jl2 = 0, p0_2, 1
              j2 = modulo(ie2 + jl2,nbase(1_8))
              bj2 = bj1*nn2(jl2)
              do jl3 = 0, p1_3, 1
                j3 = modulo(ie3 + jl3,nbase(5_8))
                bj3 = bj2*((p0_3/(t1_3(j3 + p0_3) - t1_3(j3)))*dd3(jl3))


                mat33(i1, i2, i3, p0_1 - il1 + jl1, p0_2 - il2 + jl2, &
      p0_3 - il3 + jl3) = bj3 + mat33(i1, i2, i3, p0_1 - il1 + jl1, &
      p0_2 - il2 + jl2, p0_3 - il3 + jl3)






              end do

            end do

          end do

        end do

      end do

    end do

  end do

  !evaluate inverse metric tensor
  !evaluate inverse Jacobian matrix
  !perform matrix-matrix multiplications
  !add contribution to 11 component (DNN DNN), 12 component (DNN NDN) and 13 component (DNN NND)
  !add contribution to 22 component (NDN NDN) and 23 component (NDN NND)
  !add contribution to 33 component (NND NND)
  !$omp end do  
  !$omp end parallel  
  ierr = 0_8
end subroutine
!........................................

end module