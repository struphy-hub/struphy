module ECHO_pusher

implicit none




contains

!........................................
pure subroutine mapping_matrices(q, kind, params, output, A) 

implicit none
real(kind=8), intent(in)  :: q (0:)
integer(kind=4), intent(in)  :: kind 
real(kind=8), intent(in)  :: params (0:)
integer(kind=4), intent(in)  :: output 
real(kind=8), intent(inout)  :: A (0:,0:)
real(kind=8) :: Lx  
real(kind=8) :: Ly  
real(kind=8) :: Lz  



A(:, :) = 0.0d0


!kind = 1 : slab geometry (params = [Lx, Ly, Lz], output = [DF, DF_inv, G, Ginv])
if (kind == 1 ) then


  Lx = params(0)
  Ly = params(1)
  Lz = params(2)


  if (output == 1 ) then


    A(0, 0) = Lx
    A(1, 1) = Ly
    A(2, 2) = Lz
  else if (output == 2 ) then


    A(0, 0) = 1.0d0/Lx
    A(1, 1) = 1.0d0/Ly
    A(2, 2) = 1.0d0/Lz
  else if (output == 3 ) then


    A(0, 0) = Lx**2
    A(1, 1) = Ly**2
    A(2, 2) = Lz**2
  else if (output == 4 ) then


    A(0, 0) = 1.0d0/(Lx**2)
    A(1, 1) = 1.0d0/(Ly**2)
    A(2, 2) = 1.0d0/(Lz**2)
  end if
end if
end subroutine
!........................................

!........................................
pure subroutine matrix_vector(A, b, c) 

implicit none
real(kind=8), intent(in)  :: A (0:,0:)
real(kind=8), intent(in)  :: b (0:)
real(kind=8), intent(inout)  :: c (0:)
integer(kind=4) :: i  
integer(kind=4) :: j  



c(:) = 0.0d0


do i = 0, 2, 1
do j = 0, 2, 1
  c(i) = A(i, j)*b(j) + c(i)




end do

end do

end subroutine
!........................................

!........................................
pure subroutine matrix_matrix(A, B, C) 

implicit none
real(kind=8), intent(in)  :: A (0:,0:)
real(kind=8), intent(in)  :: B (0:,0:)
real(kind=8), intent(inout)  :: C (0:,0:)
integer(kind=4) :: i  
integer(kind=4) :: j  
integer(kind=4) :: k  



C(:, :) = 0.0d0


do i = 0, 2, 1
do j = 0, 2, 1
do k = 0, 2, 1
  C(i, j) = A(i, k)*B(k, j) + C(i, j)




end do

end do

end do

end subroutine
!........................................

!........................................
pure subroutine transpose(A, B) 

implicit none
real(kind=8), intent(in)  :: A (0:,0:)
real(kind=8), intent(inout)  :: B (0:,0:)
integer(kind=4) :: i  
integer(kind=4) :: j  



B(:, :) = 0.0d0


do i = 0, 2, 1
do j = 0, 2, 1
B(i, j) = A(j, i)




end do

end do

end subroutine
!........................................

!........................................
pure real(kind=8) function det(A)  result(Dummy_8645)

implicit none
real(kind=8), intent(in)  :: A (0:,0:)
real(kind=8) :: plus  
real(kind=8) :: minus  



plus = (A(1, 1)*A(2, 2))*A(0, 0) + (A(1, 0)*A(2, 1))*A(0, 2) + (A(1, 2)* &
      A(2, 0))*A(0, 1)
minus = (A(0, 2)*A(1, 1))*A(2, 0) + (A(0, 0)*A(1, 2))*A(2, 1) + (A(0, 1) &
      *A(1, 0))*A(2, 2)


Dummy_8645 = -minus + plus
return
end function
!........................................

!........................................
subroutine pusher_step3(particles, mapping, dt, B_part, U_part) 

implicit none
real(kind=8), intent(inout)  :: particles (0:,0:)
real(kind=8), intent(in)  :: mapping (0:)
real(kind=8), intent(in)  :: dt 
real(kind=8), intent(in)  :: B_part (0:,0:)
real(kind=8), intent(in)  :: U_part (0:,0:)
real(kind=8), allocatable :: B (:) 
real(kind=8), allocatable :: U (:) 
real(kind=8), allocatable :: temp_mat1 (:,:) 
real(kind=8), allocatable :: temp_mat2 (:,:) 
real(kind=8), allocatable :: B_prod (:,:) 
real(kind=8), allocatable :: Ginv (:,:) 
real(kind=8), allocatable :: DFinv (:,:) 
real(kind=8), allocatable :: DFinv_T (:,:) 
real(kind=8), allocatable :: temp_vec (:) 
real(kind=8), allocatable :: v (:) 
real(kind=8), allocatable :: q (:) 
integer(kind=4) :: np  
integer(kind=4) :: ierr  
integer(kind=4) :: ip  







allocate(B(0:2))
allocate(U(0:2))


allocate(temp_mat1(0:2, 0:2))
allocate(temp_mat2(0:2, 0:2))


allocate(B_prod(0:2, 0:2))
B_prod = 0.0


allocate(Ginv(0:2, 0:2))


allocate(DFinv(0:2, 0:2))
allocate(DFinv_T(0:2, 0:2))


allocate(temp_vec(0:2))


allocate(v(0:2))
allocate(q(0:2))


np = size(particles(:, 0),1)


do ip = 0, np - 1, 1


B(0) = B_part(ip, 0)
B(1) = B_part(ip, 1)
B(2) = B_part(ip, 2)


U(0) = U_part(ip, 0)
U(1) = U_part(ip, 1)
U(2) = U_part(ip, 2)


B_prod(0, 1) = -B(2)
B_prod(0, 2) = B(1)


B_prod(1, 0) = B(2)
B_prod(1, 2) = -B(0)


B_prod(2, 0) = -B(1)
B_prod(2, 1) = B(0)


v(:) = particles(ip, 3:5)
q(:) = particles(ip, 0:2)


call mapping_matrices(q, 1, mapping, 2, DFinv)
call transpose(DFinv, DFinv_T)
call mapping_matrices(q, 1, mapping, 4, Ginv)
call matrix_matrix(DFinv_T, B_prod, temp_mat1)
call matrix_matrix(temp_mat1, Ginv, temp_mat2)
call matrix_vector(temp_mat2, U, temp_vec)


particles(ip, 3) = dt*(temp_vec(0)/Real(2, 8)) + particles(ip, 3)
particles(ip, 4) = dt*(temp_vec(1)/Real(2, 8)) + particles(ip, 4)
particles(ip, 5) = dt*(temp_vec(2)/Real(2, 8)) + particles(ip, 5)


end do

ierr = 0
end subroutine
!........................................

!........................................
subroutine pusher_step4(particles, mapping, dt) 

implicit none
real(kind=8), intent(inout)  :: particles (0:,0:)
real(kind=8), intent(in)  :: mapping (0:)
real(kind=8), intent(in)  :: dt 
real(kind=8), allocatable :: DFinv (:,:) 
real(kind=8), allocatable :: v (:) 
real(kind=8), allocatable :: q (:) 
real(kind=8), allocatable :: temp (:) 
integer(kind=4) :: np  
integer(kind=4) :: ierr  
integer(kind=4) :: ip  






allocate(DFinv(0:2, 0:2))
allocate(v(0:2))
allocate(q(0:2))
allocate(temp(0:2))


np = size(particles(:, 0),1)


do ip = 0, np - 1, 1


v(:) = particles(ip, 3:5)
q(:) = particles(ip, 0:2)


call mapping_matrices(q, 1, mapping, 2, DFinv)
call matrix_vector(DFinv, v, temp)


particles(ip, 0) = modulo(dt*temp(0) + q(0),mapping(0))
particles(ip, 1) = modulo(dt*temp(1) + q(1),mapping(1))
particles(ip, 2) = modulo(dt*temp(2) + q(2),mapping(2))


end do

ierr = 0
end subroutine
!........................................

!........................................
subroutine pusher_step5(particles, mapping, dt, B_part) 

implicit none
real(kind=8), intent(inout)  :: particles (0:,0:)
real(kind=8), intent(in)  :: mapping (0:)
real(kind=8), intent(in)  :: dt 
real(kind=8), intent(in)  :: B_part (0:,0:)
real(kind=8), allocatable :: B (:) 
real(kind=8), allocatable :: temp_mat1 (:,:) 
real(kind=8), allocatable :: temp_mat2 (:,:) 
real(kind=8), allocatable :: rhs (:) 
real(kind=8), allocatable :: B_prod (:,:) 
real(kind=8), allocatable :: DFinv (:,:) 
real(kind=8), allocatable :: DFinv_T (:,:) 
real(kind=8), allocatable :: I (:,:) 
real(kind=8), allocatable :: lhs (:,:) 
real(kind=8), allocatable :: lhs1 (:,:) 
real(kind=8), allocatable :: lhs2 (:,:) 
real(kind=8), allocatable :: lhs3 (:,:) 
real(kind=8), allocatable :: v (:) 
real(kind=8), allocatable :: q (:) 
integer(kind=4) :: np  
integer(kind=4) :: ierr  
integer(kind=4) :: ip  
real(kind=8) :: det_lhs  
real(kind=8) :: det_lhs1  
real(kind=8) :: det_lhs2  
real(kind=8) :: det_lhs3  







allocate(B(0:2))


allocate(temp_mat1(0:2, 0:2))
allocate(temp_mat2(0:2, 0:2))


allocate(rhs(0:2))


allocate(B_prod(0:2, 0:2))
B_prod = 0.0


allocate(DFinv(0:2, 0:2))
allocate(DFinv_T(0:2, 0:2))


allocate(I(0:2, 0:2))
I = 0.0
I(0, 0) = 1.0d0
I(1, 1) = 1.0d0
I(2, 2) = 1.0d0


allocate(lhs(0:2, 0:2))


allocate(lhs1(0:2, 0:2))
allocate(lhs2(0:2, 0:2))
allocate(lhs3(0:2, 0:2))


allocate(v(0:2))
allocate(q(0:2))


np = size(particles(:, 0),1)


do ip = 0, np - 1, 1


B(0) = B_part(ip, 0)
B(1) = B_part(ip, 1)
B(2) = B_part(ip, 2)


B_prod(0, 1) = -B(2)
B_prod(0, 2) = B(1)


B_prod(1, 0) = B(2)
B_prod(1, 2) = -B(0)


B_prod(2, 0) = -B(1)
B_prod(2, 1) = B(0)


v(:) = particles(ip, 3:5)
q(:) = particles(ip, 0:2)


call mapping_matrices(q, 1, mapping, 2, DFinv)
call matrix_matrix(B_prod, DFinv, temp_mat1)
call transpose(DFinv, DFinv_T)
call matrix_matrix(DFinv_T, temp_mat1, temp_mat2)
call matrix_vector(I + (-dt)*(temp_mat2/Real(2, 8)), v, rhs)


lhs(:, :) = I + dt*(temp_mat2/Real(2, 8))


det_lhs = det(lhs)


lhs1(:, 0) = rhs
lhs1(:, 1) = lhs(:, 1)
lhs1(:, 2) = lhs(:, 2)


lhs2(:, 0) = lhs(:, 0)
lhs2(:, 1) = rhs
lhs2(:, 2) = lhs(:, 2)


lhs3(:, 0) = lhs(:, 0)
lhs3(:, 1) = lhs(:, 1)
lhs3(:, 2) = rhs


det_lhs1 = det(lhs1)
det_lhs2 = det(lhs2)
det_lhs3 = det(lhs3)


particles(ip, 3) = det_lhs1/det_lhs
particles(ip, 4) = det_lhs2/det_lhs
particles(ip, 5) = det_lhs3/det_lhs




end do

!...
ierr = 0
end subroutine
!........................................

end module