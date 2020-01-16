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
pure real(kind=8) function det(A)  result(Dummy_6003)

implicit none
real(kind=8), intent(in)  :: A (0:,0:)
real(kind=8) :: plus  
real(kind=8) :: minus  



plus = (A(1, 1)*A(2, 2))*A(0, 0) + (A(1, 0)*A(2, 1))*A(0, 2) + (A(1, 2)* &
      A(2, 0))*A(0, 1)
minus = (A(0, 2)*A(1, 1))*A(2, 0) + (A(0, 0)*A(1, 2))*A(2, 1) + (A(0, 1) &
      *A(1, 0))*A(2, 2)


Dummy_6003 = -minus + plus
return
end function
!........................................

!........................................
subroutine pusher_step3(particles, p0, spans0, Nbase, b1, b2, b3, u1, u2 &
      , u3, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, mapping, dt, Beq, &
      Ueq)

implicit none
real(kind=8), intent(inout)  :: particles (0:,0:)
integer(kind=4), intent(in)  :: p0 (0:)
integer(kind=4), intent(in)  :: spans0 (0:,0:)
integer(kind=4), intent(in)  :: Nbase (0:)
real(kind=8), intent(in)  :: b1 (0:,0:,0:)
real(kind=8), intent(in)  :: b2 (0:,0:,0:)
real(kind=8), intent(in)  :: b3 (0:,0:,0:)
real(kind=8), intent(in)  :: u1 (0:,0:,0:)
real(kind=8), intent(in)  :: u2 (0:,0:,0:)
real(kind=8), intent(in)  :: u3 (0:,0:,0:)
real(kind=8), intent(in)  :: pp0_1 (0:,0:)
real(kind=8), intent(in)  :: pp0_2 (0:,0:)
real(kind=8), intent(in)  :: pp0_3 (0:,0:)
real(kind=8), intent(in)  :: pp1_1 (0:,0:)
real(kind=8), intent(in)  :: pp1_2 (0:,0:)
real(kind=8), intent(in)  :: pp1_3 (0:,0:)
real(kind=8), intent(in)  :: mapping (0:)
real(kind=8), intent(in)  :: dt 
real(kind=8), intent(in)  :: Beq (0:)
real(kind=8), intent(in)  :: Ueq (0:)
integer(kind=4) :: p0_1  
integer(kind=4) :: p0_2  
integer(kind=4) :: p0_3  
integer(kind=4) :: p1_1  
integer(kind=4) :: p1_2  
integer(kind=4) :: p1_3  
real(kind=8) :: delta1  
real(kind=8) :: delta2  
real(kind=8) :: delta3  
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
real(kind=8) :: pos1  
real(kind=8) :: pos2  
real(kind=8) :: pos3  
integer(kind=4) :: span0_1  
integer(kind=4) :: span0_2  
integer(kind=4) :: span0_3  
integer(kind=4) :: span1_1  
integer(kind=4) :: span1_2  
integer(kind=4) :: span1_3  
integer(kind=4) :: il1  
integer(kind=4) :: i1  
integer(kind=4) :: il2  
integer(kind=4) :: i2  
integer(kind=4) :: il3  
integer(kind=4) :: i3  
integer(kind=4) :: jl1  
integer(kind=4) :: jl2  
integer(kind=4) :: jl3  
real(kind=8) :: N1  
real(kind=8) :: D2  
real(kind=8) :: D3  
real(kind=8) :: D1  
real(kind=8) :: N2  
real(kind=8) :: N3  







p0_1 = p0(0)
p0_2 = p0(1)
p0_3 = p0(2)


p1_1 = p0_1 - 1
p1_2 = p0_2 - 1
p1_3 = p0_3 - 1


delta1 = 1.0d0/Nbase(0)
delta2 = 1.0d0/Nbase(1)
delta3 = 1.0d0/Nbase(2)


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


B(0) = Beq(0)
B(1) = Beq(1)
B(2) = Beq(2)


U(0) = Ueq(0)
U(1) = Ueq(1)
U(2) = Ueq(2)


pos1 = particles(ip, 0)
pos2 = particles(ip, 1)
pos3 = particles(ip, 2)


span0_1 = spans0(ip, 0)
span0_2 = spans0(ip, 1)
span0_3 = spans0(ip, 2)


span1_1 = span0_1 - 1
span1_2 = span0_2 - 1
span1_3 = span0_3 - 1


do il1 = 0, p0_1, 1
i1 = modulo(-il1 + span0_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


do jl1 = 0, p0_1, 1
  do jl2 = 0, p1_2, 1
    do jl3 = 0, p1_3, 1


      N1 = (delta1*(p0_1 - span0_1) + pos1)**jl1*pp0_1(-il1 + p0_1, jl1)
      D2 = (delta2*(p1_2 - span1_2) + pos2)**jl2*pp1_2(-il2 + p1_2, jl2)
      D3 = (delta3*(p1_3 - span1_3) + pos3)**jl3*pp1_3(-il3 + p1_3, jl3)


      B(0) = (N1*(D2*D3))*b1(i1, i2, i3) + B(0)




    end do

  end do

end do

end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


do jl1 = 0, p1_1, 1
  do jl2 = 0, p0_2, 1
    do jl3 = 0, p1_3, 1


      D1 = (delta1*(p1_1 - span1_1) + pos1)**jl1*pp1_1(-il1 + p1_1, jl1)
      N2 = (delta2*(p0_2 - span0_2) + pos2)**jl2*pp0_2(-il2 + p0_2, jl2)
      D3 = (delta3*(p1_3 - span1_3) + pos3)**jl3*pp1_3(-il3 + p1_3, jl3)


      B(1) = (D1*(D3*N2))*b2(i1, i2, i3) + B(1)




    end do

  end do

end do

end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))


do jl1 = 0, p1_1, 1
  do jl2 = 0, p1_2, 1
    do jl3 = 0, p0_3, 1


      D1 = (delta1*(p1_1 - span1_1) + pos1)**jl1*pp1_1(-il1 + p1_1, jl1)
      D2 = (delta2*(p1_2 - span1_2) + pos2)**jl2*pp1_2(-il2 + p1_2, jl2)
      N3 = (delta3*(p0_3 - span0_3) + pos3)**jl3*pp0_3(-il3 + p0_3, jl3)


      B(2) = (D1*(D2*N3))*b3(i1, i2, i3) + B(2)




    end do

  end do

end do

end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))


do jl1 = 0, p1_1, 1
  do jl2 = 0, p0_2, 1
    do jl3 = 0, p0_3, 1


      D1 = (delta1*(p1_1 - span1_1) + pos1)**jl1*pp1_1(-il1 + p1_1, jl1)
      N2 = (delta2*(p0_2 - span0_2) + pos2)**jl2*pp0_2(-il2 + p0_2, jl2)
      N3 = (delta3*(p0_3 - span0_3) + pos3)**jl3*pp0_3(-il3 + p0_3, jl3)


      U(0) = (D1*(N2*N3))*u1(i1, i2, i3) + U(0)




    end do

  end do

end do

end do

end do

end do

do il1 = 0, p0_1, 1
i1 = modulo(-il1 + span0_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))


do jl1 = 0, p0_1, 1
  do jl2 = 0, p1_2, 1
    do jl3 = 0, p0_3, 1


      N1 = (delta1*(p0_1 - span0_1) + pos1)**jl1*pp0_1(-il1 + p0_1, jl1)
      D2 = (delta2*(p1_2 - span1_2) + pos2)**jl2*pp1_2(-il2 + p1_2, jl2)
      N3 = (delta3*(p0_3 - span0_3) + pos3)**jl3*pp0_3(-il3 + p0_3, jl3)


      U(1) = (N1*(D2*N3))*u2(i1, i2, i3) + U(1)




    end do

  end do

end do

end do

end do

end do

do il1 = 0, p0_1, 1
i1 = modulo(-il1 + span0_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


do jl1 = 0, p0_1, 1
  do jl2 = 0, p0_2, 1
    do jl3 = 0, p1_3, 1


      N1 = (delta1*(p0_1 - span0_1) + pos1)**jl1*pp0_1(-il1 + p0_1, jl1)
      N2 = (delta2*(p0_2 - span0_2) + pos2)**jl2*pp0_2(-il2 + p0_2, jl2)
      D3 = (delta3*(p1_3 - span1_3) + pos3)**jl3*pp1_3(-il3 + p1_3, jl3)


      U(2) = (N1*(D3*N2))*u3(i1, i2, i3) + U(2)




    end do

  end do

end do

end do

end do

end do

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

!evaluation of B1 - component (NDD)
!evaluation of B2 - component (DND)
!evaluation of B3 - component (DDN)
!evaluation of U1 - component (DNN)
!evaluation of U2 - component (NDN)
!evaluation of U3 - component (NND)
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
subroutine pusher_step5(particles, p0, spans0, Nbase, b1, b2, b3, pp0_1, &
      pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, mapping, dt, Beq)

implicit none
real(kind=8), intent(inout)  :: particles (0:,0:)
integer(kind=4), intent(in)  :: p0 (0:)
integer(kind=4), intent(in)  :: spans0 (0:,0:)
integer(kind=4), intent(in)  :: Nbase (0:)
real(kind=8), intent(in)  :: b1 (0:,0:,0:)
real(kind=8), intent(in)  :: b2 (0:,0:,0:)
real(kind=8), intent(in)  :: b3 (0:,0:,0:)
real(kind=8), intent(in)  :: pp0_1 (0:,0:)
real(kind=8), intent(in)  :: pp0_2 (0:,0:)
real(kind=8), intent(in)  :: pp0_3 (0:,0:)
real(kind=8), intent(in)  :: pp1_1 (0:,0:)
real(kind=8), intent(in)  :: pp1_2 (0:,0:)
real(kind=8), intent(in)  :: pp1_3 (0:,0:)
real(kind=8), intent(in)  :: mapping (0:)
real(kind=8), intent(in)  :: dt 
real(kind=8), intent(in)  :: Beq (0:)
integer(kind=4) :: p0_1  
integer(kind=4) :: p0_2  
integer(kind=4) :: p0_3  
integer(kind=4) :: p1_1  
integer(kind=4) :: p1_2  
integer(kind=4) :: p1_3  
real(kind=8) :: delta1  
real(kind=8) :: delta2  
real(kind=8) :: delta3  
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
real(kind=8) :: pos1  
real(kind=8) :: pos2  
real(kind=8) :: pos3  
integer(kind=4) :: span0_1  
integer(kind=4) :: span0_2  
integer(kind=4) :: span0_3  
integer(kind=4) :: span1_1  
integer(kind=4) :: span1_2  
integer(kind=4) :: span1_3  
real(kind=8) :: det_lhs  
real(kind=8) :: det_lhs1  
real(kind=8) :: det_lhs2  
real(kind=8) :: det_lhs3  
integer(kind=4) :: il1  
integer(kind=4) :: i1  
integer(kind=4) :: il2  
integer(kind=4) :: i2  
integer(kind=4) :: il3  
integer(kind=4) :: i3  
integer(kind=4) :: jl1  
integer(kind=4) :: jl2  
integer(kind=4) :: jl3  
real(kind=8) :: N1  
real(kind=8) :: D2  
real(kind=8) :: D3  
real(kind=8) :: D1  
real(kind=8) :: N2  
real(kind=8) :: N3  







p0_1 = p0(0)
p0_2 = p0(1)
p0_3 = p0(2)


p1_1 = p0_1 - 1
p1_2 = p0_2 - 1
p1_3 = p0_3 - 1


delta1 = 1.0d0/Nbase(0)
delta2 = 1.0d0/Nbase(1)
delta3 = 1.0d0/Nbase(2)


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
B(0) = Beq(0)
B(1) = Beq(1)
B(2) = Beq(2)


pos1 = particles(ip, 0)
pos2 = particles(ip, 1)
pos3 = particles(ip, 2)


span0_1 = spans0(ip, 0)
span0_2 = spans0(ip, 1)
span0_3 = spans0(ip, 2)


span1_1 = span0_1 - 1
span1_2 = span0_2 - 1
span1_3 = span0_3 - 1


do il1 = 0, p0_1, 1
i1 = modulo(-il1 + span0_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


do jl1 = 0, p0_1, 1
  do jl2 = 0, p1_2, 1
    do jl3 = 0, p1_3, 1


      N1 = (delta1*(p0_1 - span0_1) + pos1)**jl1*pp0_1(-il1 + p0_1, jl1)
      D2 = (delta2*(p1_2 - span1_2) + pos2)**jl2*pp1_2(-il2 + p1_2, jl2)
      D3 = (delta3*(p1_3 - span1_3) + pos3)**jl3*pp1_3(-il3 + p1_3, jl3)


      B(0) = (N1*(D2*D3))*b1(i1, i2, i3) + B(0)




    end do

  end do

end do

end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


do jl1 = 0, p1_1, 1
  do jl2 = 0, p0_2, 1
    do jl3 = 0, p1_3, 1


      D1 = (delta1*(p1_1 - span1_1) + pos1)**jl1*pp1_1(-il1 + p1_1, jl1)
      N2 = (delta2*(p0_2 - span0_2) + pos2)**jl2*pp0_2(-il2 + p0_2, jl2)
      D3 = (delta3*(p1_3 - span1_3) + pos3)**jl3*pp1_3(-il3 + p1_3, jl3)


      B(1) = (D1*(D3*N2))*b2(i1, i2, i3) + B(1)




    end do

  end do

end do

end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))


do jl1 = 0, p1_1, 1
  do jl2 = 0, p1_2, 1
    do jl3 = 0, p0_3, 1


      D1 = (delta1*(p1_1 - span1_1) + pos1)**jl1*pp1_1(-il1 + p1_1, jl1)
      D2 = (delta2*(p1_2 - span1_2) + pos2)**jl2*pp1_2(-il2 + p1_2, jl2)
      N3 = (delta3*(p0_3 - span0_3) + pos3)**jl3*pp0_3(-il3 + p0_3, jl3)


      B(2) = (D1*(D2*N3))*b3(i1, i2, i3) + B(2)




    end do

  end do

end do

end do

end do

end do

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

!... field evaluation (wave + background)
!evaluation of B1 - component (NDD)
!evaluation of B2 - component (DND)
!evaluation of B3 - component (DDN)
!...
ierr = 0
end subroutine
!........................................

end module