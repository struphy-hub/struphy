module ECHO_accumulation

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
pure real(kind=8) function det(A)  result(Dummy_1058)

implicit none
real(kind=8), intent(in)  :: A (0:,0:)
real(kind=8) :: plus  
real(kind=8) :: minus  



plus = (A(1, 1)*A(2, 2))*A(0, 0) + (A(1, 0)*A(2, 1))*A(0, 2) + (A(1, 2)* &
      A(2, 0))*A(0, 1)
minus = (A(0, 2)*A(1, 1))*A(2, 0) + (A(0, 0)*A(1, 2))*A(2, 1) + (A(0, 1) &
      *A(1, 0))*A(2, 2)


Dummy_1058 = -minus + plus
return
end function
!........................................

!........................................
subroutine basis_funs(knots, degree, x, span, left, right, values) 

implicit none
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree 
real(kind=8), intent(in)  :: x 
integer(kind=4), intent(in)  :: span 
real(kind=8), intent(inout)  :: left (0:)
real(kind=8), intent(inout)  :: right (0:)
real(kind=8), intent(inout)  :: values (0:)
integer(kind=4) :: j  
real(kind=8) :: saved  
integer(kind=4) :: r  
real(kind=8) :: temp  



left(:) = 0.0d0
right(:) = 0.0d0


values(0) = 1.0d0


do j = 0, degree - 1, 1
left(j) = x - knots(-j + span)
right(j) = -x + knots(span + j + 1)
saved = 0.0d0
do r = 0, j, 1
temp = values(r)/(left(j - r) + right(r))
values(r) = saved + temp*right(r)
saved = temp*left(j - r)


end do

values(j + 1) = saved






end do

end subroutine
!........................................

!........................................
subroutine matrix_step1(particles, p0, spans0, Nbase, b1, b2, b3, T1, T2 &
      , T3, tt1, tt2, tt3, mapping, dt, Beq, mat12, mat13, mat23)

implicit none
real(kind=8), intent(in)  :: particles (0:,0:)
integer(kind=4), intent(in)  :: p0 (0:)
integer(kind=4), intent(in)  :: spans0 (0:,0:)
integer(kind=4), intent(in)  :: Nbase (0:)
real(kind=8), intent(in)  :: b1 (0:,0:,0:)
real(kind=8), intent(in)  :: b2 (0:,0:,0:)
real(kind=8), intent(in)  :: b3 (0:,0:,0:)
real(kind=8), intent(in)  :: T1 (0:)
real(kind=8), intent(in)  :: T2 (0:)
real(kind=8), intent(in)  :: T3 (0:)
real(kind=8), intent(in)  :: tt1 (0:)
real(kind=8), intent(in)  :: tt2 (0:)
real(kind=8), intent(in)  :: tt3 (0:)
real(kind=8), intent(in)  :: mapping (0:)
real(kind=8), intent(in)  :: dt 
real(kind=8), intent(in)  :: Beq (0:)
real(kind=8), intent(inout)  :: mat12 (0:,0:,0:,0:,0:,0:)
real(kind=8), intent(inout)  :: mat13 (0:,0:,0:,0:,0:,0:)
real(kind=8), intent(inout)  :: mat23 (0:,0:,0:,0:,0:,0:)
integer(kind=4) :: p0_1  
integer(kind=4) :: p0_2  
integer(kind=4) :: p0_3  
integer(kind=4) :: p1_1  
integer(kind=4) :: p1_2  
integer(kind=4) :: p1_3  
real(kind=8) :: delta1  
real(kind=8) :: delta2  
real(kind=8) :: delta3  
real(kind=8), allocatable :: Nl1 (:) 
real(kind=8), allocatable :: Nr1 (:) 
real(kind=8), allocatable :: N1 (:) 
real(kind=8), allocatable :: Nl2 (:) 
real(kind=8), allocatable :: Nr2 (:) 
real(kind=8), allocatable :: N2 (:) 
real(kind=8), allocatable :: Nl3 (:) 
real(kind=8), allocatable :: Nr3 (:) 
real(kind=8), allocatable :: N3 (:) 
real(kind=8), allocatable :: Dl1 (:) 
real(kind=8), allocatable :: Dr1 (:) 
real(kind=8), allocatable :: D1 (:) 
real(kind=8), allocatable :: Dl2 (:) 
real(kind=8), allocatable :: Dr2 (:) 
real(kind=8), allocatable :: D2 (:) 
real(kind=8), allocatable :: Dl3 (:) 
real(kind=8), allocatable :: Dr3 (:) 
real(kind=8), allocatable :: D3 (:) 
real(kind=8), allocatable :: B (:) 
real(kind=8), allocatable :: temp_mat1 (:,:) 
real(kind=8), allocatable :: temp_mat2 (:,:) 
real(kind=8), allocatable :: B_prod (:,:) 
real(kind=8), allocatable :: Ginv (:,:) 
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
real(kind=8) :: w  
real(kind=8) :: temp12  
real(kind=8) :: temp13  
real(kind=8) :: temp23  
integer(kind=4) :: il1  
integer(kind=4) :: i1  
integer(kind=4) :: il2  
integer(kind=4) :: i2  
integer(kind=4) :: il3  
integer(kind=4) :: i3  
integer(kind=4) :: jl3  
integer(kind=4) :: j3  
integer(kind=4) :: jl2  
integer(kind=4) :: j2  
integer(kind=4) :: jl1  
integer(kind=4) :: j1  







p0_1 = p0(0)
p0_2 = p0(1)
p0_3 = p0(2)


p1_1 = p0_1 - 1
p1_2 = p0_2 - 1
p1_3 = p0_3 - 1


delta1 = 1.0d0/Nbase(0)
delta2 = 1.0d0/Nbase(1)
delta3 = 1.0d0/Nbase(2)


allocate(Nl1(0:p0_1 - 1))
allocate(Nr1(0:p0_1 - 1))
allocate(N1(0:p0_1))
N1 = 0.0


allocate(Nl2(0:p0_2 - 1))
allocate(Nr2(0:p0_2 - 1))
allocate(N2(0:p0_2))
N2 = 0.0


allocate(Nl3(0:p0_3 - 1))
allocate(Nr3(0:p0_3 - 1))
allocate(N3(0:p0_3))
N3 = 0.0


allocate(Dl1(0:p1_1 - 1))
allocate(Dr1(0:p1_1 - 1))
allocate(D1(0:p1_1))
D1 = 0.0


allocate(Dl2(0:p1_2 - 1))
allocate(Dr2(0:p1_2 - 1))
allocate(D2(0:p1_2))
D2 = 0.0


allocate(Dl3(0:p1_3 - 1))
allocate(Dr3(0:p1_3 - 1))
allocate(D3(0:p1_3))
D3 = 0.0


allocate(B(0:2))


allocate(temp_mat1(0:2, 0:2))
allocate(temp_mat2(0:2, 0:2))


allocate(B_prod(0:2, 0:2))
B_prod = 0.0


allocate(Ginv(0:2, 0:2))


allocate(q(0:2))


np = size(particles(:, 0),1)


mat12(:, :, :, :, :, :) = 0.0d0
mat13(:, :, :, :, :, :) = 0.0d0
mat23(:, :, :, :, :, :) = 0.0d0


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


call basis_funs(T1, p0_1, pos1, span0_1, Nl1, Nr1, N1)
call basis_funs(T2, p0_2, pos2, span0_2, Nl2, Nr2, N2)
call basis_funs(T3, p0_3, pos3, span0_3, Nl3, Nr3, N3)


call basis_funs(tt1, p1_1, pos1, span1_1, Dl1, Dr1, D1)
call basis_funs(tt2, p1_2, pos2, span1_2, Dl2, Dr2, D2)
call basis_funs(tt3, p1_3, pos3, span1_3, Dl3, Dr3, D3)


D1(:) = D1/delta1
D2(:) = D2/delta2
D3(:) = D3/delta3




do il1 = 0, p0_1, 1
i1 = modulo(-il1 + span0_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


B(0) = ((D2(-il2 + p1_2)*D3(-il3 + p1_3))*N1(-il1 + p0_1))*b1(i1, i2, i3 &
      ) + B(0)




end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


B(1) = ((D3(-il3 + p1_3)*N2(-il2 + p0_2))*D1(-il1 + p1_1))*b2(i1, i2, i3 &
      ) + B(1)




end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))


B(2) = ((D2(-il2 + p1_2)*N3(-il3 + p0_3))*D1(-il1 + p1_1))*b3(i1, i2, i3 &
      ) + B(2)




end do

end do

end do

B_prod(0, 1) = -B(2)
B_prod(0, 2) = B(1)


B_prod(1, 0) = B(2)
B_prod(1, 2) = -B(0)


B_prod(2, 0) = -B(1)
B_prod(2, 1) = B(0)


q(:) = particles(ip, 0:2)
w = particles(ip, 6)


call mapping_matrices(q, 1, mapping, 4, Ginv)
call matrix_matrix(Ginv, B_prod, temp_mat1)
call matrix_matrix(temp_mat1, Ginv, temp_mat2)


temp12 = w*temp_mat2(0, 1)
temp13 = w*temp_mat2(0, 2)
temp23 = w*temp_mat2(1, 2)






do jl3 = 0, p0_3, 1
j3 = modulo(-jl3 + span0_3,Nbase(2))
do jl2 = 0, p1_2, 1
j2 = modulo(-jl2 + span1_2,Nbase(1))
do jl1 = 0, p0_1, 1
j1 = modulo(-jl1 + span0_1,Nbase(0))
do il3 = 0, p0_3, 1
  i3 = modulo(-il3 + span0_3,Nbase(2))
  do il2 = 0, p0_2, 1
    i2 = modulo(-il2 + span0_2,Nbase(1))
    do il1 = 0, p1_1, 1
      i1 = modulo(-il1 + span1_1,Nbase(0))


      mat12(i1, i2, i3, j1, j2, j3) = temp12*(((((D2(-jl2 + p1_2)*N3( &
      -jl3 + p0_3))*N1(-jl1 + p0_1))*N3(-il3 + p0_3))*N2(-il2 + p0_2))* &
      D1(-il1 + p1_1)) + mat12(i1, i2, i3, j1, j2, j3)




    end do

  end do

end do

end do

end do

end do

do jl3 = 0, p1_3, 1
j3 = modulo(-jl3 + span1_3,Nbase(2))
do jl2 = 0, p0_2, 1
j2 = modulo(-jl2 + span0_2,Nbase(1))
do jl1 = 0, p0_1, 1
j1 = modulo(-jl1 + span0_1,Nbase(0))
do il3 = 0, p0_3, 1
  i3 = modulo(-il3 + span0_3,Nbase(2))
  do il2 = 0, p0_2, 1
    i2 = modulo(-il2 + span0_2,Nbase(1))
    do il1 = 0, p1_1, 1
      i1 = modulo(-il1 + span1_1,Nbase(0))


      mat13(i1, i2, i3, j1, j2, j3) = temp13*(((((D3(-jl3 + p1_3)*N2( &
      -jl2 + p0_2))*N1(-jl1 + p0_1))*N3(-il3 + p0_3))*N2(-il2 + p0_2))* &
      D1(-il1 + p1_1)) + mat13(i1, i2, i3, j1, j2, j3)




    end do

  end do

end do

end do

end do

end do

do jl3 = 0, p1_3, 1
j3 = modulo(-jl3 + span1_3,Nbase(2))
do jl2 = 0, p0_2, 1
j2 = modulo(-jl2 + span0_2,Nbase(1))
do jl1 = 0, p0_1, 1
j1 = modulo(-jl1 + span0_1,Nbase(0))
do il3 = 0, p0_3, 1
  i3 = modulo(-il3 + span0_3,Nbase(2))
  do il2 = 0, p1_2, 1
    i2 = modulo(-il2 + span1_2,Nbase(1))
    do il1 = 0, p0_1, 1
      i1 = modulo(-il1 + span0_1,Nbase(0))


      mat23(i1, i2, i3, j1, j2, j3) = temp23*(((((D3(-jl3 + p1_3)*N2( &
      -jl2 + p0_2))*N1(-jl1 + p0_1))*N3(-il3 + p0_3))*D2(-il2 + p1_2))* &
      N1(-il1 + p0_1)) + mat23(i1, i2, i3, j1, j2, j3)




    end do

  end do

end do

end do

end do

end do

end do

!evaluation of B1 - component (NDD)
!evaluation of B2 - component (DND)
!evaluation of B3 - component (DDN)
!add contribution to 12 component (DNN NDN)
!add contribution to 13 component (DNN NND)
!add contribution to 23 component (NDN NND)   
ierr = 0
end subroutine
!........................................

!........................................
subroutine matrix_step3(particles, p0, spans0, Nbase, b1, b2, b3, T1, T2 &
      , T3, tt1, tt2, tt3, mapping, dt, Beq, mat11, mat12, mat13, mat22 &
      , mat23, mat33)

implicit none
real(kind=8), intent(in)  :: particles (0:,0:)
integer(kind=4), intent(in)  :: p0 (0:)
integer(kind=4), intent(in)  :: spans0 (0:,0:)
integer(kind=4), intent(in)  :: Nbase (0:)
real(kind=8), intent(in)  :: b1 (0:,0:,0:)
real(kind=8), intent(in)  :: b2 (0:,0:,0:)
real(kind=8), intent(in)  :: b3 (0:,0:,0:)
real(kind=8), intent(in)  :: T1 (0:)
real(kind=8), intent(in)  :: T2 (0:)
real(kind=8), intent(in)  :: T3 (0:)
real(kind=8), intent(in)  :: tt1 (0:)
real(kind=8), intent(in)  :: tt2 (0:)
real(kind=8), intent(in)  :: tt3 (0:)
real(kind=8), intent(in)  :: mapping (0:)
real(kind=8), intent(in)  :: dt 
real(kind=8), intent(in)  :: Beq (0:)
real(kind=8), intent(inout)  :: mat11 (0:,0:,0:,0:,0:,0:)
real(kind=8), intent(inout)  :: mat12 (0:,0:,0:,0:,0:,0:)
real(kind=8), intent(inout)  :: mat13 (0:,0:,0:,0:,0:,0:)
real(kind=8), intent(inout)  :: mat22 (0:,0:,0:,0:,0:,0:)
real(kind=8), intent(inout)  :: mat23 (0:,0:,0:,0:,0:,0:)
real(kind=8), intent(inout)  :: mat33 (0:,0:,0:,0:,0:,0:)
integer(kind=4) :: p0_1  
integer(kind=4) :: p0_2  
integer(kind=4) :: p0_3  
integer(kind=4) :: p1_1  
integer(kind=4) :: p1_2  
integer(kind=4) :: p1_3  
real(kind=8) :: delta1  
real(kind=8) :: delta2  
real(kind=8) :: delta3  
real(kind=8), allocatable :: Nl1 (:) 
real(kind=8), allocatable :: Nr1 (:) 
real(kind=8), allocatable :: N1 (:) 
real(kind=8), allocatable :: Nl2 (:) 
real(kind=8), allocatable :: Nr2 (:) 
real(kind=8), allocatable :: N2 (:) 
real(kind=8), allocatable :: Nl3 (:) 
real(kind=8), allocatable :: Nr3 (:) 
real(kind=8), allocatable :: N3 (:) 
real(kind=8), allocatable :: Dl1 (:) 
real(kind=8), allocatable :: Dr1 (:) 
real(kind=8), allocatable :: D1 (:) 
real(kind=8), allocatable :: Dl2 (:) 
real(kind=8), allocatable :: Dr2 (:) 
real(kind=8), allocatable :: D2 (:) 
real(kind=8), allocatable :: Dl3 (:) 
real(kind=8), allocatable :: Dr3 (:) 
real(kind=8), allocatable :: D3 (:) 
real(kind=8), allocatable :: B (:) 
real(kind=8), allocatable :: temp_mat1 (:,:) 
real(kind=8), allocatable :: temp_mat2 (:,:) 
real(kind=8), allocatable :: B_prod (:,:) 
real(kind=8), allocatable :: B_prod_T (:,:) 
real(kind=8), allocatable :: Ginv (:,:) 
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
real(kind=8) :: w  
real(kind=8) :: temp11  
real(kind=8) :: temp12  
real(kind=8) :: temp13  
real(kind=8) :: temp22  
real(kind=8) :: temp23  
real(kind=8) :: temp33  
integer(kind=4) :: il1  
integer(kind=4) :: i1  
integer(kind=4) :: il2  
integer(kind=4) :: i2  
integer(kind=4) :: il3  
integer(kind=4) :: i3  
integer(kind=4) :: jl1  
integer(kind=4) :: j1  
integer(kind=4) :: jl2  
integer(kind=4) :: j2  
integer(kind=4) :: jl3  
integer(kind=4) :: j3  







p0_1 = p0(0)
p0_2 = p0(1)
p0_3 = p0(2)


p1_1 = p0_1 - 1
p1_2 = p0_2 - 1
p1_3 = p0_3 - 1


delta1 = 1.0d0/Nbase(0)
delta2 = 1.0d0/Nbase(1)
delta3 = 1.0d0/Nbase(2)


allocate(Nl1(0:p0_1 - 1))
allocate(Nr1(0:p0_1 - 1))
allocate(N1(0:p0_1))
N1 = 0.0


allocate(Nl2(0:p0_2 - 1))
allocate(Nr2(0:p0_2 - 1))
allocate(N2(0:p0_2))
N2 = 0.0


allocate(Nl3(0:p0_3 - 1))
allocate(Nr3(0:p0_3 - 1))
allocate(N3(0:p0_3))
N3 = 0.0


allocate(Dl1(0:p1_1 - 1))
allocate(Dr1(0:p1_1 - 1))
allocate(D1(0:p1_1))
D1 = 0.0


allocate(Dl2(0:p1_2 - 1))
allocate(Dr2(0:p1_2 - 1))
allocate(D2(0:p1_2))
D2 = 0.0


allocate(Dl3(0:p1_3 - 1))
allocate(Dr3(0:p1_3 - 1))
allocate(D3(0:p1_3))
D3 = 0.0


allocate(B(0:2))


allocate(temp_mat1(0:2, 0:2))
allocate(temp_mat2(0:2, 0:2))


allocate(B_prod(0:2, 0:2))
B_prod = 0.0
allocate(B_prod_T(0:2, 0:2))
B_prod_T = 0.0


allocate(Ginv(0:2, 0:2))


allocate(q(0:2))


np = size(particles(:, 0),1)


mat11(:, :, :, :, :, :) = 0.0d0
mat12(:, :, :, :, :, :) = 0.0d0
mat13(:, :, :, :, :, :) = 0.0d0
mat22(:, :, :, :, :, :) = 0.0d0
mat23(:, :, :, :, :, :) = 0.0d0
mat33(:, :, :, :, :, :) = 0.0d0


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


call basis_funs(T1, p0_1, pos1, span0_1, Nl1, Nr1, N1)
call basis_funs(T2, p0_2, pos2, span0_2, Nl2, Nr2, N2)
call basis_funs(T3, p0_3, pos3, span0_3, Nl3, Nr3, N3)


call basis_funs(tt1, p1_1, pos1, span1_1, Dl1, Dr1, D1)
call basis_funs(tt2, p1_2, pos2, span1_2, Dl2, Dr2, D2)
call basis_funs(tt3, p1_3, pos3, span1_3, Dl3, Dr3, D3)


D1(:) = D1/delta1
D2(:) = D2/delta2
D3(:) = D3/delta3




do il1 = 0, p0_1, 1
i1 = modulo(-il1 + span0_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


B(0) = ((D2(-il2 + p1_2)*D3(-il3 + p1_3))*N1(-il1 + p0_1))*b1(i1, i2, i3 &
      ) + B(0)




end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


B(1) = ((D3(-il3 + p1_3)*N2(-il2 + p0_2))*D1(-il1 + p1_1))*b2(i1, i2, i3 &
      ) + B(1)




end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))


B(2) = ((D2(-il2 + p1_2)*N3(-il3 + p0_3))*D1(-il1 + p1_1))*b3(i1, i2, i3 &
      ) + B(2)




end do

end do

end do

B_prod(0, 1) = -B(2)
B_prod(0, 2) = B(1)


B_prod(1, 0) = B(2)
B_prod(1, 2) = -B(0)


B_prod(2, 0) = -B(1)
B_prod(2, 1) = B(0)


q(:) = particles(ip, 0:2)
w = particles(ip, 6)


call mapping_matrices(q, 1, mapping, 4, Ginv)
call matrix_matrix(Ginv, B_prod, temp_mat1)
call matrix_matrix(temp_mat1, Ginv, temp_mat2)
call transpose(B_prod, B_prod_T)
call matrix_matrix(temp_mat2, B_prod_T, temp_mat1)
call matrix_matrix(temp_mat1, Ginv, temp_mat2)


temp11 = w*temp_mat2(0, 0)
temp12 = w*temp_mat2(0, 1)
temp13 = w*temp_mat2(0, 2)
temp22 = w*temp_mat2(1, 1)
temp23 = w*temp_mat2(1, 2)
temp33 = w*temp_mat2(2, 2)


do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))
do jl1 = 0, p1_1, 1
  j1 = modulo(-jl1 + span1_1,Nbase(0))
  do jl2 = 0, p0_2, 1
    j2 = modulo(-jl2 + span0_2,Nbase(1))
    do jl3 = 0, p0_3, 1
      j3 = modulo(-jl3 + span0_3,Nbase(2))


      mat11(i1, i2, i3, j1, j2, j3) = temp11*(((((N2(-jl2 + p0_2)*N3( &
      -jl3 + p0_3))*D1(-jl1 + p1_1))*N3(-il3 + p0_3))*N2(-il2 + p0_2))* &
      D1(-il1 + p1_1)) + mat11(i1, i2, i3, j1, j2, j3)




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
do jl1 = 0, p0_1, 1
  j1 = modulo(-jl1 + span0_1,Nbase(0))
  do jl2 = 0, p1_2, 1
    j2 = modulo(-jl2 + span1_2,Nbase(1))
    do jl3 = 0, p0_3, 1
      j3 = modulo(-jl3 + span0_3,Nbase(2))


      mat12(i1, i2, i3, j1, j2, j3) = temp12*(((((D2(-jl2 + p1_2)*N3( &
      -jl3 + p0_3))*N1(-jl1 + p0_1))*N3(-il3 + p0_3))*N2(-il2 + p0_2))* &
      D1(-il1 + p1_1)) + mat12(i1, i2, i3, j1, j2, j3)




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
do jl1 = 0, p0_1, 1
  j1 = modulo(-jl1 + span0_1,Nbase(0))
  do jl2 = 0, p0_2, 1
    j2 = modulo(-jl2 + span0_2,Nbase(1))
    do jl3 = 0, p1_3, 1
      j3 = modulo(-jl3 + span1_3,Nbase(2))


      mat13(i1, i2, i3, j1, j2, j3) = temp13*(((((D3(-jl3 + p1_3)*N2( &
      -jl2 + p0_2))*N1(-jl1 + p0_1))*N3(-il3 + p0_3))*N2(-il2 + p0_2))* &
      D1(-il1 + p1_1)) + mat13(i1, i2, i3, j1, j2, j3)




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
  j1 = modulo(-jl1 + span0_1,Nbase(0))
  do jl2 = 0, p1_2, 1
    j2 = modulo(-jl2 + span1_2,Nbase(1))
    do jl3 = 0, p0_3, 1
      j3 = modulo(-jl3 + span0_3,Nbase(2))


      mat22(i1, i2, i3, j1, j2, j3) = temp22*(((((D2(-jl2 + p1_2)*N3( &
      -jl3 + p0_3))*N1(-jl1 + p0_1))*N3(-il3 + p0_3))*D2(-il2 + p1_2))* &
      N1(-il1 + p0_1)) + mat22(i1, i2, i3, j1, j2, j3)




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
  j1 = modulo(-jl1 + span0_1,Nbase(0))
  do jl2 = 0, p0_2, 1
    j2 = modulo(-jl2 + span0_2,Nbase(1))
    do jl3 = 0, p1_3, 1
      j3 = modulo(-jl3 + span1_3,Nbase(2))


      mat23(i1, i2, i3, j1, j2, j3) = temp23*(((((D3(-jl3 + p1_3)*N2( &
      -jl2 + p0_2))*N1(-jl1 + p0_1))*N3(-il3 + p0_3))*D2(-il2 + p1_2))* &
      N1(-il1 + p0_1)) + mat23(i1, i2, i3, j1, j2, j3)




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
  j1 = modulo(-jl1 + span0_1,Nbase(0))
  do jl2 = 0, p0_2, 1
    j2 = modulo(-jl2 + span0_2,Nbase(1))
    do jl3 = 0, p1_3, 1
      j3 = modulo(-jl3 + span1_3,Nbase(2))


      mat33(i1, i2, i3, j1, j2, j3) = temp33*(((((D3(-jl3 + p1_3)*N2( &
      -jl2 + p0_2))*N1(-jl1 + p0_1))*D3(-il3 + p1_3))*N2(-il2 + p0_2))* &
      N1(-il1 + p0_1)) + mat33(i1, i2, i3, j1, j2, j3)




    end do

  end do

end do

end do

end do

end do

end do

!evaluation of B1 - component (NDD)
!evaluation of B2 - component (DND)
!evaluation of B3 - component (DDN)
!add contribution to 11 component (DNN DNN)
!add contribution to 12 component (DNN NDN)
!add contribution to 13 component (DNN NND)
!add contribution to 22 component (NDN NDN)
!add contribution to 23 component (NDN NND)
!add contribution to 33 component (NND NND)
ierr = 0
end subroutine
!........................................

!........................................
subroutine vector_step3(particles, p0, spans0, Nbase, b1, b2, b3, T1, T2 &
      , T3, tt1, tt2, tt3, mapping, Beq, vec1, vec2, vec3)

implicit none
real(kind=8), intent(in)  :: particles (0:,0:)
integer(kind=4), intent(in)  :: p0 (0:)
integer(kind=4), intent(in)  :: spans0 (0:,0:)
integer(kind=4), intent(in)  :: Nbase (0:)
real(kind=8), intent(in)  :: b1 (0:,0:,0:)
real(kind=8), intent(in)  :: b2 (0:,0:,0:)
real(kind=8), intent(in)  :: b3 (0:,0:,0:)
real(kind=8), intent(in)  :: T1 (0:)
real(kind=8), intent(in)  :: T2 (0:)
real(kind=8), intent(in)  :: T3 (0:)
real(kind=8), intent(in)  :: tt1 (0:)
real(kind=8), intent(in)  :: tt2 (0:)
real(kind=8), intent(in)  :: tt3 (0:)
real(kind=8), intent(in)  :: mapping (0:)
real(kind=8), intent(in)  :: Beq (0:)
real(kind=8), intent(inout)  :: vec1 (0:,0:,0:)
real(kind=8), intent(inout)  :: vec2 (0:,0:,0:)
real(kind=8), intent(inout)  :: vec3 (0:,0:,0:)
integer(kind=4) :: p0_1  
integer(kind=4) :: p0_2  
integer(kind=4) :: p0_3  
integer(kind=4) :: p1_1  
integer(kind=4) :: p1_2  
integer(kind=4) :: p1_3  
real(kind=8) :: delta1  
real(kind=8) :: delta2  
real(kind=8) :: delta3  
real(kind=8), allocatable :: Nl1 (:) 
real(kind=8), allocatable :: Nr1 (:) 
real(kind=8), allocatable :: N1 (:) 
real(kind=8), allocatable :: Nl2 (:) 
real(kind=8), allocatable :: Nr2 (:) 
real(kind=8), allocatable :: N2 (:) 
real(kind=8), allocatable :: Nl3 (:) 
real(kind=8), allocatable :: Nr3 (:) 
real(kind=8), allocatable :: N3 (:) 
real(kind=8), allocatable :: Dl1 (:) 
real(kind=8), allocatable :: Dr1 (:) 
real(kind=8), allocatable :: D1 (:) 
real(kind=8), allocatable :: Dl2 (:) 
real(kind=8), allocatable :: Dr2 (:) 
real(kind=8), allocatable :: D2 (:) 
real(kind=8), allocatable :: Dl3 (:) 
real(kind=8), allocatable :: Dr3 (:) 
real(kind=8), allocatable :: D3 (:) 
real(kind=8), allocatable :: B (:) 
real(kind=8), allocatable :: temp_mat1 (:,:) 
real(kind=8), allocatable :: temp_mat2 (:,:) 
real(kind=8), allocatable :: B_prod (:,:) 
real(kind=8), allocatable :: Ginv (:,:) 
real(kind=8), allocatable :: DFinv (:,:) 
real(kind=8), allocatable :: temp_vec (:) 
real(kind=8), allocatable :: q (:) 
real(kind=8), allocatable :: v (:) 
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
real(kind=8) :: w  
real(kind=8) :: temp1  
real(kind=8) :: temp2  
real(kind=8) :: temp3  
integer(kind=4) :: il1  
integer(kind=4) :: i1  
integer(kind=4) :: il2  
integer(kind=4) :: i2  
integer(kind=4) :: il3  
integer(kind=4) :: i3  







p0_1 = p0(0)
p0_2 = p0(1)
p0_3 = p0(2)


p1_1 = p0_1 - 1
p1_2 = p0_2 - 1
p1_3 = p0_3 - 1


delta1 = 1.0d0/Nbase(0)
delta2 = 1.0d0/Nbase(1)
delta3 = 1.0d0/Nbase(2)


allocate(Nl1(0:p0_1 - 1))
allocate(Nr1(0:p0_1 - 1))
allocate(N1(0:p0_1))
N1 = 0.0


allocate(Nl2(0:p0_2 - 1))
allocate(Nr2(0:p0_2 - 1))
allocate(N2(0:p0_2))
N2 = 0.0


allocate(Nl3(0:p0_3 - 1))
allocate(Nr3(0:p0_3 - 1))
allocate(N3(0:p0_3))
N3 = 0.0


allocate(Dl1(0:p1_1 - 1))
allocate(Dr1(0:p1_1 - 1))
allocate(D1(0:p1_1))
D1 = 0.0


allocate(Dl2(0:p1_2 - 1))
allocate(Dr2(0:p1_2 - 1))
allocate(D2(0:p1_2))
D2 = 0.0


allocate(Dl3(0:p1_3 - 1))
allocate(Dr3(0:p1_3 - 1))
allocate(D3(0:p1_3))
D3 = 0.0


allocate(B(0:2))


allocate(temp_mat1(0:2, 0:2))
allocate(temp_mat2(0:2, 0:2))


allocate(B_prod(0:2, 0:2))
B_prod = 0.0


allocate(Ginv(0:2, 0:2))
allocate(DFinv(0:2, 0:2))


allocate(temp_vec(0:2))


allocate(q(0:2))
allocate(v(0:2))


np = size(particles(:, 0),1)


vec1(:, :, :) = 0.0d0
vec2(:, :, :) = 0.0d0
vec3(:, :, :) = 0.0d0


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


call basis_funs(T1, p0_1, pos1, span0_1, Nl1, Nr1, N1)
call basis_funs(T2, p0_2, pos2, span0_2, Nl2, Nr2, N2)
call basis_funs(T3, p0_3, pos3, span0_3, Nl3, Nr3, N3)


call basis_funs(tt1, p1_1, pos1, span1_1, Dl1, Dr1, D1)
call basis_funs(tt2, p1_2, pos2, span1_2, Dl2, Dr2, D2)
call basis_funs(tt3, p1_3, pos3, span1_3, Dl3, Dr3, D3)


D1(:) = D1/delta1
D2(:) = D2/delta2
D3(:) = D3/delta3




do il1 = 0, p0_1, 1
i1 = modulo(-il1 + span0_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


B(0) = ((D2(-il2 + p1_2)*D3(-il3 + p1_3))*N1(-il1 + p0_1))*b1(i1, i2, i3 &
      ) + B(0)




end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


B(1) = ((D3(-il3 + p1_3)*N2(-il2 + p0_2))*D1(-il1 + p1_1))*b2(i1, i2, i3 &
      ) + B(1)




end do

end do

end do

do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))


B(2) = ((D2(-il2 + p1_2)*N3(-il3 + p0_3))*D1(-il1 + p1_1))*b3(i1, i2, i3 &
      ) + B(2)




end do

end do

end do

B_prod(0, 1) = -B(2)
B_prod(0, 2) = B(1)


B_prod(1, 0) = B(2)
B_prod(1, 2) = -B(0)


B_prod(2, 0) = -B(1)
B_prod(2, 1) = B(0)


q(:) = particles(ip, 0:2)
v(:) = particles(ip, 3:5)


call mapping_matrices(q, 1, mapping, 4, Ginv)
call mapping_matrices(q, 1, mapping, 2, DFinv)


call matrix_matrix(Ginv, B_prod, temp_mat1)
call matrix_matrix(temp_mat1, DFinv, temp_mat2)
call matrix_vector(temp_mat2, v, temp_vec)


w = particles(ip, 6)


temp1 = w*temp_vec(0)
temp2 = w*temp_vec(1)
temp3 = w*temp_vec(2)


do il1 = 0, p1_1, 1
i1 = modulo(-il1 + span1_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))


vec1(i1, i2, i3) = temp1*((N2(-il2 + p0_2)*N3(-il3 + p0_3))*D1(-il1 + &
      p1_1)) + vec1(i1, i2, i3)


end do

end do

end do

do il1 = 0, p0_1, 1
i1 = modulo(-il1 + span0_1,Nbase(0))
do il2 = 0, p1_2, 1
i2 = modulo(-il2 + span1_2,Nbase(1))
do il3 = 0, p0_3, 1
i3 = modulo(-il3 + span0_3,Nbase(2))


vec2(i1, i2, i3) = temp2*((D2(-il2 + p1_2)*N3(-il3 + p0_3))*N1(-il1 + &
      p0_1)) + vec2(i1, i2, i3)


end do

end do

end do

do il1 = 0, p0_1, 1
i1 = modulo(-il1 + span0_1,Nbase(0))
do il2 = 0, p0_2, 1
i2 = modulo(-il2 + span0_2,Nbase(1))
do il3 = 0, p1_3, 1
i3 = modulo(-il3 + span1_3,Nbase(2))


vec3(i1, i2, i3) = temp3*((D3(-il3 + p1_3)*N2(-il2 + p0_2))*N1(-il1 + &
      p0_1)) + vec3(i1, i2, i3)






end do

end do

end do

end do

!evaluation of B1 - component (NDD)
!evaluation of B2 - component (DND)
!evaluation of B3 - component (DDN)
!add contribution to vector 1 (DNN)
!add contribution to vector 2 (NDN)
!add contribution to vector 3 (NND)
ierr = 0
end subroutine
!........................................

end module