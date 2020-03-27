module STRUPHY_fields

implicit none




contains

!........................................
subroutine evaluate_1form(particles_pos, p0, spans0, Nbase, Np, u1, u2, &
      u3, Ueq, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, U_part)

  implicit none
  real(kind=8), intent(in)  :: particles_pos (0:,0:)
  integer(kind=4), intent(in)  :: p0 (0:)
  integer(kind=4), intent(in)  :: spans0 (0:,0:)
  integer(kind=4), intent(in)  :: Nbase (0:)
  integer(kind=4), intent(in)  :: Np 
  real(kind=8), intent(in)  :: u1 (0:,0:,0:)
  real(kind=8), intent(in)  :: u2 (0:,0:,0:)
  real(kind=8), intent(in)  :: u3 (0:,0:,0:)
  real(kind=8), intent(in)  :: Ueq (0:)
  real(kind=8), intent(in)  :: pp0_1 (0:,0:)
  real(kind=8), intent(in)  :: pp0_2 (0:,0:)
  real(kind=8), intent(in)  :: pp0_3 (0:,0:)
  real(kind=8), intent(in)  :: pp1_1 (0:,0:)
  real(kind=8), intent(in)  :: pp1_2 (0:,0:)
  real(kind=8), intent(in)  :: pp1_3 (0:,0:)
  real(kind=8), intent(inout)  :: U_part (0:,0:)
  integer(kind=4) :: p0_1  
  integer(kind=4) :: p0_2  
  integer(kind=4) :: p0_3  
  integer(kind=4) :: p1_1  
  integer(kind=4) :: p1_2  
  integer(kind=4) :: p1_3  
  real(kind=8) :: delta1  
  real(kind=8) :: delta2  
  real(kind=8) :: delta3  
  integer(kind=4) :: ierr  
  integer(kind=4) :: ip  
  integer(kind=4) :: span0_1  
  integer(kind=4) :: span0_2  
  integer(kind=4) :: span0_3  
  integer(kind=4) :: span1_1  
  integer(kind=4) :: span1_2  
  integer(kind=4) :: span1_3  
  real(kind=8) :: posloc_1  
  real(kind=8) :: posloc_2  
  real(kind=8) :: posloc_3  
  integer(kind=4) :: jl3  
  real(kind=8) :: pow3  
  integer(kind=4) :: jl2  
  real(kind=8) :: pow2  
  integer(kind=4) :: jl1  
  real(kind=8) :: pow1  
  integer(kind=4) :: il3  
  integer(kind=4) :: i3  
  real(kind=8) :: N3  
  integer(kind=4) :: il2  
  integer(kind=4) :: i2  
  real(kind=8) :: N2  
  integer(kind=4) :: il1  
  integer(kind=4) :: i1  
  real(kind=8) :: D1  
  real(kind=8) :: D2  
  real(kind=8) :: N1  
  real(kind=8) :: D3  



  p0_1 = p0(0)
  p0_2 = p0(1)
  p0_3 = p0(2)


  p1_1 = p0_1 - 1
  p1_2 = p0_2 - 1
  p1_3 = p0_3 - 1


  delta1 = 1.0d0/Nbase(0)
  delta2 = 1.0d0/Nbase(1)
  delta3 = 1.0d0/Nbase(2)




  !$omp parallel
  !$omp do private(ip, span0_1, span0_2, span0_3, span1_1, span1_2, span1_3, pos&
      !$omp &loc_1, posloc_2, posloc_3, jl3, jl2, jl1, il3, il2, il1, pow1, pow2, pow&
      !$omp &3, i3, i2, i1, N3, N2, N1, D3, D2, D1)
  do ip = 0, Np - 1, 1


    U_part(ip, 0) = Ueq(0)
    U_part(ip, 1) = Ueq(1)
    U_part(ip, 2) = Ueq(2)


    span0_1 = spans0(ip, 0)
    span0_2 = spans0(ip, 1)
    span0_3 = spans0(ip, 2)


    span1_1 = span0_1 - 1
    span1_2 = span0_2 - 1
    span1_3 = span0_3 - 1


    posloc_1 = delta1*(p0_1 - span0_1) + particles_pos(ip, 0)
    posloc_2 = delta2*(p0_2 - span0_2) + particles_pos(ip, 1)
    posloc_3 = delta3*(p0_3 - span0_3) + particles_pos(ip, 2)


    do jl3 = 0, p0_3, 1
      pow3 = posloc_3**jl3
      do jl2 = 0, p0_2, 1
        pow2 = posloc_2**jl2
        do jl1 = 0, p1_1, 1
          pow1 = posloc_1**jl1


          do il3 = 0, p0_3, 1
            i3 = modulo(-il3 + span0_3,Nbase(2))
            N3 = pow3*pp0_3(-il3 + p0_3, jl3)
            do il2 = 0, p0_2, 1
              i2 = modulo(-il2 + span0_2,Nbase(1))
              N2 = pow2*pp0_2(-il2 + p0_2, jl2)
              do il1 = 0, p1_1, 1
                i1 = modulo(-il1 + span1_1,Nbase(0))
                D1 = pow1*pp1_1(-il1 + p1_1, jl1)


                U_part(ip, 0) = (D1*(N2*N3))*u1(i1, i2, i3) + U_part(ip, &
      0)




              end do

            end do

          end do

        end do

      end do

    end do

    do jl3 = 0, p0_3, 1
      pow3 = posloc_3**jl3
      do jl2 = 0, p1_2, 1
        pow2 = posloc_2**jl2
        do jl1 = 0, p0_1, 1
          pow1 = posloc_1**jl1


          do il3 = 0, p0_3, 1
            i3 = modulo(-il3 + span0_3,Nbase(2))
            N3 = pow3*pp0_3(-il3 + p0_3, jl3)
            do il2 = 0, p1_2, 1
              i2 = modulo(-il2 + span1_2,Nbase(1))
              D2 = pow2*pp1_2(-il2 + p1_2, jl2)
              do il1 = 0, p0_1, 1
                i1 = modulo(-il1 + span0_1,Nbase(0))
                N1 = pow1*pp0_1(-il1 + p0_1, jl1)


                U_part(ip, 1) = (N1*(D2*N3))*u2(i1, i2, i3) + U_part(ip, &
      1)




              end do

            end do

          end do

        end do

      end do

    end do

    do jl3 = 0, p1_3, 1
      pow3 = posloc_3**jl3
      do jl2 = 0, p0_2, 1
        pow2 = posloc_2**jl2
        do jl1 = 0, p0_1, 1
          pow1 = posloc_1**jl1


          do il3 = 0, p1_3, 1
            i3 = modulo(-il3 + span1_3,Nbase(2))
            D3 = pow3*pp1_3(-il3 + p1_3, jl3)
            do il2 = 0, p0_2, 1
              i2 = modulo(-il2 + span0_2,Nbase(1))
              N2 = pow2*pp0_2(-il2 + p0_2, jl2)
              do il1 = 0, p0_1, 1
                i1 = modulo(-il1 + span0_1,Nbase(0))
                N1 = pow1*pp0_1(-il1 + p0_1, jl1)


                U_part(ip, 2) = (N1*(D3*N2))*u3(i1, i2, i3) + U_part(ip, &
      2)




              end do

            end do

          end do

        end do

      end do

    end do

  end do

  !evaluation of 1 - component (DNN)
  !evaluation of 2 - component (NDN)
  !evaluation of 3 - component (NND)
  !$omp end do  
  !$omp end parallel  
  ierr = 0
end subroutine
!........................................

!........................................
subroutine evaluate_2form(particles_pos, p0, spans0, Nbase, Np, b1, b2, &
      b3, Beq, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, B_part)

  implicit none
  real(kind=8), intent(in)  :: particles_pos (0:,0:)
  integer(kind=4), intent(in)  :: p0 (0:)
  integer(kind=4), intent(in)  :: spans0 (0:,0:)
  integer(kind=4), intent(in)  :: Nbase (0:)
  integer(kind=4), intent(in)  :: Np 
  real(kind=8), intent(in)  :: b1 (0:,0:,0:)
  real(kind=8), intent(in)  :: b2 (0:,0:,0:)
  real(kind=8), intent(in)  :: b3 (0:,0:,0:)
  real(kind=8), intent(in)  :: Beq (0:)
  real(kind=8), intent(in)  :: pp0_1 (0:,0:)
  real(kind=8), intent(in)  :: pp0_2 (0:,0:)
  real(kind=8), intent(in)  :: pp0_3 (0:,0:)
  real(kind=8), intent(in)  :: pp1_1 (0:,0:)
  real(kind=8), intent(in)  :: pp1_2 (0:,0:)
  real(kind=8), intent(in)  :: pp1_3 (0:,0:)
  real(kind=8), intent(inout)  :: B_part (0:,0:)
  integer(kind=4) :: p0_1  
  integer(kind=4) :: p0_2  
  integer(kind=4) :: p0_3  
  integer(kind=4) :: p1_1  
  integer(kind=4) :: p1_2  
  integer(kind=4) :: p1_3  
  real(kind=8) :: delta1  
  real(kind=8) :: delta2  
  real(kind=8) :: delta3  
  integer(kind=4) :: ierr  
  integer(kind=4) :: ip  
  integer(kind=4) :: span0_1  
  integer(kind=4) :: span0_2  
  integer(kind=4) :: span0_3  
  integer(kind=4) :: span1_1  
  integer(kind=4) :: span1_2  
  integer(kind=4) :: span1_3  
  real(kind=8) :: posloc_1  
  real(kind=8) :: posloc_2  
  real(kind=8) :: posloc_3  
  integer(kind=4) :: jl3  
  real(kind=8) :: pow3  
  integer(kind=4) :: jl2  
  real(kind=8) :: pow2  
  integer(kind=4) :: jl1  
  real(kind=8) :: pow1  
  integer(kind=4) :: il3  
  integer(kind=4) :: i3  
  real(kind=8) :: D3  
  integer(kind=4) :: il2  
  integer(kind=4) :: i2  
  real(kind=8) :: D2  
  integer(kind=4) :: il1  
  integer(kind=4) :: i1  
  real(kind=8) :: N1  
  real(kind=8) :: N2  
  real(kind=8) :: D1  
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


  !$omp parallel
  !$omp do private(ip, span0_1, span0_2, span0_3, span1_1, span1_2, span1_3, pos&
      !$omp &loc_1, posloc_2, posloc_3, jl3, jl2, jl1, il3, il2, il1, pow1, pow2, pow&
      !$omp &3, i3, i2, i1, N3, N2, N1, D3, D2, D1)
  do ip = 0, Np - 1, 1


    B_part(ip, 0) = Beq(0)
    B_part(ip, 1) = Beq(1)
    B_part(ip, 2) = Beq(2)


    span0_1 = spans0(ip, 0)
    span0_2 = spans0(ip, 1)
    span0_3 = spans0(ip, 2)


    span1_1 = span0_1 - 1
    span1_2 = span0_2 - 1
    span1_3 = span0_3 - 1


    posloc_1 = delta1*(p0_1 - span0_1) + particles_pos(ip, 0)
    posloc_2 = delta2*(p0_2 - span0_2) + particles_pos(ip, 1)
    posloc_3 = delta3*(p0_3 - span0_3) + particles_pos(ip, 2)


    do jl3 = 0, p1_3, 1
      pow3 = posloc_3**jl3
      do jl2 = 0, p1_2, 1
        pow2 = posloc_2**jl2
        do jl1 = 0, p0_1, 1
          pow1 = posloc_1**jl1


          do il3 = 0, p1_3, 1
            i3 = modulo(-il3 + span1_3,Nbase(2))
            D3 = pow3*pp1_3(-il3 + p1_3, jl3)
            do il2 = 0, p1_2, 1
              i2 = modulo(-il2 + span1_2,Nbase(1))
              D2 = pow2*pp1_2(-il2 + p1_2, jl2)
              do il1 = 0, p0_1, 1
                i1 = modulo(-il1 + span0_1,Nbase(0))
                N1 = pow1*pp0_1(-il1 + p0_1, jl1)


                B_part(ip, 0) = (N1*(D2*D3))*b1(i1, i2, i3) + B_part(ip, &
      0)




              end do

            end do

          end do

        end do

      end do

    end do

    do jl3 = 0, p1_3, 1
      pow3 = posloc_3**jl3
      do jl2 = 0, p0_2, 1
        pow2 = posloc_2**jl2
        do jl1 = 0, p1_1, 1
          pow1 = posloc_1**jl1


          do il3 = 0, p1_3, 1
            i3 = modulo(-il3 + span1_3,Nbase(2))
            D3 = pow3*pp1_3(-il3 + p1_3, jl3)
            do il2 = 0, p0_2, 1
              i2 = modulo(-il2 + span0_2,Nbase(1))
              N2 = pow2*pp0_2(-il2 + p0_2, jl2)
              do il1 = 0, p1_1, 1
                i1 = modulo(-il1 + span1_1,Nbase(0))
                D1 = pow1*pp1_1(-il1 + p1_1, jl1)


                B_part(ip, 1) = (D1*(D3*N2))*b2(i1, i2, i3) + B_part(ip, &
      1)




              end do

            end do

          end do

        end do

      end do

    end do

    do jl3 = 0, p0_3, 1
      pow3 = posloc_3**jl3
      do jl2 = 0, p1_2, 1
        pow2 = posloc_2**jl2
        do jl1 = 0, p1_1, 1
          pow1 = posloc_1**jl1


          do il3 = 0, p0_3, 1
            i3 = modulo(-il3 + span0_3,Nbase(2))
            N3 = pow3*pp0_3(-il3 + p0_3, jl3)
            do il2 = 0, p1_2, 1
              i2 = modulo(-il2 + span1_2,Nbase(1))
              D2 = pow2*pp1_2(-il2 + p1_2, jl2)
              do il1 = 0, p1_1, 1
                i1 = modulo(-il1 + span1_1,Nbase(0))
                D1 = pow1*pp1_1(-il1 + p1_1, jl1)


                B_part(ip, 2) = (D1*(D2*N3))*b3(i1, i2, i3) + B_part(ip, &
      2)


              end do

            end do

          end do

        end do

      end do

    end do

  end do

  !evaluation of 1 - component (NDD)
  !evaluation of 2 - component (DND)
  !evaluation of 3 - component (DDN)
  !$omp end do  
  !$omp end parallel  
  ierr = 0
end subroutine
!........................................

end module