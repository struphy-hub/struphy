module STRUPHY_fields

use interface, only: inter_zyksv4_b3_eq => b3_eq
use interface, only: inter_zyksv4_u3_eq => u3_eq
use interface, only: inter_zyksv4_b1_eq => b1_eq
use interface, only: inter_zyksv4_b2_eq => b2_eq
use interface, only: inter_zyksv4_u1_eq => u1_eq
use interface, only: inter_zyksv4_u2_eq => u2_eq
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
subroutine evaluate_1form(particles_pos, t0_1, t0_2, t0_3, t1_1, t1_2, &
      t1_3, p0, nel, nbase, np, u1, u2, u3, pp0_1, pp0_2, pp0_3, pp1_1, &
      pp1_2, pp1_3, u_part, kind_map, params_map)

  implicit none
  real(kind=8), intent(in)  :: particles_pos (0:,0:)
  real(kind=8), intent(in)  :: t0_1 (0:)
  real(kind=8), intent(in)  :: t0_2 (0:)
  real(kind=8), intent(in)  :: t0_3 (0:)
  real(kind=8), intent(in)  :: t1_1 (0:)
  real(kind=8), intent(in)  :: t1_2 (0:)
  real(kind=8), intent(in)  :: t1_3 (0:)
  integer(kind=8), intent(in)  :: p0 (0:)
  integer(kind=8), intent(in)  :: nel (0:)
  integer(kind=8), intent(in)  :: nbase (0:,0:)
  integer(kind=8), value  :: np
  real(kind=8), intent(in)  :: u1 (0:,0:,0:)
  real(kind=8), intent(in)  :: u2 (0:,0:,0:)
  real(kind=8), intent(in)  :: u3 (0:,0:,0:)
  real(kind=8), intent(in)  :: pp0_1 (0:,0:)
  real(kind=8), intent(in)  :: pp0_2 (0:,0:)
  real(kind=8), intent(in)  :: pp0_3 (0:,0:)
  real(kind=8), intent(in)  :: pp1_1 (0:,0:)
  real(kind=8), intent(in)  :: pp1_2 (0:,0:)
  real(kind=8), intent(in)  :: pp1_3 (0:,0:)
  real(kind=8), intent(inout)  :: u_part (0:,0:)
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  integer(kind=8)  :: p0_1  
  integer(kind=8)  :: p0_2  
  integer(kind=8)  :: p0_3  
  integer(kind=8)  :: p1_1  
  integer(kind=8)  :: p1_2  
  integer(kind=8)  :: p1_3  
  real(kind=8)  :: delta1  
  real(kind=8)  :: delta2  
  real(kind=8)  :: delta3  
  real(kind=8)  :: bl1  
  real(kind=8)  :: bl2  
  real(kind=8)  :: bl3  
  real(kind=8)  :: br1  
  real(kind=8)  :: br2  
  real(kind=8)  :: br3  
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
  integer(kind=8)  :: ierr  
  integer(kind=8)  :: ip  
  integer(kind=8)  :: span0_1  
  integer(kind=8)  :: span0_2  
  integer(kind=8)  :: span0_3  
  integer(kind=8)  :: span1_1  
  integer(kind=8)  :: span1_2  
  integer(kind=8)  :: span1_3  
  real(kind=8)  :: posloc_1  
  real(kind=8)  :: posloc_2  
  real(kind=8)  :: posloc_3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: i1  
  real(kind=8)  :: d1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: i2  
  real(kind=8)  :: n2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: i3  
  real(kind=8)  :: n3  
  real(kind=8)  :: n1  
  real(kind=8)  :: d2  
  real(kind=8)  :: d3  
  integer(kind=8)  :: jl3  
  real(kind=8)  :: pow3  
  integer(kind=8)  :: jl2  
  real(kind=8)  :: pow2  
  integer(kind=8)  :: jl1  
  real(kind=8)  :: pow1  







  p0_1 = p0(0_8)
  p0_2 = p0(1_8)
  p0_3 = p0(2_8)


  p1_1 = p0_1 - 1_8
  p1_2 = p0_2 - 1_8
  p1_3 = p0_3 - 1_8


  delta1 = 1.0d0/Real(nel(0_8), 8)
  delta2 = 1.0d0/Real(nel(1_8), 8)
  delta3 = 1.0d0/Real(nel(2_8), 8)


  bl1 = delta1*p1_1
  bl2 = delta2*p1_2
  bl3 = delta3*p1_3


  br1 = delta1*(-p1_1) + 1_8
  br2 = delta2*(-p1_2) + 1_8
  br3 = delta3*(-p1_3) + 1_8


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




  !$omp parallel
  !$omp do private(ip, span0_1, span0_2, span0_3, span1_1, span1_2, span1_3, pos&
      !$omp &loc_1, posloc_2, posloc_3, nl1, nr1, nn1, nl2, nr2, nn2, nl3, nr3, nn3, &
      !$omp &dl1, dr1, dd1, dl2, dr2, dd2, dl3, dr3, dd3, jl3, jl2, jl1, il3, il2, il&
      !$omp &1, pow1, pow2, pow3, i3, i2, i1, n3, n2, n1, d3, d2, d1)
  do ip = 0, np - 1_8, 1


    u_part(ip, 0_8) = inter_zyksv4_u1_eq(particles_pos(ip, 0_8), &
      particles_pos(ip, 1_8), particles_pos(ip, 2_8), kind_map, &
      params_map)
    u_part(ip, 1_8) = inter_zyksv4_u2_eq(particles_pos(ip, 0_8), &
      particles_pos(ip, 1_8), particles_pos(ip, 2_8), kind_map, &
      params_map)
    u_part(ip, 2_8) = inter_zyksv4_u3_eq(particles_pos(ip, 0_8), &
      particles_pos(ip, 1_8), particles_pos(ip, 2_8), kind_map, &
      params_map)


    span0_1 = p0_1 + Int(nel(0_8)*particles_pos(ip, 0_8), 8)
    span0_2 = p0_2 + Int(nel(1_8)*particles_pos(ip, 1_8), 8)
    span0_3 = p0_3 + Int(nel(2_8)*particles_pos(ip, 2_8), 8)


    span1_1 = span0_1 - 1_8
    span1_2 = span0_2 - 1_8
    span1_3 = span0_3 - 1_8


    posloc_1 = delta1*(p0_1 - span0_1) + particles_pos(ip, 0_8)
    posloc_2 = delta2*(p0_2 - span0_2) + particles_pos(ip, 1_8)
    posloc_3 = delta3*(p0_3 - span0_3) + particles_pos(ip, 2_8)




    if (particles_pos(ip, 0_8) > br1 .or. particles_pos(ip, 1_8) > br2 &
      .or. particles_pos(ip, 2_8) > br3 .or. particles_pos(ip, 0_8) < &
      bl1 .or. particles_pos(ip, 1_8) < bl2 .or. particles_pos(ip, 2_8 &
      ) < bl3) then


      call basis_funs(t0_1, p0_1, particles_pos(ip, 0_8), span0_1, nl1, &
      nr1, nn1)
      call basis_funs(t0_2, p0_2, particles_pos(ip, 1_8), span0_2, nl2, &
      nr2, nn2)
      call basis_funs(t0_3, p0_3, particles_pos(ip, 2_8), span0_3, nl3, &
      nr3, nn3)


      call basis_funs(t1_1, p1_1, particles_pos(ip, 0_8), span1_1, dl1, &
      dr1, dd1)
      call basis_funs(t1_2, p1_2, particles_pos(ip, 1_8), span1_2, dl2, &
      dr2, dd2)
      call basis_funs(t1_3, p1_3, particles_pos(ip, 2_8), span1_3, dl3, &
      dr3, dd3)


      !evaluation of 1 - component (DNN)
      do il1 = 0, p1_1, 1
        i1 = modulo(-il1 + span1_1,nbase(0_8, 0_8))
        d1 = (p0_1/(t1_1(i1 + p0_1) - t1_1(i1)))*dd1(-il1 + p1_1)
        do il2 = 0, p0_2, 1
          i2 = modulo(-il2 + span0_2,nbase(0_8, 1_8))
          n2 = nn2(-il2 + p0_2)
          do il3 = 0, p0_3, 1
            i3 = modulo(-il3 + span0_3,nbase(0_8, 2_8))
            n3 = nn3(-il3 + p0_3)


            u_part(ip, 0_8) = (d1*(n2*n3))*u1(i1, i2, i3) + u_part(ip, &
      0_8)


          end do

        end do

      end do

      !evaluation of 2 - component (NDN)
      do il1 = 0, p0_1, 1
        i1 = modulo(-il1 + span0_1,nbase(1_8, 0_8))
        n1 = nn1(-il1 + p0_1)
        do il2 = 0, p1_2, 1
          i2 = modulo(-il2 + span1_2,nbase(1_8, 1_8))
          d2 = (p0_2/(t1_2(i2 + p0_2) - t1_2(i2)))*dd2(-il2 + p1_2)
          do il3 = 0, p0_3, 1
            i3 = modulo(-il3 + span0_3,nbase(1_8, 2_8))
            n3 = nn3(-il3 + p0_3)


            u_part(ip, 1_8) = (n1*(d2*n3))*u2(i1, i2, i3) + u_part(ip, &
      1_8)


          end do

        end do

      end do

      !evaluation of 3 - component (NND)
      do il1 = 0, p0_1, 1
        i1 = modulo(-il1 + span0_1,nbase(2_8, 0_8))
        n1 = nn1(-il1 + p0_1)
        do il2 = 0, p0_2, 1
          i2 = modulo(-il2 + span0_2,nbase(2_8, 1_8))
          n2 = nn2(-il2 + p0_2)
          do il3 = 0, p1_3, 1
            i3 = modulo(-il3 + span1_3,nbase(2_8, 2_8))
            d3 = (p0_3/(t1_3(i3 + p0_3) - t1_3(i3)))*dd3(-il3 + p1_3)


            u_part(ip, 2_8) = (n1*(d3*n2))*u3(i1, i2, i3) + u_part(ip, &
      2_8)




          end do

        end do

      end do

    else
      !evaluation of 1 - component (DNN)
      do jl3 = 0, p0_3, 1
        pow3 = posloc_3**jl3
        do jl2 = 0, p0_2, 1
          pow2 = posloc_2**jl2
          do jl1 = 0, p1_1, 1
            pow1 = posloc_1**jl1


            do il3 = 0, p0_3, 1
              i3 = modulo(-il3 + span0_3,nbase(0_8, 2_8))
              n3 = pow3*pp0_3(-il3 + p0_3, jl3)
              do il2 = 0, p0_2, 1
                i2 = modulo(-il2 + span0_2,nbase(0_8, 1_8))
                n2 = pow2*pp0_2(-il2 + p0_2, jl2)
                do il1 = 0, p1_1, 1
                  i1 = modulo(-il1 + span1_1,nbase(0_8, 0_8))
                  d1 = pow1*pp1_1(-il1 + p1_1, jl1)


                  u_part(ip, 0_8) = (d1*(n2*n3))*u1(i1, i2, i3) + u_part &
      (ip, 0_8)




                end do

              end do

            end do

          end do

        end do

      end do

      !evaluation of 2 - component (NDN)
      do jl3 = 0, p0_3, 1
        pow3 = posloc_3**jl3
        do jl2 = 0, p1_2, 1
          pow2 = posloc_2**jl2
          do jl1 = 0, p0_1, 1
            pow1 = posloc_1**jl1


            do il3 = 0, p0_3, 1
              i3 = modulo(-il3 + span0_3,nbase(0_8, 2_8))
              n3 = pow3*pp0_3(-il3 + p0_3, jl3)
              do il2 = 0, p1_2, 1
                i2 = modulo(-il2 + span1_2,nbase(2_8, 1_8))
                d2 = pow2*pp1_2(-il2 + p1_2, jl2)
                do il1 = 0, p0_1, 1
                  i1 = modulo(-il1 + span0_1,nbase(0_8, 0_8))
                  n1 = pow1*pp0_1(-il1 + p0_1, jl1)


                  u_part(ip, 1_8) = (n1*(d2*n3))*u2(i1, i2, i3) + u_part &
      (ip, 1_8)




                end do

              end do

            end do

          end do

        end do

      end do

      !evaluation of 3 - component (NND)
      do jl3 = 0, p1_3, 1
        pow3 = posloc_3**jl3
        do jl2 = 0, p0_2, 1
          pow2 = posloc_2**jl2
          do jl1 = 0, p0_1, 1
            pow1 = posloc_1**jl1


            do il3 = 0, p1_3, 1
              i3 = modulo(-il3 + span1_3,nbase(0_8, 2_8))
              d3 = pow3*pp1_3(-il3 + p1_3, jl3)
              do il2 = 0, p0_2, 1
                i2 = modulo(-il2 + span0_2,nbase(0_8, 1_8))
                n2 = pow2*pp0_2(-il2 + p0_2, jl2)
                do il1 = 0, p0_1, 1
                  i1 = modulo(-il1 + span0_1,nbase(0_8, 0_8))
                  n1 = pow1*pp0_1(-il1 + p0_1, jl1)


                  u_part(ip, 2_8) = (n1*(d3*n2))*u3(i1, i2, i3) + u_part &
      (ip, 2_8)




                end do

              end do

            end do

          end do

        end do

      end do

    end if
  end do

  !evaluation of equilibrium field
  !boundary region with recursive evaluation
  !$omp end do  
  !$omp end parallel  
  !interior with pp-form evaluation
  ierr = 0_8
end subroutine
!........................................

!........................................
subroutine evaluate_2form(particles_pos, t0_1, t0_2, t0_3, t1_1, t1_2, &
      t1_3, p0, nel, nbase, np, b1, b2, b3, pp0_1, pp0_2, pp0_3, pp1_1, &
      pp1_2, pp1_3, b_part, kind_map, params_map)

  implicit none
  real(kind=8), intent(in)  :: particles_pos (0:,0:)
  real(kind=8), intent(in)  :: t0_1 (0:)
  real(kind=8), intent(in)  :: t0_2 (0:)
  real(kind=8), intent(in)  :: t0_3 (0:)
  real(kind=8), intent(in)  :: t1_1 (0:)
  real(kind=8), intent(in)  :: t1_2 (0:)
  real(kind=8), intent(in)  :: t1_3 (0:)
  integer(kind=8), intent(in)  :: p0 (0:)
  integer(kind=8), intent(in)  :: nel (0:)
  integer(kind=8), intent(in)  :: nbase (0:,0:)
  integer(kind=8), value  :: np
  real(kind=8), intent(in)  :: b1 (0:,0:,0:)
  real(kind=8), intent(in)  :: b2 (0:,0:,0:)
  real(kind=8), intent(in)  :: b3 (0:,0:,0:)
  real(kind=8), intent(in)  :: pp0_1 (0:,0:)
  real(kind=8), intent(in)  :: pp0_2 (0:,0:)
  real(kind=8), intent(in)  :: pp0_3 (0:,0:)
  real(kind=8), intent(in)  :: pp1_1 (0:,0:)
  real(kind=8), intent(in)  :: pp1_2 (0:,0:)
  real(kind=8), intent(in)  :: pp1_3 (0:,0:)
  real(kind=8), intent(inout)  :: b_part (0:,0:)
  integer(kind=8), value  :: kind_map
  real(kind=8), intent(in)  :: params_map (0:)
  integer(kind=8)  :: p0_1  
  integer(kind=8)  :: p0_2  
  integer(kind=8)  :: p0_3  
  integer(kind=8)  :: p1_1  
  integer(kind=8)  :: p1_2  
  integer(kind=8)  :: p1_3  
  real(kind=8)  :: delta1  
  real(kind=8)  :: delta2  
  real(kind=8)  :: delta3  
  real(kind=8)  :: bl1  
  real(kind=8)  :: bl2  
  real(kind=8)  :: bl3  
  real(kind=8)  :: br1  
  real(kind=8)  :: br2  
  real(kind=8)  :: br3  
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
  integer(kind=8)  :: ierr  
  integer(kind=8)  :: ip  
  integer(kind=8)  :: span0_1  
  integer(kind=8)  :: span0_2  
  integer(kind=8)  :: span0_3  
  integer(kind=8)  :: span1_1  
  integer(kind=8)  :: span1_2  
  integer(kind=8)  :: span1_3  
  real(kind=8)  :: posloc_1  
  real(kind=8)  :: posloc_2  
  real(kind=8)  :: posloc_3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: i1  
  real(kind=8)  :: n1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: i2  
  real(kind=8)  :: d2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: i3  
  real(kind=8)  :: d3  
  real(kind=8)  :: d1  
  real(kind=8)  :: n2  
  real(kind=8)  :: n3  
  integer(kind=8)  :: jl3  
  real(kind=8)  :: pow3  
  integer(kind=8)  :: jl2  
  real(kind=8)  :: pow2  
  integer(kind=8)  :: jl1  
  real(kind=8)  :: pow1  







  p0_1 = p0(0_8)
  p0_2 = p0(1_8)
  p0_3 = p0(2_8)


  p1_1 = p0_1 - 1_8
  p1_2 = p0_2 - 1_8
  p1_3 = p0_3 - 1_8


  delta1 = 1.0d0/Real(nel(0_8), 8)
  delta2 = 1.0d0/Real(nel(1_8), 8)
  delta3 = 1.0d0/Real(nel(2_8), 8)


  bl1 = delta1*p1_1
  bl2 = delta2*p1_2
  bl3 = delta3*p1_3


  br1 = delta1*(-p1_1) + 1_8
  br2 = delta2*(-p1_2) + 1_8
  br3 = delta3*(-p1_3) + 1_8


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




  !$omp parallel
  !$omp do private(ip, span0_1, span0_2, span0_3, span1_1, span1_2, span1_3, pos&
      !$omp &loc_1, posloc_2, posloc_3, nl1, nr1, nn1, nl2, nr2, nn2, nl3, nr3, nn3, &
      !$omp &dl1, dr1, dd1, dl2, dr2, dd2, dl3, dr3, dd3, jl3, jl2, jl1, il3, il2, il&
      !$omp &1, pow1, pow2, pow3, i3, i2, i1, n3, n2, n1, d3, d2, d1)
  do ip = 0, np - 1_8, 1


    b_part(ip, 0_8) = inter_zyksv4_b1_eq(particles_pos(ip, 0_8), &
      particles_pos(ip, 1_8), particles_pos(ip, 2_8), kind_map, &
      params_map)
    b_part(ip, 1_8) = inter_zyksv4_b2_eq(particles_pos(ip, 0_8), &
      particles_pos(ip, 1_8), particles_pos(ip, 2_8), kind_map, &
      params_map)
    b_part(ip, 2_8) = inter_zyksv4_b3_eq(particles_pos(ip, 0_8), &
      particles_pos(ip, 1_8), particles_pos(ip, 2_8), kind_map, &
      params_map)


    span0_1 = p0_1 + Int(nel(0_8)*particles_pos(ip, 0_8), 8)
    span0_2 = p0_2 + Int(nel(1_8)*particles_pos(ip, 1_8), 8)
    span0_3 = p0_3 + Int(nel(2_8)*particles_pos(ip, 2_8), 8)


    span1_1 = span0_1 - 1_8
    span1_2 = span0_2 - 1_8
    span1_3 = span0_3 - 1_8


    posloc_1 = delta1*(p0_1 - span0_1) + particles_pos(ip, 0_8)
    posloc_2 = delta2*(p0_2 - span0_2) + particles_pos(ip, 1_8)
    posloc_3 = delta3*(p0_3 - span0_3) + particles_pos(ip, 2_8)




    if (particles_pos(ip, 0_8) > br1 .or. particles_pos(ip, 1_8) > br2 &
      .or. particles_pos(ip, 2_8) > br3 .or. particles_pos(ip, 0_8) < &
      bl1 .or. particles_pos(ip, 1_8) < bl2 .or. particles_pos(ip, 2_8 &
      ) < bl3) then


      call basis_funs(t0_1, p0_1, particles_pos(ip, 0_8), span0_1, nl1, &
      nr1, nn1)
      call basis_funs(t0_2, p0_2, particles_pos(ip, 1_8), span0_2, nl2, &
      nr2, nn2)
      call basis_funs(t0_3, p0_3, particles_pos(ip, 2_8), span0_3, nl3, &
      nr2, nn3)


      call basis_funs(t1_1, p1_1, particles_pos(ip, 0_8), span1_1, dl1, &
      dr1, dd1)
      call basis_funs(t1_2, p1_2, particles_pos(ip, 1_8), span1_2, dl2, &
      dr2, dd2)
      call basis_funs(t1_3, p1_3, particles_pos(ip, 2_8), span1_3, dl3, &
      dr2, dd3)


      !evaluation of 1 - component (NDD)
      do il1 = 0, p0_1, 1
        i1 = modulo(-il1 + span0_1,nbase(0_8, 0_8))
        n1 = nn1(-il1 + p0_1)
        do il2 = 0, p1_2, 1
          i2 = modulo(-il2 + span1_2,nbase(0_8, 1_8))
          d2 = (p0_2/(t1_2(i2 + p0_2) - t1_2(i2)))*dd2(-il2 + p1_2)
          do il3 = 0, p1_3, 1
            i3 = modulo(-il3 + span1_3,nbase(0_8, 2_8))
            d3 = (p0_3/(t1_3(i3 + p0_3) - t1_3(i3)))*dd3(-il3 + p1_3)


            b_part(ip, 0_8) = (n1*(d2*d3))*b1(i1, i2, i3) + b_part(ip, &
      0_8)


          end do

        end do

      end do

      !evaluation of 2 - component (DND)
      do il1 = 0, p1_1, 1
        i1 = modulo(-il1 + span1_1,nbase(1_8, 0_8))
        d1 = (p0_1/(t1_1(i1 + p0_1) - t1_1(i1)))*dd1(-il1 + p1_1)
        do il2 = 0, p0_2, 1
          i2 = modulo(-il2 + span0_2,nbase(1_8, 1_8))
          n2 = nn2(-il2 + p0_2)
          do il3 = 0, p1_3, 1
            i3 = modulo(-il3 + span1_3,nbase(1_8, 2_8))
            d3 = (p0_3/(t1_3(i3 + p0_3) - t1_3(i3)))*dd3(-il3 + p1_3)


            b_part(ip, 1_8) = (d1*(d3*n2))*b2(i1, i2, i3) + b_part(ip, &
      1_8)


          end do

        end do

      end do

      !evaluation of 3 - component (DDN)
      do il1 = 0, p1_1, 1
        i1 = modulo(-il1 + span1_1,nbase(2_8, 0_8))
        d1 = (p0_1/(t1_1(i1 + p0_1) - t1_1(i1)))*dd1(-il1 + p1_1)
        do il2 = 0, p1_2, 1
          i2 = modulo(-il2 + span1_2,nbase(2_8, 1_8))
          d2 = (p0_2/(t1_2(i2 + p0_2) - t1_2(i2)))*dd2(-il2 + p1_2)
          do il3 = 0, p0_3, 1
            i3 = modulo(-il3 + span0_3,nbase(2_8, 2_8))
            n3 = nn3(-il3 + p0_3)


            b_part(ip, 2_8) = (d1*(d2*n3))*b3(i1, i2, i3) + b_part(ip, &
      2_8)




          end do

        end do

      end do

    else
      !evaluation of 1 - component (NDD)
      do jl3 = 0, p1_3, 1
        pow3 = posloc_3**jl3
        do jl2 = 0, p1_2, 1
          pow2 = posloc_2**jl2
          do jl1 = 0, p0_1, 1
            pow1 = posloc_1**jl1


            do il3 = 0, p1_3, 1
              i3 = modulo(-il3 + span1_3,nbase(0_8, 2_8))
              d3 = pow3*pp1_3(-il3 + p1_3, jl3)
              do il2 = 0, p1_2, 1
                i2 = modulo(-il2 + span1_2,nbase(0_8, 1_8))
                d2 = pow2*pp1_2(-il2 + p1_2, jl2)
                do il1 = 0, p0_1, 1
                  i1 = modulo(-il1 + span0_1,nbase(0_8, 0_8))
                  n1 = pow1*pp0_1(-il1 + p0_1, jl1)


                  b_part(ip, 0_8) = (n1*(d2*d3))*b1(i1, i2, i3) + b_part &
      (ip, 0_8)




                end do

              end do

            end do

          end do

        end do

      end do

      !evaluation of 2 - component (DND)
      do jl3 = 0, p1_3, 1
        pow3 = posloc_3**jl3
        do jl2 = 0, p0_2, 1
          pow2 = posloc_2**jl2
          do jl1 = 0, p1_1, 1
            pow1 = posloc_1**jl1


            do il3 = 0, p1_3, 1
              i3 = modulo(-il3 + span1_3,nbase(1_8, 2_8))
              d3 = pow3*pp1_3(-il3 + p1_3, jl3)
              do il2 = 0, p0_2, 1
                i2 = modulo(-il2 + span0_2,nbase(1_8, 1_8))
                n2 = pow2*pp0_2(-il2 + p0_2, jl2)
                do il1 = 0, p1_1, 1
                  i1 = modulo(-il1 + span1_1,nbase(1_8, 0_8))
                  d1 = pow1*pp1_1(-il1 + p1_1, jl1)


                  b_part(ip, 1_8) = (d1*(d3*n2))*b2(i1, i2, i3) + b_part &
      (ip, 1_8)




                end do

              end do

            end do

          end do

        end do

      end do

      !evaluation of 3 - component (DDN)
      do jl3 = 0, p0_3, 1
        pow3 = posloc_3**jl3
        do jl2 = 0, p1_2, 1
          pow2 = posloc_2**jl2
          do jl1 = 0, p1_1, 1
            pow1 = posloc_1**jl1


            do il3 = 0, p0_3, 1
              i3 = modulo(-il3 + span0_3,nbase(2_8, 2_8))
              n3 = pow3*pp0_3(-il3 + p0_3, jl3)
              do il2 = 0, p1_2, 1
                i2 = modulo(-il2 + span1_2,nbase(2_8, 1_8))
                d2 = pow2*pp1_2(-il2 + p1_2, jl2)
                do il1 = 0, p1_1, 1
                  i1 = modulo(-il1 + span1_1,nbase(2_8, 0_8))
                  d1 = pow1*pp1_1(-il1 + p1_1, jl1)


                  b_part(ip, 2_8) = (d1*(d2*n3))*b3(i1, i2, i3) + b_part &
      (ip, 2_8)




                end do

              end do

            end do

          end do

        end do

      end do

    end if
  end do

  !evaluation of equilibrium field
  !boundary region with recursive evaluation
  !$omp end do  
  !$omp end parallel  
  !interior with pp-form evaluation
  ierr = 0_8
end subroutine
!........................................

end module