module kernels_projectors_global_mhd

implicit none




contains

!........................................
subroutine kernel_pi0(n1, n2, n3, pl1, pl2, pl3, b1, b2, b3, mat, rhs) 

  implicit none
  integer(kind=8), value  :: n1
  integer(kind=8), value  :: n2
  integer(kind=8), value  :: n3
  integer(kind=8), value  :: pl1
  integer(kind=8), value  :: pl2
  integer(kind=8), value  :: pl3
  real(kind=8), intent(in)  :: b1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b3 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat (0:,0:,0:)
  real(kind=8), intent(inout)  :: rhs (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  



  do ie1 = 0, n1 - 1_8, 1
    do ie2 = 0, n2 - 1_8, 1
      do ie3 = 0, n3 - 1_8, 1


        do il1 = 0, pl1 - 1_8, 1
          do il2 = 0, pl2 - 1_8, 1
            do il3 = 0, pl3 - 1_8, 1


              rhs(ie1, ie2, ie3, il1, il2, il3) = ((b3(ie3, il3, 0_8, &
      0_8)*mat(ie1, ie2, ie3))*b2(ie2, il2, 0_8, 0_8))*b1(ie1, il1, 0_8 &
      , 0_8)






            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi1_1(n1, n2, n3, pl1, pl2, pl3, ies_1, il_add_1, nq1, &
      w1, b1, b2, b3, mat, rhs_1)

  implicit none
  integer(kind=8), value  :: n1
  integer(kind=8), value  :: n2
  integer(kind=8), value  :: n3
  integer(kind=8), value  :: pl1
  integer(kind=8), value  :: pl2
  integer(kind=8), value  :: pl3
  integer(kind=8), intent(in)  :: ies_1 (0:)
  integer(kind=8), intent(in)  :: il_add_1 (0:)
  integer(kind=8), value  :: nq1
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: b1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b3 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat (0:,0:,0:)
  real(kind=8), intent(inout)  :: rhs_1 (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: q1  



  do ie1 = 0, n1 - 1_8, 1
    do ie2 = 0, n2 - 1_8, 1
      do ie3 = 0, n3 - 1_8, 1


        do il1 = 0, pl1 - 1_8, 1
          do il2 = 0, pl2 - 1_8, 1
            do il3 = 0, pl3 - 1_8, 1


              do q1 = 0, nq1 - 1_8, 1
                rhs_1(ies_1(ie1), ie2, ie3, il1 + il_add_1(ie1), il2, &
      il3) = (((b3(ie3, il3, 0_8, 0_8)*mat(ie1*nq1 + q1, ie2, ie3))*b2( &
      ie2, il2, 0_8, 0_8))*b1(ie1, il1, 0_8, q1))*w1(ie1, q1) + rhs_1( &
      ies_1(ie1), ie2, ie3, il1 + il_add_1(ie1), il2, il3)






              end do

            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi1_2(n1, n2, n3, pl1, pl2, pl3, ies_2, il_add_2, nq2, &
      w2, b1, b2, b3, mat, rhs_2)

  implicit none
  integer(kind=8), value  :: n1
  integer(kind=8), value  :: n2
  integer(kind=8), value  :: n3
  integer(kind=8), value  :: pl1
  integer(kind=8), value  :: pl2
  integer(kind=8), value  :: pl3
  integer(kind=8), intent(in)  :: ies_2 (0:)
  integer(kind=8), intent(in)  :: il_add_2 (0:)
  integer(kind=8), value  :: nq2
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: b1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b3 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat (0:,0:,0:)
  real(kind=8), intent(inout)  :: rhs_2 (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: q2  



  do ie1 = 0, n1 - 1_8, 1
    do ie2 = 0, n2 - 1_8, 1
      do ie3 = 0, n3 - 1_8, 1


        do il1 = 0, pl1 - 1_8, 1
          do il2 = 0, pl2 - 1_8, 1
            do il3 = 0, pl3 - 1_8, 1


              do q2 = 0, nq2 - 1_8, 1
                rhs_2(ie1, ies_2(ie2), ie3, il1, il2 + il_add_2(ie2), &
      il3) = (((b3(ie3, il3, 0_8, 0_8)*mat(ie1, ie2*nq2 + q2, ie3))*b2( &
      ie2, il2, 0_8, q2))*b1(ie1, il1, 0_8, 0_8))*w2(ie2, q2) + rhs_2( &
      ie1, ies_2(ie2), ie3, il1, il2 + il_add_2(ie2), il3)






              end do

            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi1_3(n1, n2, n3, pl1, pl2, pl3, ies_3, il_add_3, nq3, &
      w3, b1, b2, b3, mat, rhs_3)

  implicit none
  integer(kind=8), value  :: n1
  integer(kind=8), value  :: n2
  integer(kind=8), value  :: n3
  integer(kind=8), value  :: pl1
  integer(kind=8), value  :: pl2
  integer(kind=8), value  :: pl3
  integer(kind=8), intent(in)  :: ies_3 (0:)
  integer(kind=8), intent(in)  :: il_add_3 (0:)
  integer(kind=8), value  :: nq3
  real(kind=8), intent(in)  :: w3 (0:,0:)
  real(kind=8), intent(in)  :: b1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b3 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat (0:,0:,0:)
  real(kind=8), intent(inout)  :: rhs_3 (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: q3  



  do ie1 = 0, n1 - 1_8, 1
    do ie2 = 0, n2 - 1_8, 1
      do ie3 = 0, n3 - 1_8, 1


        do il1 = 0, pl1 - 1_8, 1
          do il2 = 0, pl2 - 1_8, 1
            do il3 = 0, pl3 - 1_8, 1


              do q3 = 0, nq3 - 1_8, 1
                rhs_3(ie1, ie2, ies_3(ie3), il1, il2, il3 + il_add_3(ie3 &
      )) = (((b3(ie3, il3, 0_8, q3)*mat(ie1, ie2, ie3*nq3 + q3))*b2(ie2 &
      , il2, 0_8, 0_8))*b1(ie1, il1, 0_8, 0_8))*w3(ie3, q3) + rhs_3(ie1 &
      , ie2, ies_3(ie3), il1, il2, il3 + il_add_3(ie3))






              end do

            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi2_1(n1, n2, n3, pl1, pl2, pl3, ies_2, ies_3, &
      il_add_2, il_add_3, nq2, nq3, w2, w3, b1, b2, b3, mat, rhs_1)

  implicit none
  integer(kind=8), value  :: n1
  integer(kind=8), value  :: n2
  integer(kind=8), value  :: n3
  integer(kind=8), value  :: pl1
  integer(kind=8), value  :: pl2
  integer(kind=8), value  :: pl3
  integer(kind=8), intent(in)  :: ies_2 (0:)
  integer(kind=8), intent(in)  :: ies_3 (0:)
  integer(kind=8), intent(in)  :: il_add_2 (0:)
  integer(kind=8), intent(in)  :: il_add_3 (0:)
  integer(kind=8), value  :: nq2
  integer(kind=8), value  :: nq3
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: w3 (0:,0:)
  real(kind=8), intent(in)  :: b1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b3 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat (0:,0:,0:)
  real(kind=8), intent(inout)  :: rhs_1 (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: q2  
  integer(kind=8)  :: q3  



  do ie1 = 0, n1 - 1_8, 1
    do ie2 = 0, n2 - 1_8, 1
      do ie3 = 0, n3 - 1_8, 1


        do il1 = 0, pl1 - 1_8, 1
          do il2 = 0, pl2 - 1_8, 1
            do il3 = 0, pl3 - 1_8, 1


              do q2 = 0, nq2 - 1_8, 1
                do q3 = 0, nq3 - 1_8, 1


                  rhs_1(ie1, ies_2(ie2), ies_3(ie3), il1, il2 + il_add_2 &
      (ie2), il3 + il_add_3(ie3)) = ((((b3(ie3, il3, 0_8, q3)*mat(ie1, &
      ie2*nq2 + q2, ie3*nq3 + q3))*b2(ie2, il2, 0_8, q2))*b1(ie1, il1, &
      0_8, 0_8))*w3(ie3, q3))*w2(ie2, q2) + rhs_1(ie1, ies_2(ie2), &
      ies_3(ie3), il1, il2 + il_add_2(ie2), il3 + il_add_3(ie3))






                end do

              end do

            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi2_2(n1, n2, n3, pl1, pl2, pl3, ies_1, ies_3, &
      il_add_1, il_add_3, nq1, nq3, w1, w3, b1, b2, b3, mat, rhs_2)

  implicit none
  integer(kind=8), value  :: n1
  integer(kind=8), value  :: n2
  integer(kind=8), value  :: n3
  integer(kind=8), value  :: pl1
  integer(kind=8), value  :: pl2
  integer(kind=8), value  :: pl3
  integer(kind=8), intent(in)  :: ies_1 (0:)
  integer(kind=8), intent(in)  :: ies_3 (0:)
  integer(kind=8), intent(in)  :: il_add_1 (0:)
  integer(kind=8), intent(in)  :: il_add_3 (0:)
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq3
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: w3 (0:,0:)
  real(kind=8), intent(in)  :: b1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b3 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat (0:,0:,0:)
  real(kind=8), intent(inout)  :: rhs_2 (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q3  



  do ie1 = 0, n1 - 1_8, 1
    do ie2 = 0, n2 - 1_8, 1
      do ie3 = 0, n3 - 1_8, 1


        do il1 = 0, pl1 - 1_8, 1
          do il2 = 0, pl2 - 1_8, 1
            do il3 = 0, pl3 - 1_8, 1


              do q1 = 0, nq1 - 1_8, 1
                do q3 = 0, nq3 - 1_8, 1


                  rhs_2(ies_1(ie1), ie2, ies_3(ie3), il1 + il_add_1(ie1) &
      , il2, il3 + il_add_3(ie3)) = ((((b3(ie3, il3, 0_8, q3)*mat(ie1* &
      nq1 + q1, ie2, ie3*nq3 + q3))*b2(ie2, il2, 0_8, 0_8))*b1(ie1, il1 &
      , 0_8, q1))*w3(ie3, q3))*w1(ie1, q1) + rhs_2(ies_1(ie1), ie2, &
      ies_3(ie3), il1 + il_add_1(ie1), il2, il3 + il_add_3(ie3))






                end do

              end do

            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

!........................................
subroutine kernel_pi2_3(n1, n2, n3, pl1, pl2, pl3, ies_1, ies_2, &
      il_add_1, il_add_2, nq1, nq2, w1, w2, b1, b2, b3, mat, rhs_3)

  implicit none
  integer(kind=8), value  :: n1
  integer(kind=8), value  :: n2
  integer(kind=8), value  :: n3
  integer(kind=8), value  :: pl1
  integer(kind=8), value  :: pl2
  integer(kind=8), value  :: pl3
  integer(kind=8), intent(in)  :: ies_1 (0:)
  integer(kind=8), intent(in)  :: ies_2 (0:)
  integer(kind=8), intent(in)  :: il_add_1 (0:)
  integer(kind=8), intent(in)  :: il_add_2 (0:)
  integer(kind=8), value  :: nq1
  integer(kind=8), value  :: nq2
  real(kind=8), intent(in)  :: w1 (0:,0:)
  real(kind=8), intent(in)  :: w2 (0:,0:)
  real(kind=8), intent(in)  :: b1 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b2 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: b3 (0:,0:,0:,0:)
  real(kind=8), intent(in)  :: mat (0:,0:,0:)
  real(kind=8), intent(inout)  :: rhs_3 (0:,0:,0:,0:,0:,0:)
  integer(kind=8)  :: ie1  
  integer(kind=8)  :: ie2  
  integer(kind=8)  :: ie3  
  integer(kind=8)  :: il1  
  integer(kind=8)  :: il2  
  integer(kind=8)  :: il3  
  integer(kind=8)  :: q1  
  integer(kind=8)  :: q2  



  do ie1 = 0, n1 - 1_8, 1
    do ie2 = 0, n2 - 1_8, 1
      do ie3 = 0, n3 - 1_8, 1


        do il1 = 0, pl1 - 1_8, 1
          do il2 = 0, pl2 - 1_8, 1
            do il3 = 0, pl3 - 1_8, 1


              do q1 = 0, nq1 - 1_8, 1
                do q2 = 0, nq2 - 1_8, 1


                  rhs_3(ies_1(ie1), ies_2(ie2), ie3, il1 + il_add_1(ie1) &
      , il2 + il_add_2(ie2), il3) = ((((b3(ie3, il3, 0_8, 0_8)*mat(ie1* &
      nq1 + q1, ie2*nq2 + q2, ie3))*b2(ie2, il2, 0_8, q2))*b1(ie1, il1, &
      0_8, q1))*w2(ie2, q2))*w1(ie1, q1) + rhs_3(ies_1(ie1), ies_2(ie2) &
      , ie3, il1 + il_add_1(ie1), il2 + il_add_2(ie2), il3)
                end do

              end do

            end do

          end do

        end do

      end do

    end do

  end do

end subroutine
!........................................

end module