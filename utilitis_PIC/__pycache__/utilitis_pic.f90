module utilitis_pic

implicit none




contains

!........................................
pure subroutine cross(a, b, r) 

implicit none
real(kind=8), intent(in)  :: a (0:)
real(kind=8), intent(in)  :: b (0:)
real(kind=8), intent(inout)  :: r (0:)

r(0) = a(1)*b(2) + (-a(2))*b(1)
r(1) = (-a(0))*b(2) + a(2)*b(0)
r(2) = a(0)*b(1) + (-a(1))*b(0)
end subroutine
!........................................

!........................................
pure integer(kind=4) function find_span(knots, degree, x)  result( &
      returnVal)

implicit none
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree 
real(kind=8), intent(in)  :: x 
integer(kind=4) :: low  
integer(kind=4) :: high  
integer(kind=4) :: span  

!__________________________________________________CommentBlock__________________________________________________!
!                                                                                                                !
!    Determine the knot span index at location x, given the B-Splines' knot sequence and polynomial degree.      !
!                                                                                                                !
!    For a degree p, the knot span index i identifies the indices [i-p:i] of all p+1 non-zero basis functions at a!
!    given location x.                                                                                           !
!                                                                                                                !
!    Parameters                                                                                                  !
!    ----------                                                                                                  !
!    knots : array_like                                                                                          !
!        Knots sequence.                                                                                         !
!                                                                                                                !
!    degree : int                                                                                                !
!        Polynomial degree of B-splines.                                                                         !
!                                                                                                                !
!    x : float                                                                                                   !
!        Location of interest.                                                                                   !
!                                                                                                                !
!    Returns                                                                                                     !
!    -------                                                                                                     !
!    span : int                                                                                                  !
!        Knot span index.                                                                                        !
!                                                                                                                !
!                                                                                                                !
!________________________________________________________________________________________________________________!
!Knot index at left/right boundary
low = degree
high = 0
high = -degree - 1 + size(knots,1)


!Check if point is exactly on left/right boundary, or outside domain
if (x <= knots(low)) then
returnVal = low
else if (x >= knots(high)) then
returnVal = high - 1
else
!Perform binary search
span = floor((high + low)/Real(2, 8))
do while (x >= knots(span + 1) .or. x < knots(span)) 
  if (x < knots(span)) then
    high = span
  else
    low = span
  end if
  span = floor((high + low)/Real(2, 8))
end do
returnVal = span
end if
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
integer(kind=4) :: p  
integer(kind=4) :: j  
real(kind=8) :: saved  
integer(kind=4) :: r  
real(kind=8) :: temp  

!__________________________________________________CommentBlock__________________________________________________!
!                                                                                                                !
!    Compute the non-vanishing B-splines at location x, given the knot sequence, polynomial degree and knot span.!
!                                                                                                                !
!    Parameters                                                                                                  !
!    ----------                                                                                                  !
!    knots : array_like                                                                                          !
!        Knots sequence.                                                                                         !
!                                                                                                                !
!    degree : int                                                                                                !
!        Polynomial degree of B-splines.                                                                         !
!                                                                                                                !
!    x : float                                                                                                   !
!        Evaluation point.                                                                                       !
!                                                                                                                !
!    span : int                                                                                                  !
!        Knot span index.                                                                                        !
!                                                                                                                !
!    Results                                                                                                     !
!    -------                                                                                                     !
!    values : numpy.ndarray                                                                                      !
!        Values of p+1 non-vanishing B-Splines at location x.                                                    !
!                                                                                                                !
!                                                                                                                !
!________________________________________________________________________________________________________________!
!to avoid degree being intent(inout)
!TODO improve
p = degree


!from numpy      import empty
!left   = empty( p  , dtype=float )
!right  = empty( p  , dtype=float )
left(:) = 0.0d0
right(:) = 0.0d0


values(0) = 1.0d0
do j = 0, p - 1, 1
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

end module