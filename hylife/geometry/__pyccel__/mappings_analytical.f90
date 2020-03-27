module mappings_analytical

implicit none




contains

!........................................
function f(xi1, xi2, xi3, kind, params, component) result(value)

  implicit none
  real(kind=8)  :: value  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  integer(kind=8), value  :: component
  real(kind=8)  :: Lx  
  real(kind=8)  :: Ly  
  real(kind=8)  :: Lz  
  real(kind=8)  :: R1  
  real(kind=8)  :: R2  
  real(kind=8)  :: dR  
  real(kind=8)  :: alpha  

  !__________________________CommentBlock__________________________!
  !'''                                                             !
  !    defines an analytical mapping x = f(xi) in three dimensions. !
  !                                                                !
  !    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella)      !
  !                                                                !
  !    params    : slab            --> Lx, Ly, Lz                  !
  !              : hollow cylinder --> R1, R2, Lz                  !
  !              : colella         --> Lx, Ly, alpha, Lz           !
  !                                                                !
  !    component : 1 (x), 2 (y), 3 (z)                             !
  !    '''                                                         !
  !________________________________________________________________!


  value = 0.0d0


  if (kind == 1_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    Lz = params(2_8)


    if (component == 1_8 ) then
      value = Lx*xi1
    else if (component == 2_8 ) then
      value = Ly*xi2
    else if (component == 3_8 ) then
      value = Lz*xi3
    end if
  else if (kind == 2_8 ) then


    R1 = params(0_8)
    R2 = params(1_8)
    Lz = params(2_8)
    dR = -R1 + R2


    if (component == 1_8 ) then
      value = (R1 + dR*xi1)*cos(2_8*(3.14159265358979d0*xi2))
    else if (component == 2_8 ) then
      value = (R1 + dR*xi1)*sin(2_8*(3.14159265358979d0*xi2))
    else if (component == 3_8 ) then
      value = Lz*xi3
    end if
  else if (kind == 3_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    alpha = params(2_8)
    Lz = params(3_8)


    if (component == 1_8 ) then
      value = Lx*(alpha*(sin(2_8*(3.14159265358979d0*xi1))*sin(2_8*( &
      3.14159265358979d0*xi2))) + xi1)
    else if (component == 2_8 ) then
      value = Ly*(alpha*(sin(2_8*(3.14159265358979d0*xi1))*sin(2_8*( &
      3.14159265358979d0*xi2))) + xi2)
    else if (component == 3_8 ) then
      value = Lz*xi3
    end if
  end if
  return
end function
!........................................

!........................................
function df(xi1, xi2, xi3, kind, params, component) result(value)

  implicit none
  real(kind=8)  :: value  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  integer(kind=8), value  :: component
  real(kind=8)  :: Lx  
  real(kind=8)  :: Ly  
  real(kind=8)  :: Lz  
  real(kind=8)  :: calue  
  real(kind=8)  :: R1  
  real(kind=8)  :: R2  
  real(kind=8)  :: dR  
  real(kind=8)  :: alpha  

  !_______________________________________________CommentBlock_______________________________________________!
  !'''                                                                                                       !
  !    returns the components of the Jacobian matrix of an analytical mapping x = f(xi) in three dimensions. !
  !                                                                                                          !
  !    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella)                                                !
  !                                                                                                          !
  !    params    : slab            --> Lx, Ly, Lz                                                            !
  !              : hollow cylinder --> R1, R2, Lz                                                            !
  !              : colella         --> Lx, Ly, alpha, Lz                                                     !
  !                                                                                                          !
  !    component : 11 (dfx/dxi1), 12 (dfx/dxi2), 13 (dfx/dxi3)                                               !
  !                21 (dfy/dxi1), 22 (dfy/dxi2), 23 (dfy/dxi3)                                               !
  !                31 (dfz/dxi1), 32 (dfz/dxi2), 33 (dfz/dxi3)                                               !
  !    '''                                                                                                   !
  !__________________________________________________________________________________________________________!


  value = 0.0d0


  if (kind == 1_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    Lz = params(2_8)


    if (component == 11_8 ) then
      value = Lx
    else if (component == 12_8 ) then
      value = 0.0d0
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      calue = 0.0d0
    else if (component == 22_8 ) then
      value = Ly
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = Lz
    end if
  else if (kind == 2_8 ) then


    R1 = params(0_8)
    R2 = params(1_8)
    Lz = params(2_8)
    dR = -R1 + R2


    if (component == 11_8 ) then
      value = dR*cos(2_8*(3.14159265358979d0*xi2))
    else if (component == 12_8 ) then
      value = -2_8*3.14159265358979d0*(R1 + dR*xi1)*sin(2_8*( &
      3.14159265358979d0*xi2))
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      value = dR*sin(2_8*(3.14159265358979d0*xi2))
    else if (component == 22_8 ) then
      value = 2_8*(3.14159265358979d0*((R1 + dR*xi1)*cos(2_8*( &
      3.14159265358979d0*xi2))))
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = Lz
    end if
  else if (kind == 3_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    alpha = params(2_8)
    Lz = params(3_8)


    if (component == 11_8 ) then
      value = Lx*(alpha*(((2_8*3.14159265358979d0)*sin(2_8*( &
      3.14159265358979d0*xi2)))*cos(2_8*(3.14159265358979d0*xi1))) + &
      1_8)
    else if (component == 12_8 ) then
      value = Lx*(alpha*(((2_8*3.14159265358979d0)*cos(2_8*( &
      3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1))))
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      value = Ly*(alpha*(((2_8*3.14159265358979d0)*sin(2_8*( &
      3.14159265358979d0*xi2)))*cos(2_8*(3.14159265358979d0*xi1))))
    else if (component == 22_8 ) then
      value = Ly*(alpha*(((2_8*3.14159265358979d0)*cos(2_8*( &
      3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1))) + &
      1_8)
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = Lz
    end if
  end if
  return
end function
!........................................

!........................................
function det_df(xi1, xi2, xi3, kind, params) result(value)

  implicit none
  real(kind=8)  :: value  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  real(kind=8)  :: Lx  
  real(kind=8)  :: Ly  
  real(kind=8)  :: Lz  
  real(kind=8)  :: R1  
  real(kind=8)  :: R2  
  real(kind=8)  :: dR  
  real(kind=8)  :: alpha  

  !________________________________________CommentBlock________________________________________!
  !'''                                                                                         !
  !    returns the jacobian determinant of an analytical mapping x = f(xi) in three dimensions. !
  !                                                                                            !
  !    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella)                                  !
  !                                                                                            !
  !    params    : slab            --> Lx, Ly, Lz                                              !
  !              : hollow cylinder --> R1, R2, Lz                                              !
  !              : colella         --> Lx, Ly, alpha, Lz                                       !
  !    '''                                                                                     !
  !____________________________________________________________________________________________!


  value = 0.0d0


  if (kind == 1_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    Lz = params(2_8)


    value = Lx*(Ly*Lz)
  else if (kind == 2_8 ) then


    R1 = params(0_8)
    R2 = params(1_8)
    Lz = params(2_8)
    dR = -R1 + R2


    value = dR*(Lz*(2_8*(3.14159265358979d0*(R1 + dR*xi1))))
  else if (kind == 3_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    alpha = params(2_8)
    Lz = params(3_8)


    value = Lx*(Ly*(Lz*(alpha*(((2_8*3.14159265358979d0)*cos(2_8*( &
      3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1))) + &
      alpha*(((2_8*3.14159265358979d0)*sin(2_8*(3.14159265358979d0*xi2 &
      )))*cos(2_8*(3.14159265358979d0*xi1))) + 1_8)))
  end if




  return
end function
!........................................

!........................................
function df_inv(xi1, xi2, xi3, kind, params, component) result( &
      Dummy_4403)

  implicit none
  real(kind=8)  :: Dummy_4403  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  integer(kind=8), value  :: component
  real(kind=8)  :: value  
  real(kind=8)  :: Lx  
  real(kind=8)  :: Ly  
  real(kind=8)  :: Lz  
  real(kind=8)  :: calue  
  real(kind=8)  :: R1  
  real(kind=8)  :: R2  
  real(kind=8)  :: dR  
  real(kind=8)  :: alpha  

  !_______________________________________________CommentBlock_______________________________________________!
  !'''                                                                                                       !
  !    returns the components of the Jacobian matrix of an analytical mapping x = f(xi) in three dimensions. !
  !                                                                                                          !
  !    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella)                                                !
  !                                                                                                          !
  !    params    : slab            --> Lx, Ly, Lz                                                            !
  !              : hollow cylinder --> R1, R2, Lz                                                            !
  !              : colella         --> Lx, Ly, alpha, Lz                                                     !
  !                                                                                                          !
  !    component : 11 (dfx/dxi1), 12 (dfx/dxi2), 13 (dfx/dxi3)                                               !
  !                21 (dfy/dxi1), 22 (dfy/dxi2), 23 (dfy/dxi3)                                               !
  !                31 (dfz/dxi1), 32 (dfz/dxi2), 33 (dfz/dxi3)                                               !
  !    '''                                                                                                   !
  !__________________________________________________________________________________________________________!


  value = 0.0d0


  if (kind == 1_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    Lz = params(2_8)


    if (component == 11_8 ) then
      value = Ly*Lz
    else if (component == 12_8 ) then
      value = 0.0d0
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      calue = 0.0d0
    else if (component == 22_8 ) then
      value = Lx*Lz
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = Lx*Ly
    end if
  else if (kind == 2_8 ) then


    R1 = params(0_8)
    R2 = params(1_8)
    Lz = params(2_8)
    dR = -R1 + R2


    if (component == 11_8 ) then
      value = 2_8*(3.14159265358979d0*((Lz*cos(2_8*(3.14159265358979d0* &
      xi2)))*(R1 + dR*xi1)))
    else if (component == 12_8 ) then
      value = 2_8*(3.14159265358979d0*((Lz*sin(2_8*(3.14159265358979d0* &
      xi2)))*(R1 + dR*xi1)))
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      value = (-dR)*(Lz*sin(2_8*(3.14159265358979d0*xi2)))
    else if (component == 22_8 ) then
      value = dR*(Lz*cos(2_8*(3.14159265358979d0*xi2)))
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = dR*(2_8*(3.14159265358979d0*(R1 + dR*xi1)))
    end if
  else if (kind == 3_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    alpha = params(2_8)
    Lz = params(3_8)


    if (component == 11_8 ) then
      value = Ly*(Lz*(alpha*(((2_8*3.14159265358979d0)*cos(2_8*( &
      3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1))) + &
      1_8))
    else if (component == 12_8 ) then
      value = (-Lx)*(alpha*(((2_8*(3.14159265358979d0*Lz))*cos(2_8*( &
      3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1))))
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      value = (-Ly)*(alpha*(((2_8*(3.14159265358979d0*Lz))*sin(2_8*( &
      3.14159265358979d0*xi2)))*cos(2_8*(3.14159265358979d0*xi1))))
    else if (component == 22_8 ) then
      value = Lx*(Lz*(alpha*(((2_8*3.14159265358979d0)*sin(2_8*( &
      3.14159265358979d0*xi2)))*cos(2_8*(3.14159265358979d0*xi1))) + &
      1_8))
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = Lx*(Ly*(alpha*(((2_8*3.14159265358979d0)*cos(2_8*( &
      3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1))) + &
      alpha*(((2_8*3.14159265358979d0)*sin(2_8*(3.14159265358979d0*xi2 &
      )))*cos(2_8*(3.14159265358979d0*xi1))) + 1_8))
    end if
  end if
  Dummy_4403 = value/det_df(xi1, xi2, xi3, kind, params)
  return
end function
!........................................

!........................................
function g(xi1, xi2, xi3, kind, params, component) result(value)

  implicit none
  real(kind=8)  :: value  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  integer(kind=8), value  :: component
  real(kind=8)  :: Lx  
  real(kind=8)  :: Ly  
  real(kind=8)  :: Lz  
  real(kind=8)  :: calue  
  real(kind=8)  :: R1  
  real(kind=8)  :: R2  
  real(kind=8)  :: dR  
  real(kind=8)  :: alpha  

  !___________________________________________________CommentBlock___________________________________________________!
  !'''                                                                                                               !
  !    returns the components of the metric tensor (df)^T*df of an analytical mapping x = f(xi) in three dimensions. !
  !                                                                                                                  !
  !    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella)                                                        !
  !                                                                                                                  !
  !    params    : slab            --> Lx, Ly, Lz                                                                    !
  !              : hollow cylinder --> R1, R2, Lz                                                                    !
  !              : colella         --> Lx, Ly, alpha, Lz                                                             !
  !                                                                                                                  !
  !    component : 11 (dfx/dxi1), 12 (dfx/dxi2), 13 (dfx/dxi3)                                                       !
  !                21 (dfy/dxi1), 22 (dfy/dxi2), 23 (dfy/dxi3)                                                       !
  !                31 (dfz/dxi1), 32 (dfz/dxi2), 33 (dfz/dxi3)                                                       !
  !    '''                                                                                                           !
  !__________________________________________________________________________________________________________________!


  value = 0.0d0


  if (kind == 1_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    Lz = params(2_8)


    if (component == 11_8 ) then
      value = Lx**2_8
    else if (component == 12_8 ) then
      value = 0.0d0
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      calue = 0.0d0
    else if (component == 22_8 ) then
      value = Ly**2_8
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = Lz**2_8
    end if
  else if (kind == 2_8 ) then


    R1 = params(0_8)
    R2 = params(1_8)
    Lz = params(2_8)
    dR = -R1 + R2


    if (component == 11_8 ) then
      value = dR**2_8
    else if (component == 12_8 ) then
      value = 0.0d0
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      value = 0.0d0
    else if (component == 22_8 ) then
      value = (2_8*3.14159265358979d0)**2_8*(R1 + dR*xi1)**2_8
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = Lz**2_8
    end if
  else if (kind == 3_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    alpha = params(2_8)
    Lz = params(3_8)


    if (component == 11_8 ) then
      value = Lx**2_8*(alpha*(((2_8*3.14159265358979d0)*sin(2_8*( &
      3.14159265358979d0*xi2)))*cos(2_8*(3.14159265358979d0*xi1))) + &
      1_8)**2_8 + Ly**2_8*(alpha**2_8*(((2_8*3.14159265358979d0)**2_8* &
      sin(2_8*(3.14159265358979d0*xi2))**2_8)*cos(2_8*( &
      3.14159265358979d0*xi1))**2_8))
    else if (component == 12_8 ) then
      value = (alpha**2_8*(((((2_8*3.14159265358979d0)**2_8*cos(2_8*( &
      3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1)))*sin( &
      2_8*(3.14159265358979d0*xi2)))*cos(2_8*(3.14159265358979d0*xi1 &
      ))))*(Lx**2_8 + Ly**2_8) + Lx**2_8*(alpha*(((2_8* &
      3.14159265358979d0)*cos(2_8*(3.14159265358979d0*xi2)))*sin(2_8*( &
      3.14159265358979d0*xi1)))) + Ly**2_8*(alpha*(((2_8* &
      3.14159265358979d0)*sin(2_8*(3.14159265358979d0*xi2)))*cos(2_8*( &
      3.14159265358979d0*xi1))))
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      value = (alpha**2_8*(((((2_8*3.14159265358979d0)**2_8*cos(2_8*( &
      3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1)))*sin( &
      2_8*(3.14159265358979d0*xi2)))*cos(2_8*(3.14159265358979d0*xi1 &
      ))))*(Lx**2_8 + Ly**2_8) + Lx**2_8*(alpha*(((2_8* &
      3.14159265358979d0)*cos(2_8*(3.14159265358979d0*xi2)))*sin(2_8*( &
      3.14159265358979d0*xi1)))) + Ly**2_8*(alpha*(((2_8* &
      3.14159265358979d0)*sin(2_8*(3.14159265358979d0*xi2)))*cos(2_8*( &
      3.14159265358979d0*xi1))))
    else if (component == 22_8 ) then
      value = Lx**2_8*(alpha**2_8*(((2_8*3.14159265358979d0)**2_8*cos( &
      2_8*(3.14159265358979d0*xi2))**2_8)*sin(2_8*(3.14159265358979d0* &
      xi1))**2_8)) + Ly**2_8*(alpha*(((2_8*3.14159265358979d0)*cos(2_8* &
      (3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1))) + &
      1_8)**2_8
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = Lz**2_8
    end if
  end if
  return
end function
!........................................

!........................................
function g_inv(xi1, xi2, xi3, kind, params, component) result(Dummy_7961 &
      )

  implicit none
  real(kind=8)  :: Dummy_7961  
  real(kind=8), value  :: xi1
  real(kind=8), value  :: xi2
  real(kind=8), value  :: xi3
  integer(kind=8), value  :: kind
  real(kind=8), intent(in)  :: params (0:)
  integer(kind=8), value  :: component
  real(kind=8)  :: value  
  real(kind=8)  :: Lx  
  real(kind=8)  :: Ly  
  real(kind=8)  :: Lz  
  real(kind=8)  :: calue  
  real(kind=8)  :: R1  
  real(kind=8)  :: R2  
  real(kind=8)  :: dR  
  real(kind=8)  :: alpha  

  !___________________________________________________________CommentBlock___________________________________________________________!
  !'''                                                                                                                               !
  !    returns the components of the inverse metric tensor (df)^(-1)*df^(-T) of an analytical mapping x = f(xi) in three dimensions. !
  !                                                                                                                                  !
  !    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella)                                                                        !
  !                                                                                                                                  !
  !    params    : slab            --> Lx, Ly, Lz                                                                                    !
  !              : hollow cylinder --> R1, R2, Lz                                                                                    !
  !              : colella         --> Lx, Ly, alpha, Lz                                                                             !
  !                                                                                                                                  !
  !    component : 11 (dfx/dxi1), 12 (dfx/dxi2), 13 (dfx/dxi3)                                                                       !
  !                21 (dfy/dxi1), 22 (dfy/dxi2), 23 (dfy/dxi3)                                                                       !
  !                31 (dfz/dxi1), 32 (dfz/dxi2), 33 (dfz/dxi3)                                                                       !
  !    '''                                                                                                                           !
  !__________________________________________________________________________________________________________________________________!


  value = 0.0d0


  if (kind == 1_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    Lz = params(2_8)


    if (component == 11_8 ) then
      value = Ly**2_8*Lz**2_8
    else if (component == 12_8 ) then
      value = 0.0d0
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      calue = 0.0d0
    else if (component == 22_8 ) then
      value = Lx**2_8*Lz**2_8
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = Lx**2_8*Ly**2_8
    end if
  else if (kind == 2_8 ) then


    R1 = params(0_8)
    R2 = params(1_8)
    Lz = params(2_8)
    dR = -R1 + R2


    if (component == 11_8 ) then
      value = (2_8*3.14159265358979d0)**2_8*(Lz**2_8*(R1 + dR*xi1)**2_8)
    else if (component == 12_8 ) then
      value = 0.0d0
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      value = 0.0d0
    else if (component == 22_8 ) then
      value = Lz**2_8*dR**2_8
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = dR**2_8*((2_8*3.14159265358979d0)**2_8*(R1 + dR*xi1)**2_8)
    end if
  else if (kind == 3_8 ) then


    Lx = params(0_8)
    Ly = params(1_8)
    alpha = params(2_8)
    Lz = params(3_8)


    if (component == 11_8 ) then
      value = Lx**2_8*(Lz**2_8*(alpha**2_8*(((2_8*3.14159265358979d0)** &
      2_8*cos(2_8*(3.14159265358979d0*xi2))**2_8)*sin(2_8*( &
      3.14159265358979d0*xi1))**2_8))) + Ly**2_8*(Lz**2_8*(alpha*(((2_8 &
      *3.14159265358979d0)*cos(2_8*(3.14159265358979d0*xi2)))*sin(2_8*( &
      3.14159265358979d0*xi1))) + 1_8)**2_8)
    else if (component == 12_8 ) then
      value = Lz**2_8*(-4_8*3.14159265358979d0**2_8*alpha**2_8*(Lx**2_8 &
      + Ly**2_8)*sin(2_8*(3.14159265358979d0*xi1))*sin(2_8*( &
      3.14159265358979d0*xi2))*cos(2_8*(3.14159265358979d0*xi1))*cos( &
      2_8*(3.14159265358979d0*xi2)) - 2_8*3.14159265358979d0*Lx**2_8* &
      alpha*sin(2_8*(3.14159265358979d0*xi1))*cos(2_8*( &
      3.14159265358979d0*xi2)) - 2_8*3.14159265358979d0*Ly**2_8*alpha* &
      sin(2_8*(3.14159265358979d0*xi2))*cos(2_8*(3.14159265358979d0*xi1 &
      )))
    else if (component == 13_8 ) then
      value = 0.0d0
    else if (component == 21_8 ) then
      value = Lz**2_8*(-4_8*3.14159265358979d0**2_8*alpha**2_8*(Lx**2_8 &
      + Ly**2_8)*sin(2_8*(3.14159265358979d0*xi1))*sin(2_8*( &
      3.14159265358979d0*xi2))*cos(2_8*(3.14159265358979d0*xi1))*cos( &
      2_8*(3.14159265358979d0*xi2)) - 2_8*3.14159265358979d0*Lx**2_8* &
      alpha*sin(2_8*(3.14159265358979d0*xi1))*cos(2_8*( &
      3.14159265358979d0*xi2)) - 2_8*3.14159265358979d0*Ly**2_8*alpha* &
      sin(2_8*(3.14159265358979d0*xi2))*cos(2_8*(3.14159265358979d0*xi1 &
      )))
    else if (component == 22_8 ) then
      value = Lx**2_8*(Lz**2_8*(alpha*(((2_8*3.14159265358979d0)*sin(2_8 &
      *(3.14159265358979d0*xi2)))*cos(2_8*(3.14159265358979d0*xi1))) + &
      1_8)**2_8) + Ly**2_8*(Lz**2_8*(alpha**2_8*(((2_8* &
      3.14159265358979d0)**2_8*sin(2_8*(3.14159265358979d0*xi2))**2_8)* &
      cos(2_8*(3.14159265358979d0*xi1))**2_8)))
    else if (component == 23_8 ) then
      value = 0.0d0
    else if (component == 31_8 ) then
      value = 0.0d0
    else if (component == 32_8 ) then
      value = 0.0d0
    else if (component == 33_8 ) then
      value = (Lx*(Ly*(alpha*(((2_8*3.14159265358979d0)*cos(2_8*( &
      3.14159265358979d0*xi2)))*sin(2_8*(3.14159265358979d0*xi1))) + &
      alpha*(((2_8*3.14159265358979d0)*sin(2_8*(3.14159265358979d0*xi2 &
      )))*cos(2_8*(3.14159265358979d0*xi1))) + 1_8)))**2_8
    end if
  end if
  Dummy_7961 = value/det_df(xi1, xi2, xi3, kind, params)**2_8
  return
end function
!........................................

end module