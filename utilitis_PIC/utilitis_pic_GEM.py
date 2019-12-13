from pyccel.decorators import types
from pyccel.decorators import pure
from pyccel.decorators import external_call

#==============================================================================
@pure
@types('double[:]','double[:]','double[:]')
def cross(a, b, r):
    r[0] = a[1]*b[2] - a[2]*b[1]
    r[1] = a[2]*b[0] - a[0]*b[2]
    r[2] = a[0]*b[1] - a[1]*b[0]


#==============================================================================
@pure
@types('double[:]','int','double')
def find_span(knots, degree, x):
    """
    Determine the knot span index at location x, given the B-Splines' knot sequence and polynomial degree.

    For a degree p, the knot span index i identifies the indices [i-p:i] of all p+1 non-zero basis functions at a
    given location x.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Location of interest.

    Returns
    -------
    span : int
        Knot span index.

    """
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots) - 1 - degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high - 1
    else:
        # Perform binary search
        span = (low + high)//2
        while x < knots[span] or x >= knots[span + 1]:
            if x < knots[span]:
                high = span
            else:
                low  = span
            span = (low + high)//2
        returnVal = span

    return returnVal


#==============================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:]')
def basis_funs(knots, degree, x, span, left, right, values):
    """
    Compute the non-vanishing B-splines at location x, given the knot sequence, polynomial degree and knot span.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    Results
    -------
    values : numpy.ndarray
        Values of p+1 non-vanishing B-Splines at location x.

    """
    # to avoid degree being intent(inout)
    # TODO improve
    p = degree

#    from numpy      import empty
#    left   = empty( p  , dtype=float )
#    right  = empty( p  , dtype=float )
    left[:] = 0.
    right[:] = 0.

    values[0] = 1.0
    for j in range(0, p):
        left [j] = x - knots[span - j]
        right[j] = knots[span + 1 + j] - x
        saved    = 0.0
        for r in range(0, j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r]*temp
            saved     = left[j - r]*temp
        values[j + 1] = saved


#==============================================================================
@external_call
@types('double[:]','double[:,:](order=F)','double[:]','int','double','double[:]','double','int')
def hotCurrentRel_bc_1(particles_pos, particles_vel_w, knots, p, qe, jh, c, Nb):
    
    jh[:] = 0.

    # ... needed for splines evaluation
    from numpy      import empty
    from numpy      import zeros

    left   = empty(p,     dtype=float )
    right  = empty(p,     dtype=float )
    values = zeros(p + 1, dtype=float )

    span = 0
    # ...

    npart = len(particles_pos)

    #$ omp parallel
    #$ omp do reduction ( + : jh ) private ( ipart, pos, span, left, right, values, wk, vx, vy, vz, il, i, ii, bi )
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        span = find_span(knots, p, pos)
        basis_funs(knots, p, pos, span, left, right, values)

        wk = particles_vel_w[ipart, 3]
        
        vx = particles_vel_w[ipart, 0]
        vy = particles_vel_w[ipart, 1]
        vz = particles_vel_w[ipart, 2]
        
        gamma = (1 + (vx**2 + vy**2 + vz**2)/c**2)**(1/2)

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p - il]

            jh[2*ii] += vx*wk*bi/gamma
            jh[2*ii + 1] += vy*wk*bi/gamma

    #$ omp end do
    #$ omp end parallel

    jh = qe/npart*jh



#==============================================================================
@external_call
@types('int','double[:,:](order=F)','double','double','double','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','int','int','int','int','int','int','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double','double','double')
def boris_bc_1(npart, particles, dt, q, m, T1, T2, T3, tt1, tt2, tt3, p1, p2, p3, Nb1, Nb2, Nb3, e1, e2, e3, b1, b2, b3, B01, B02, B03):

    from numpy      import empty
    from numpy      import zeros

    Nl1 = empty(p1,     dtype=float)
    Nr1 = empty(p1,     dtype=float)
    N1  = zeros(p1 + 1, dtype=float)
    
    Nl2 = empty(p2,     dtype=float)
    Nr2 = empty(p2,     dtype=float)
    N2  = zeros(p2 + 1, dtype=float)
    
    Nl3 = empty(p3,     dtype=float)
    Nr3 = empty(p3,     dtype=float)
    N3  = zeros(p3 + 1, dtype=float)
    
    Dl1 = empty(p1 - 1, dtype=float)
    Dr1 = empty(p1 - 1, dtype=float)
    D1  = zeros(p1,     dtype=float)
    
    Dl2 = empty(p2 - 1, dtype=float)
    Dr2 = empty(p2 - 1, dtype=float)
    D2  = zeros(p2,     dtype=float)
    
    Dl3 = empty(p3 - 1, dtype=float)
    Dr3 = empty(p3 - 1, dtype=float)
    D3  = zeros(p3,     dtype=float)
    
    u   = zeros(3,      dtype=float)
    up  = zeros(3,      dtype=float)
    uxb = zeros(3,      dtype=float)
    tmp = zeros(3,      dtype=float)
    E   = zeros(3,      dtype=float)
    B   = zeros(3,      dtype=float)
    S   = zeros(3,      dtype=float)

    span1 = 0
    span2 = 0
    span3 = 0

    qprime = dt*q/(2*m)
    
    L1 = 1.
    L2 = 1.
    L3 = 1.
    
    delta1 = L1/Nb1
    delta2 = L2/Nb2
    delta3 = L3/Nb3

    
    for ip in range(0, npart):
        
        
        # ... field interpolation
        E[:] = 0.
        
        B[0] = B01
        B[1] = B02
        B[2] = B03

        pos1 = particles[ip, 0]
        pos2 = particles[ip, 1]
        pos3 = particles[ip, 2]
        
        span1 = find_span(T1, p1, pos1)
        span2 = find_span(T2, p2, pos2)
        span3 = find_span(T3, p3, pos3)
        
        basis_funs(T1, p1, pos1, span1, Nl1, Nr1, N1)
        basis_funs(T2, p2, pos2, span2, Nl2, Nr2, N2)
        basis_funs(T3, p3, pos3, span3, Nl3, Nr3, N3)
        
        basis_funs(tt1, p1 - 1, pos1, span1 - 1, Dl1, Dr1, D1)
        basis_funs(tt2, p2 - 1, pos2, span2 - 1, Dl2, Dr2, D2)
        basis_funs(tt3, p3 - 1, pos3, span3 - 1, Dl3, Dr3, D3)
        
            
        for il1 in range(0, p1):
            for il2 in range(0, p2 + 1):
                for il3 in range(0, p3 + 1):
                    
                    i1 = (span1 - 1 - il1)%Nb1
                    i2 = (span2 - il2)%Nb2
                    i3 = (span3 - il3)%Nb3
                    
                    E[0] += e1[i1, i2, i3] * D1[p1 - 1 - il1]/delta1 * N2[p2 - il2] * N3[p3 - il3]
                    
        
        for il1 in range(0, p1 + 1):
            for il2 in range(0, p2):
                for il3 in range(0, p3 + 1):
                    
                    i1 = (span1 - il1)%Nb1
                    i2 = (span2 - 1 - il2)%Nb2
                    i3 = (span3 - il3)%Nb3
                    
                    E[1] += e2[i1, i2, i3] * N1[p1 - il1] * D2[p2 - 1 - il2]/delta2 * N3[p3 - il3]
                    
        
        for il1 in range(0, p1 + 1):
            for il2 in range(0, p2 + 1):
                for il3 in range(0, p3):
                    
                    i1 = (span1 - il1)%Nb1
                    i2 = (span2 - il2)%Nb2
                    i3 = (span3 - 1 - il3)%Nb3
                    
                    E[2] += e3[i1, i2, i3] * N1[p1 - il1] * N2[p2 - il2] * D3[p3 - 1 - il3]/delta3 
                    
        
        for il1 in range(0, p1 + 1):
            for il2 in range(0, p2):
                for il3 in range(0, p3):
                    
                    i1 = (span1 - il1)%Nb1
                    i2 = (span2 - 1 - il2)%Nb2
                    i3 = (span3 - 1 - il3)%Nb3
                    
                    B[0] += b1[i1, i2, i3] * N1[p1 - il1] * D2[p2 - 1 - il2]/delta2 * D3[p3 - 1 - il3]/delta3
                    
                    
        for il1 in range(0, p1):
            for il2 in range(0, p2 + 1):
                for il3 in range(0, p3):
                    
                    i1 = (span1 - 1 - il1)%Nb1
                    i2 = (span2 - il2)%Nb2
                    i3 = (span3 - 1 - il3)%Nb3
                    
                    B[1] += b2[i1, i2, i3] * D1[p1 - 1 - il1]/delta1 * N2[p2 - il2] * D3[p3 - 1 - il3]/delta3
                    
                    
        for il1 in range(0, p1):
            for il2 in range(0, p2):
                for il3 in range(0, p3 + 1):
                    
                    i1 = (span1 - 1 - il1)%Nb1
                    i2 = (span2 - 1 - il2)%Nb2
                    i3 = (span3 - il3)%Nb3
                    
                    B[2] += b3[i1, i2, i3] * D1[p1 - 1 - il1]/delta1 * D2[p2 - 1 - il2]/delta2 * N3[p3 - il3]  
        # ...

        
        
        
        normB = B[0]**2 + B[1]**2 + B[2]**2
        r = 1 + (qprime**2)*normB
        S[:] = 2*qprime*B[:]/r
        u = particles[3:6, ip] + qprime*E[:]
        cross(u, B[:], uxb)
        uxb[:] = qprime*uxb[:]
        cross(u + uxb, S, tmp)
        up = u + tmp
        particles[3:6, ip] = up + qprime*E[:]
        particles[0:3, ip] += dt*particles[3:6, ip]
        particles[0:3, ip] = particles[0:3, ip]%L1 


    # TODO remove this line later, once fixed in pyccel
    ierr = 0