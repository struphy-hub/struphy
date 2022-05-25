def index_to_domain(gl_start, gl_end, pad, ind_mat, breaks):
    '''Transform the decomposition of spline indices into a domain decomposition.
    
    Parameters
    ----------
        gl_start : int
            Global start index on mpi process.
            
        gl_end : int
            Global end index on mpi process.
            
        pads : int
            Padding on mpi process (size of ghost region in spline coeffs).

        ind_mat : np.array
            2d array [element, spline index] of indices of non-vanishing splines in each element (or cell).
            Either DerhamBuild.indN_psy[n] or DerhamBuild.indD_psy[n].

        breaks : list
            Break points (=cell interfaces) in [0, 1].
            
    Returns
    -------
        Left and right boundary [le, ri] of local 1d domain.'''

    ind_le = gl_start
    ind_ri = gl_end + pad

    le = None
    ri = None
    for n in range(ind_mat.shape[0]):

        #print(f'rank : {rank}, ind_le/ri: {ind_le, ind_ri}, {ind_mat[n, :]}')

        if ind_le == ind_mat[n, 0]:
            le = breaks[n]
            #print(f'rank: {rank}, found!')

        if ind_ri == ind_mat[n, -1]:
            ri = breaks[n + 1]
            #print(f'rank: {rank}, found!')
            
    return le, ri