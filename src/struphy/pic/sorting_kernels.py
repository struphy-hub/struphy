from numpy import floor

def flatten_index(n1 : 'int', 
                  n2 : 'int',
                  n3 : 'int', 
                  nx : 'int',
                  ny : 'int',
                  nz : 'int',):
    """Find the global index of a box based on its index in all 3 direction.
    At the moment this is simply sorted according to x then y then z but in the future 
    more evolved indexing could be implemented.
    """
    return n1 + n2*(nx+2) + n3*(nx+2)*(ny+2)

def initialize_neighbours(neighbour : 'int[:,:]',
                          nx : 'int',
                          ny : 'int',
                          nz : 'int',):
    """Initialize the list of neighbours of boxes (find the index of the 
    neighbours and put the in the right line of the neighbouring array)."""
    for k in range(nz+2):
        for j in range(ny+2):
            for i in range(nx+2):
                counter = 0
                loc_box = flatten_index(i, j, k, nx, ny, nz)
                for kk in range(k-1, k+2):
                    k_box = kk%(nz+2)
                    for jj in range(j-1, j+2):
                        j_box = jj%(ny+2)
                        for ii in range(i-1, i+2):
                            i_box = ii%(nx+2)
                            neig_box = flatten_index(i_box, j_box, k_box, nx, ny, nz)
                            neighbour[loc_box, counter] = neig_box
                            counter +=1

def find_box(eta1 : float,
             eta2 : float,
             eta3 : float,
             nx : 'int', 
             ny : 'int', 
             nz : 'int',
             domain_array : 'float[:]', 
             ):
    """Return the number of the box in which the point (eta1, eta2, eta3) is located, or -1 if the point is not on the process."""
    # Leave some room before and after the end of the domain for the box coming from neighbouring processes
    
    x_l = domain_array[0]-(domain_array[1]-domain_array[0])/nx
    x_r = domain_array[1]+(domain_array[1]-domain_array[0])/nx
    y_l = domain_array[3]-(domain_array[4]-domain_array[3])/ny
    y_r = domain_array[4]+(domain_array[4]-domain_array[3])/ny
    z_l = domain_array[6]-(domain_array[7]-domain_array[6])/nz
    z_r = domain_array[7]+(domain_array[7]-domain_array[6])/nz
    if eta1<x_l or eta1>x_r or eta2<y_l or eta2>y_r or eta3<z_l or eta3>z_r:
        return -1
    n1 = int(floor((eta1-x_l)/(x_r-x_l)*(nx+2)))
    n2 = int(floor((eta2-y_l)/(y_r-y_l)*(ny+2)))
    n3 = int(floor((eta3-z_l)/(z_r-z_l)*(nz+2)))
    return flatten_index(n1, n2 ,n3, nx, ny, nz)

def put_particles_in_boxes(markers : 'float[:,:]', 
                           holes : 'bool[:]', 
                           nx : 'int', 
                           ny : 'int', 
                           nz : 'int',
                           boxes : 'int[:,:]', 
                           next_index : 'int[:]',
                           domain_array : 'float[:]', 
                           box_index : 'int' =-2):
    """Assign the right box to all particles."""
    boxes[:,:] = -1
    next_index[:] = 0  
    l = markers.shape[1]
    for p in range(markers.shape[0]):
        if holes[p]:
            n_box = (nx+2)*(ny+2)*(nz+2)
        else:
            a = find_box(markers[p, 0],markers[p, 1],markers[p, 2],nx,ny,nz,domain_array)
            if a>=(nx+2)*(ny+2)*(nz+2) or a<0:
                n_box = (nx+2)*(ny+2)*(nz+2)
            else:
                n_box = a
                boxes[n_box, next_index[n_box]] = p
        next_index[n_box] += 1
        b = float(n_box)
        markers[p, l+box_index] = b

def get_next_index(box_nb : 'float',
                   next_index : 'int[:]',
                   cumul_next_index : 'int[:]'):
    """Utilities for sorting """
    int_bnb = int(box_nb)
    index = cumul_next_index[int_bnb]+next_index[int_bnb]
    next_index[int_bnb] +=1
    return index

def sort_boxed_particles(markers : 'float[:,:]', 
                         swap_line_1 : 'float[:]', 
                         swap_line_2 : 'float[:]', 
                         n_boxes : 'int', 
                         next_index : 'int[:]', 
                         cumul_next_index : 'int[:]',
                         box_index : 'int' =-2):
    """Sort the particules taking advantage of the boxes"""
    c = 0
    for i in range(next_index.shape[0]):
        cumul_next_index[i] = c
        c += next_index[i] 
    
    cumul_next_index[-1] = c
    next_index[:] = 0
    l = len(swap_line_2)
    loc_i = 0
    
    for box in range(n_boxes):
        # skip the ones that are already in good position
        loc_i+=next_index[box]
        # loop through the rest of the box
        while loc_i<cumul_next_index[box+1]:
            box_i = int(markers[loc_i,l+box_index])
            if box_i==box:
                # is it in good position? if yes go to the next one 
                # and say that this position is no longer available
                loc_i+=1
                next_index[box]+=1
            else:
                # swap elements until comming back
                swap_i = -1
                swap_line_1[:] = markers[loc_i,:]
                while swap_i != loc_i:
                    # where should the actual particule go, put it and save data from the particle that should now move
                    swap_i = get_next_index(swap_line_1[l+box_index],next_index,cumul_next_index)
                    swap_line_2[:] = markers[swap_i,:]
                    markers[swap_i,:] = swap_line_1[:]
                    swap_line_1[:] = swap_line_2[:]
                loc_i+=1

def reassigne_boxes(markers : 'float[:,:]', 
                    holes : 'bool[:]',
                    boxes : 'int[:,:]', 
                    next_index : 'int[:]',
                    box_index : 'int' =-2):
    """Reloop over the particles after communication to update the neighbouring boxes 
    with the right particles and the next_index for later sorting."""
    boxes[:,:] = -1
    next_index[:] = 0  
    l = markers.shape[1]
    for p in range(markers.shape[0]):
        if holes[p]:
            continue
        else:
            #print(markers[p,l+box_index])
            a = int(markers[p,l+box_index])
            boxes[a, next_index[a]] = p
            next_index[a] += 1
