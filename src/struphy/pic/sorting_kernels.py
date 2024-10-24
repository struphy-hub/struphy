from numpy import floor

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
        x_l = domain_array[0]
        x_r = domain_array[1]
        y_l = domain_array[3]
        y_r = domain_array[4]
        z_l = domain_array[6]
        z_r = domain_array[7]
        boxes[:,:] = -1
        next_index[:] = 0  
        l = markers.shape[1]
        for p in range(markers.shape[0]):
            if holes[p]:
                n_box = nx*ny*nz
            else:
                a = floor((markers[p, 0]-x_l)/(x_r-x_l)*nx) + floor((markers[p, 1]-y_l)/(y_r-y_l)*ny)*nx + floor((markers[p, 2]-z_l)/(z_r-z_l)*nz)*ny*nx
                if a>=nx*ny*nz:
                    n_box = nx*ny*nz
                else:
                    n_box = int(a)
                    boxes[n_box, next_index[n_box]] = p
            next_index[n_box] += 1
            a = float(n_box)
            markers[p, l+box_index] = a

def get_next_index(box_nb : 'float',
                   next_index : 'int[:]',
                   cumul_next_index : 'int[:]'):
    """utilities for sorting """
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
            
