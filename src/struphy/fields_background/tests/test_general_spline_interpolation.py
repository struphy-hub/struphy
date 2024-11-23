import numpy as np
from struphy.fields_background.coil_fields import spline_interpolation_nd


def test_spline_interpolation():

    ### Definition of grid ###

    # dimension of grid
    d = 3
    # define interval (bounds) for each axis
    intervals = [[-10,10] for _ in range(d)]
    # number of grid points per axis
    grid_size = 100
    # define random grid points along each axis
    grids_1d = np.array([intervals[i][0]+(intervals[i][1]-intervals[i][0])*np.sort(np.random.rand(grid_size)) for i in range(d)])
    for dim in range(d):
        grids_1d[dim,0]=intervals[dim][0]
    # create meshgrid
    meshgrid_tuple = np.meshgrid(*grids_1d, indexing='ij')
    
    ### Definition of function to interpolate ###
    # define values of function at interpolation points
    values = np.ones(meshgrid_tuple[0].shape)
    tmp = [np.sin(2*np.pi*meshgrid_tuple[i]/(intervals[i][1]-intervals[i][0])) for i in range(len(meshgrid_tuple))]
    for val in tmp:
        values *= val

    ### Get coefficients and knots of interpolated function ###
    # interpolate function
    p_order = [2 for _ in range(d)]
    spl_kind = [True for _ in range(d)]
    coeffs, T, indN = spline_interpolation_nd(p_order, spl_kind, grids_1d, values, intervals)
    print(coeffs.shape)

    ### Reconstruct function at regular grid ###
    new_grid_size = 200
    x_grids = np.array([linspace(intervals[i][0],intervals[i][1],new_grid_size) for i in range(d)])
    I_mat = [bsp.collocation_matrix(T[i], p_order[i], x_grids[i], spl_kind[i]) for i in range(d)]
    # I am not sure how to continue from this

if __name__ == '__main__':
    test_spline_interpolation()
