import numpy as np
#from struphy.fields_background.coil_fields import spline_interpolation_nd
from struphy.fields_background.coil_fields.coil_fields import RatGUI



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
    

def coil_fields_test():
    #file_path = "//wsl.localhost/Ubuntu/home/ohad/struphy/src/struphy/fields_background/tests/vector_field.csv"
    file_path = "~/struphy/src/struphy/fields_background/tests/vector_field.csv"
    ratGUI = RatGUI(file_path, [2,2,2], [False,False,False])
    xaxis = np.linspace(-10,10,30)
    yaxis = np.linspace(-10,10,30)
    result = ratGUI.b_xyz(xaxis,yaxis,0.)
    from matplotlib import pyplot as plt

    # plt.plot(xaxis,result[0][:,0,0])
    # plt.grid(True)
    # plt.show()
    print("test on")
    print(result[0].shape)
    z_index = 0  # Index corresponding to the Z value you want
    Bx_slice = result[0][:,:,z_index]
    By_slice = result[1][:,:,z_index]

    # Create a 2D plot
    plt.figure(figsize=(8, 6))
    plt.quiver(xaxis, xaxis, Bx_slice, By_slice, 
            scale=30, pivot='middle', color='blue')

    # Add labels and title
    plt.title(f"2D Slice of Vector Field at z_index = {z_index:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')  # Equal scaling for better visualization
    plt.grid(True)
    plt.show()

def coil_fields_test2():
    file_path = "~/struphy/src/struphy/fields_background/tests/polar_grid_data_B0_RAT.csv"
    ratGUI = RatGUI(file_path, [4,4,4], [False,True,False])
    xaxis = np.linspace(min(ratGUI._x),max(ratGUI._x),300)
    yaxis = np.linspace(min(ratGUI._y),max(ratGUI._y),50)
    zaxis = np.linspace(min(ratGUI._z),max(ratGUI._z),300)
    result = ratGUI.b_xyz(xaxis,yaxis,zaxis)
    from matplotlib import pyplot as plt

    # pick phi=0 slice
    Brho_slice = result[0][:,0,:]
    Bphi_slice = result[1][:,0,:]
    Bz_slice = result[2][:,0,:]

    Bmagnitude_slice = np.squeeze(np.sqrt(Brho_slice**2+Bphi_slice**2+Bz_slice**2))

    R, Z = np.meshgrid(xaxis, zaxis)     # Create a 2D grid
    # # Create a 2D plot
    # plt.figure(figsize=(8, 6))
    # contour = plt.contourf(R, Z, Bmagnitude_slice, levels=50, cmap='viridis')  # Filled contours
    # plt.colorbar(contour, label='B[T]]')  # Add a colorbar
    # plt.title('Filled Contour Plot of B[T]')
    # plt.xlabel('R')
    # plt.ylabel('Z')
    # plt.show()

    titles = ['|B|', r"$B_\rho$", r"$B_\phi$", r"$B_z$"]
    functions = [Bmagnitude_slice, np.squeeze(Brho_slice), np.squeeze(Bphi_slice), np.squeeze(Bz_slice)]
    # Create the subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid of subplots
    axes = axes.ravel()  # Flatten to 1D array for easier iteration

    # Loop through each function and plot
    for ax, func, title in zip(axes, functions, titles):
        contour = ax.contourf(R, Z, func, levels=50, cmap='viridis')  # Contourf plot
        fig.colorbar(contour, ax=ax)  # Add colorbar to each subplot
        ax.set_title(title)  # Add title
        ax.set_xlabel('R')  # Label x-axis
        ax.set_ylabel('Z')  # Label y-axis
    
    plt.show()

if __name__ == '__main__':
    coil_fields_test2()
