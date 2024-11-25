import warnings

import numpy as np
import pandas as pd

from struphy.fields_background.coil_fields.base import (
    CartesianCoilField,
    LogicalCoilField,
)
from struphy.fields_background.coil_fields.base import spline_interpolation_nd
from struphy.geometry.base import Domain
from struphy.bsplines.evaluation_kernels_3d import evaluate_matrix

class RatGUI(CartesianCoilField):

    def __init__(
        self,
        file_path,
        spline_degree,
        spline_kind,
    ):
            # dimension of grid
            d = 3
            #
            x, y, z, Bx, By, Bz = self.read_vector_field(file_path)
            self._x = x
            self._y = y
            self._z = z
            # define interval (bounds) for each axis
            intervals = [[axis[0],axis[-1]+int(spl_kind)*np.abs((axis[1]-axis[0]))] for axis, spl_kind in zip([x,y,z], spline_kind)]
            # interpolate function
            self._p = spline_degree
            self._spl_kind = spline_kind
            self._coeffs_Bx, self._T_Bx, self._indN_Bx = spline_interpolation_nd(self._p, self._spl_kind, [x,y,z], Bx, intervals)
            self._coeffs_By, self._T_By, self._indN_By = spline_interpolation_nd(self._p, self._spl_kind, [x,y,z], By, intervals)
            self._coeffs_Bz, self._T_Bz, self._indN_Bz = spline_interpolation_nd(self._p, self._spl_kind, [x,y,z], Bz, intervals)


    def old_init(
        self,
        spline_degree,
        spline_kind,
    ):
            ## TODO - replace with imported csv file
            # dimension of grid
            d = 3
            # define interval (bounds) for each axis
            intervals = [[-10+i,10+i] for i in range(d)]
            # number of grid points per axis
            grid_size = 10
            # define random grid points along each axis
            grids_1d = np.array([intervals[i][0]+(intervals[i][1]-intervals[i][0])*np.sort(np.random.rand(grid_size)) for i in range(d)])
            for dim in range(d):
                grids_1d[dim,0]=intervals[dim][0]
            # create meshgrid
            meshgrid_tuple = np.meshgrid(*grids_1d, indexing='ij')
            print(intervals)
            print(grids_1d[0,:])
            
            ### Definition of function to interpolate ###
            # define values of function at interpolation points
            fun = lambda x, y, z: np.sin(2*np.pi*(x+10)/20)
            values = np.ones(meshgrid_tuple[0].shape)
            # values = fun(*meshgrid_tuple)
            tmp = [np.sin(2*np.pi*(meshgrid_tuple[i]-intervals[i][0])/(intervals[i][1]-intervals[i][0])) for i in range(len(meshgrid_tuple))]
            for val in tmp:
                values *= val
            
            ### Get coefficients and knots of interpolated function ###
            # interpolate function
            self._p = spline_degree
            self._spl_kind = spline_kind
            self._coeffs, self._T, self._indN = spline_interpolation_nd(self._p, self._spl_kind, grids_1d, values, intervals)

    def read_vector_field(self, file_path):
        """
        Reads a CSV file containing a 3D vector field, extracts sorted grids and field values.

        Parameters:
            file_path (str): Path to the input CSV file.

        Returns:
            unique_x, unique_y, unique_z: Sorted grid axes for x, y, and z coordinates (3D arrays).
            Bx, By, Bz: Corresponding vector field components (3D arrays).
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Apply numeric conversion only to rows with data
        df.iloc[:, :] = df.iloc[:, :].apply(pd.to_numeric, errors='coerce')

        # Extract the columns
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        z = df['z'].to_numpy()
        Bx = df['Bx'].to_numpy()
        By = df['By'].to_numpy()
        Bz = df['Bz'].to_numpy()

        # Sort the unique grid points for each axis
        unique_x = np.sort(np.unique(x))
        unique_y = np.sort(np.unique(y))
        unique_z = np.sort(np.unique(z))

        # Map the original data to the sorted indices
        shape = (len(unique_x), len(unique_y), len(unique_z))
        X, Y, Z = np.meshgrid(unique_x, unique_y, unique_z, indexing='ij')

        # Create a mapping from (x, y, z) to grid indices
        x_indices = np.searchsorted(unique_x, x)
        y_indices = np.searchsorted(unique_y, y)
        z_indices = np.searchsorted(unique_z, z)

        # Initialize empty arrays for the field components
        Bx_sorted = np.zeros(shape)
        By_sorted = np.zeros(shape)
        Bz_sorted = np.zeros(shape)

        # Populate the sorted field arrays
        Bx_sorted[x_indices, y_indices, z_indices] = Bx
        By_sorted[x_indices, y_indices, z_indices] = By
        Bz_sorted[x_indices, y_indices, z_indices] = Bz

        return unique_x, unique_y, unique_z, Bx_sorted, By_sorted, Bz_sorted
    
    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================

    # coil magnetic field
    def b_xyz(self, x, y, z):
        """ Magnetic field.
        """
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(x,y,z)
        kind = 0
        bx = np.zeros_like(E1)
        by = np.zeros_like(E1)
        bz = np.zeros_like(E1)
        print(E1.shape)
        print(type(E1))
        print(E1.dtype)

        T = self._T_Bx
        indN = self._indN_Bx
        coeffs = self._coeffs_Bx
        evaluate_matrix(T[0], T[1], T[2], self._p[0],  self._p[1], self._p[2], indN[0], indN[1], indN[2], coeffs, E1,E2,E3, bx, kind)
        
        T = self._T_By
        indN = self._indN_By
        coeffs = self._coeffs_By
        evaluate_matrix(T[0], T[1], T[2], self._p[0],  self._p[1], self._p[2], indN[0], indN[1], indN[2], coeffs, E1,E2,E3, by, kind)
        
        T = self._T_Bz
        indN = self._indN_Bz
        coeffs = self._coeffs_Bz
        evaluate_matrix(T[0], T[1], T[2], self._p[0],  self._p[1], self._p[2], indN[0], indN[1], indN[2], coeffs, E1,E2,E3, bz, kind)
        
        return bx, by, bz

    # coil magnetic field current (curl of equilibrium magnetic field)
    def j_xyz(self, x, y, z):
        """ Current density.
        """
        jx = 0*x
        jy = 0*x
        jz = 0*x

        return jx, jy, jz

    def gradB_xyz(self, x, y, z):
        """ Cartesian gradient of coil magnetic field in physical space. Must return the components as a tuple.
        """
        pass
