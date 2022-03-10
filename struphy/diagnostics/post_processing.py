import numpy                    as np
from struphy.feec               import spline_space
from struphy.geometry           import domain_3d
from struphy.diagnostics        import data_module
import os

class Post_processing:
    """
    Evaluate the data from a simulation at specific points

    Parameters
    ----------
        params : dict
            parameters from parameters.yml 

        data : hdf5 file
            data of a simulation from data.hdf5
    """

    def __init__(self, params, data):
        self.params         = params
        self.data           = data
        
    def construct_FEEC_spaces(self, pts=None):

        """
        Construct the spaces object for the evaluation

        Parameters
        ----------
            pts : ndarray(dtype=float, ndim=3) [optional]
                Points for the evaluation in each direction (default is element boundaries).
        
        Notes
        -----
            If pts not given the variables from the SPACES object will be used.
        """


        # mesh parameters
        Nel             = self.params['grid']['Nel']
        spl_kind        = self.params['grid']['spl_kind']
        p               = self.params['grid']['p']
        nq_el           = self.params['grid']['nq_el']
        bc              = self.params['grid']['bc']

        # domain object
        domain_type     = self.params['geometry']['type']
        self.DOMAIN     = domain_3d.Domain(domain_type, self.params['geometry']['params_' + domain_type])

        # spline spaces
        spaces_FEM_1    = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], bc) 
        spaces_FEM_2    = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1])
        spaces_FEM_3    = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2], nq_el[2])

        # tensor spline space
        self.SPACES     = spline_space.Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

        # get time and space variables
        self.t          = self.data['time'][:].reshape((self.data['time'][:].shape[0], ))
        if pts is None:
            self.eta1       = self.SPACES.el_b[0]
            self.eta2       = self.SPACES.el_b[1]
            self.eta3       = self.SPACES.el_b[2]
        else:
            self.eta1       = pts[0]
            self.eta2       = pts[1]
            self.eta3       = pts[2]

        # get time and space sizes
        self.Nt         = int(self.t.shape[0])
        self.Nx         = int(self.eta1.shape[0])
        self.Ny         = int(self.eta2.shape[0])
        self.Nz         = int(self.eta3.shape[0])
        self.sizes      = np.array([self.Nt, self.Nx, self.Ny, self.Nz]).astype(int)

    def __evaluate_data(self, form, quantity):

        """
        Evaluate the data from a simulation for a given quantity and the associated form

        Parameters
        ----------
            quantity : string
                key for the data.hdf5 file where the quantity is saved

            form : string
                from of the desired quantity

        Returns
        -------
            eval_values: ndarray
                evaluated_values in the physical space at (t, eta1, eta2, eta3) for each coordinate

        Notes
        -----
            Possible choices for form:
                
                * '0_form', 
                * '1_form_1',
                * '1_form_2',
                * '2_form_3',
                * '2_form_1',
                * '2_form_2',
                * '2_form_3',
                * '3_form'
        """

        print('Evaluate ' + quantity + ' as ' + form)
    
        if form == '0_form':

            quantity        = self.data[quantity]
            eval_values     = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs                          = self.SPACES.extract_0(quantity[tn])
                eval_spline                     = self.SPACES.evaluate_NNN(self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_values[tn, :, :, :]        = self.DOMAIN.push(eval_spline, self.eta1, self.eta2, self.eta3, '0_form')

                frac    = int(np.ceil(((tn/self.Nt))*100))
                message = 'Processing: ' '{0:03}'.format(frac) + " %: "
                print('\r', message, end='')

            return eval_values

        elif form == '1_form_1':

            quantity        = self.data[quantity]
            eval_values     = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3    = self.SPACES.extract_1(quantity[tn])
                eval_spline_1                   = self.SPACES.evaluate_DNN(self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2                   = self.SPACES.evaluate_NDN(self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3                   = self.SPACES.evaluate_NND(self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :]        = self.DOMAIN.push([eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '1_form_1')

                frac    = int(np.ceil(((tn/self.Nt))*100))
                message = 'Processing: ' '{0:03}'.format(frac) + " %: "
                print('\r', message, end='')

            return eval_values

        elif form == '1_form_2':

            quantity        = self.data[quantity]
            eval_values     = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3    = self.SPACES.extract_1(quantity[tn])
                eval_spline_1                   = self.SPACES.evaluate_DNN(self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2                   = self.SPACES.evaluate_NDN(self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3                   = self.SPACES.evaluate_NND(self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :]        = self.DOMAIN.push([eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '1_form_2')

                frac    = int(np.ceil(((tn/self.Nt))*100))
                message = 'Processing: ' '{0:03}'.format(frac) + " %: "
                print('\r', message, end='')

            return eval_values

        elif form == '1_form_3':

            quantity        = self.data[quantity]
            eval_values     = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3    = self.SPACES.extract_1(quantity[tn])
                eval_spline_1                   = self.SPACES.evaluate_DNN(self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2                   = self.SPACES.evaluate_NDN(self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3                   = self.SPACES.evaluate_NND(self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :]        = self.DOMAIN.push([eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '1_form_3')

                frac    = int(np.ceil(((tn/self.Nt))*100))
                message = 'Processing: ' '{0:03}'.format(frac) + " %: "
                print('\r', message, end='')

            return eval_values

        elif form == '2_form_1':

            quantity        = self.data[quantity]
            eval_values     = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3    = self.SPACES.extract_2(quantity[tn])
                eval_spline_1                   = self.SPACES.evaluate_NDD(self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2                   = self.SPACES.evaluate_DND(self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3                   = self.SPACES.evaluate_DDN(self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :]        = self.DOMAIN.push([eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '2_form_1')

                frac    = int(np.ceil(((tn/self.Nt))*100))
                message = 'Processing: ' '{0:03}'.format(frac) + " %: "
                print('\r', message, end='')

            return eval_values

        elif form == '2_form_2':

            quantity        = self.data[quantity]
            eval_values     = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3    = self.SPACES.extract_2(quantity[tn])
                eval_spline_1                   = self.SPACES.evaluate_NDD(self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2                   = self.SPACES.evaluate_DND(self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3                   = self.SPACES.evaluate_DDN(self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :]        = self.DOMAIN.push([eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '2_form_2')

                frac    = int(np.ceil(((tn/self.Nt))*100))
                message = 'Processing: ' '{0:03}'.format(frac) + " %: "
                print('\r', message, end='')

            return eval_values

        elif form == '2_form_3':

            quantity        = self.data[quantity]
            eval_values     = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3    = self.SPACES.extract_2(quantity[tn])
                eval_spline_1                   = self.SPACES.evaluate_NDD(self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2                   = self.SPACES.evaluate_DND(self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3                   = self.SPACES.evaluate_DDN(self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :]        = self.DOMAIN.push([eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '2_form_3')

                frac    = int(np.ceil(((tn/self.Nt))*100))
                message = 'Processing: ' '{0:03}'.format(frac) + " %: "
                print('\r', message, end='')

            return eval_values

        elif form == '3_form':

            quantity        = self.data[quantity]
            eval_values     = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs                          = self.SPACES.extract_3(quantity[tn])
                eval_spline                     = self.SPACES.evaluate_DDD(self.eta1, self.eta2, self.eta3, coeffs)
                eval_values[tn, :, :, :]        = self.DOMAIN.push(eval_spline, self.eta1, self.eta2, self.eta3, '3_form')

                frac    = int(np.ceil(((tn/self.Nt))*100))
                message = 'Processing: ' '{0:03}'.format(frac) + " %: "
                print('\r', message, end='')

            return eval_values

        else:
            print('Wrong form')
            pass

    def save_evaluated_data(self, quantities, path_out=None, data_name=None):

        """
        Evaluate the data from the simulation for all quantities in quantities dict

        Parameters
        ----------
            quantities : dict
                dict where the keys are the quantities and the values the associated forms

            path_out : string
                path string where the data will be saved

            path_out : string
                name of the evaluated data
        """
        
        if path_out is None:    path_out = os.getcwd()
        if data_name is None:   data_name = 'eval_data.hdf5'

        name = {'0_form':'',
                '1_form_1':' x',
                '1_form_2':' y',
                '1_form_3':' z',
                '2_form_1':' x',
                '2_form_2':' y',
                '2_form_3':' z',
                '3_form':''}

        print(quantities)

        DATA = data_module.Data_container(path_out=path_out, data_name=data_name)

        for form, quantity in quantities.items():

            eval_values = self.__evaluate_data(form, quantity)
            
            DATA.add_data({quantity + name[form]  : eval_values})

        DATA.file.close()
 