from struphy.feec.psydac_derham import Derham
from struphy.feec import preconditioner
from struphy.fields_background.mhd_equil.equils import set_defaults
import struphy.feec.utilities_kernels as util_kernels

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector
from psydac.linalg.solvers import inverse

import numpy as np


class L2_Projector:
    r"""
    A projector into the discrete de Rham spaces based on the L2-scalar product.
    
    Computes the L2 scalar product with basis functions via Gauss-Legendre quadrature:

    .. math::

        <f, \Lambda^\alpha_i> = \int_{[0,1]^3} f(\eta) \Lambda^\alpha_i(\eta) \, \text{d} \eta \approx \sum_k w_k f(\eta_k) \Lambda^\alpha_i(\eta_k)

    TODO (for now only a=0, i.e. space=H1)
    and solves the following system for the FE-coefficients

    .. math::

        M^\alpha_{ij} f_j = <f, \Lambda^\alpha_i>

    Parameters:
    -----------
    """

    def __init__(self, mass_mat, derham: Derham, **params):
        
        params_default = {
            'space': 'H1',
            'solver_params': {
                'type': ('pcg', 'MassMatrixPreconditioner'),
                'tol': 1.e-14,
                'maxiter': 3000,
                'info': False,
                'verbose': False,
            }
        }

        set_defaults(params, params_default)

        self._params = params
        self._solver_params = params['solver_params']

        # TODO generalize
        assert params['space'] == 'H1'
        self._space_id = params['space']

        assert isinstance(derham, Derham)
        self._derham = derham
        self._lhs_mat = mass_mat

        self._space_key = self.derham.space_to_form[self.space_id]
        self._space = derham.Vh_fem[self.space_key]

        if self.space_id in ("H1", "L2"):
            self._lhs = StencilVector(
                derham.Vh_fem[self.space_key].vector_space)
            self._rhs = StencilVector(
                derham.Vh_fem[self.space_key].vector_space)

        elif self.space_id in ("Hcurl", "Hdiv", "H1vec"):
            self._lhs = BlockVector(derham.Vh_fem[self.space_key].vector_space)
            self._rhs = BlockVector(derham.Vh_fem[self.space_key].vector_space)

        # Preconditioner
        if self.solver_params['type'][1] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, self.solver_params['type'][1])
            self._pc = pc_class(mass_mat)

        self._solver = inverse(mass_mat, 
                               self.solver_params['type'][0],
                               pc=self._pc,
                               tol=self.solver_params['tol'], 
                               maxiter=self.solver_params['maxiter'],
                               verbose=self.solver_params['verbose'])

        self._kernel = getattr(util_kernels, 'l2_projection_V0')

    @property
    def derham(self):
        return self._derham

    @property
    def space_id(self):
        """ The ID of the space, one of [H1, Hcurl, Hdiv, L2, H1vec] """
        return self._space_id

    @property
    def space_key(self):
        """ The key of the space, one of [0, 1, 2, 3, 4] """
        return self._space_key

    @property
    def space(self):
        return self._space

    @property
    def params(self):
        return self._params

    @property
    def params(self):
        return self._params

    @property
    def solver_params(self):
        return self._solver_params

    @property
    def lhs_mat(self):
        return self._lhs_mat

    def __call__(self, fun):
        """
        Performs the L2-projection of fun.

        Parameters
        ----------
        fun : callable or np.ndarray
            TODO
        """
        if callable(fun):
            locs, scaled_wts = self.generate_quad_points()
            etas = np.meshgrid(*locs, indexing='ij')

            fun_vals = fun(*etas)

            self._kernel(np.array(self.derham.p),
                         self.derham.Vh_fem['0'].knots[0],
                         self.derham.Vh_fem['0'].knots[1],
                         self.derham.Vh_fem['0'].knots[2],
                         np.array(self.derham.Vh['0'].starts),
                         self._rhs._data,
                         locs[0], locs[1], locs[2],
                         scaled_wts[0], scaled_wts[1], scaled_wts[2],
                         fun_vals)

            self._rhs.exchange_assembly_data()
            self._rhs.update_ghost_regions()

        elif isinstance(fun, np.ndarray):
            raise NotImplementedError('This option is not supported yet!')
        else:
            raise ValueError(
                f'fun must be either a callable function or a numpy array but is of type {type(fun)}')

        x = self._solver.solve(self._rhs, out=self._lhs)

        return x

    def generate_quad_points(self):
        """ Generate the quadrature points and weights for all cells.

        Returns
        -------
        locs : list of np.ndarrays
            array with eta-directions in first axis and flattened quad points along the second axis.

        scaled_weights : list of np.ndarrays
            array with eta-directions in first axis and flattened quad weights multiplied by cell length
            along the second axis.
        """
        pts = [None]*3
        wts = [None]*3

        for k in range(3):
            pts[k], wts[k] = np.polynomial.legendre.leggauss(
                self.derham.nquads[k])

        locs = [None]*3
        scaled_weights = [None]*3

        for k in range(3):
            amb = (
                self.derham.breaks_loc[k][1:] - self.derham.breaks_loc[k][:-1]) / 2
            apb = (
                self.derham.breaks_loc[k][1:] + self.derham.breaks_loc[k][:-1]) / 2

            locs[k] = (np.multiply.outer(amb, pts[k]) + apb[:, None]).ravel()
            scaled_weights[k] = np.multiply.outer(amb, wts[k]).ravel()

        return locs, scaled_weights
