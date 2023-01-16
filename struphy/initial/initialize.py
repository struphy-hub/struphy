from struphy.kinetic_background import moments_kernels
from struphy.initial import perturbations

import numpy as np


class KineticPerturbation:
    '''Set initial conditions for a kinetic species by perturbing the background distribution.

    Parameters
    ----------
        background_params : dict
            From parameters.yml, see :ref:`kinetic`.

        perturb_params : dict
            From parameters.yml, see :ref:`kinetic`.

        marker_type : str
            'full_f' or 'delta_f'
    '''

    def __init__(self, marker_type, background_params, perturb_params):

        self.marker_type = marker_type
        self.back_type = background_params['type']
        self.moms_spec = background_params['moms_spec']
        self.moms_params = background_params['moms_params']

        self.p_type = perturb_params['type']

        if self.p_type is not None:
            self.p_moms = perturb_params['moms']
            self.p_params = perturb_params[self.p_type]
        else:
            self.p_moms = []
            self.p_params = {}

    def __call__(self, eta, v):
        '''
        Parameters
        ----------
            eta : array-like
                Logical position in [0, 1]^3.

            v : array-like
                Velocity in R^3.

        Returns
        -------
            The initial distribution evaluated at (eta1, eta2, eta3, vx, vy, vz).
        '''

        if self.marker_type == 'full_f':
            # unperturbed moments
            n0 = np.empty(len(eta[:, 0]), dtype=float)
            u0x = np.empty(len(eta[:, 0]), dtype=float)
            u0y = np.empty(len(eta[:, 0]), dtype=float)
            u0z = np.empty(len(eta[:, 0]), dtype=float)
            vth0x = np.empty(len(eta[:, 0]), dtype=float)
            vth0y = np.empty(len(eta[:, 0]), dtype=float)
            vth0z = np.empty(len(eta[:, 0]), dtype=float)

            moments_kernels.array_moments(np.array(eta),
                                          np.array(self.moms_spec),
                                          np.array(self.moms_params),
                                          n0, u0x, u0y, u0z, vth0x, vth0y, vth0z)

            # add perturbation
            if self.p_type is None:
                n = n0
                ux = u0x
                uy = u0y
                uz = u0z
                vthx = vth0x
                vthy = vth0y
                vthz = vth0z

            else:
                p_class = getattr(perturbations, self.p_type)
                perturb = p_class(**self.p_params)

                if 'n' in self.p_moms:
                    n = n0 + perturb

                if 'ux' in self.p_moms:
                    ux = u0x + perturb

                if 'uy' in self.p_moms:
                    uy = u0y + perturb

                if 'uz' in self.p_moms:
                    uz = u0z + perturb

                if 'vthx' in self.p_moms:
                    vthx = vth0x + perturb

                if 'vthy' in self.p_moms:
                    vthy = vth0y + perturb

                if 'vthz' in self.p_moms:
                    vthz = vth0z + perturb

            if self.back_type == 0:  # maxwellian_6d, see kinetic_background/f0_kernels.py

                Gx = np.exp(-(v[:, 0] - ux)**2 / (vthx**2)) / \
                    (np.sqrt(np.pi) * vthx)
                Gy = np.exp(-(v[:, 1] - uy)**2 / (vthy**2)) / \
                    (np.sqrt(np.pi) * vthy)
                Gz = np.exp(-(v[:, 2] - uz)**2 / (vthz**2)) / \
                    (np.sqrt(np.pi) * vthz)

                return n * Gx * Gy * Gz

            else:

                raise NotImplementedError(
                    f'Background of type {self.back_type} not implemented.')

        elif self.marker_type == 'delta_f':

            if self.p_type is None:
                return 0.

            else:
                p_class = getattr(perturbations, self.p_type)
                perturb = p_class(**self.p_params)

                return perturb

        else:
            raise NotImplementedError(
                f'Marker type {self.marker_type} is not implemented')
