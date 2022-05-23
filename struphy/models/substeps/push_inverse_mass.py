from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert
from struphy.psydac_api.preconditioner import MassMatrixPreConditioner as MassPre


class InvertMassMatrices:
    '''
    Invert the mass matrices M0, M1, M2 and M3 of the 3d de Rham complex.

    Parameters
    ----------    
        DR: obj
            From struphy/psydac_api/fields.Field_init.

        params: dict
            Solver parameters for this splitting step. 
    '''

    def __init__(self, DR, params):

        self._DR = DR
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._verbose = params['verbose']
        self._info = params['info']

        # Preconditioner
        if params['pc'] == None:
            _pc0 = None
            _pc1 = None
            _pc2 = None
            _pc3 = None
        elif params['pc'] == 'fft':
            _pc0 = MassPre(DR.V0)
            _pc1 = MassPre(DR.V1)
            _pc2 = MassPre(DR.V2)
            _pc3 = MassPre(DR.V3)
        else:
            raise ValueError(f'Preconditioner "{params["pc"]}" not implemented.')

        # Objects
        self._M0inv = Invert(self._DR.M0, pc=_pc0, tol=self._tol, maxiter=self._maxiter, verbose=self._verbose)
        self._M1inv = Invert(self._DR.M1, pc=_pc1, tol=self._tol, maxiter=self._maxiter, verbose=self._verbose)
        self._M2inv = Invert(self._DR.M2, pc=_pc2, tol=self._tol, maxiter=self._maxiter, verbose=self._verbose)
        self._M3inv = Invert(self._DR.M3, pc=_pc3, tol=self._tol, maxiter=self._maxiter, verbose=self._verbose)

    def __call__(self, v0, v1, v2, v3):

        # in place update
        v0[:] = self.step_M0inv(v0)[:]

        tmp1 = self.step_M1inv(v1)
        for v1i, tmp1i in zip(v1, tmp1):
            v1i[:] = tmp1i[:]

        tmp2 = self.step_M2inv(v2)
        for v2i, tmp2i in zip(v2, tmp2):
            v2i[:] = tmp2i[:]

        v3[:] = self.step_M3inv(v3)[:]

    def step_M0inv(self, v0):
        tmp = self._M0inv.dot(v0)
        if self._info:
            print('Status     for Invert_M0:', self._M0inv.info['success'])
            print('Iterations for Invert_M0:', self._M0inv.info['niter'])
            print()
        return tmp

    def step_M1inv(self, v1):
        tmp = self._M1inv.dot(v1)
        if self._info:
            print('Status     for Invert_M1:', self._M1inv.info['success'])
            print('Iterations for Invert_M1:', self._M1inv.info['niter'])
            print()
        return tmp

    def step_M2inv(self, v2):
        tmp = self._M2inv.dot(v2)
        if self._info:
            print('Status     for Invert_M0:', self._M2inv.info['success'])
            print('Iterations for Invert_M0:', self._M2inv.info['niter'])
            print()
        return tmp

    def step_M3inv(self, v3):
        tmp = self._M3inv.dot(v3)
        if self._info:
            print('Status     for Invert_M0:', self._M3inv.info['success'])
            print('Iterations for Invert_M0:', self._M3inv.info['niter'])
            print()
        return tmp
        

