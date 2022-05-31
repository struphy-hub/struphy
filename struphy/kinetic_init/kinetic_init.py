class KineticInit6D:
    """
    TODO
    """

    def __init__(self, init, params, DOMAIN, EQ_Kinetic):
        self._init = init
        self._params = params
        self._DOMAIN = DOMAIN
        self._EQ_Kinetic = EQ_Kinetic

    # initial distribution function on logical domain (with possible density perturbation)
    def f0(self, eta1, eta2, eta3, vx, vy, vz):

        assert eta1.shape == eta2.shape == eta3.shape == vx.shape == vy.shape == vz.shape

        if self._init['coords'] == 'logical':
            value = self.nh_tot(eta1, eta2, eta3) * \
                self.EQ.fh0_eq(eta1, eta2, eta3, vx, vy, vz)

        elif self._init['coords'] == 'physical':

            # must do evaluation here, because pull needs an array as input (not a 6d callable)
            X = self._DOMAIN.evaluate(eta1, eta2, eta3, 'x', flat_eval=True)
            Y = self._DOMAIN.evaluate(eta1, eta2, eta3, 'y', flat_eval=True)
            Z = self._DOMAIN.evaluate(eta1, eta2, eta3, 'z', flat_eval=True)
            fun = self.nh_tot(X, Y, Z) * \
                self._EQ_Kinetic.fh_eq_phys(X, Y, Z, vx, vy, vz)

            value = self._DOMAIN.pull(
                fun, eta1, eta2, eta3, '0_form', flat_eval=True)

        else:
            raise ValueError('Coordinates for f0 not supported.')

        return value


class KineticInit5d:
    """
    TODO
    """
    def __init__(self):
        pass