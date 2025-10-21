from inspect import signature

from psydac.linalg.block import BlockVector
from psydac.linalg.stencil import StencilVector

from struphy.ode.utils import ButcherTableau
from struphy.utils.arrays import xp


class ODEsolverFEEC:
    r"""
    Solver for FEEC coefficients based on explicit s-stage Runge-Kutta methods.

    .. image:: ../../pics/explicit_rk_methods.png
        :align: center
        :scale: 70%

    Parameters
    ----------
    vector_field : dict
        The vector field of the ODE as a dictionary.
        Keys are the variables to be updated (i.e. Stencil- or BlockVectors),
        values are callables representing the respective component of the vector field.
        That means dy_i/dt = f_i(y_1, ..., y_n) for i = 1,...,n, where n is the number of
        variables.

    algo : str
        See :class:`~struphy.ode.utils.ButcherTableau` for available algorithms.
    """

    def __init__(
        self,
        vector_field: dict,
        butcher: ButcherTableau = ButcherTableau(),
    ):
        # get algorithm
        self._butcher = butcher

        # check arguments and allocate k for each stage
        self._k = {}
        for vec, f in vector_field.items():
            assert isinstance(vec, (StencilVector, BlockVector))
            assert callable(f)
            sig = signature(f)
            assert len(sig.parameters) == len(vector_field) + 2, (
                "Number of arguments of each callable must match the number of unknows plus two (for time and out)."
            )

            self._k[vec] = []
            for s in range(self.butcher.n_stages):
                self._k[vec] += [vec.space.zeros()]

        self._vector_field = vector_field

        # collect unknows in list
        self._y = list(self.vector_field.keys())

        # allocate space for initial condition and intermediate values
        self._yn = [v.copy() for v in self.y]
        self._ystar = [v.copy() for v in self.y]

    def __call__(self, tn, h):
        a = self.butcher.a
        b = self.butcher.b
        c = self.butcher.c

        # keep initial condition
        for v, vn in zip(self.y, self.yn):
            v.copy(out=vn)

        # evaluate vector field for each stage
        for i in range(self.butcher.n_stages):
            # new intermediate y* (stored at self.y)
            for v, vn, vec in zip(self.y, self.yn, self.vector_field):
                # start with yn
                vn.copy(out=v)
                # add already computed k's
                for j in range(i):
                    v += h * a[i, j] * self.k[vec][j]
            # compute new k_i
            for vec, f in self.vector_field.items():
                self.k[vec][i] *= 0.0
                self.k[vec][i] += f(tn + c[i] * h, *self.y)

        # final addition, start with vn
        for v, vn, vec in zip(self.y, self.yn, self.vector_field):
            vn.copy(out=v)
            for i in range(self.butcher.n_stages):
                v += h * b[i] * self.k[vec][i]

    @property
    def vector_field(self):
        """The vector field of the ode as a dictionary.
        Keys are the variables to be updated (i.e. Stencil- or BlockVectors),
        values are callables representing the respective component of the vector field."""
        return self._vector_field

    @property
    def y(self):
        """List of variables to be updated."""
        return self._y

    @property
    def yn(self):
        """List of allocated space for initial conditions for each variable."""
        return self._yn

    @property
    def butcher(self):
        """See :class:`~struphy.ode.utils.ButcherTableau`."""
        return self._butcher

    @property
    def k(self):
        """Dictionary of k values for each stage;
        keys are the variables and values are lists with one allocated k-vector
        for each stage."""
        return self._k
