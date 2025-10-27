import copy

from struphy.fields_background.base import (
    CartesianFluidEquilibrium,
    CartesianFluidEquilibriumWithB,
    CartesianMHDequilibrium,
    LogicalFluidEquilibrium,
    LogicalFluidEquilibriumWithB,
    LogicalMHDequilibrium,
)


class GenericCartesianFluidEquilibrium(CartesianFluidEquilibrium):
    """Allows to pass callables at init."""

    def __init__(
        self,
        u_xyz: callable = None,
        p_xyz: callable = None,
        n_xyz: callable = None,
    ):
        # use params setter
        self.params = copy.deepcopy(locals())

        if u_xyz is None:
            u_xyz = lambda x, y, z: (0.0 * x, 0.0 * x, 0.0 * x)
        else:
            assert callable(u_xyz)

        if p_xyz is None:
            p_xyz = lambda x, y, z: 1.0 * x
        else:
            assert callable(p_xyz)

        if n_xyz is None:
            n_xyz = lambda x, y, z: 1.0 * x
        else:
            assert callable(n_xyz)

        self._u_xyz = u_xyz
        self._p_xyz = p_xyz
        self._n_xyz = n_xyz

    def u_xyz(self, x, y, z):
        return self._u_xyz(x, y, z)

    def p_xyz(self, x, y, z):
        return self._p_xyz(x, y, z)

    def n_xyz(self, x, y, z):
        return self._n_xyz(x, y, z)


class GenericCartesianFluidEquilibriumWithB(GenericCartesianFluidEquilibrium):
    """Allows to pass callables at init."""

    def __init__(
        self,
        u_xyz: callable = None,
        p_xyz: callable = None,
        n_xyz: callable = None,
        b_xyz: callable = None,
        gradB_xyz: callable = None,
    ):
        # use params setter
        self.params = copy.deepcopy(locals())

        super().__init__(u_xyz=u_xyz, p_xyz=p_xyz, n_xyz=n_xyz)

        if b_xyz is None:
            b_xyz = lambda x, y, z: (0.0 * x, 0.0 * x, 0.0 * x)
        else:
            assert callable(b_xyz)

        if gradB_xyz is None:
            gradB_xyz = lambda x, y, z: (0.0 * x, 0.0 * x, 0.0 * x)
        else:
            assert callable(gradB_xyz)

        self._b_xyz = b_xyz
        self._gradB_xyz = gradB_xyz

    def b_xyz(self, x, y, z):
        return self._b_xyz(x, y, z)

    def gradB_xyz(self, x, y, z):
        return self._gradB_xyz(x, y, z)
