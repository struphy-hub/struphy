"Base classes for MHD equilibria."

from abc import ABCMeta, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from pyevtk.hl import gridToVTK

from struphy.geometry.base import Domain


class FluidEquilibrium(metaclass=ABCMeta):
    """
    Base class for callable fluid equilibria consisting of at least
    u (velocity), p (pressure) and n (density).

    Any child class must provide  the following callables:

    * either ``u_xyz`` or override ``uv``
    * either ``p_xyz`` or override ``p0``
    * either ``n_xyz`` or override ``n0``
    """

    @property
    def params(self):
        """Parameters dictionary."""
        return self._params

    def set_params(self, **params):
        """Generates self.params dictionary from keyword arguments."""
        self._params = params

    @property
    def domain(self):
        """Domain object that characterizes the mapping from the logical to the physical domain."""
        assert hasattr(self, "_domain"), (
            "Domain for FluidEquilibrium not set; do obj.domain = ... to have access to all transformations."
        )
        return self._domain

    @domain.setter
    def domain(self, new_domain):
        assert isinstance(new_domain, Domain) or new_domain is None
        self._domain = new_domain

    ###########################
    # Vector-valued callables #
    ###########################

    def u1(self, *etas, squeeze_out=False):
        """1-form components of velocity on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.uv(*etas, squeeze_out=False),
            *etas,
            kind="v_to_1",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def u2(self, *etas, squeeze_out=False):
        """2-form components of velocity on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.uv(*etas, squeeze_out=False),
            *etas,
            kind="v_to_2",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def uv(self, *etas, squeeze_out=False):
        """Contra-variant components of velocity on logical cube [0, 1]^3."""
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.u_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind="v", squeeze_out=squeeze_out)

    def u_cart(self, *etas, squeeze_out=False):
        """Cartesian components of velocity evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        out = self.domain.push(
            self.uv(*etas, squeeze_out=False),
            *etas,
            kind="v",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )
        return out, self.domain(*etas, squeeze_out=squeeze_out)

    ###########################
    # Scalar-valued callables #
    ###########################

    def p0(self, *etas, squeeze_out=False):
        """0-form pressure on logical cube [0, 1]^3."""
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.p_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind="0", squeeze_out=squeeze_out)

    def p3(self, *etas, squeeze_out=False):
        """3-form pressure on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.p0(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def n0(self, *etas, squeeze_out=False):
        """0-form number density on logical cube [0, 1]^3."""
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.n_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind="0", squeeze_out=squeeze_out)

    def n3(self, *etas, squeeze_out=False):
        """3-form number density on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.n0(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def t0(self, *etas, squeeze_out=False):
        """0-form temperature on logical cube [0, 1]^3."""
        return self.p0(*etas, squeeze_out=squeeze_out) / self.n0(*etas, squeeze_out=squeeze_out)

    def t3(self, *etas, squeeze_out=False):
        """3-form temperature on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.t0(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def vth0(self, *etas, squeeze_out=False):
        """0-form thermal velocity on logical cube [0, 1]^3."""
        return np.sqrt(self.t0(*etas, squeeze_out=squeeze_out))

    def vth3(self, *etas, squeeze_out=False):
        """3-form thermal velocity on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.vth0(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def q0(self, *etas, squeeze_out=False):
        """0-form square root of the pressure on logical cube [0, 1]^3."""
        # xyz = self.domain(*etas, squeeze_out=False)
        p = self.p0(*etas)
        q = np.sqrt(p)
        return self.domain.pull(q, *etas, kind="0", squeeze_out=squeeze_out)

    def q3(self, *etas, squeeze_out=False):
        """3-form square root of the pressure on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.q0(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def s0_monoatomic(self, *etas, squeeze_out=False):
        """0-form entropy density on logical cube [0, 1]^3.
        Hard coded assumption : gamma = 5/3 (monoatomic perfect gaz)
        """
        # xyz = self.domain(*etas, squeeze_out=False)
        p = self.p0(*etas)
        n = self.n0(*etas)
        s = n * np.log(p / (2 / 3 * np.power(n, 5 / 3)))
        return self.domain.pull(s, *etas, kind="0", squeeze_out=squeeze_out)

    def s3_monoatomic(self, *etas, squeeze_out=False):
        """3-form entropy density on logical cube [0, 1]^3.
        Hard coded assumption : gamma = 5/3 (monoatomic perfect gaz)
        """
        return self.domain.transform(
            self.s0_monoatomic(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def s0_diatomic(self, *etas, squeeze_out=False):
        """0-form entropy density on logical cube [0, 1]^3.
        Hard coded assumption : gamma = 7/5 (diatomic perfect gaz)
        """
        # xyz = self.domain(*etas, squeeze_out=False)
        p = self.p0(*etas)
        n = self.n0(*etas)
        s = n * np.log(p / (2 / 5 * np.power(n, 7 / 5)))
        return self.domain.pull(s, *etas, kind="0", squeeze_out=squeeze_out)

    def s3_diatomic(self, *etas, squeeze_out=False):
        """3-form entropy density on logical cube [0, 1]^3.
        Hard coded assumption : gamma = 5/3 (monoatomic perfect gaz)
        """
        return self.domain.transform(
            self.s0_diatomic(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    #########################################################
    # Single components (for input of commuting projectors) #
    #########################################################

    def u1_1(self, *etas, squeeze_out=False):
        return self.u1(*etas, squeeze_out=squeeze_out)[0]

    def u1_2(self, *etas, squeeze_out=False):
        return self.u1(*etas, squeeze_out=squeeze_out)[1]

    def u1_3(self, *etas, squeeze_out=False):
        return self.u1(*etas, squeeze_out=squeeze_out)[2]

    def u2_1(self, *etas, squeeze_out=False):
        return self.u2(*etas, squeeze_out=squeeze_out)[0]

    def u2_2(self, *etas, squeeze_out=False):
        return self.u2(*etas, squeeze_out=squeeze_out)[1]

    def u2_3(self, *etas, squeeze_out=False):
        return self.u2(*etas, squeeze_out=squeeze_out)[2]

    def uv_1(self, *etas, squeeze_out=False):
        return self.uv(*etas, squeeze_out=squeeze_out)[0]

    def uv_2(self, *etas, squeeze_out=False):
        return self.uv(*etas, squeeze_out=squeeze_out)[1]

    def uv_3(self, *etas, squeeze_out=False):
        return self.uv(*etas, squeeze_out=squeeze_out)[2]

    def u_cart_1(self, *etas, squeeze_out=False):
        return self.u_cart(*etas, squeeze_out=squeeze_out)[0][0]

    def u_cart_2(self, *etas, squeeze_out=False):
        return self.u_cart(*etas, squeeze_out=squeeze_out)[0][1]

    def u_cart_3(self, *etas, squeeze_out=False):
        return self.u_cart(*etas, squeeze_out=squeeze_out)[0][2]


class CartesianFluidEquilibrium(FluidEquilibrium):
    r"""
    The callables ``u_xyz``, ``p_xyz`` and ``n_xyz`` must be provided in Cartesian coordinates.
    """

    @abstractmethod
    def u_xyz(self, x, y, z):
        """Cartesian velocity in physical space.
        Must return the components as a tuple."""
        pass

    @abstractmethod
    def p_xyz(self, x, y, z):
        """Equilibrium pressure in physical space."""
        pass

    @abstractmethod
    def n_xyz(self, x, y, z):
        """Equilibrium number density in physical space."""
        pass

    @FluidEquilibrium.domain.setter
    def domain(self, new_domain):
        super(CartesianFluidEquilibrium, type(self)).domain.fset(self, new_domain)


class LogicalFluidEquilibrium(FluidEquilibrium):
    r"""
    The callables ``uv``, ``p0`` and ``n0`` must be provided on the logical cube [0, 1]^3.
    """

    @abstractmethod
    def uv(self, *etas, squeeze_out=False):
        """Contra-variant (vector field) velocity on logical cube [0, 1]^3.
        Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def p0(self, *etas, squeeze_out=False):
        """0-form pressure on logical cube [0, 1]^3."""
        pass

    @abstractmethod
    def n0(self, *etas, squeeze_out=False):
        """0-form density on logical cube [0, 1]^3."""
        pass

    @FluidEquilibrium.domain.setter
    def domain(self, new_domain):
        super(LogicalFluidEquilibrium, type(self)).domain.fset(self, new_domain)


class NumericalFluidEquilibrium(LogicalFluidEquilibrium):
    r"""
    Must provide a (numerical) mapping from the logical cube [0, 1]^3 to the physical domain.
    Overrides base class domain.
    """

    @property
    @abstractmethod
    def numerical_domain(self):
        """Numerically computed mapping from the logical cube [0, 1]^3 to the physical domain
        in the form of a :class:`~struphy.geometry.base.Domain` object."""
        pass

    @property
    def domain(self):
        return self.numerical_domain


class FluidEquilibriumWithB(FluidEquilibrium):
    """
    :ref:`FluidEquilibrium` with B (magnetic field) in addition.

    Any child class must provide the following callables:

    * either ``b_xyz`` or override ``bv``
    * either ``gradB_xyz`` or override ``gradB1``
    """

    @FluidEquilibrium.domain.setter
    def domain(self, new_domain):
        super(FluidEquilibriumWithB, type(self)).domain.fset(self, new_domain)

    ###########################
    # Vector-valued callables #
    ###########################

    def b1(self, *etas, squeeze_out=False):
        """1-form components of magnetic field on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.bv(*etas, squeeze_out=False),
            *etas,
            kind="v_to_1",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def b2(self, *etas, squeeze_out=False):
        """2-form components of magnetic field on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.bv(*etas, squeeze_out=False),
            *etas,
            kind="v_to_2",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def bv(self, *etas, squeeze_out=False):
        """Contra-variant components of  magnetic field on logical cube [0, 1]^3."""
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.b_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind="v", squeeze_out=squeeze_out)

    def b_cart(self, *etas, squeeze_out=False):
        """Cartesian components of magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        b_out = self.domain.push(
            self.bv(*etas, squeeze_out=False),
            *etas,
            kind="v",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )
        return b_out, self.domain(*etas, squeeze_out=squeeze_out)

    def unit_b1(self, *etas, squeeze_out=False):
        """Unit vector components of magnetic field (1-form) on logical cube [0, 1]^3."""
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind="1", squeeze_out=squeeze_out)

    def unit_b2(self, *etas, squeeze_out=False):
        """Unit vector components of magnetic field (2-form) on logical cube [0, 1]^3."""
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind="2", squeeze_out=squeeze_out)

    def unit_bv(self, *etas, squeeze_out=False):
        """Unit vector components of magnetic field (contra-variant) on logical cube [0, 1]^3."""
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind="v", squeeze_out=squeeze_out)

    def unit_b_cart(self, *etas, squeeze_out=False):
        """Unit vector Cartesian components of magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        absB = self.absB0(*etas, squeeze_out=squeeze_out)
        out = np.array([b[0] / absB, b[1] / absB, b[2] / absB], dtype=float)
        return out, xyz

    def gradB1(self, *etas, squeeze_out=False):
        """1-form components of gradient of magnetic field strength evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.gradB_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind="1", squeeze_out=squeeze_out)

    def gradB2(self, *etas, squeeze_out=False):
        """2-form components of gradient of magnetic field strength evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        return self.domain.transform(
            self.gradB1(*etas, squeeze_out=False),
            *etas,
            kind="1_to_2",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def gradBv(self, *etas, squeeze_out=False):
        """Contra-variant components of gradient of magnetic field strength evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        return self.domain.transform(
            self.gradB1(*etas, squeeze_out=False),
            *etas,
            kind="1_to_v",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def gradB_cart(self, *etas, squeeze_out=False):
        """Cartesian components of gradient of magnetic field strength evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        gradB_out = self.domain.push(
            self.gradB1(*etas, squeeze_out=False),
            *etas,
            kind="1",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )
        return gradB_out, self.domain(*etas)

    def a1(self, *etas, squeeze_out=False):
        """1-form components of vector potential on logical cube [0, 1]^3."""
        avail_list = ["HomogenSlab"]
        assert self.__class__.__name__ in avail_list, (
            f'Vector potential currently available only for {avail_list}, but mhd_equil is "{self.__class__.__name__}".'
        )

        return self.domain.transform(
            self.a2(*etas, squeeze_out=False),
            *etas,
            kind="2_to_1",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def a2(self, *etas, squeeze_out=False):
        """2-form components of vector potential on logical cube [0, 1]^3."""
        avail_list = ["HomogenSlab"]
        assert self.__class__.__name__ in avail_list, (
            f'Vector potential currently available only for {avail_list}, but mhd_equil is "{self.__class__.__name__}".'
        )

        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.a_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind="2", squeeze_out=squeeze_out)

    def av(self, *etas, squeeze_out=False):
        """Contra-variant components of vector potneital on logical cube [0, 1]^3."""
        avail_list = ["HomogenSlab"]
        assert self.__class__.__name__ in avail_list, (
            f'Vector potential currently available only for {avail_list}, but mhd_equil is "{self.__class__.__name__}".'
        )

        return self.domain.transform(
            self.a2(*etas, squeeze_out=False),
            *etas,
            kind="2_to_v",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    ###########################
    # Scalar-valued callables #
    ###########################

    def absB0(self, *etas, squeeze_out=False):
        """0-form absolute value of magnetic field on logical cube [0, 1]^3."""
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        return np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)

    def absB3(self, *etas, squeeze_out=False):
        """3-form absolute value of magnetic field on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.absB0(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def u_para0(self, *etas, squeeze_out=False):
        """0-form parallel velocity on logical cube [0, 1]^3."""
        tmp_uv = self.uv(*etas, squeeze_out=squeeze_out)
        tmp_unit_b1 = self.unit_b1(*etas, squeeze_out=squeeze_out)
        return sum([ji * bi for ji, bi in zip(tmp_uv, tmp_unit_b1)])

    def u_para3(self, *etas, squeeze_out=False):
        """3-form parallel velocity on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.u_para0(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    #########################################################
    # Single components (for input of commuting projectors) #
    #########################################################

    def b1_1(self, *etas, squeeze_out=False):
        return self.b1(*etas, squeeze_out=squeeze_out)[0]

    def b1_2(self, *etas, squeeze_out=False):
        return self.b1(*etas, squeeze_out=squeeze_out)[1]

    def b1_3(self, *etas, squeeze_out=False):
        return self.b1(*etas, squeeze_out=squeeze_out)[2]

    def b2_1(self, *etas, squeeze_out=False):
        return self.b2(*etas, squeeze_out=squeeze_out)[0]

    def b2_2(self, *etas, squeeze_out=False):
        return self.b2(*etas, squeeze_out=squeeze_out)[1]

    def b2_3(self, *etas, squeeze_out=False):
        return self.b2(*etas, squeeze_out=squeeze_out)[2]

    def bv_1(self, *etas, squeeze_out=False):
        return self.bv(*etas, squeeze_out=squeeze_out)[0]

    def bv_2(self, *etas, squeeze_out=False):
        return self.bv(*etas, squeeze_out=squeeze_out)[1]

    def bv_3(self, *etas, squeeze_out=False):
        return self.bv(*etas, squeeze_out=squeeze_out)[2]

    def unit_b1_1(self, *etas, squeeze_out=False):
        return self.unit_b1(*etas, squeeze_out=squeeze_out)[0]

    def unit_b1_2(self, *etas, squeeze_out=False):
        return self.unit_b1(*etas, squeeze_out=squeeze_out)[1]

    def unit_b1_3(self, *etas, squeeze_out=False):
        return self.unit_b1(*etas, squeeze_out=squeeze_out)[2]

    def unit_b2_1(self, *etas, squeeze_out=False):
        return self.unit_b2(*etas, squeeze_out=squeeze_out)[0]

    def unit_b2_2(self, *etas, squeeze_out=False):
        return self.unit_b2(*etas, squeeze_out=squeeze_out)[1]

    def unit_b2_3(self, *etas, squeeze_out=False):
        return self.unit_b2(*etas, squeeze_out=squeeze_out)[2]

    def unit_bv_1(self, *etas, squeeze_out=False):
        return self.unit_bv(*etas, squeeze_out=squeeze_out)[0]

    def unit_bv_2(self, *etas, squeeze_out=False):
        return self.unit_bv(*etas, squeeze_out=squeeze_out)[1]

    def unit_bv_3(self, *etas, squeeze_out=False):
        return self.unit_bv(*etas, squeeze_out=squeeze_out)[2]

    def gradB1_1(self, *etas, squeeze_out=False):
        return self.gradB1(*etas, squeeze_out=squeeze_out)[0]

    def gradB1_2(self, *etas, squeeze_out=False):
        return self.gradB1(*etas, squeeze_out=squeeze_out)[1]

    def gradB1_3(self, *etas, squeeze_out=False):
        return self.gradB1(*etas, squeeze_out=squeeze_out)[2]

    def gradB2_1(self, *etas, squeeze_out=False):
        return self.gradB2(*etas, squeeze_out=squeeze_out)[0]

    def gradB2_2(self, *etas, squeeze_out=False):
        return self.gradB2(*etas, squeeze_out=squeeze_out)[1]

    def gradB2_3(self, *etas, squeeze_out=False):
        return self.gradB2(*etas, squeeze_out=squeeze_out)[2]

    def gradBv_1(self, *etas, squeeze_out=False):
        return self.gradBv(*etas, squeeze_out=squeeze_out)[0]

    def gradBv_2(self, *etas, squeeze_out=False):
        return self.gradBv(*etas, squeeze_out=squeeze_out)[1]

    def gradBv_3(self, *etas, squeeze_out=False):
        return self.gradBv(*etas, squeeze_out=squeeze_out)[2]

    def a1_1(self, *etas, squeeze_out=False):
        return self.a1(*etas, squeeze_out=squeeze_out)[0]

    def a1_2(self, *etas, squeeze_out=False):
        return self.a1(*etas, squeeze_out=squeeze_out)[1]

    def a1_3(self, *etas, squeeze_out=False):
        return self.a1(*etas, squeeze_out=squeeze_out)[2]

    def a2_1(self, *etas, squeeze_out=False):
        return self.a2(*etas, squeeze_out=squeeze_out)[0]

    def a2_2(self, *etas, squeeze_out=False):
        return self.a2(*etas, squeeze_out=squeeze_out)[1]

    def a2_3(self, *etas, squeeze_out=False):
        return self.a2(*etas, squeeze_out=squeeze_out)[2]

    def av_1(self, *etas, squeeze_out=False):
        return self.av(*etas, squeeze_out=squeeze_out)[0]

    def av_2(self, *etas, squeeze_out=False):
        return self.av(*etas, squeeze_out=squeeze_out)[1]

    def av_3(self, *etas, squeeze_out=False):
        return self.av(*etas, squeeze_out=squeeze_out)[2]


class CartesianFluidEquilibriumWithB(CartesianFluidEquilibrium):
    r"""
    The callables ``b_xyz`` and ``gradB_xyz`` must be provided in Cartesian coordinates.
    """

    @abstractmethod
    def b_xyz(self, x, y, z):
        """Cartesian magnetic field in physical space.
        Must return the components as a tuple."""
        pass

    @abstractmethod
    def gradB_xyz(self, x, y, z):
        """Cartesian gradient of magnetic field strength in physical space. Must return the components as a tuple."""
        pass

    @CartesianFluidEquilibrium.domain.setter
    def domain(self, new_domain):
        super(CartesianFluidEquilibriumWithB, type(self)).domain.fset(self, new_domain)


class LogicalFluidEquilibriumWithB(LogicalFluidEquilibrium):
    r"""
    The callable ``bv`` must be provided on the logical cube [0, 1]^3.
    """

    @abstractmethod
    def bv(self, *etas, squeeze_out=False):
        """Contra-variant (vector field) magnetic field on logical cube [0, 1]^3.
        Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def gradB1(self, *etas, squeeze_out=False):
        """Co-variant (1-from) gradient of magnetic field strength on logical cube [0, 1]^3.
        Must return the components as a tuple.
        """
        pass

    @LogicalFluidEquilibrium.domain.setter
    def domain(self, new_domain):
        super(LogicalFluidEquilibriumWithB, type(self)).domain.fset(self, new_domain)


class NumericalFluidEquilibriumWithB(LogicalFluidEquilibriumWithB):
    r"""
    Must provide a (numerical) mapping from the logical cube [0, 1]^3 to the physical domain.
    Overrides base class domain.
    """

    @property
    @abstractmethod
    def numerical_domain(self):
        """Numerically computed mapping from the logical cube [0, 1]^3 to the physical domain
        in the form of a :class:`~struphy.geometry.base.Domain` object."""
        pass

    @property
    def domain(self):
        return self.numerical_domain


class MHDequilibrium(FluidEquilibriumWithB):
    """
    :ref:`FluidEquilibriumWithB` with j (current density) in addition.
    The mean velocity is returned as j/n (overriding the base class).

    Any child class must provide  the following callables:

    * either ``j_xyz`` or override ``jv``
    """

    @FluidEquilibriumWithB.domain.setter
    def domain(self, new_domain):
        super(MHDequilibrium, type(self)).domain.fset(self, new_domain)

    ###########################
    # Vector-valued callables #
    ###########################

    def j1(self, *etas, squeeze_out=False):
        """1-form components of current on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.jv(*etas, squeeze_out=False),
            *etas,
            kind="v_to_1",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def j2(self, *etas, squeeze_out=False):
        """2-form components of current on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.jv(*etas, squeeze_out=False),
            *etas,
            kind="v_to_2",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def jv(self, *etas, squeeze_out=False):
        """Contra-variant components of current on logical cube [0, 1]^3."""
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.j_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind="v", squeeze_out=squeeze_out)

    def j_cart(self, *etas, squeeze_out=False):
        """Cartesian components of current evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        j_out = self.domain.push(
            self.jv(*etas, squeeze_out=False),
            *etas,
            kind="v",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )
        return j_out, self.domain(*etas, squeeze_out=squeeze_out)

    def u1(self, *etas, squeeze_out=False):
        """1-form components of mean velocity evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        return self.j1(
            *etas,
            squeeze_out=squeeze_out,
        ) / self.n0(
            *etas,
            squeeze_out=squeeze_out,
        )

    def u2(self, *etas, squeeze_out=False):
        """2-form components of mean velocity evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        return self.j2(
            *etas,
            squeeze_out=squeeze_out,
        ) / self.n0(
            *etas,
            squeeze_out=squeeze_out,
        )

    def uv(self, *etas, squeeze_out=False):
        """Contra-variant components of mean velocity evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        return self.jv(
            *etas,
            squeeze_out=squeeze_out,
        ) / self.n0(
            *etas,
            squeeze_out=squeeze_out,
        )

    def u_cart(self, *etas, squeeze_out=False):
        """Cartesian components of mean velocity evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        u_out = self.j_cart(
            *etas,
            squeeze_out=squeeze_out,
        )[0] / self.n0(
            *etas,
            squeeze_out=squeeze_out,
        )
        return u_out, self.domain(*etas, squeeze_out=squeeze_out)

    def curl_unit_b1(self, *etas, squeeze_out=False):
        """1-form components of curl of unit magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        return self.domain.pull(
            self.curl_unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind="1", squeeze_out=squeeze_out
        )

    def curl_unit_b2(self, *etas, squeeze_out=False):
        """2-form components of curl of unit magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        return self.domain.pull(
            self.curl_unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind="2", squeeze_out=squeeze_out
        )

    def curl_unit_bv(self, *etas, squeeze_out=False):
        """Contra-variant components of curl of unit magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        return self.domain.pull(
            self.curl_unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind="v", squeeze_out=squeeze_out
        )

    def curl_unit_b_cart(self, *etas, squeeze_out=False):
        """Cartesian components of curl of unit magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        j, xyz = self.j_cart(*etas, squeeze_out=squeeze_out)
        gradB, xyz = self.gradB_cart(*etas, squeeze_out=squeeze_out)
        absB = self.absB0(*etas, squeeze_out=squeeze_out)
        out = np.array(
            [
                j[0] / absB + (b[1] * gradB[2] - b[2] * gradB[1]) / absB**2,
                j[1] / absB + (b[2] * gradB[0] - b[0] * gradB[2]) / absB**2,
                j[2] / absB + (b[0] * gradB[1] - b[1] * gradB[0]) / absB**2,
            ],
            dtype=float,
        )
        return out, xyz

    ###########################
    # Scalar-valued callables #
    ###########################

    def curl_unit_b_dot_b0(self, *etas, squeeze_out=False):
        r"""0-form of :math:`(\nabla \times \mathbf b_0) \times \mathbf b_0` evaluated on logical cube [0, 1]^3."""
        curl_b, xyz = self.curl_unit_b_cart(*etas, squeeze_out=squeeze_out)
        b, xyz = self.unit_b_cart(*etas, squeeze_out=squeeze_out)
        out = curl_b[0] * b[0] + curl_b[1] * b[1] + curl_b[2] * b[2]
        return out

    ###################
    # Single components
    ###################

    def j1_1(self, *etas, squeeze_out=False):
        return self.j1(*etas, squeeze_out=squeeze_out)[0]

    def j1_2(self, *etas, squeeze_out=False):
        return self.j1(*etas, squeeze_out=squeeze_out)[1]

    def j1_3(self, *etas, squeeze_out=False):
        return self.j1(*etas, squeeze_out=squeeze_out)[2]

    def j2_1(self, *etas, squeeze_out=False):
        return self.j2(*etas, squeeze_out=squeeze_out)[0]

    def j2_2(self, *etas, squeeze_out=False):
        return self.j2(*etas, squeeze_out=squeeze_out)[1]

    def j2_3(self, *etas, squeeze_out=False):
        return self.j2(*etas, squeeze_out=squeeze_out)[2]

    def jv_1(self, *etas, squeeze_out=False):
        return self.jv(*etas, squeeze_out=squeeze_out)[0]

    def jv_2(self, *etas, squeeze_out=False):
        return self.jv(*etas, squeeze_out=squeeze_out)[1]

    def jv_3(self, *etas, squeeze_out=False):
        return self.jv(*etas, squeeze_out=squeeze_out)[2]

    def curl_unit_b1_1(self, *etas, squeeze_out=False):
        return self.curl_unit_b1(*etas, squeeze_out=squeeze_out)[0]

    def curl_unit_b1_2(self, *etas, squeeze_out=False):
        return self.curl_unit_b1(*etas, squeeze_out=squeeze_out)[1]

    def curl_unit_b1_3(self, *etas, squeeze_out=False):
        return self.curl_unit_b1(*etas, squeeze_out=squeeze_out)[2]

    def curl_unit_b2_1(self, *etas, squeeze_out=False):
        return self.curl_unit_b2(*etas, squeeze_out=squeeze_out)[0]

    def curl_unit_b2_2(self, *etas, squeeze_out=False):
        return self.curl_unit_b2(*etas, squeeze_out=squeeze_out)[1]

    def curl_unit_b2_3(self, *etas, squeeze_out=False):
        return self.curl_unit_b2(*etas, squeeze_out=squeeze_out)[2]

    def curl_unit_bv_1(self, *etas, squeeze_out=False):
        return self.curl_unit_bv(*etas, squeeze_out=squeeze_out)[0]

    def curl_unit_bv_2(self, *etas, squeeze_out=False):
        return self.curl_unit_bv(*etas, squeeze_out=squeeze_out)[1]

    def curl_unit_bv_3(self, *etas, squeeze_out=False):
        return self.curl_unit_bv(*etas, squeeze_out=squeeze_out)[2]

    ##########
    # Plotting
    ##########

    def show(self, n1=16, n2=33, n3=21, n_planes=5):
        """Generate vtk files of equilibirum and do some 2d plots with matplotlib.

        Parameters
        ----------
        n1, n2, n3 : int
            Evaluation points of mapping in each direcion.

        n_planes : int
            Number of planes to show perpendicular to eta3."""

        import struphy

        torus_mappings = (
            "Tokamak",
            "GVECunit",
            "DESCunit",
            "IGAPolarTorus",
            "HollowTorus",
        )

        e1 = np.linspace(0.0001, 1, n1)
        e2 = np.linspace(0, 1, n2)
        e3 = np.linspace(0, 1, n3)

        if self.domain.__class__.__name__ in ("GVECunit", "DESCunit"):
            if n_planes > 1:
                jump = (n3 - 1) / (n_planes - 1)
            else:
                jump = 0
        else:
            n_planes = 1
            jump = 0

        x, y, z = self.domain(e1, e2, e3)
        print("Evaluation of mapping done.")
        det_df = self.domain.jacobian_det(e1, e2, e3)
        p = self.p0(e1, e2, e3)
        print("Computation of pressure done.")

        # ori 240624
        n_dens = self.n0(e1, e2, e3)
        print("Computation of density done.")

        absB = self.absB0(e1, e2, e3)
        print("Computation of abs(B) done.")
        j_cart, xyz = self.j_cart(e1, e2, e3)
        print("Computation of current density done.")
        absJ = np.sqrt(j_cart[0] ** 2 + j_cart[1] ** 2 + j_cart[2] ** 2)

        _path = struphy.__path__[0] + "/fields_background/mhd_equil/gvec/output/"
        gridToVTK(
            _path + "vtk/gvec_equil",
            x,
            y,
            z,
            pointData={"det_df": det_df, "pressure": p, "absB": absB},
        )
        print("Generation of vtk files done.")

        # show params
        print("\nEquilibrium parameters:")
        for key, val in self.params.items():
            print(key, ": ", val)

        print("\nMapping parameters:")
        for key, val in self.domain.params_map.items():
            if key not in {"cx", "cy", "cz"}:
                print(key, ": ", val)

        # poloidal plane grid
        fig = plt.figure(figsize=(13, np.ceil(n_planes / 2) * 6.5))
        for n in range(n_planes):
            xp = x[:, :, int(n * jump)].squeeze()
            yp = y[:, :, int(n * jump)].squeeze()
            zp = z[:, :, int(n * jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = "R"
                l2 = "Z"
            else:
                pc1 = xp
                pc2 = yp
                l1 = "x"
                l2 = "y"

            ax = fig.add_subplot(int(np.ceil(n_planes / 2)), 2, n + 1)
            for i in range(pc1.shape[0]):
                for j in range(pc1.shape[1] - 1):
                    if i < pc1.shape[0] - 1:
                        ax.plot(
                            [pc1[i, j], pc1[i + 1, j]],
                            [pc2[i, j], pc2[i + 1, j]],
                            "b",
                            linewidth=0.6,
                        )
                    if j < pc1.shape[1] - 1:
                        ax.plot(
                            [pc1[i, j], pc1[i, j + 1]],
                            [pc2[i, j], pc2[i, j + 1]],
                            "b",
                            linewidth=0.6,
                        )

            ax.scatter(pc1[0, 0], pc2[0, 0], 20, "red", zorder=10)
            # ax.scatter(pc1[0, 32], pc2[0, 32], 20, 'red', zorder=10)

            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis("equal")
            ax.set_title(
                r"Poloidal plane at $\eta_3$={0:4.3f}".format(e3[int(n * jump)]),
            )

        # top view
        e1 = np.linspace(0, 1, n1)  # radial coordinate in [0, 1]
        e2 = np.linspace(0, 1, 3)  # poloidal angle in [0, 1]
        e3 = np.linspace(0, 1, n3)  # toroidal angle in [0, 1]

        xt, yt, zt = self.domain(e1, e2, e3)

        fig = plt.figure(figsize=(13, 2 * 6.5))
        ax = fig.add_subplot()
        for m in range(2):
            xp = xt[:, m, :].squeeze()
            yp = yt[:, m, :].squeeze()
            zp = zt[:, m, :].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                tc1 = xp
                tc2 = yp
                l1 = "x"
                l2 = "y"
            else:
                tc1 = xp
                tc2 = zp
                l1 = "x"
                l2 = "z"

            for i in range(tc1.shape[0]):
                for j in range(tc1.shape[1] - 1):
                    if i < tc1.shape[0] - 1:
                        ax.plot(
                            [tc1[i, j], tc1[i + 1, j]],
                            [tc2[i, j], tc2[i + 1, j]],
                            "b",
                            linewidth=0.6,
                        )
                    if j < tc1.shape[1] - 1:
                        if i == 0:
                            ax.plot(
                                [tc1[i, j], tc1[i, j + 1]],
                                [tc2[i, j], tc2[i, j + 1]],
                                "r",
                                linewidth=1,
                            )
                        else:
                            ax.plot(
                                [tc1[i, j], tc1[i, j + 1]],
                                [tc2[i, j], tc2[i, j + 1]],
                                "b",
                                linewidth=0.6,
                            )
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis("equal")
            ax.set_title("Device top view")

        # Jacobian determinant
        fig = plt.figure(figsize=(13, np.ceil(n_planes / 2) * 6.5))
        for n in range(n_planes):
            xp = x[:, :, int(n * jump)].squeeze()
            yp = y[:, :, int(n * jump)].squeeze()
            zp = z[:, :, int(n * jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = "R"
                l2 = "Z"
            else:
                pc1 = xp
                pc2 = yp
                l1 = "x"
                l2 = "y"

            detp = det_df[:, :, int(n * jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes / 2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, detp, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis("equal")
            ax.set_title(
                r"Jacobian determinant at $\eta_3$={0:4.3f}".format(e3[int(n * jump)]),
            )
            fig.colorbar(map, ax=ax, location="right")

        # pressure
        fig = plt.figure(figsize=(15, np.ceil(n_planes / 2) * 6.5))
        for n in range(n_planes):
            xp = x[:, :, int(n * jump)].squeeze()
            yp = y[:, :, int(n * jump)].squeeze()
            zp = z[:, :, int(n * jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = "R"
                l2 = "Z"
            else:
                pc1 = xp
                pc2 = yp
                l1 = "x"
                l2 = "y"

            pp = p[:, :, int(n * jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes / 2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, pp, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis("equal")
            ax.set_title(
                r"Pressure at $\eta_3$={0:4.3f}".format(e3[int(n * jump)]),
            )
            fig.colorbar(map, ax=ax, location="right")

        # density
        fig = plt.figure(figsize=(15, np.ceil(n_planes / 2) * 6.5))
        for n in range(n_planes):
            xp = x[:, :, int(n * jump)].squeeze()
            yp = y[:, :, int(n * jump)].squeeze()
            zp = z[:, :, int(n * jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = "R"
                l2 = "Z"
            else:
                pc1 = xp
                pc2 = yp
                l1 = "x"
                l2 = "y"

            nn = n_dens[:, :, int(n * jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes / 2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, nn, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis("equal")
            ax.set_title(
                r"Equilibrium density at $\eta_3$={0:4.3f}".format(e3[int(n * jump)]),
            )
            fig.colorbar(map, ax=ax, location="right")

        # magnetic field strength
        fig = plt.figure(figsize=(15, np.ceil(n_planes / 2) * 6.5))
        for n in range(n_planes):
            xp = x[:, :, int(n * jump)].squeeze()
            yp = y[:, :, int(n * jump)].squeeze()
            zp = z[:, :, int(n * jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = "R"
                l2 = "Z"
            else:
                pc1 = xp
                pc2 = yp
                l1 = "x"
                l2 = "y"

            ab = absB[:, :, int(n * jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes / 2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, ab, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis("equal")
            ax.set_title(
                r"Magnetic field strength at $\eta_3$={0:4.3f}".format(e3[int(n * jump)]),
            )
            fig.colorbar(map, ax=ax, location="right")

        # current density
        fig = plt.figure(figsize=(15, np.ceil(n_planes / 2) * 6.5))
        for n in range(n_planes):
            xp = x[:, :, int(n * jump)].squeeze()
            yp = y[:, :, int(n * jump)].squeeze()
            zp = z[:, :, int(n * jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = "R"
                l2 = "Z"
            else:
                pc1 = xp
                pc2 = yp
                l1 = "x"
                l2 = "y"

            ab = absJ[:, :, int(n * jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes / 2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, ab, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis("equal")
            ax.set_title(
                r"Current density (abs) at $\eta_3$={0:4.3f}".format(e3[int(n * jump)]),
            )
            fig.colorbar(map, ax=ax, location="right")

        plt.show()


class CartesianMHDequilibrium(MHDequilibrium):
    r"""
    The callables ``b_xyz``, ``j_xyz``, ``p_xyz``, ``n_xyz`` and ``gradB_xyz``
    must be provided in Cartesian coordinates.
    """

    @abstractmethod
    def b_xyz(self, x, y, z):
        """Cartesian magnetic field in physical space.
        Must return the components as a tuple."""
        pass

    @abstractmethod
    def j_xyz(self, x, y, z):
        """Cartesian current (curl of magnetic field) in physical space.
        Must return the components as a tuple."""
        pass

    @abstractmethod
    def p_xyz(self, x, y, z):
        """Equilibrium pressure in physical space."""
        pass

    @abstractmethod
    def n_xyz(self, x, y, z):
        """Equilibrium number density in physical space."""
        pass

    @abstractmethod
    def gradB_xyz(self, x, y, z):
        """Cartesian gradient of magnetic field strength in physical space.
        Must return the components as a tuple."""
        pass

    @MHDequilibrium.domain.setter
    def domain(self, new_domain):
        super(CartesianMHDequilibrium, type(self)).domain.fset(self, new_domain)


class AxisymmMHDequilibrium(CartesianMHDequilibrium):
    r"""
    Base class for ideal axisymmetric MHD equilibria based on a poloidal flux function
    :math:`\psi(R, Z)` and a toroidal field function :math:`g_{tor}(R, Z)`
    in a cylindrical coordinate system :math:`(R, \phi, Z)`.

    The magnetic field and current density are then given by

    .. math::

        \mathbf B = \nabla \psi \times \nabla \phi + g_{tor} \nabla \phi\,,\qquad \mathbf j = \nabla \times \mathbf B\,.

    The pressure and density profiles need to be implemented by child classes.
    """

    @abstractmethod
    def psi(self, R, Z, dR=0, dZ=0):
        """Poloidal flux function per radian. First AND second derivatives dR=0,1,2 and dZ=0,1,2 must be implemented."""
        pass

    @abstractmethod
    def g_tor(self, R, Z, dR=0, dZ=0):
        """Toroidal field function. First derivatives dR=0,1 and dZ=0,1 must be implemented."""
        pass

    @property
    @abstractmethod
    def psi_range(self):
        """Psi on-axis and at plasma boundary returned as list [psi_axis, psi_boundary]."""
        pass

    @property
    @abstractmethod
    def psi_axis_RZ(self):
        """Location of magnetic axis in R-Z-coordinates returned as list [psi_axis_R, psi_axis_Z]."""
        pass

    @abstractmethod
    def p_xyz(self, x, y, z):
        """Equilibrium pressure in physical space."""
        pass

    @abstractmethod
    def n_xyz(self, x, y, z):
        """Equilibrium number density in physical space."""
        pass

    def b_xyz(self, x, y, z):
        """Cartesian B-field components calculated as BR = -(dpsi/dZ)/R, BPhi = g_tor/R, BZ = (dpsi/dR)/R."""

        R, Phi, Z = self.inverse_map(x, y, z)

        # at phi = 0°
        BR = -self.psi(R, Z, dZ=1) / R
        BP = self.g_tor(R, Z) / R
        BZ = self.psi(R, Z, dR=1) / R

        # push-forward to Cartesian components
        Bx = BR * np.cos(Phi) - BP * np.sin(Phi)
        By = BR * np.sin(Phi) + BP * np.cos(Phi)
        Bz = 1 * BZ

        return Bx, By, Bz

    def j_xyz(self, x, y, z):
        """Cartesian current density components calculated as curl(B)."""

        R, Phi, Z = self.inverse_map(x, y, z)

        # at phi = 0° (j = curl(B))
        jR = -self.g_tor(R, Z, dZ=1) / R
        jP = -self.psi(R, Z, dZ=2) / R + self.psi(R, Z, dR=1) / R**2 - self.psi(R, Z, dR=2) / R
        jZ = self.g_tor(R, Z, dR=1) / R

        # push-forward to Cartesian components
        jx = jR * np.cos(Phi) - jP * np.sin(Phi)
        jy = jR * np.sin(Phi) + jP * np.cos(Phi)
        jz = 1 * jZ

        return jx, jy, jz

    def gradB_xyz(self, x, y, z):
        """Cartesian gradient |B| components calculated as grad(sqrt(BR**2 + BPhi**2 + BZ**2))."""

        R, Phi, Z = self.inverse_map(x, y, z)

        RabsB = np.sqrt(
            self.psi(R, Z, dZ=1) ** 2 + self.g_tor(R, Z) ** 2 + self.psi(R, Z, dR=1) ** 2,
        )

        # at phi = 0° (gradB = grad(absB))
        gradBR = (
            -RabsB / R**2
            + (
                self.psi(R, Z, dZ=1)
                * self.psi(
                    R,
                    Z,
                    dR=1,
                    dZ=1,
                )
                + self.psi(R, Z, dR=1) * self.psi(R, Z, dR=2)
            )
            / RabsB
            / R
        )
        gradBP = 0.0
        gradBZ = (
            (self.psi(R, Z, dZ=1) * self.psi(R, Z, dZ=2) + self.psi(R, Z, dR=1) * self.psi(R, Z, dR=1, dZ=1))
            / RabsB
            / R
        )

        # push-forward to Cartesian components
        gradBx = gradBR * np.cos(Phi) - gradBP * np.sin(Phi)
        gradBy = gradBR * np.sin(Phi) + gradBP * np.cos(Phi)
        gradBz = 1 * gradBZ

        return gradBx, gradBy, gradBz

    @staticmethod
    def inverse_map(x, y, z):
        """Inverse cylindrical mapping."""

        R = np.sqrt(x**2 + y**2)
        P = np.arctan2(y, x)
        Z = 1 * z

        return R, P, Z

    @CartesianMHDequilibrium.domain.setter
    def domain(self, new_domain):
        super(AxisymmMHDequilibrium, type(self)).domain.fset(self, new_domain)


class LogicalMHDequilibrium(MHDequilibrium):
    r"""
    The callables ``bv``, ``jv``, ``p0``, ``n0`` and ``gradB1``
    must be provided on the logical cube [0, 1]^3.
    """

    @abstractmethod
    def bv(self, *etas, squeeze_out=False):
        """Contra-variant (vector field) magnetic field on logical cube [0, 1]^3.
        Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def jv(self, *etas, squeeze_out=False):
        """Contra-variant (vector field) current density (=curl B) on logical cube [0, 1]^3.
        Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def p0(self, *etas, squeeze_out=False):
        """0-form pressure on logical cube [0, 1]^3.
        Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def n0(self, *etas, squeeze_out=False):
        """0-form density on logical cube [0, 1]^3."""
        pass

    @abstractmethod
    def gradB1(self, *etas, squeeze_out=False):
        """1-form gradient of magnetic field strength strength on logical cube [0, 1]^3.
        Must return the components as a tuple.
        """
        pass

    @MHDequilibrium.domain.setter
    def domain(self, new_domain):
        super(LogicalMHDequilibrium, type(self)).domain.fset(self, new_domain)


class NumericalMHDequilibrium(LogicalMHDequilibrium):
    r"""
    Must provide a (numerical) mapping from the logical cube [0, 1]^3 to the physical domain.
    Overrides base class domain.
    """

    @property
    @abstractmethod
    def numerical_domain(self):
        """Numerically computed mapping from the logical cube [0, 1]^3 to the physical domain
        in the form of a :class:`~struphy.geometry.base.Domain` object."""
        pass

    @property
    def domain(self):
        return self.numerical_domain
