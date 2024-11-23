'Base classes for coil fields.'


from abc import ABCMeta, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from pyevtk.hl import gridToVTK

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import struphy.bsplines.bsplines as bsp
from struphy.linear_algebra import linalg_kron


class CoilField(metaclass=ABCMeta):
    """
    Base class for Struphy coil fields.
    The callables B, and J have to be provided through the child classes `CartesianCoilField` or `LogicalCoilField`.
    The base class provides transformations of callables to different representations or coordinates.
    For logical coil fields, the methods bv, and jv are overidden by the child class.   
    """

    @property
    def params(self):
        """ Parameters dictionary.
        """
        return self._params

    def set_params(self, **params):
        '''Generates self.params dictionary.'''
        self._params = params

    #########################
    # Coil field callables #
    #########################

    def absB0(self, *etas, squeeze_out=False):
        """ 0-form absolute value of coil magnetic field on logical cube [0, 1]^3.
        """
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        return np.sqrt(b[0]**2 + b[1]**2 + b[2]**2)

    def absB3(self, *etas, squeeze_out=False):
        """ 3-form absolute value of coil magnetic field on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.absB0(*etas, squeeze_out=False), *etas, kind='0_to_3', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def b1(self, *etas, squeeze_out=False):
        """ 1-form components of coil magnetic field on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.bv(*etas, squeeze_out=False), *etas, kind='v_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def b2(self, *etas, squeeze_out=False):
        """ 2-form components of coil magnetic field on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.bv(*etas, squeeze_out=False), *etas, kind='v_to_2', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def bv(self, *etas, squeeze_out=False):
        """ Contra-variant components of coil magnetic field on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.b_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='v', squeeze_out=squeeze_out)

    def b_cart(self, *etas, squeeze_out=False):
        """ Cartesian components of coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        b_out = self.domain.push(
            self.bv(*etas, squeeze_out=False), *etas,
            kind='v', a_kwargs={'squeeze_out': False}, squeeze_out=squeeze_out,
        )
        return b_out, self.domain(*etas, squeeze_out=squeeze_out)

    def unit_b1(self, *etas, squeeze_out=False):
        """ Unit vector components of coil magnetic field (1-form) on logical cube [0, 1]^3.
        """
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='1', squeeze_out=squeeze_out)

    def unit_b2(self, *etas, squeeze_out=False):
        """ Unit vector components of coil magnetic field (2-form) on logical cube [0, 1]^3.
        """
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='2', squeeze_out=squeeze_out)

    def unit_bv(self, *etas, squeeze_out=False):
        """ Unit vector components of  coil magnetic field (contra-variant) on logical cube [0, 1]^3.
        """
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='v', squeeze_out=squeeze_out)

    def unit_b_cart(self, *etas, squeeze_out=False):
        """ Unit vector Cartesian components of coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        absB = self.absB0(*etas, squeeze_out=squeeze_out)
        out = np.array([b[0]/absB, b[1]/absB, b[2]/absB], dtype=float)
        return out, xyz

    def curl_unit_b1(self, *etas, squeeze_out=False):
        """ 1-form components of curl of unit coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        return self.domain.pull(self.curl_unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='1', squeeze_out=squeeze_out)

    def curl_unit_b2(self, *etas, squeeze_out=False):
        """ 2-form components of curl of unit coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        return self.domain.pull(self.curl_unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='2', squeeze_out=squeeze_out)

    def curl_unit_bv(self, *etas, squeeze_out=False):
        """ Contra-variant components of curl of unit coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        return self.domain.pull(self.curl_unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='v', squeeze_out=squeeze_out)

    def curl_unit_b_cart(self, *etas, squeeze_out=False):
        """ Cartesian components of curl of unit coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        j, xyz = self.j_cart(*etas, squeeze_out=squeeze_out)
        gradB, xyz = self.gradB_cart(*etas, squeeze_out=squeeze_out)
        absB = self.absB0(*etas, squeeze_out=squeeze_out)
        out = np.array(
            [
                j[0]/absB + (b[1]*gradB[2] - b[2]*gradB[1])/absB**2,
                j[1]/absB + (b[2]*gradB[0] - b[0]*gradB[2])/absB**2,
                j[2]/absB + (b[0]*gradB[1] - b[1]*gradB[0])/absB**2,
            ], dtype=float,
        )
        return out, xyz

    def curl_unit_b_dot_b0(self, *etas, squeeze_out=False):
        r'''0-form of :math:`(\nabla \times \mathbf b_0) \times \mathbf b_0` evaluated on logical cube [0, 1]^3.'''
        curl_b, xyz = self.curl_unit_b_cart(*etas, squeeze_out=squeeze_out)
        b, xyz = self.unit_b_cart(*etas, squeeze_out=squeeze_out)
        out = curl_b[0]*b[0] + curl_b[1]*b[1] + curl_b[2]*b[2]
        return out

    def a1(self, *etas, squeeze_out=False):
        """ 1-form components of coil vector potential on logical cube [0, 1]^3.
        """
        avail_list = ['HomogenSlab']
        assert self.__class__.__name__ in avail_list, f'Vector potential currently available only for {avail_list}, but coil_fields is "{self.__class__.__name__}".'

        return self.domain.transform(self.a2(*etas, squeeze_out=False), *etas, kind='2_to_1', a_kwargs={'squeeze_out': False}, squeeze_out=squeeze_out)

    def a2(self, *etas, squeeze_out=False):
        """ 2-form components of coil vector potential on logical cube [0, 1]^3.
        """
        avail_list = ['HomogenSlab']
        assert self.__class__.__name__ in avail_list, f'Vector potential currently available only for {avail_list}, but coil_fields is "{self.__class__.__name__}".'

        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.a_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='2', squeeze_out=squeeze_out)

    def av(self, *etas, squeeze_out=False):
        """ Contra-variant components of coil vector potneital on logical cube [0, 1]^3.
        """
        avail_list = ['HomogenSlab']
        assert self.__class__.__name__ in avail_list, f'Vector potential currently available only for {avail_list}, but coil_fields is "{self.__class__.__name__}".'

        return self.domain.transform(self.a2(*etas, squeeze_out=False), *etas, kind='2_to_v', a_kwargs={'squeeze_out': False}, squeeze_out=squeeze_out)

    def j1(self, *etas, squeeze_out=False):
        """ 1-form components of coil current on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.jv(*etas, squeeze_out=False), *etas, kind='v_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def j2(self, *etas, squeeze_out=False):
        """ 2-form components of coil current on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.jv(*etas, squeeze_out=False), *etas, kind='v_to_2', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def jv(self, *etas, squeeze_out=False):
        """ Contra-variant components of coil current on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.j_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='v', squeeze_out=squeeze_out)

    def j_cart(self, *etas, squeeze_out=False):
        """ Cartesian components of coil current evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        j_out = self.domain.push(
            self.jv(*etas, squeeze_out=False), *etas,
            kind='v', a_kwargs={'squeeze_out': False}, squeeze_out=squeeze_out,
        )
        return j_out, self.domain(*etas, squeeze_out=squeeze_out)

    def gradB1(self, *etas, squeeze_out=False):
        """ 1-form components of gradient of coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.gradB_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='1', squeeze_out=squeeze_out)

    def gradB2(self, *etas, squeeze_out=False):
        """ 2-form components of gradient of coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        return self.domain.transform(self.gradB1(*etas, squeeze_out=False), *etas, kind='1_to_2', a_kwargs={'squeeze_out': False}, squeeze_out=squeeze_out)

    def gradBv(self, *etas, squeeze_out=False):
        """ Contra-variant components of gradient of coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        return self.domain.transform(self.gradB1(*etas, squeeze_out=False), *etas, kind='1_to_v', a_kwargs={'squeeze_out': False}, squeeze_out=squeeze_out)

    def gradB_cart(self, *etas, squeeze_out=False):
        """ Cartesian components of gradient of coil magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        gradB_out = self.domain.push(
            self.gradB1(*etas, squeeze_out=False), *etas,
            kind='1', a_kwargs={'squeeze_out': False}, squeeze_out=squeeze_out,
        )
        return gradB_out, self.domain(*etas)

    ###################
    # Single components
    ###################

    def b1_1(self, *etas, squeeze_out=False):
        return self.b1(*etas, squeeze_out=squeeze_out)[0]

    def b1_2(self, *etas, squeeze_out=False):
        return self.b1(*etas, squeeze_out=squeeze_out)[1]

    def b1_3(self, *etas, squeeze_out=False):
        return self.b1(*etas, squeeze_out=squeeze_out)[2]

    def a1_1(self, *etas, squeeze_out=False):
        return self.a1(*etas, squeeze_out=squeeze_out)[0]

    def a1_2(self, *etas, squeeze_out=False):
        return self.a1(*etas, squeeze_out=squeeze_out)[1]

    def a1_3(self, *etas, squeeze_out=False):
        return self.a1(*etas, squeeze_out=squeeze_out)[2]

    def b2_1(self, *etas, squeeze_out=False):
        return self.b2(*etas, squeeze_out=squeeze_out)[0]

    def b2_2(self, *etas, squeeze_out=False):
        return self.b2(*etas, squeeze_out=squeeze_out)[1]

    def b2_3(self, *etas, squeeze_out=False):
        return self.b2(*etas, squeeze_out=squeeze_out)[2]

    def a2_1(self, *etas, squeeze_out=False):
        return self.a2(*etas, squeeze_out=squeeze_out)[0]

    def a2_2(self, *etas, squeeze_out=False):
        return self.a2(*etas, squeeze_out=squeeze_out)[1]

    def a2_3(self, *etas, squeeze_out=False):
        return self.a2(*etas, squeeze_out=squeeze_out)[2]

    def bv_1(self, *etas, squeeze_out=False):
        return self.bv(*etas, squeeze_out=squeeze_out)[0]

    def bv_2(self, *etas, squeeze_out=False):
        return self.bv(*etas, squeeze_out=squeeze_out)[1]

    def bv_3(self, *etas, squeeze_out=False):
        return self.bv(*etas, squeeze_out=squeeze_out)[2]

    def av_1(self, *etas, squeeze_out=False):
        return self.av(*etas, squeeze_out=squeeze_out)[0]

    def av_2(self, *etas, squeeze_out=False):
        return self.av(*etas, squeeze_out=squeeze_out)[1]

    def av_3(self, *etas, squeeze_out=False):
        return self.av(*etas, squeeze_out=squeeze_out)[2]

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

    def b_cart_1(self, *etas, squeeze_out=False):
        return self.b_cart(*etas, squeeze_out=squeeze_out)[0][0]

    def b_cart_2(self, *etas, squeeze_out=False):
        return self.b_cart(*etas, squeeze_out=squeeze_out)[0][1]

    def b_cart_3(self, *etas, squeeze_out=False):
        return self.b_cart(*etas, squeeze_out=squeeze_out)[0][2]

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
        '''Generate vtk files of coil fields and do some 2d plots with matplotlib.

        Parameters
        ----------
        n1, n2, n3 : int
            Evaluation points of mapping in each direcion.

        n_planes : int
            Number of planes to show perpendicular to eta3.'''

        import struphy

        torus_mappings = (
            'Tokamak', 'GVECunit', 'DESCunit',
            'IGAPolarTorus', 'HollowTorus',
        )

        e1 = np.linspace(0.0001, 1, n1)
        e2 = np.linspace(0, 1, n2)
        e3 = np.linspace(0, 1, n3)

        if self.domain.__class__.__name__ in ('GVECunit', 'DESCunit'):
            if n_planes > 1:
                jump = (n3 - 1)/(n_planes - 1)
            else:
                jump = 0
        else:
            n_planes = 1
            jump = 0

        x, y, z = self.domain(e1, e2, e3)
        print('Evaluation of mapping done.')
        det_df = self.domain.jacobian_det(e1, e2, e3)

        absB = self.absB0(e1, e2, e3)
        print('Computation of abs(B) done.')
        j_cart, xyz = self.j_cart(e1, e2, e3)
        absJ = np.sqrt(j_cart[0]**2 + j_cart[1]**2 + j_cart[2]**2)

        _path = struphy.__path__[0] + \
            '/fields_background/coil_fields/gvec/output/'
        gridToVTK(
            _path + 'vtk/gvec_coil', x, y, z,
            pointData={'det_df': det_df, 'absB': absB},
        )
        print('Generation of vtk files done.')

        # show params
        print('\ncoil parameters:')
        for key, val in self.params.items():
            print(key, ': ', val)

        print('\nMapping parameters:')
        for key, val in self.domain.params_map.items():
            if key not in {'cx', 'cy', 'cz'}:
                print(key, ': ', val)

        # poloidal plane grid
        fig = plt.figure(figsize=(13, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):
            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = 'R'
                l2 = 'Z'
            else:
                pc1 = xp
                pc2 = yp
                l1 = 'x'
                l2 = 'y'

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            for i in range(pc1.shape[0]):
                for j in range(pc1.shape[1] - 1):
                    if i < pc1.shape[0] - 1:
                        ax.plot(
                            [pc1[i, j], pc1[i + 1, j]],
                            [pc2[i, j], pc2[i + 1, j]], 'b', linewidth=.6,
                        )
                    if j < pc1.shape[1] - 1:
                        ax.plot(
                            [pc1[i, j], pc1[i, j + 1]],
                            [pc2[i, j], pc2[i, j + 1]], 'b', linewidth=.6,
                        )

            ax.scatter(pc1[0, 0], pc2[0, 0], 20, 'red', zorder=10)
            # ax.scatter(pc1[0, 32], pc2[0, 32], 20, 'red', zorder=10)

            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title(
                'Poloidal plane at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]),
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
                l1 = 'x'
                l2 = 'y'
            else:
                tc1 = xp
                tc2 = zp
                l1 = 'x'
                l2 = 'z'

            for i in range(tc1.shape[0]):
                for j in range(tc1.shape[1] - 1):
                    if i < tc1.shape[0] - 1:
                        ax.plot(
                            [tc1[i, j], tc1[i + 1, j]],
                            [tc2[i, j], tc2[i + 1, j]], 'b', linewidth=.6,
                        )
                    if j < tc1.shape[1] - 1:
                        if i == 0:
                            ax.plot(
                                [tc1[i, j], tc1[i, j + 1]],
                                [tc2[i, j], tc2[i, j + 1]], 'r', linewidth=1,
                            )
                        else:
                            ax.plot(
                                [tc1[i, j], tc1[i, j + 1]],
                                [tc2[i, j], tc2[i, j + 1]], 'b', linewidth=.6,
                            )
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title('Device top view')

        # Jacobian determinant
        fig = plt.figure(figsize=(13, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = 'R'
                l2 = 'Z'
            else:
                pc1 = xp
                pc2 = yp
                l1 = 'x'
                l2 = 'y'

            detp = det_df[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, detp, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title(
                'Jacobian determinant at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]),
            )
            fig.colorbar(map, ax=ax, location='right')

        # magnetic field strength
        fig = plt.figure(figsize=(15, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = 'R'
                l2 = 'Z'
            else:
                pc1 = xp
                pc2 = yp
                l1 = 'x'
                l2 = 'y'

            ab = absB[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, ab, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title(
                'Magnetic field strength at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]),
            )
            fig.colorbar(map, ax=ax, location='right')

        # current density
        fig = plt.figure(figsize=(15, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = 'R'
                l2 = 'Z'
            else:
                pc1 = xp
                pc2 = yp
                l1 = 'x'
                l2 = 'y'

            ab = absJ[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, ab, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title(
                'Current density (abs) at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]),
            )
            fig.colorbar(map, ax=ax, location='right')


class CartesianCoilField(CoilField):
    """
    Base class for coil fields where B, and J are specified in Cartesian coordinates.  
    The domain must be set using the setter method.    
    """

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        assert hasattr(self, '_domain'), 'Domain for Cartesian coil field not set. Only b_xyz and j_xyz available at this stage. Please do obj.domain = ... to have access to all transformations (1-form, 2-form, etc.)'
        return self._domain

    @domain.setter
    def domain(self, domain):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        self._domain = domain

    @abstractmethod
    def b_xyz(self, x, y, z):
        """ Cartesian coil magnetic field in physical space. Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def j_xyz(self, x, y, z):
        """ Cartesian coil current (curl of coil magnetic field) in physical space. Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def gradB_xyz(self, x, y, z):
        """ Cartesian gradient of coil magnetic field in physical space. Must return the components as a tuple.
        """
        pass


class LogicalCoilField(CoilField):
    """
    Base class for coil fields where B and J are specified on the logical cube [0, 1]^3. 
    Must prescribe B, J and grad(|B|) as 1-forms (covariant).      
    """

    @property
    @abstractmethod
    def domain(self):
        """ Domain object that characterizes the mapping from the logical cube [0, 1]^3 to the physical domain.
        """
        pass

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
    def gradB1(self, *etas, squeeze_out=False):
        """1-form gradient of coil magnetic field strength on logical cube [0, 1]^3. 
        Must return the components as a tuple.
        """
        pass

def spline_interpolation_nd(p: list, spl_kind: list, grids_1d: list, values: np.ndarray, intervals: list):
    """
    n-dimensional tensor-product spline interpolation with discrete input. 

    The interpolation points are passed as a list of 1d arrays, each array i with increasing entries g[0]=interval[i,0] < g[1] < ...
    The last element must be g[-1] = interval[i,1] for clamped interpolation and g[-1] < interval[i,1] for periodic interpolation.

    Parameters
    -----------
    p : list[int]
        Spline degree.

    grids_1d : list[array]
        Interpolation points ford grids_1d[i] are in [interval[i,0], interval[i,1]].

    spl_kind : list[bool]
        True: periodic splines, False: clamped splines.

    values: array
        Function values at interpolation points. values.shape = (grid1.size, ..., gridn.size).

    interval : list[array]
        The interpolation intervals, each must be of size 2.

    Returns
    --------
    coeffs : np.array
        spline coefficients as nd array.

    T : list[array]
        Knot vector of spline interpolant.

    indN : list[array]
        Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
    """

    T = []
    indN = []
    I_mat = []
    I_LU = []
    for sh, x_grid, p_i, kind_i, interval in zip(values.shape, grids_1d, p, spl_kind, intervals):
        assert isinstance(x_grid, np.ndarray)
        assert sh == x_grid.size
        assert np.all(
            np.roll(x_grid, 1)[1:] <
            x_grid[1:],
        ) and x_grid[-1] > x_grid[-2]
        assert len(interval) == 2
        assert np.abs(x_grid[0] - interval[0]) < 1e-14

        if kind_i:
            assert x_grid[-1] < interval[1], 'Interpolation points must be < interval[1] for periodic interpolation.'
            breaks = interval[1]*np.ones(x_grid.size + 1)

            if p_i % 2 == 0:
                breaks[1:-1] = (x_grid[1:] + np.roll(x_grid, 1)[1:]) / 2.
                breaks[0] = interval[0]
            else:
                breaks[:-1] = x_grid

        else:
            assert np.abs(
                x_grid[-1] - interval[1],
            ) < 1e-14, 'Interpolation points must include x=interval[1] for clamped interpolation.'
            # dimension of the 1d spline spaces: dim = breaks.size - 1 + p = x_grid.size
            if p_i == 1:
                breaks = x_grid
            elif p_i % 2 == 0:
                breaks = x_grid[p_i//2 - 1:-p_i//2].copy()
            else:
                breaks = x_grid[(p_i - 1)//2:-(p_i - 1)//2].copy()

            # cells must be in interval
            breaks[0] = interval[0]
            breaks[-1] = interval[1]

        # breaks = np.linspace(0., 1., x_grid.size - (not kind_i)*p_i + 1)

        T += [bsp.make_knots(breaks, p_i, periodic=kind_i)]

        indN += [
            (
                np.indices((breaks.size - 1, p_i + 1))[1] +
                np.arange(breaks.size - 1)[:, None]
            ) % x_grid.size,
        ]

        I_mat += [bsp.collocation_matrix(T[-1], p_i, x_grid, periodic=kind_i)]

        I_LU += [splu(csc_matrix(I_mat[-1]))]

    # dimension check
    for I, x_grid in zip(I_mat, grids_1d):
        assert I.shape[0] == x_grid.size
        assert I.shape[0] == I.shape[1]

    # solve system
    if len(p) == 1:
        return I_LU[0].solve(values), T, indN
    if len(p) == 2:
        return linalg_kron.kron_lusolve_2d(I_LU, values), T, indN
    elif len(p) == 3:
        return linalg_kron.kron_lusolve_3d(I_LU, values), T, indN
    else:
        raise AssertionError("Only dimensions < 4 are supported.")
