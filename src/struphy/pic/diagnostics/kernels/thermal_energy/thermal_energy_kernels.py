"""Calculate thermal energy of electron."""

from numpy import abs, empty, log, sqrt
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments


@stack_array("det_df", "dfm")
def thermal_energy(
    res: "float[:]",
    density: "float[:,:,:,:,:,:]",
    pads1: int,
    pads2: int,
    pads3: int,
    nel1: "int",
    nel2: "int",
    nel3: "int",
    nq1: int,
    nq2: int,
    nq3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    pts1: "float[:,:]",
    pts2: "float[:,:]",
    pts3: "float[:,:]",
    args_domain: "DomainArguments",
):
    r"""
    Calculate thermal energy of electron.

    Parameters
    ----------
        res : array[float]
            array to store the thermal energy of electrons

        density : array[float]
            array to store values of density at quadrature points in each cell

        pads1 - pads3 : int
            size of ghost region in each direction

        nel1 - nel3 : array[int]
            number of cells in each direction

        nq1 - nq3 : array[int]
            number of quadrature points in each direction of each cell

        w1 - w3: array[float]
            quadrature weights in each cell

        pts1 - pts3: array[float]
            quadrature points in each cell

        starts1 : array[int]
            starts of the stencil objects

        kind_map ->  cz:
            domain information

    .. math::
        \begin{align*}
            \int \hat{n}^0 \ln \hat{n}^0 \sqrt{g} \mathrm{d}{\boldsymbol \eta}.
        \end{align*}
    """

    res[:] = 0.0
    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # fmt: off
    #$ omp parallel private (iel1, iel2, iel3, q1, q2, q3, eta1, eta2, eta3, wvol, vv, dfm, det_df)
    #$ omp for reduction( + : res)
    # fmt: on
    for iel1 in range(nel1):
        for iel2 in range(nel2):
            for iel3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            eta1 = pts1[iel1, q1]
                            eta2 = pts2[iel2, q2]
                            eta3 = pts3[iel3, q3]

                            wvol = w1[iel1, q1] * w2[iel2, q2] * w3[iel3, q3]

                            vv = density[
                                pads1 + iel1,
                                pads2 + iel2,
                                pads3 + iel3,
                                q1,
                                q2,
                                q3,
                            ]

                            if abs(vv) < 0.00001:
                                vv = 1.0

                            # evaluate Jacobian, result in dfm
                            evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

                            det_df = linalg_kernels.det(dfm)

                            res[0] += vv * det_df * log(vv) * wvol
