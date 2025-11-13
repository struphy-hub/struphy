import os

import cunumpy as xp
import yaml
from psydac.api.discretization import discretize
from sympde.topology import Derham, Line

from struphy.fields_background.equils import set_defaults


class InitialMHDAxisymHdivEigFun:
    r"""
    Defines the initial condition via a 2-form MHD velocity field eigenfunction on the logical domain and setting the magnetic field and pressure to zero.

    Parameters
    ----------
    derham : struphy.feec.psydac_derham.Derham
        Discrete Derham complex.

    **params
        Parameters for loading and selecting the desired eigenfunction.

        * spec : str, path of the .npy eigenspectrum relative to <install_path>/io/out/
        * spec_abs : str, absolute path of the .npy eigenspectrum
        * eig_freq_upper : float, upper search limit of squared eigenfrequency
        * eig_freq_lower : float, lower search limit of squared eigenfrequency
        * kind : str, whether to use real (r) or imaginary (i) part of eigenfunction
        * scaling : float, scaling factor that is multiplied with the eigenfunction
    """

    def __init__(self, derham, **params):
        import struphy.utils.utils as utils

        # Read struphy state file
        state = utils.read_state()

        o_path = state["o_path"]

        params_default = {
            "spec": "sim_1/spec_n_-1.npy",
            "spec_abs": None,
            "eig_freq_upper": 0.02,
            "eig_freq_lower": 0.03,
            "kind": "r",
            "scaling": 1.0,
        }

        params = set_defaults(params, params_default)

        # absolute path of spectrum
        if params["spec_abs"] is None:
            spec_path = os.path.join(o_path, params["spec"])
        else:
            spec_path = params["spec_abs"]

        # load eigenvector for velocity field
        omega2, U2_eig = xp.split(xp.load(spec_path), [1], axis=0)
        omega2 = omega2.flatten()

        # find eigenvector corresponding to given squared eigenfrequency range
        mode = xp.where((xp.real(omega2) < params["eig_freq_upper"]) & (xp.real(omega2) > params["eig_freq_lower"]))[0]

        assert mode.size == 1
        mode = mode[0]

        nnz_pol = derham.boundary_ops["2"].dim_nz_pol
        nnz_tor = derham.boundary_ops["2"].dim_nz_tor

        eig_vec_1 = U2_eig[
            0 * nnz_pol[0] + 0 * nnz_pol[1] + 0 * nnz_pol[2] : 1 * nnz_pol[0] + 0 * nnz_pol[1] + 0 * nnz_pol[2],
            mode,
        ]
        eig_vec_2 = U2_eig[
            1 * nnz_pol[0] + 0 * nnz_pol[1] + 0 * nnz_pol[2] : 1 * nnz_pol[0] + 1 * nnz_pol[1] + 0 * nnz_pol[2],
            mode,
        ]
        eig_vec_3 = U2_eig[
            1 * nnz_pol[0] + 1 * nnz_pol[1] + 0 * nnz_pol[2] : 1 * nnz_pol[0] + 1 * nnz_pol[1] + 1 * nnz_pol[2],
            mode,
        ]

        del omega2, U2_eig

        # project toroidal Fourier modes
        domain_log = Line("L", bounds=(0, 1))
        derham_sym = Derham(domain_log)

        domain_log_h = discretize(domain_log, ncells=[derham.Nel[2]], periodic=[True])
        derham_1d = discretize(derham_sym, domain_log_h, degree=[derham.p[2]], nquads=[derham.nquads[2]])

        p0, p1 = derham_1d.projectors(nquads=[derham.nq_pr[2]])

        n_tor = int(os.path.split(spec_path)[-1][-6:-4])

        N_cos = p0(lambda phi: xp.cos(2 * xp.pi * n_tor * phi)).coeffs.toarray()
        N_sin = p0(lambda phi: xp.sin(2 * xp.pi * n_tor * phi)).coeffs.toarray()

        D_cos = p1(lambda phi: xp.cos(2 * xp.pi * n_tor * phi)).coeffs.toarray()
        D_sin = p1(lambda phi: xp.sin(2 * xp.pi * n_tor * phi)).coeffs.toarray()

        # select real part or imaginary part
        assert params["kind"] == "r" or params["kind"] == "i"

        if params["kind"] == "r":
            eig_vec_1 = (xp.outer(xp.real(eig_vec_1), D_cos) - xp.outer(xp.imag(eig_vec_1), D_sin)).flatten()
            eig_vec_2 = (xp.outer(xp.real(eig_vec_2), D_cos) - xp.outer(xp.imag(eig_vec_2), D_sin)).flatten()
            eig_vec_3 = (xp.outer(xp.real(eig_vec_3), N_cos) - xp.outer(xp.imag(eig_vec_3), N_sin)).flatten()
        else:
            eig_vec_1 = (xp.outer(xp.imag(eig_vec_1), D_cos) + xp.outer(xp.real(eig_vec_1), D_sin)).flatten()
            eig_vec_2 = (xp.outer(xp.imag(eig_vec_2), D_cos) + xp.outer(xp.real(eig_vec_2), D_sin)).flatten()
            eig_vec_3 = (xp.outer(xp.imag(eig_vec_3), N_cos) + xp.outer(xp.real(eig_vec_3), N_sin)).flatten()

        # set coefficients in full space
        eigvec_1_ten = xp.zeros(derham.nbasis["2"][0], dtype=float)
        eigvec_2_ten = xp.zeros(derham.nbasis["2"][1], dtype=float)
        eigvec_3_ten = xp.zeros(derham.nbasis["2"][2], dtype=float)

        bc1_1 = derham.dirichlet_bc[0][0]
        bc1_2 = derham.dirichlet_bc[0][1]

        bc2_1 = derham.dirichlet_bc[1][0]
        bc2_2 = derham.dirichlet_bc[1][1]

        bc3_1 = derham.dirichlet_bc[2][0]
        bc3_2 = derham.dirichlet_bc[2][1]

        if derham.polar_ck == -1:
            n_v2_0 = [
                [derham.nbasis["2"][0][0] - bc1_1 - bc1_2, derham.nbasis["2"][0][1], derham.nbasis["2"][0][2]],
                [derham.nbasis["2"][1][0], derham.nbasis["2"][1][1] - bc2_1 - bc2_2, derham.nbasis["2"][1][2]],
                [derham.nbasis["2"][2][0], derham.nbasis["2"][2][1], derham.nbasis["2"][2][2] - bc3_1 - bc3_2],
            ]

            eigvec_1_ten[bc1_1 : derham.nbasis["2"][0][0] - bc1_2, :, :] = eig_vec_1.reshape(n_v2_0[0])
            eigvec_2_ten[:, bc2_1 : derham.nbasis["2"][1][1] - bc2_2, :] = eig_vec_2.reshape(n_v2_0[1])
            eigvec_3_ten[:, :, bc3_1 : derham.nbasis["2"][2][2] - bc3_2] = eig_vec_3.reshape(n_v2_0[2])

            self._eigvec_1 = eigvec_1_ten * params["scaling"]
            self._eigvec_2 = eigvec_2_ten * params["scaling"]
            self._eigvec_3 = eigvec_3_ten * params["scaling"]

        else:
            # split into polar/tensor product parts
            eig_vec_1 = xp.split(
                eig_vec_1,
                [
                    derham.Vh_pol["2"].n_polar[0] * nnz_tor[0],
                ],
            )
            eig_vec_2 = xp.split(
                eig_vec_2,
                [
                    derham.Vh_pol["2"].n_polar[1] * nnz_tor[1],
                ],
            )
            eig_vec_3 = xp.split(
                eig_vec_3,
                [
                    derham.Vh_pol["2"].n_polar[2] * nnz_tor[2],
                ],
            )

            # reshape polar coeffs
            eig_vec_1[0] = eig_vec_1[0].reshape(derham.Vh_pol["2"].n_polar[0], nnz_tor[0])
            eig_vec_2[0] = eig_vec_2[0].reshape(derham.Vh_pol["2"].n_polar[1], nnz_tor[1])
            eig_vec_3[0] = eig_vec_3[0].reshape(derham.Vh_pol["2"].n_polar[2], nnz_tor[2])

            # reshape tensor product coeffs
            n_v2_0 = [
                [
                    derham.nbasis["2"][0][0] - derham.Vh_pol["2"].n_rings[0] - bc1_2,
                    derham.nbasis["2"][0][1],
                    derham.nbasis["2"][0][2],
                ],
                [
                    derham.nbasis["2"][1][0] - derham.Vh_pol["2"].n_rings[1],
                    derham.nbasis["2"][1][1],
                    derham.nbasis["2"][1][2],
                ],
                [
                    derham.nbasis["2"][2][0] - derham.Vh_pol["2"].n_rings[2],
                    derham.nbasis["2"][2][1],
                    derham.nbasis["2"][2][2],
                ],
            ]

            eigvec_1_ten[derham.Vh_pol["2"].n_rings[0] : derham.nbasis["2"][0][0] - bc1_2, :, :] = eig_vec_1[1].reshape(
                n_v2_0[0],
            )
            eigvec_2_ten[derham.Vh_pol["2"].n_rings[1] :, :, :] = eig_vec_2[1].reshape(n_v2_0[1])
            eigvec_3_ten[derham.Vh_pol["2"].n_rings[2] :, :, :] = eig_vec_3[1].reshape(n_v2_0[2])

            self._eigvec_1 = [eig_vec_1[0] * params["scaling"], eigvec_1_ten * params["scaling"]]
            self._eigvec_2 = [eig_vec_2[0] * params["scaling"], eigvec_2_ten * params["scaling"]]
            self._eigvec_3 = [eig_vec_3[0] * params["scaling"], eigvec_3_ten * params["scaling"]]

    @property
    def u2(self):
        """List of eigenvectors"""
        return self._eigvec_1, self._eigvec_2, self._eigvec_3
