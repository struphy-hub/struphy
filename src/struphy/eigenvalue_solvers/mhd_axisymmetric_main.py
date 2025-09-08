def solve_mhd_ev_problem_2d(num_params, eq_mhd, n_tor, basis_tor="i", path_out=None):
    """
    Numerical solution of the ideal MHD eigenvalue problem for a given 2D axisymmetric equilibrium and a fixed toroial mode number.

    Parameters
    ----------
    num_params : dictionary
        numerical parameters :
            * Nel      : list of ints, number of elements in [s, chi] direction
            * p        : list of ints, spline degrees in [s, chi] direction
            * spl_kind : list of booleans, kind of splines in [s, chi] direction
            * nq_el    : list of ints, number of quadrature points per element in [s, chi] direction
            * nq_pr    : list of ints, number of quadrature points per projection interval in [s, chi] direction
            * bc       : list of strings, boundary conditions in [s, chi] direction
            * polar_ck : int, C^k continuity at pole

    eq_mhd : MHD equilibrium object
        the MHD equilibrium for which the spectrum shall be computed

    n_tor : int
        toroidal mode number

    basis_tor : string
        basis in toroidal direction :
            * r : A(s, chi)*cos(n_tor*2*pi*phi) + B(s, chi)*sin(n_tor*2*pi*phi),
            * i : A(s, chi)*exp(n_tor*2*pi*phi*i)

    path_out : string, optional
        if given, directory where to save the .npy eigenspectrum.
    """

    import os
    import time

    import numpy as np
    import scipy.sparse as spa

    from struphy.eigenvalue_solvers.mhd_operators import MHDOperators
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space

    print("\nStart of eigenspectrum calculation for toroidal mode number", n_tor)
    print("")
    print("MHD equilibrium:".ljust(20), eq_mhd)
    print("domain:".ljust(20), eq_mhd.domain)

    # print grid info
    print("\nGrid parameters:")
    print(f"number of elements :", num_params["Nel"])
    print(f"spline degrees     :", num_params["p"])
    print(f"periodic bcs       :", num_params["spl_kind"])
    print(f"hom. Dirichlet bc  :", num_params["bc"])
    print(f"GL quad pts (L2)   :", num_params["nq_el"])
    print(f"GL quad pts (hist) :", num_params["nq_pr"])
    print(f"polar Ck           :", num_params["polar_ck"])
    print("")

    # extract numerical parameters
    Nel = num_params["Nel"]
    p = num_params["p"]
    spl_kind = num_params["spl_kind"]
    nq_el = num_params["nq_el"]
    nq_pr = num_params["nq_pr"]
    bc = num_params["bc"]
    polar_ck = num_params["polar_ck"]

    # set up 1d spline spaces and corresponding projectors
    space_1d_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], bc[0])
    space_1d_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1], bc[1])

    space_1d_1.set_projectors(nq_pr[0])
    space_1d_2.set_projectors(nq_pr[1])

    # set up 2d tensor-product space
    space_2d = Tensor_spline_space(
        [space_1d_1, space_1d_2], polar_ck, eq_mhd.domain.cx[:, :, 0], eq_mhd.domain.cy[:, :, 0], n_tor, basis_tor
    )

    # set up 2d projectors
    space_2d.set_projectors("general")

    print("Initialization of FEM spaces done")

    # assemble mass matrix in V2 and V3 and apply boundary operators
    space_2d.assemble_Mk(eq_mhd.domain, "V2")
    space_2d.assemble_Mk(eq_mhd.domain, "V3")

    M2_0 = space_2d.B2.dot(space_2d.M2_mat.dot(space_2d.B2.T))
    M3_0 = space_2d.B3.dot(space_2d.M3_mat.dot(space_2d.B3.T))

    print("Assembly of mass matrices done")

    # create linear MHD operators
    mhd_ops = MHDOperators(space_2d, eq_mhd, 2)

    # assemble right-hand sides of degree of freedom projection matrices
    mhd_ops.assemble_dofs("EF")
    mhd_ops.assemble_dofs("MF")
    mhd_ops.assemble_dofs("PF")
    mhd_ops.assemble_dofs("PR")

    print("Assembly of projection matrices done")

    # assemble mass matrix weighted with 0-form density
    timea = time.time()
    mhd_ops.assemble_Mn()
    timeb = time.time()

    print("Assembly of weighted mass matrix done (density), time : ", timeb - timea)

    # assemble mass matrix weighted with J_eq x
    timea = time.time()
    mhd_ops.assemble_MJ()
    timeb = time.time()

    print("Assembly of weighted mass matrix done (current), time : ", timeb - timea)

    # final operators
    I1_11 = spa.kron(space_2d.projectors.I1_pol_0, space_2d.projectors.I_tor)
    I1_22 = spa.kron(space_2d.projectors.I0_pol_0, space_2d.projectors.H_tor)

    I2_11 = spa.kron(space_2d.projectors.I2_pol_0, space_2d.projectors.H_tor)
    I2_22 = spa.kron(space_2d.projectors.I3_pol_0, space_2d.projectors.I_tor)

    I3 = spa.kron(space_2d.projectors.I3_pol_0, space_2d.projectors.H_tor)

    I1 = spa.bmat([[I1_11, None], [None, I1_22]], format="csc")
    I2 = spa.bmat([[I2_11, None], [None, I2_22]], format="csc")
    I3 = I3.tocsc()

    EF = spa.linalg.inv(I1).dot(mhd_ops.dofs_EF).tocsr()
    PF = spa.linalg.inv(I2).dot(mhd_ops.dofs_PF).tocsr()
    PR = spa.linalg.inv(I3).dot(mhd_ops.dofs_PR).tocsr()

    L = -space_2d.D0.dot(PF) - (5 / 3 - 1) * PR.dot(space_2d.D0)

    print("Application of inverse interpolation matrices on projection matrices done")

    # set up eigenvalue problem MAT*u = omega^2*u
    MAT = (
        spa.linalg.inv(mhd_ops.Mn_mat.tocsc())
        .dot(
            EF.T.dot(space_2d.C0.conjugate().T.dot(M2_0.dot(space_2d.C0.dot(EF))))
            + mhd_ops.MJ_mat.dot(space_2d.C0.dot(EF))
            - space_2d.D0.conjugate().T.dot(M3_0.dot(L))
        )
        .toarray()
    )

    print("Assembly of final system matrix done --> start of eigenvalue calculation")

    omega2, U2_eig = np.linalg.eig(MAT)

    print("Eigenstates calculated")

    # save spectrum as .npy
    if path_out is not None:
        assert isinstance(path_out, str)

        if n_tor < 0:
            n_tor_str = str(n_tor)
        else:
            n_tor_str = "+" + str(n_tor)

        np.save(
            os.path.join(path_out, "spec_n_" + n_tor_str + ".npy"), np.vstack((omega2.reshape(1, omega2.size), U2_eig))
        )

    # or return eigenfrequencies, eigenvectors and system matrix
    else:
        return omega2, U2_eig, MAT


# command line interface
if __name__ == "__main__":
    import argparse
    import os
    import shutil

    import yaml

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Computes the complete eigenspectrum for a given axisymmetric MHD equilibrium."
    )

    parser.add_argument("n_tor", type=int, help="the toroidal mode number")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        metavar="FILE",
        help="parameter file (.yml) in current I/O path (default=parameters.yml)",
        default="parameters.yml",
    )

    parser.add_argument("--input-abs", type=str, metavar="FILE", help="parameter file (.yml), absolute path")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="DIR",
        help="output directory relative to current I/O path (default=sim_1)",
        default="sim_1",
    )

    parser.add_argument("--output-abs", type=str, metavar="DIR", help="output directory, absolute path")

    args = parser.parse_args()

    import struphy.utils.utils as utils

    # Read struphy state file
    state = utils.read_state()

    i_path = state["i_path"]
    o_path = state["o_path"]

    # create absolute i/o paths
    if args.input_abs is None:
        input_abs = os.path.join(i_path, args.input)
    else:
        input_abs = args.input_abs

    if args.output_abs is None:
        output_abs = os.path.join(o_path, args.output)
    else:
        output_abs = args.output_abs

    # create output folder (if it does not already exist)
    if not os.path.exists(output_abs):
        os.mkdir(output_abs)
        print("\nCreated folder " + output_abs)

    # copy parameter file to output folder
    if input_abs != os.path.join(output_abs, "parameters.yml"):
        shutil.copy2(input_abs, os.path.join(output_abs, "parameters.yml"))

    # load parameter file
    with open(input_abs) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # create domain and MHD equilibrium
    from struphy.io.setup import setup_domain_and_equil

    domain, mhd_equil = setup_domain_and_equil(params)

    # load grid parameters
    num_params = {
        "Nel": params["grid"]["Nel"][:2],
        "p": params["grid"]["p"][:2],
        "spl_kind": params["grid"]["spl_kind"][:2],
        "nq_el": params["grid"]["nq_el"][:2],
        "nq_pr": params["grid"]["nq_pr"][:2],
        "bc": params["grid"]["bc"][:2],
        "polar_ck": params["grid"]["polar_ck"],
    }

    # calculate eigenspectrum for given toroidal mode number and save result
    solve_mhd_ev_problem_2d(num_params, mhd_equil, n_tor=args.n_tor, basis_tor="i", path_out=output_abs)
