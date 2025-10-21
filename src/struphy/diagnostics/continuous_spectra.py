def get_mhd_continua_2d(space, domain, omega2, U_eig, m_range, omega_A, div_tol, comp_sound):
    """
    Get the eigenfrequencies omega^2/omega_A^2 in the range (0, 1) sorted by shear Alfvén modes and slow sound modes.

    Parameters
    ----------
    space : struphy.eigenvalue_solvers.spline_space.Tensor_spline_space
        2d finite element B-spline space.

    domain : struphy.geometry.base.Domain
        The domain in which the eigenvalue problem has been solved.

    omega2 : array-like
        Eigenfrequencies obtained from eigenvalue solver.

    U_eig : array-like
        Eigenvectors obtained from eigenvalue solver.

    m_range : list
        Range of poloidal mode numbers that shall be identified.

    omega_A : float
        On-axis Alfvén frequency B0/R0.

    div_tol : float
        Threshold for the maximum divergence of an eigenmode below which it is considered to be an Alfvénic mode.

    comp_sound : int
        The component that is used for the slow sound mode analysis (2 : 2nd component or 3 : third component).

    Returns
    -------
    a_spec : list of 2d numpy arrays
        the radial location a_spec[m][0], squared eigenfrequencis a_spec[m][1] and global mode index a_spec[m][2] corresponding to shear Alfvén modes for each poloidal mode number m in m_range.

    s_spec : list of 2d numpy arrays
        the radial location s_spec[m][0], squared eigenfrequencis s_spec[m][1] and global mode index s_spec[m][2] corresponding to slow sound modes for each poloidal mode number m in m_range.
    """

    import struphy.bsplines.bsplines as bsp
    from struphy.utils.arrays import xp as np

    # greville points in radial direction (s)
    gN_1 = bsp.greville(space.T[0], space.p[0], space.spl_kind[0])
    gD_1 = bsp.greville(space.t[0], space.p[0] - 1, space.spl_kind[0])

    # greville points in angular direction (chi)
    gN_2 = bsp.greville(space.T[1], space.p[1], space.spl_kind[1])
    gD_2 = bsp.greville(space.t[1], space.p[1] - 1, space.spl_kind[1])

    # poloidal mode numbers
    ms = np.arange(m_range[1] - m_range[0] + 1) + m_range[0]

    # grid for normalized Jacobian determinant
    det_df = domain.jacobian_det(gD_1, gD_2, 0.0)

    # remove singularity for polar domains
    if domain.pole:
        det_df = det_df[1:, :]

    det_df_norm = det_df / det_df.max()

    # Alfvén and sound spectra (location, squared frequency, mode number)
    a_spec = [[[], [], []] for m in ms]
    s_spec = [[[], [], []] for m in ms]

    # only consider eigenmodes in range omega^2/omega_A^2 = [0, 1]
    modes_ind = np.where((np.real(omega2) / omega_A**2 < 1.0) & (np.real(omega2) / omega_A**2 > 0.0))[0]

    for i in range(modes_ind.size):
        # determine whether it's an Alfvén branch or sound branch by checking DIV(U)
        if space.ck == 0:
            divU = space.D0.dot(U_eig[:, modes_ind[i]])[space.NbaseD[1] :]
        else:
            divU = space.D0.dot(U_eig[:, modes_ind[i]])

        # Alfvén branch
        if abs(divU / det_df_norm.flatten()).max() < div_tol:
            # get FEM coefficients (1st component)
            U2_1_coeff = space.extract_2(U_eig[:, modes_ind[i]])[0]

            if space.basis_tor == "i":
                U2_1_coeff = U2_1_coeff[:, :, 0]
            else:
                U2_1_coeff = (U2_1_coeff[:, :, 0] - 1j * U2_1_coeff[:, :, 1]) / 2

            # determine radial location of singularity by looking for a peak in eigenfunction U2_1
            s_ind = np.unravel_index(np.argmax(abs(U2_1_coeff)), U2_1_coeff.shape)[0]
            s = gN_1[s_ind]

            # perform fft to determine m
            U2_1_fft = np.fft.fft(U2_1_coeff)

            # determine m by looking for peak in Fourier spectrum at singularity
            m = int((np.fft.fftfreq(U2_1_fft[s_ind].size) * U2_1_fft[s_ind].size)[np.argmax(abs(U2_1_fft[s_ind]))])

            ## perform shift for negative m
            # if m >= (space.Nel[1] + 1)//2:
            #    m -= space.Nel[1]

            # add to spectrum if found m is inside m_range
            for j in range(ms.size):
                if ms[j] == m:
                    a_spec[j][0].append(s)
                    a_spec[j][1].append(np.real(omega2[modes_ind[i]]))
                    a_spec[j][2].append(modes_ind[i])

        # Sound branch
        else:
            # get FEM coefficients (2nd component or 3rd component)
            U2_coeff = space.extract_2(U_eig[:, modes_ind[i]])[comp_sound - 1]

            if space.basis_tor == "i":
                U2_coeff = U2_coeff[:, :, 0]
            else:
                U2_coeff = (U2_coeff[:, :, 0] - 1j * U2_coeff[:, :, 1]) / 2

            # determine radial location of singularity by looking for a peak in eigenfunction (U2_2 or U2_3)
            s_ind = np.unravel_index(np.argmax(abs(U2_coeff)), U2_coeff.shape)[0]
            s = gD_1[s_ind]

            # perform fft to determine m
            U2_fft = np.fft.fft(U2_coeff)

            # determine m by looking for peak in Fourier spectrum at singularity
            m = int((np.fft.fftfreq(U2_fft[s_ind].size) * U2_fft[s_ind].size)[np.argmax(abs(U2_fft[s_ind]))])

            ## perform shift for negative m
            # if m >= (space.Nel[1] + 1)//2:
            #    m -= space.Nel[1]

            # add to spectrum if found m is inside m_range
            for j in range(ms.size):
                if ms[j] == m:
                    s_spec[j][0].append(s)
                    s_spec[j][1].append(np.real(omega2[modes_ind[i]]))
                    s_spec[j][2].append(modes_ind[i])

    # convert to array
    for j in range(ms.size):
        a_spec[j] = np.array(a_spec[j])
        s_spec[j] = np.array(s_spec[j])

    return a_spec, s_spec


# command line interface
if __name__ == "__main__":
    import argparse
    import glob
    import os
    import shutil

    import yaml

    from struphy.utils.arrays import xp as np

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Looks for eigenmodes in a given MHD eigenspectrum in a certain poloidal mode number range and plots the continuous shear Alfvén and slow sound spectra (frequency versus radial-like coordinate)."
    )

    parser.add_argument("m_l_alfvén", type=int, help="lower bound of poloidal mode number range for Alfvénic modes")

    parser.add_argument("m_u_alfvén", type=int, help="upper bound of poloidal mode number range for Alfvénic modes")

    parser.add_argument("m_l_sound", type=int, help="lower bound of poloidal mode number range for slow sound modes")

    parser.add_argument("m_u_sound", type=int, help="upper bound of poloidal mode number range for slow sound modes")

    parser.add_argument("-n", "--name", type=str, metavar="FILE", help="name of .npy file to analyze", required=True)

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        metavar="DIR",
        help="directory with eigenspectrum (.npy) and parameter (.yml) file, relative to <install_path>/struphy/io/out/ (default=sim_1)",
        default="sim_1",
    )

    parser.add_argument(
        "--input-abs",
        type=str,
        metavar="DIR",
        help="directory with eigenspectrum (.npy) and parameter (.yml) file, absolute path",
    )

    parser.add_argument(
        "-t",
        "--tol",
        type=float,
        metavar="tol",
        help="threshold for the maximum divergence of an eigenmode below which it is considered to be an Alfvénic mode (default=0.05)",
        default=0.05,
    )

    parser.add_argument(
        "-c",
        "--comp-sound",
        type=int,
        metavar="n",
        help="the component that is used for the slow sound mode analysis (2 : 2nd component or 3 : 3rd component, default=3)",
        default=3,
    )

    args = parser.parse_args()

    import struphy.utils.utils as utils

    # Read struphy state file
    state = utils.read_state()

    o_path = state["o_path"]

    # create absolute input folder path
    if args.input_abs is None:
        input_path = os.path.join(o_path, args.input)
    else:
        input_path = args.input_abs

    # absolute path of .npy spectrum and toroidal mode number
    spec_path = os.path.join(input_path, args.name)
    n_tor = int(os.path.split(spec_path)[-1][-6:-4])

    # load parameter file
    with open(os.path.join(input_path, "parameters.yml")) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # create domain and MHD equilibrium
    from struphy.io.setup import setup_domain_and_equil

    domain, mhd_equil = setup_domain_and_equil(params)

    # get MHD equilibrium parameters
    for k, v in params["fluid_background"].items():
        params_mhd = v

    # set up spline spaces
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space

    print("Toroidal mode number : ", n_tor)

    Nel = params["grid"]["Nel"]
    p = params["grid"]["p"]
    spl_kind = params["grid"]["spl_kind"]
    nq_el = params["grid"]["nq_el"]
    dirichlet_bc = params["grid"]["dirichlet_bc"]
    polar_ck = params["grid"]["polar_ck"]

    fem_1d_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], dirichlet_bc[0])
    fem_1d_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1], dirichlet_bc[1])

    fem_2d = Tensor_spline_space(
        [fem_1d_1, fem_1d_2], polar_ck, domain.cx[:, :, 0], domain.cy[:, :, 0], n_tor=n_tor, basis_tor="i"
    )

    # load and analyze spectrum
    omega2, U2_eig = np.split(np.load(spec_path), [1], axis=0)
    omega2 = omega2.flatten()

    m_range_alfven = [args.m_l_alfvén, args.m_u_alfvén]
    m_range_sound = [args.m_l_sound, args.m_u_sound]

    omegaA = params_mhd["B0"] / params_mhd["R0"]

    A, S = get_mhd_continua_2d(
        fem_2d,
        domain,
        omega2,
        U2_eig,
        [min(m_range_alfven[0], m_range_sound[0]), max(m_range_alfven[1], m_range_sound[1])],
        omegaA,
        args.tol,
        args.comp_sound,
    )

    # plot results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3)
    fig.set_figheight(12)
    fig.set_figwidth(14)

    etaplot = [np.linspace(0.0, 1.0, 201), np.linspace(0.0, 1.0, 101)]

    etaplot[0][0] += 1e-5

    xplot = domain(etaplot[0], etaplot[1], 0.0)[0]
    yplot = domain(etaplot[0], etaplot[1], 0.0)[1]

    # plot equilibrium profiles for (s, chi=0)
    ax[0, 0].plot(etaplot[0], mhd_equil.b2_3(etaplot[0], 0.0, 0.0) / mhd_equil.b2_2(etaplot[0], 0.0, 0.0))
    ax[0, 1].plot(etaplot[0], mhd_equil.p_xyz(xplot[:, 0], 0.0, 0.0).squeeze())
    ax[0, 2].plot(etaplot[0], mhd_equil.n_xyz(xplot[:, 0], 0.0, 0.0).squeeze())

    ax[0, 0].set_xlabel("$s$")
    ax[0, 1].set_xlabel("$s$")
    ax[0, 2].set_xlabel("$s$")

    ax[0, 0].set_ylabel("$q$")
    ax[0, 1].set_ylabel("$p$")
    ax[0, 2].set_ylabel("$n$")

    ax[0, 0].set_title("Safety factor")
    ax[0, 1].set_title("Pressure")
    ax[0, 2].set_title("Number density")

    # plot grid
    domain_name = domain.__class__.__name__

    xgrid = domain(fem_2d.el_b[0], fem_2d.el_b[1], 0.0)

    if "Torus" in domain_name or domain_name == "GVECunit" or domain_name == "Tokamak":
        for i in range(xgrid[0].shape[0]):
            ax[1, 0].plot(xgrid[0][i, :], xgrid[2][i, :], "tab:blue", alpha=0.5)

        for i in range(xgrid[0].shape[1]):
            ax[1, 0].plot(xgrid[0][:, i], xgrid[2][:, i], "tab:blue", alpha=0.5)

        ax[1, 0].set_xlabel("x [m]")
        ax[1, 0].set_ylabel("z [m]")

    else:
        for i in range(xgrid[0].shape[0]):
            ax[1, 0].plot(xgrid[0][i, :], xgrid[1][i, :], "tab:blue", alpha=0.5)

        for i in range(xgrid[0].shape[1]):
            ax[1, 0].plot(xgrid[0][:, i], xgrid[1][:, i], "tab:blue", alpha=0.5)

        ax[1, 0].set_xlabel("x [m]")
        ax[1, 0].set_ylabel("y [m]")

    ax[1, 0].set_title(r"Grid : $N_\mathrm{el}=$" + str(fem_2d.Nel[:2]), pad=10)

    # plot shear Alfvén continuum in range omega^2 = [0, omega_A^2]
    for m in range(m_range_alfven[0], m_range_alfven[1] + 1):
        ax[1, 1].plot(A[m][0], A[m][1], "+", label="m = " + str(m))

    ax[1, 1].set_xlabel("$s$")
    ax[1, 1].set_ylabel(r"$\omega^2$")
    ax[1, 1].set_xlim((0.0, 1.0))
    ax[1, 1].set_ylim((0.0, omegaA**2 + 0.02 * omegaA**2))
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("Shear Alfvén continuum", pad=10)
    ax[1, 1].set_xticks([0.0, 0.5, 1.0])

    # plot shear Alfvén continuum in given range % of omega_A
    for m in range(m_range_sound[0], m_range_sound[1] + 1):
        ax[1, 2].plot(S[m][0], S[m][1], "+", label="m = " + str(m))

    ax[1, 2].set_xlabel("$s$")
    ax[1, 2].set_ylabel(r"$\omega^2$")
    ax[1, 2].set_xlim((0.0, 1.0))
    ax[1, 2].set_ylim((0.0, 0.1 * omegaA**2))
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].set_title("Slow sound continuum", pad=10)
    ax[1, 2].set_xticks([0.0, 0.5, 1.0])
    # =========================================================================

    plt.subplots_adjust(wspace=0.4)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
