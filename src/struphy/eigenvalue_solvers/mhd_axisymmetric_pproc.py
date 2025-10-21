# command line interface
def main():
    import argparse
    import os

    import yaml

    from struphy.utils.arrays import xp as np

    # parse arguments
    parser = argparse.ArgumentParser(description="Restrict a full .npy eigenspectrum to a range of eigenfrequencies.")

    parser.add_argument("-n", type=int, help="toroidal mode number", required=True)

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        metavar="DIR",
        help="directory with eigenspectrum (.npy), relative to current I/O path (default=sim_1)",
        default="sim_1",
    )

    parser.add_argument(
        "--input-abs", type=str, metavar="DIR", help="directory with eigenspectrum (.npy) file, absolute path"
    )

    parser.add_argument("lower", type=float, help="lower range of squared eigenfrequency")

    parser.add_argument("upper", type=float, help="upper range of squared eigenfrequency")

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

    # load spectrum and restrict to range
    if args.n < 0:
        n_tor_str = str(args.n)
    else:
        n_tor_str = "+" + str(args.n)

    spec_path = os.path.join(input_path, "spec_n_" + n_tor_str + ".npy")

    omega2, U2_eig = np.split(np.load(spec_path), [1], axis=0)
    omega2 = omega2.flatten()

    modes_ind = np.where((np.real(omega2) < args.upper) & (np.real(omega2) > args.lower))[0]

    omega2 = omega2[modes_ind]
    U2_eig = U2_eig[:, modes_ind]

    # save restricted spectrum
    np.save(
        os.path.join(input_path, "spec_" + str(args.lower) + "_" + str(args.upper) + "_n_" + n_tor_str + ".npy"),
        np.vstack((omega2.reshape(1, omega2.size), U2_eig)),
    )


if __name__ == "__main__":
    main()
