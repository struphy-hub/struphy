import logging

logger = logging.getLogger("struphy")


def struphy_profile(dirs, replace, all, n_lines, print_callers, savefig):
    """
    Profile finished Struphy runs.
    """

    import os
    import pickle

    import cunumpy as xp
    import yaml
    from matplotlib import pyplot as plt

    from struphy.post_processing.cprofile_analyser import get_cprofile_data, replace_keys
    from struphy.utils import utils

    # Read struphy state file
    state = utils.read_state()

    o_path = state["o_path"]

    # absolute paths
    abs_paths = []
    for d in dirs:
        abs_paths += [os.path.join(o_path, d)]

    # define the function filter
    list_of_funcs = [
        "assemble_",
        "propagator",
        "accumulate",
        "_fill",
        "pusher",
        "update_ghost_regions",
        "mpi_sort_markers",
        "apply_kinetic_bc",
        "solver",
        "class ",
        "stencil",
        "block",
        "integrate",
    ]

    # check --all option
    if all:
        list_of_funcs = None
    else:
        logger.info("\nKeyword search enabled with the following filter:")
        logger.info("-------------------------------------------------")
        logger.info(list_of_funcs)

    logger.info("\nLoad profiling data:")
    logger.info("--------------------")

    # load data
    sim_names = []
    dicts_pre = []
    nproc = []
    num_elements = []
    for path in abs_paths:
        logger.info("")
        get_cprofile_data(path, print_callers)

        sim_names += [path.split("/")[-1]]

        with open(os.path.join(path, "profile_dict.sav"), "rb") as f:
            dicts_pre += [pickle.load(f)]

        with open(os.path.join(path, "meta.txt"), "r") as f:
            lines = f.readlines()

        nproc += [int(lines[4].split()[-1])]

        with open(os.path.join(path, "parameters.yml"), "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        if "grid" in params:
            num_elements += [params["grid"]["num_elements"]]
        else:
            num_elements += [0]

    # Nicer key names for output:
    dicts = []
    for d in dicts_pre:
        tmp = {}
        for key, val in d.items():
            # tmp[key] = float(val['cumtime'])
            tmp[key] = val

        if replace:
            tmp2 = replace_keys(tmp)
        else:
            tmp2 = tmp

        dicts += [tmp2]

    # runtime of the main
    runtime = dicts[0]["main.py:1(<module>)"]["cumtime"]

    # loop over keys (should be same in each dict)
    d_saved = {}
    logger.info(
        "simulation".ljust(20)
        + "#proc".ljust(7)
        + "pos".ljust(5)
        + "function".ljust(70)
        + "ncalls".ljust(15)
        + "tottime".ljust(15)
        + "percall".ljust(15)
        + "cumtime".ljust(15),
    )
    logger.info("-" * 154)
    for position, key in enumerate(dicts[0].keys()):
        if list_of_funcs is None:
            for dict, sim_name, n, dim in zip(dicts, sim_names, nproc, num_elements):
                string = f"{sim_name}".ljust(20) + f"{n}".ljust(7) + f"{position:2d}".ljust(5) + str(key.ljust(70))
                for value in dict[key].values():
                    string += str(value).ljust(15)
                    # if len(str(value)) < 7:
                    #     string += '\t\t'
                    # else:
                    #     string += '\t'
                logger.info(string)
            logger.info("")

            if position == 50:
                break

        elif any(func in key for func in list_of_funcs) and "dependencies_" not in key and "_dot" not in key:
            d_saved[key] = {"mpi_size": [], "num_elements": [], "time": [], "ncalls": []}

            for dict, sim_name, n, dim in zip(dicts, sim_names, nproc, num_elements):
                string = f"{sim_name}".ljust(20) + f"{n}".ljust(7) + f"{position:2d}".ljust(5) + str(key.ljust(70))
                for value in dict[key].values():
                    string += str(value).ljust(15)
                    # string += '\t\t'
                logger.info(string)

                d_saved[key]["mpi_size"] += [n]
                d_saved[key]["num_elements"] += [dim]
                d_saved[key]["time"] += [dict[key]["cumtime"]]
                d_saved[key]["ncalls"] += [dict[key]["ncalls"]]
            logger.info("")

            if position >= 200:
                break

    # save profiling date in each sim path
    for path in abs_paths:
        with open(os.path.join(path, "comparison_dict.sav"), "w+b") as f:
            pickle.dump(d_saved, f)

    # plot results
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({"font.size": 10})

    for n, (key, val) in enumerate(d_saved.items()):
        if n < n_lines and "__init__" not in key and "mass" not in key and "set_backend" not in key:
            # runtime of the main
            runtime = float(dicts[0]["main.py:1(<module>)"]["cumtime"])

            # calculate relative cumtime and the ratio (cumtime/runtime)
            min_time = float(val["time"][0])
            relative_times = []
            ratio = []
            for t in val["time"]:
                relative_times.append(float(t) / min_time)
                ratio.append(str(int(float(t) / runtime * 100)) + "%")

            # strong scaling plot
            if xp.all([num_elements == val["num_elements"][0] for num_elements in val["num_elements"]]):
                # ideal scaling
                if n == 0:
                    ax.loglog(val["mpi_size"], 1 / 2 ** xp.arange(len(val["time"])), "k--", alpha=0.3, label="ideal")

                # print average time per one time step
                if "integrate" in key:
                    textstr = "\nAverage time per" + " Δt :" + "\n"

                    for m in range(len(val["mpi_size"])):
                        avg_time_dt = round(float(val["time"][m]) / float(val["ncalls"][m]), 2)
                        textstr += "MPI #" + str(val["mpi_size"][m]) + " : " + str(avg_time_dt) + " s \n"

                    ax.text(
                        0.97,
                        0.91,
                        textstr,
                        fontsize=13,
                        transform=ax.transAxes,
                        verticalalignment="center",
                        horizontalalignment="right",
                    )

                    continue

                ax.loglog(val["mpi_size"], relative_times, "o-", label=key + ", " + "".join(ratio[0]))
                # plt.loglog(val['mpi_size'], val['time'], label=key)
                ax.set_xlabel("MPI #", fontsize=13)
                ax.set_ylabel("Relative time [Total time with MPI #" + str(val["mpi_size"][0]) + "]", fontsize=13)
                ax.set(title="Strong scaling for num_elements=" + str(val["num_elements"][0]) + " cells")
                ax.legend(loc="lower left")

            # weak scaling plot
            else:
                ax.plot(val["mpi_size"], val["time"], label=key)
                ax.set_xlabel("mpi_size")
                ax.set_ylabel("time [s]")
                ax.set(
                    title="Weak scaling for cells/mpi_size="
                    + str(xp.prod(val["num_elements"][0]) / val["mpi_size"][0])
                    + "=const.",
                )
                ax.legend(loc="upper left")
                # ax.loglog(val['mpi_size'], val['time'][0]*xp.ones_like(val['time']), 'k--', alpha=0.3)
                ax.set_xscale("log")

    if savefig is None:
        plt.show()

    else:
        # savefig paths
        save_path = os.path.join(o_path, savefig)

        plt.savefig(save_path)
