import pickle
import sys

import cunumpy as xp
import yaml
from matplotlib import pyplot as plt

from struphy.post_processing.cprofile_analyser import get_cprofile_data, replace_keys


def main():
    """
    TODO
    """
    print(sys.argv)

    # check --all option
    if sys.argv[1] == "true":
        list_of_funcs = None
    else:
        list_of_funcs = [
            "assemble_",
            "propagator",
            "accumulate",
            "_fill",
            "pusher",
            "update_ghost_regions",
            "solver",
            "class ",
            "stencil",
            "block",
            "integrate_in_time",
        ]
        print("\nKeyword search enabled with keywords:")
        print("-------------------------------------")
        print(list_of_funcs)

    print("\nLoad profiling data:")
    print("--------------------")

    # replace propagator keys or not
    do_replace_keys = sys.argv[2] == "true"

    # plot n_lines most time consuming calls in profiling analysis
    n_lines = int(sys.argv[3])

    # load data
    sim_names = []
    dicts_pre = []
    nproc = []
    Nel = []
    for path in sys.argv[4:]:
        print("")
        get_cprofile_data(path)

        sim_names += [path.split("/")[-2]]

        with open(path + "profile_dict.sav", "rb") as f:
            dicts_pre += [pickle.load(f)]

        with open(path + "meta.txt", "r") as f:
            lines = f.readlines()

        nproc += [int(lines[4].split()[-1])]

        with open(path + "parameters.yml", "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        Nel += [params["grid"]["Nel"]]

    # Nicer key names for output:
    dicts = []
    for d in dicts_pre:
        tmp = {}
        for key, val in d.items():
            # tmp[key] = float(val['cumtime'])
            tmp[key] = val

        if do_replace_keys:
            tmp2 = replace_keys(tmp)
        else:
            tmp2 = tmp

        dicts += [tmp2]

    # loop over keys (should be same in each dict)
    d_saved = {}
    print(
        "simulation".ljust(20)
        + "#proc".ljust(7)
        + "pos".ljust(5)
        + "function".ljust(70)
        + "ncalls".ljust(15)
        + "totime".ljust(15)
        + "percall".ljust(15)
        + "cumtime".ljust(15),
    )
    print("-" * 154)
    for position, key in enumerate(dicts[0].keys()):
        if list_of_funcs == None:
            for dict, sim_name, n, dim in zip(dicts, sim_names, nproc, Nel):
                string = f"{sim_name}".ljust(20) + f"{n}".ljust(7) + f"{position:2d}".ljust(5) + str(key.ljust(70))
                for value in dict[key].values():
                    string += str(value).ljust(15)
                    # if len(str(value)) < 7:
                    #     string += '\t\t'
                    # else:
                    #     string += '\t'
                print(string)
            print("")

            # TODO: Write an explanation of what this means
            if position == 50:
                sys.exit(1)

        elif any(func in key for func in list_of_funcs) and "dependencies_" not in key and "_dot" not in key:
            d_saved[key] = {"mpi_size": [], "Nel": [], "time": []}

            for dict, sim_name, n, dim in zip(dicts, sim_names, nproc, Nel):
                string = f"{sim_name}".ljust(20) + f"{n}".ljust(7) + f"{position:2d}".ljust(5) + str(key.ljust(70))
                for value in dict[key].values():
                    string += str(value).ljust(15)
                    # string += '\t\t'
                print(string)

                d_saved[key]["mpi_size"] += [n]
                d_saved[key]["Nel"] += [dim]
                d_saved[key]["time"] += [dict[key]]
            print("")

            # TODO: Write an explanation of what this means
            if position >= 200:
                sys.exit(1)

    # save profiling date in each sim path
    for path in sys.argv[4:]:
        with open(path + "comparison_dict.sav", "w+b") as f:
            pickle.dump(d_saved, f)

    # plot results
    fig = plt.figure(figsize=(10, 10))
    for n, (key, val) in enumerate(d_saved.items()):
        if n < n_lines and "__init__" not in key and "mass" not in key and "set_backend" not in key:
            # print(key, val)

            # strong scaling plot
            if all([Nel == val["Nel"][0] for Nel in val["Nel"]]):
                plt.loglog(val["mpi_size"], val["time"], label=key)
                plt.xlabel("mpi_size")
                plt.ylabel("time [s]")
                plt.title("Strong scaling for Nel=" + str(val["Nel"][0]) + " cells")
                plt.legend(loc="lower left")
                plt.loglog(val["mpi_size"], val["time"][0] / 2 ** xp.arange(len(val["time"])), "k--", alpha=0.3)
            # weak scaling plot
            else:
                plt.plot(val["mpi_size"], val["time"], label=key)
                plt.xlabel("mpi_size")
                plt.ylabel("time [s]")
                plt.title(
                    "Weak scaling for cells/mpi_size=" + str(xp.prod(val["Nel"][0]) / val["mpi_size"][0]) + "=const.",
                )
                plt.legend(loc="upper left")
                # plt.loglog(val['mpi_size'], val['time'][0]*xp.ones_like(val['time']), 'k--', alpha=0.3)
                plt.xscale("log")

    plt.show()


if __name__ == "__main__":
    main()
