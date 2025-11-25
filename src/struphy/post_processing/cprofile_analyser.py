def get_cprofile_data(path, print_callers=None):
    """Prepare Cprofile data and save to "profile_dict.sav".

    Parameters
    ----------
    path : str
        Path to file "profile_tmp" (usually in output folder).

    print_callers : str
        Part of function name for which to show calling functions.
    """

    import os
    import pickle
    import pstats
    from pstats import SortKey

    p = pstats.Stats(os.path.join(path, "profile_tmp"))
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(0)

    if print_callers is not None:
        print("Print callers:")
        print("--------------")
        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_callers(print_callers)

    stdout = open(os.path.join(path, "profile_out.txt"), "w+")
    p = pstats.Stats(os.path.join(path, "profile_tmp"), stream=stdout)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
    stdout.close()
    name_li = None
    data_cprofile = dict()
    with open(os.path.join(path, "profile_out.txt")) as f:
        lines = f.readlines()
        # print(len(lines))
        search = False
        for n, line in enumerate(lines):
            if search:
                li = line.split()
                # print(li)
                # print(len(name_li), len(li))
                if len(li) == 0:
                    search = False
                    continue
                # print(name_li[0], li[0])
                # print(name_li[1], li[1])
                # print(name_li[2], li[2])
                # print(name_li[3], li[3])
                # print(name_li[4], li[4])
                data_cprofile[li[-1]] = {
                    name_li[0]: li[0],
                    name_li[1]: li[1],
                    name_li[2]: li[2],
                    name_li[3]: li[3],
                    name_li[4]: li[4],
                }
                # time.sleep(1)

            if "filename:lineno" in line:
                # print(n, repr(line))
                name_li = line.split()
                # print(name_li)
                search = True

    with open(os.path.join(path, "profile_dict.sav"), "w+b") as f:
        pickle.dump(data_cprofile, f)


def compare_cprofile_data(path, list_of_funcs=None):
    """Print Cprofile data from "profile_dict.sav" to screen (see get_cprofile_data).

    Parameters
    ----------
        path : str
            Path to file "profile_dict.sav" (usually in output folder).

        list_of_funcs : list
            Strings to watch for in "function name" of Cprofile data, allows to look at data of specific functions.
            If "None", the 50 functions with the longest cumtime are listed.
    """

    import os
    import pickle

    with open(os.path.join(path, "profile_dict.sav"), "rb") as f:
        data_cprofile = pickle.load(f)

    if list_of_funcs is None:
        print("-" * 76)
        print("function name".ljust(60), "cumulative time")
        print("-" * 76)
    else:
        print("-" * 76)
        print("function name, keywords: {}".format(list_of_funcs).ljust(60), "cumulative time")
        print("-" * 76)

    counter = 0
    for k, v in data_cprofile.items():
        counter += 1
        if list_of_funcs is None:
            print(k.ljust(60), v["cumtime"])
            if counter > 49:
                break
        elif any(func in k for func in list_of_funcs) and "dependencies_" not in k:
            print(k.ljust(60), v["cumtime"])


def replace_keys(d):
    """Replace keys from cprofile data with corresponding class names.

    Parameters
    ----------
        d : dict
            Dictionary with keys from cprofile_analyser.get_cprofile_data().

        list_of_funcs : list[str]
            Names for keyword search.
    """

    import os

    import psydac

    import struphy

    struphy_path = struphy.__path__[0]
    psydac_path = psydac.__path__[0]

    key_list = []
    for key in d.keys():
        key_list += [key]

    for key in key_list:
        if "propagators" in key or "stencil" in key or "block" in key:
            p1 = key.find(":")
            p2 = key.find("(")
            if p1 == -1 or p2 == -1:
                continue
            f_name = key[:p1]
            l_nr = int(key[p1 + 1 : p2])
            new_routine = key[p2:]

            # print(key, p1, f_name, l_nr, new_routine)

            found = False
            for root, dirs, files in os.walk(struphy_path):
                for name in files:
                    if name == f_name:
                        f_path = os.path.abspath(os.path.join(root, name))
                        # print(f_path)

                        li = []
                        with open(f_path, "r") as fp:
                            for n, line in enumerate(fp):
                                if line[0] == "c":
                                    li += [line[: line.find(":")]]
                                if n == l_nr - 1 and len(li) > 0:
                                    new_key = li[-1] + new_routine
                                    found = True
                                    # print(new_key)
                                    # print('xxx')
                                    break

            if not found:
                for root, dirs, files in os.walk(psydac_path):
                    for name in files:
                        if name == f_name:
                            f_path = os.path.abspath(os.path.join(root, name))
                            # print(f_path)

                            li = []
                            with open(f_path, "r") as fp:
                                for n, line in enumerate(fp):
                                    if line[0] == "c":
                                        li += [line[: line.find(":")]]
                                    if n == l_nr - 1 and len(li) > 0:
                                        new_key = li[-1] + new_routine
                                        found = True
                                        # print(new_key)
                                        # print('xxx')
                                        break

            if found:
                d[new_key] = d.pop(key)

    # sort dictionary by cumulative time
    return dict(sorted(d.items(), key=lambda item: float(item[1]["cumtime"]), reverse=True))
