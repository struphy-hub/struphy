import sys

from struphy.utils.utils import subp_run


def struphy_compile(
    language,
    compiler,
    compiler_config,
    omp_pic,
    omp_feec,
    delete,
    status,
    verbose,
    dependencies,
    time_execution,
    yes,
):
    """Compile Struphy kernels. All files that contain "kernels" are detected automatically and saved to state.yml.

    Parameters
    ----------
    language : str
        Either "c" (default) or "fortran".

    compiler : str
        Either "GNU" (default), "intel", "PGI", "nvidia", or "LLVM"
        Only "GNU" is regularly tested at the moment.

    compiler_config : str
        Path to a JSON compiler file.

    omp_pic : bool
        Whether to compile PIC kernels with OpenMP (default=False).

    omp_feec : bool
        WHether to compile FEEC kernels with OpenMP (default=False).

    delete : bool
        If True, deletes generated Fortran/C files and .so files (default=False).

    status : bool
        If true, prints the current Struphy compilation status on screen.

    verbose : bool
        Call pyccel in verbose mode (default=False).

    dependencies : bool
        Whether to print Struphy kernels (to be compiled) and their dependencies on screen.

    time_execution: bool
        Prints the time spent in each section of the pyccelization (default=False).

    yes : bool
        Whether to say yes to prompt when changing the language.
    """

    import importlib.metadata
    import importlib.util
    import os
    import re
    import sysconfig

    import pyccel

    import struphy
    import struphy.dependencies as depmod
    import struphy.utils.utils as utils

    libpath = struphy.__path__[0]

    so_suffix = sysconfig.get_config_var("EXT_SUFFIX")

    if any([s == " " for s in libpath]):
        raise NameError(
            f'Stuphy installation path MUST NOT contain blank spaces. Please rename "{libpath}".',
        )

    # Read struphy state file
    state = utils.read_state()

    # collect kernels
    if "kernels" not in state:
        state["kernels"] = []
        for subdir, dirs, files in os.walk(libpath):
            for file in files:
                if (
                    "kernels" in file
                    and ".py" in file
                    and "_tmp.py" not in file
                    and "test" not in file
                    and "__pycache__" not in subdir
                ):
                    state["kernels"] += [os.path.join(subdir, file)]

        # set initial compiler infos to None
        state["last_used_language"] = None
        state["last_used_compiler"] = None
        state["last_used_omp_pic"] = None
        state["last_used_omp_feec"] = None

        utils.save_state(state)
    # source files
    sources = " ".join(state["kernels"])

    # actions
    if delete:
        # (change dir not to be in source path)
        print("\nDeleting .f90/.c and .so files ...")
        cmd = [
            "make",
            "clean",
            "-f",
            "compile_struphy.mk",
            "sources=" + sources,
        ]
        subp_run(cmd)
        print("Done.")

        print("\nDeleting psydac kernels ...")
        cmd = [
            "psydac-accelerate",
            "--cleanup",
        ]
        subp_run(cmd)
        print("Done.")

        print("\nDeleting state.yml ...")
        os.remove(os.path.join(libpath, "state.yml"))
        print("Done.")

    elif status:
        # update status
        count_c = 0
        count_f90 = 0
        list_not_compiled = [s for s in state["kernels"]]
        for subdir, _, files in os.walk(libpath):
            # print(f'{subdir = }')
            if subdir[-10:] == "__pyccel__" and "__epyccel__" not in subdir:
                dir_stem = "/".join(subdir.split("/")[:-1])
                # print(f'{dir_stem = }')
                for file in files:
                    if file[-2:] == ".c" and "wrapper" not in file and "bind_c_" not in file:
                        stem = file[:-2]
                        is_c = True
                    elif file[-4:] == ".f90" and "wrapper" not in file and "bind_c_" not in file:
                        stem = file[:-4]
                        is_c = False
                    else:
                        continue

                    py_file = stem + ".py"
                    matches = [ker for ker in state["kernels"] if py_file in ker and dir_stem in ker]
                    # print(f'{matches = }')
                    matching = None
                    for match in matches:
                        py_ker = match.split("/")[-1]
                        if py_ker == py_file:
                            matching = match
                    matching_so = matching.replace(".py", so_suffix)
                    # print(f'{matching_so = }')
                    if os.path.isfile(matching_so):
                        if is_c and state["last_used_language"] == "c":
                            count_c += 1
                        elif not is_c and state["last_used_language"] == "fortran":
                            count_f90 += 1
                        if matching in list_not_compiled:
                            list_not_compiled.remove(matching)

        n_kernels = len(state["kernels"])
        print("")
        print(f"{count_c} of {n_kernels} Struphy kernels are compiled with language C.")
        print(
            f"{count_f90} of {n_kernels} Struphy kernels are compiled with language Fortran.",
        )
        print(f"{n_kernels - count_c - count_f90} of {n_kernels} Struphy kernels are not compiled (pure Python).")
        print(
            f"\ncompiler={state['last_used_compiler']}\nflags_omp_pic={state['last_used_omp_pic']}\nflags_omp_feec={state['last_used_omp_feec']}",
        )
        if len(list_not_compiled) > 0:
            print("\nPure Python kernels (not compiled) are:")
            for ker in list_not_compiled:
                print(ker)

        state["kernels_n"] = n_kernels
        state["compiled_in_c"] = count_c
        state["compiled_in_fortran"] = count_f90
        state["compiled_not_n"] = n_kernels - count_c - count_f90
        state["compiled_not"] = list_not_compiled

        utils.save_state(state)

    elif dependencies:
        print("\nAuto-detect dependencies ...")
        for ker in state["kernels"]:
            deps = depmod.get_dependencies(ker.replace(".py", so_suffix))
            deps_li = deps.split(" ")
            print("-" * 28)
            print(f"{ker =}")
            for dep in deps_li:
                print(f"{dep =}")

    else:
        # struphy and psydac (change dir not to be in source path)
        flag_omp_pic = ""
        flag_omp_feec = ""
        if omp_pic:
            flag_omp_pic = " --openmp"
        if omp_feec:
            flag_omp_feec = " --openmp"

        # pyccel flags
        flags = "--language=" + language

        if compiler_config:
            flags += " --compiler-config=" + compiler_config
        else:
            flags += " --compiler-family=" + compiler

        if time_execution:
            flags += " --time-execution"

        # state
        if state["last_used_language"] not in (language, None):
            if yes:
                yesno = "Y"
            else:
                yesno = input(
                    f"Kernels compiled in language {state['last_used_language']} exist, will be deleted, continue (Y/n)?",
                )

            if yesno in ("", "Y", "y", "yes"):
                cmd = [
                    "struphy",
                    "compile",
                    "--delete",
                ]
                subp_run(cmd)
            else:
                return

        state["last_used_language"] = language
        state["last_used_compiler"] = compiler
        state["last_used_omp_pic"] = flag_omp_pic
        state["last_used_omp_feec"] = flag_omp_feec

        utils.save_state(state)

        # Compile psydac kernels, note that this is a special function call in psydac-for-struphy.
        # Otherwise, psydac only allows for recompiling the kernels when installed in editable mode.

        print("\nCompiling Psydac kernels ...")
        cmd = [
            "psydac-accelerate",
            "--language=" + language,
        ]
        if compiler_config:
            cmd += ["--compiler-config=" + compiler_config]
        else:
            cmd += ["--compiler-family=" + compiler]

        subp_run(cmd)

        # Compile struphy kernels
        _li = pyccel.__version__.split(".")
        _num = int(_li[0]) * 100 + int(_li[1]) * 10 + int(_li[2])
        if _num >= 180:
            flags += " --conda-warnings=off"

        if verbose:
            flags += " --verbose"

        # compilation
        print("\nCompiling Struphy kernels ...")
        cmd = [
            "make",
            "-f",
            "compile_struphy.mk",
            "sources=" + sources,
            "flags=" + flags,
            "flags_openmp_pic=" + flag_omp_pic,
            "flags_openmp_mhd=" + flag_omp_feec,
        ]
        subp_run(cmd)
        print("Done.")

        cmd = [
            "struphy",
            "compile",
            "--status",
        ]
        subp_run(cmd)

        # collect available models
        print("")
        cmd = [
            "struphy",
            "--refresh-models",
        ]
        subp_run(cmd)
