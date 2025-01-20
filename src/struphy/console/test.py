from struphy.console.run import subp_run


def struphy_test(group, mpi=2, fast=False, with_desc=False, Tend=None, vrbose=False):
    """
    Run Struphy unit and/or code tests.

    Parameters
    ----------
    mpi : int
        Number of MPI processes used in tests (must be >1, default=2).

    fast : bool
        Whether to test models just in slab geometry.

    with_desc : bool
        Whether to include DESC equilibrium in unit tests (mem consuming).

    Tend : float
        If group is a model name, simulation end time in units of the model (default=0.015 with dt=0.005). Data is only saved at Tend if set.

    vrbose : bool
        Show full screen output.
    """

    if "unit" in group:
        # first run only tests that require single process
        cmd = [
            "pytest",
            "-k",
            "not _models and not _tutorial and not pproc",
        ]
        if with_desc:
            cmd += ["--with-desc"]
        if vrbose:
            cmd += ["--vrbose"]
        subp_run(cmd)

        # now run parallel unit tests
        cmd = [
            "mpirun",
            "-n",
            str(mpi),
            "pytest",
            "-k",
            "not _models and not _tutorial and not pproc",
            "--with-mpi",
        ]
        if with_desc:
            cmd += ["--with-desc"]
        if vrbose:
            cmd += ["--vrbose"]
        subp_run(cmd)

    elif "models" in group:
        cmd = [
            "mpirun",
            "-n",
            str(mpi),
            "pytest",
            "-k",
            "_models",
            "-s",
            "--with-mpi",
        ]
        if fast:
            cmd += ["--fast"]
        if vrbose:
            cmd += ["--vrbose"]
        subp_run(cmd)

        # test post processing of models
        cmd = [
            "pytest",
            "-k",
            "pproc",
            "-s",
        ]
        subp_run(cmd)

    elif group in {"fluid", "kinetic", "hybrid", "toy"}:
        cmd = [
            "mpirun",
            "-n",
            str(mpi),
            "pytest",
            "-k",
            group + "_models",
            "-s",
            "--with-mpi",
        ]
        if fast:
            cmd += ["--fast"]
        if vrbose:
            cmd += ["--vrbose"]
        subp_run(cmd)

        from struphy.models.tests.test_xxpproc import test_pproc_codes

        test_pproc_codes(group=group)

    else:
        from struphy.models.tests import test_fluid_models, test_hybrid_models, test_kinetic_models, test_toy_models
        from struphy.models.tests.test_xxpproc import test_pproc_codes

        objs = [
            test_toy_models.test_toy,
            test_fluid_models.test_fluid,
            test_kinetic_models.test_kinetic,
            test_hybrid_models.test_hybrid,
        ]
        for obj in objs:
            try:
                obj(("Cuboid", "HomogenSlab"), fast, vrbose, model=group, Tend=Tend)
                if fast:
                    print(
                        f"Fast is enabled, mappings other than Cuboid are skipped ...",
                    )
                else:
                    obj(
                        ("HollowTorus", "AdhocTorus"),
                        fast,
                        vrbose,
                        model=group,
                        Tend=Tend,
                    )
                    obj(
                        ("Tokamak", "EQDSKequilibrium"),
                        fast,
                        vrbose,
                        model=group,
                        Tend=Tend,
                    )
            except AttributeError:
                pass

        test_pproc_codes(group)
