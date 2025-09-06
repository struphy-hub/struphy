import inspect

import pytest


# @pytest.mark.mpi(min_size=2)
# @pytest.mark.parametrize('combine_comps', [('f0', 'f1'), ('f0', 'f3'), ('f1', 'f2'), ('fvec', 'f3'), ('f1', 'fvec', 'f0')])
@pytest.mark.parametrize("Nel", [[16, 16, 16]])
@pytest.mark.parametrize("p", [[2, 3, 4]])
@pytest.mark.parametrize("spl_kind", [[False, True, True]])
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 0.0, "r1": 4.0, "l2": 0.0, "r2": 5.0, "l3": 0.0, "r3": 6.0}],
        ["Colella", {"Lx": 4.0, "Ly": 5.0, "alpha": 0.07, "Lz": 6.0}],
        ["HollowCylinder", {"a1": 0.1}],
        ["HollowTorus", {"tor_period": 1}],
    ],
)
def test_init_modes(Nel, p, spl_kind, mapping, combine_comps=None, do_plot=False):
    """Test the initialization Field.initialize_coeffs with all "Modes" classes in perturbations.py."""

    import numpy as np
    from matplotlib import pyplot as plt
    from mpi4py import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.geometry.base import Domain
    from struphy.initial import perturbations

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Domain
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])
    assert isinstance(domain, Domain)

    # Derham
    derham = Derham(Nel, p, spl_kind, comm=comm)

    fields = {}
    for space, form in derham.space_to_form.items():
        fields[form] = derham.create_spline_function(form, space)

    form_scalar = ["0", "3", "physical_at_eta"]
    form_vector = ["1", "2", "v", "norm", "physical_at_eta"]

    # evaluation points
    e1 = np.linspace(0.0, 1.0, 30)
    e2 = np.linspace(0.0, 1.0, 40)
    e3 = np.linspace(0.0, 1.0, 50)
    eee1, eee2, eee3 = np.meshgrid(e1, e2, e3, indexing="ij")

    # mode paramters
    kwargs = {}
    kwargs["ms"] = [1, 0]
    kwargs["ns"] = [2, 0]
    kwargs["amps"] = [0.01, 0.0]

    ls = [0, 0]
    pfuns = ["sin", "sin"]

    pmap = domain.params
    if isinstance(domain, domains.Cuboid):
        Lx = pmap["r1"] - pmap["l1"]
        Ly = pmap["r2"] - pmap["l2"]
        Lz = pmap["r3"] - pmap["l3"]
        form_scalar += ["physical"]
        form_vector += ["physical"]
    elif isinstance(domain, domains.Colella):
        Lx = pmap["Lx"]
        Ly = pmap["Ly"]
        Lz = pmap["Lz"]
        form_scalar += ["physical"]
        form_vector += ["physical"]

    for key, val in inspect.getmembers(perturbations):
        if inspect.isclass(val):
            print(key, val)

            if "Modes" not in key:
                continue

            # skip impossible combinations
            if "Torus" not in key and (
                isinstance(domain, domains.HollowTorus) or isinstance(domain, domains.HollowCylinder)
            ):
                continue

            # functions to compare to
            if "Torus" in key:
                fun = val(**kwargs, pfuns=pfuns)
            else:
                fun = val(**kwargs, ls=ls)
                if isinstance(domain, domains.Cuboid) or isinstance(domain, domains.Colella):
                    fun_xyz = val(**kwargs, ls=ls, Lx=Lx, Ly=Ly, Lz=Lz)

            # single component is initialized
            for space, name in derham.space_to_form.items():
                if do_plot:
                    plt.figure(key + "_" + name + "-form_e1e2 " + mapping[0], figsize=(24, 16))
                    plt.figure(key + "_" + name + "-form_e1e3 " + mapping[0], figsize=(24, 16))

                if name in ("0", "3"):
                    for n, fun_form in enumerate(form_scalar):
                        params = {key: {"given_in_basis": fun_form}}

                        if "Modes" in key:
                            params[key]["ls"] = ls
                            params[key]["ms"] = kwargs["ms"]
                            params[key]["ns"] = kwargs["ns"]
                            params[key]["amps"] = kwargs["amps"]
                            if fun_form == "physical":
                                params[key]["Lx"] = Lx
                                params[key]["Ly"] = Ly
                                params[key]["Lz"] = Lz
                        else:
                            raise ValueError(f'Perturbation {key} not implemented, only "Modes" are testes.')

                        if "Torus" in key:
                            params[key].pop("ls")
                            if fun_form == "physical":
                                continue
                            params[key]["pfuns"] = pfuns

                        field = derham.create_spline_function(name, space, pert_params=params)
                        field.initialize_coeffs(domain=domain)

                        field_vals_xyz = domain.push(field, e1, e2, e3, kind=name)

                        x, y, z = domain(e1, e2, e3)
                        r = np.sqrt(x**2 + y**2)

                        if fun_form == "physical":
                            fun_vals_xyz = fun_xyz(x, y, z)
                        elif fun_form == "physical_at_eta":
                            fun_vals_xyz = fun(eee1, eee2, eee3)
                        else:
                            fun_vals_xyz = domain.push(fun, eee1, eee2, eee3, kind=fun_form)

                        error = np.max(np.abs(field_vals_xyz - fun_vals_xyz)) / np.max(np.abs(fun_vals_xyz))
                        print(f"{rank=}, {key=}, {name=}, {fun_form=}, {error=}")
                        assert error < 0.02

                        if do_plot:
                            plt.figure(key + "_" + name + "-form_e1e2 " + mapping[0])
                            plt.subplot(2, 4, n + 1)
                            if isinstance(domain, domains.HollowTorus):
                                plt.contourf(r[:, :, 0], z[:, :, 0], field_vals_xyz[:, :, 0])
                                plt.xlabel("R")
                                plt.ylabel("Z")
                            else:
                                plt.contourf(x[:, :, 0], y[:, :, 0], field_vals_xyz[:, :, 0])
                                plt.xlabel("x")
                                plt.ylabel("y")
                            plt.colorbar()
                            plt.title(f"init was {fun_form}, (m,n)=({kwargs['ms'][0]},{kwargs['ns'][0]})")
                            ax = plt.gca()
                            ax.set_aspect("equal", adjustable="box")

                            plt.subplot(2, 4, 4 + n + 1)
                            if isinstance(domain, domains.HollowTorus):
                                plt.contourf(r[:, :, 0], z[:, :, 0], fun_vals_xyz[:, :, 0])
                                plt.xlabel("R")
                                plt.ylabel("Z")
                            else:
                                plt.contourf(x[:, :, 0], y[:, :, 0], fun_vals_xyz[:, :, 0])
                                plt.xlabel("x")
                                plt.ylabel("y")
                            plt.colorbar()
                            plt.title(f"exact function")
                            ax = plt.gca()
                            ax.set_aspect("equal", adjustable="box")

                            plt.figure(key + "_" + name + "-form_e1e3 " + mapping[0])
                            plt.subplot(2, 4, n + 1)
                            if isinstance(domain, domains.HollowTorus):
                                plt.contourf(x[:, 0, :], y[:, 0, :], field_vals_xyz[:, 0, :])
                                plt.xlabel("x")
                                plt.ylabel("y")
                            else:
                                plt.contourf(x[:, 0, :], z[:, 0, :], field_vals_xyz[:, 0, :])
                                plt.xlabel("x")
                                plt.ylabel("z")
                            plt.colorbar()
                            plt.title(f"init was {fun_form}, (m,n)=({kwargs['ms'][0]},{kwargs['ns'][0]})")
                            ax = plt.gca()
                            ax.set_aspect("equal", adjustable="box")

                            plt.subplot(2, 4, 4 + n + 1)
                            if isinstance(domain, domains.HollowTorus):
                                plt.contourf(x[:, 0, :], y[:, 0, :], fun_vals_xyz[:, 0, :])
                                plt.xlabel("x")
                                plt.ylabel("y")
                            else:
                                plt.contourf(x[:, 0, :], z[:, 0, :], fun_vals_xyz[:, 0, :])
                                plt.xlabel("x")
                                plt.ylabel("z")
                            plt.colorbar()
                            plt.title(f"exact function")
                            ax = plt.gca()
                            ax.set_aspect("equal", adjustable="box")

                else:
                    for n, fun_form in enumerate(form_vector):
                        params = {
                            key: {
                                "given_in_basis": [fun_form] * 3,
                            }
                        }

                        if "Modes" in key:
                            params[key]["ms"] = [kwargs["ms"]] * 3
                            params[key]["ns"] = [kwargs["ns"]] * 3
                            params[key]["amps"] = [kwargs["amps"]] * 3
                        else:
                            raise ValueError(f'Perturbation {key} not implemented, only "Modes" are testes.')

                        if "Torus" in key:
                            # params[key].pop('ls')
                            if fun_form == "physical":
                                continue
                            params[key]["pfuns"] = [pfuns] * 3
                        else:
                            params[key]["ls"] = [ls] * 3
                            if fun_form == "physical":
                                params[key]["Lx"] = Lx
                                params[key]["Ly"] = Ly
                                params[key]["Lz"] = Lz
                            if isinstance(domain, domains.HollowTorus):
                                continue

                        field = derham.create_spline_function(name, space, pert_params=params)
                        field.initialize_coeffs(domain=domain)

                        f1_xyz, f2_xyz, f3_xyz = domain.push(field, e1, e2, e3, kind=name)
                        f_xyz = [f1_xyz, f2_xyz, f3_xyz]

                        x, y, z = domain(e1, e2, e3)
                        r = np.sqrt(x**2 + y**2)

                        # exact values
                        if fun_form == "physical":
                            fun1_xyz = fun_xyz(x, y, z)
                            fun2_xyz = fun_xyz(x, y, z)
                            fun3_xyz = fun_xyz(x, y, z)
                        elif fun_form == "physical_at_eta":
                            fun1_xyz = fun(eee1, eee2, eee3)
                            fun2_xyz = fun(eee1, eee2, eee3)
                            fun3_xyz = fun(eee1, eee2, eee3)
                        elif fun_form == "norm":
                            tmp1, tmp2, tmp3 = domain.transform(
                                [fun, fun, fun], eee1, eee2, eee3, kind=fun_form + "_to_v"
                            )
                            fun1_xyz, fun2_xyz, fun3_xyz = domain.push([tmp1, tmp2, tmp3], eee1, eee2, eee3, kind="v")
                        else:
                            fun1_xyz, fun2_xyz, fun3_xyz = domain.push([fun, fun, fun], eee1, eee2, eee3, kind=fun_form)

                        fun_xyz_vec = [fun1_xyz, fun2_xyz, fun3_xyz]

                        error = 0.0
                        for fi, funi in zip(f_xyz, fun_xyz_vec):
                            error += np.max(np.abs(fi - funi)) / np.max(np.abs(funi))
                        error /= 3.0
                        print(f"{rank=}, {key=}, {name=}, {fun_form=}, {error=}")
                        assert error < 0.02

                        if do_plot:
                            rn = len(form_vector)
                            for c, (fi, f) in enumerate(zip(f_xyz, fun_xyz_vec)):
                                plt.figure(key + "_" + name + "-form_e1e2 " + mapping[0])
                                plt.subplot(3, rn, rn * c + n + 1)
                                if isinstance(domain, domains.HollowTorus):
                                    plt.contourf(r[:, :, 0], z[:, :, 0], fi[:, :, 0])
                                    plt.xlabel("R")
                                    plt.ylabel("Z")
                                else:
                                    plt.contourf(x[:, :, 0], y[:, :, 0], fi[:, :, 0])
                                    plt.xlabel("x")
                                    plt.ylabel("y")
                                plt.colorbar()
                                plt.title(
                                    f"component {c + 1}, init was {fun_form}, (m,n)=({kwargs['ms'][0]},{kwargs['ns'][0]})"
                                )
                                ax = plt.gca()
                                ax.set_aspect("equal", adjustable="box")

                                plt.figure(key + "_" + name + "-form_e1e3 " + mapping[0])
                                plt.subplot(3, rn, rn * c + n + 1)
                                if isinstance(domain, domains.HollowTorus):
                                    plt.contourf(x[:, 0, :], y[:, 0, :], fi[:, 0, :])
                                    plt.xlabel("x")
                                    plt.ylabel("y")
                                else:
                                    plt.contourf(x[:, 0, :], z[:, 0, :], fi[:, 0, :])
                                    plt.xlabel("x")
                                    plt.ylabel("z")
                                plt.colorbar()
                                plt.title(
                                    f"component {c + 1}, init was {fun_form}, (m,n)=({kwargs['ms'][0]},{kwargs['ns'][0]})"
                                )
                                ax = plt.gca()
                                ax.set_aspect("equal", adjustable="box")

    if do_plot and rank == 0:
        plt.show()


if __name__ == "__main__":
    # mapping = ['Colella', {'Lx': 4., 'Ly': 5., 'alpha': .07, 'Lz': 6.}]
    # mapping = ['HollowCylinder', {'a1': 0.1}]
    # mapping = ['Cuboid', {'l1': 0., 'r1': 4., 'l2': 0., 'r2': 5., 'l3': 0., 'r3': 6.}]
    # test_init_modes([16, 16, 16], [2, 3, 4], [False, True, True],
    #                 mapping,
    #                 combine_comps=None,
    #                 do_plot=False)
    mapping = ["HollowTorus", {"tor_period": 1}]
    test_init_modes([16, 14, 14], [2, 3, 4], [False, True, True], mapping, combine_comps=None, do_plot=True)
