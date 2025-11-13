import inspect
from copy import deepcopy

import pytest


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

    import cunumpy as xp
    from matplotlib import pyplot as plt
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.geometry.base import Domain
    from struphy.initial import perturbations
    from struphy.initial.base import Perturbation
    from struphy.models.variables import FEECVariable

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
    e1 = xp.linspace(0.0, 1.0, 30)
    e2 = xp.linspace(0.0, 1.0, 40)
    e3 = xp.linspace(0.0, 1.0, 50)
    eee1, eee2, eee3 = xp.meshgrid(e1, e2, e3, indexing="ij")

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
        if inspect.isclass(val) and val.__module__ == perturbations.__name__:
            print(key, val)

            if key not in ("ModesCos", "ModesSin", "TorusModesCos", "TorusModesSin"):
                continue

            # skip impossible combinations
            if "Torus" not in key and (
                isinstance(domain, domains.HollowTorus) or isinstance(domain, domains.HollowCylinder)
            ):
                continue

            # instance of perturbation
            if "Torus" in key:
                perturbation = val(**kwargs, pfuns=pfuns)
            else:
                perturbation = val(**kwargs, ls=ls)
                if isinstance(domain, domains.Cuboid) or isinstance(domain, domains.Colella):
                    perturbation_xyz = val(**kwargs, ls=ls, Lx=Lx, Ly=Ly, Lz=Lz)
            assert isinstance(perturbation, Perturbation)

            # single component is initialized
            for space, form in derham.space_to_form.items():
                if do_plot:
                    plt.figure(key + "_" + form + "-form_e1e2 " + mapping[0], figsize=(24, 16))
                    plt.figure(key + "_" + form + "-form_e1e3 " + mapping[0], figsize=(24, 16))

                if form in ("0", "3"):
                    for n, fun_form in enumerate(form_scalar):
                        if "Torus" in key and fun_form == "physical":
                            continue

                        if "Modes" in key and fun_form == "physical":
                            perturbation._Lx = Lx
                            perturbation._Ly = Ly
                            perturbation._Lz = Lz
                        else:
                            perturbation._Lx = 1.0
                            perturbation._Ly = 1.0
                            perturbation._Lz = 1.0
                        # use the setter
                        perturbation.given_in_basis = fun_form

                        var = FEECVariable(space=space)
                        var.add_perturbation(perturbation)
                        var.allocate(derham, domain)
                        field = var.spline

                        field_vals_xyz = domain.push(field, e1, e2, e3, kind=form)

                        x, y, z = domain(e1, e2, e3)
                        r = xp.sqrt(x**2 + y**2)

                        if fun_form == "physical":
                            fun_vals_xyz = perturbation_xyz(x, y, z)
                        elif fun_form == "physical_at_eta":
                            fun_vals_xyz = perturbation(eee1, eee2, eee3)
                        else:
                            fun_vals_xyz = domain.push(perturbation, eee1, eee2, eee3, kind=fun_form)

                        error = xp.max(xp.abs(field_vals_xyz - fun_vals_xyz)) / xp.max(xp.abs(fun_vals_xyz))
                        print(f"{rank=}, {key=}, {form=}, {fun_form=}, {error=}")
                        assert error < 0.02

                        if do_plot:
                            plt.figure(key + "_" + form + "-form_e1e2 " + mapping[0])
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
                            plt.title("exact function")
                            ax = plt.gca()
                            ax.set_aspect("equal", adjustable="box")

                            plt.figure(key + "_" + form + "-form_e1e3 " + mapping[0])
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
                            plt.title("exact function")
                            ax = plt.gca()
                            ax.set_aspect("equal", adjustable="box")

                else:
                    for n, fun_form in enumerate(form_vector):
                        if "Torus" in key and fun_form == "physical":
                            continue

                        if "Modes" in key and fun_form == "physical":
                            perturbation._Lx = Lx
                            perturbation._Ly = Ly
                            perturbation._Lz = Lz
                        else:
                            perturbation._Lx = 1.0
                            perturbation._Ly = 1.0
                            perturbation._Lz = 1.0
                        perturbation_0 = perturbation
                        perturbation_1 = deepcopy(perturbation)
                        perturbation_2 = deepcopy(perturbation)

                        params = {
                            key: {
                                "given_in_basis": [fun_form] * 3,
                            },
                        }

                        if "Modes" in key:
                            params[key]["ms"] = [kwargs["ms"]] * 3
                            params[key]["ns"] = [kwargs["ns"]] * 3
                            params[key]["amps"] = [kwargs["amps"]] * 3
                        else:
                            raise ValueError(f'Perturbation {key} not implemented, only "Modes" are testes.')

                        if "Torus" not in key and isinstance(domain, domains.HollowTorus):
                            continue

                        # use the setters
                        perturbation_0.given_in_basis = fun_form
                        perturbation_0.comp = 0
                        perturbation_1.given_in_basis = fun_form
                        perturbation_1.comp = 1
                        perturbation_2.given_in_basis = fun_form
                        perturbation_2.comp = 2

                        var = FEECVariable(space=space)
                        var.add_perturbation(perturbation_0)
                        var.add_perturbation(perturbation_1)
                        var.add_perturbation(perturbation_2)
                        var.allocate(derham, domain)
                        field = var.spline

                        f1_xyz, f2_xyz, f3_xyz = domain.push(field, e1, e2, e3, kind=form)
                        f_xyz = [f1_xyz, f2_xyz, f3_xyz]

                        x, y, z = domain(e1, e2, e3)
                        r = xp.sqrt(x**2 + y**2)

                        # exact values
                        if fun_form == "physical":
                            fun1_xyz = perturbation_xyz(x, y, z)
                            fun2_xyz = perturbation_xyz(x, y, z)
                            fun3_xyz = perturbation_xyz(x, y, z)
                        elif fun_form == "physical_at_eta":
                            fun1_xyz = perturbation(eee1, eee2, eee3)
                            fun2_xyz = perturbation(eee1, eee2, eee3)
                            fun3_xyz = perturbation(eee1, eee2, eee3)
                        elif fun_form == "norm":
                            tmp1, tmp2, tmp3 = domain.transform(
                                [perturbation, perturbation, perturbation],
                                eee1,
                                eee2,
                                eee3,
                                kind=fun_form + "_to_v",
                            )
                            fun1_xyz, fun2_xyz, fun3_xyz = domain.push([tmp1, tmp2, tmp3], eee1, eee2, eee3, kind="v")
                        else:
                            fun1_xyz, fun2_xyz, fun3_xyz = domain.push(
                                [perturbation, perturbation, perturbation],
                                eee1,
                                eee2,
                                eee3,
                                kind=fun_form,
                            )

                        fun_xyz_vec = [fun1_xyz, fun2_xyz, fun3_xyz]

                        error = 0.0
                        for fi, funi in zip(f_xyz, fun_xyz_vec):
                            error += xp.max(xp.abs(fi - funi)) / xp.max(xp.abs(funi))
                        error /= 3.0
                        print(f"{rank=}, {key=}, {form=}, {fun_form=}, {error=}")
                        assert error < 0.02

                        if do_plot:
                            rn = len(form_vector)
                            for c, (fi, f) in enumerate(zip(f_xyz, fun_xyz_vec)):
                                plt.figure(key + "_" + form + "-form_e1e2 " + mapping[0])
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
                                    f"component {c + 1}, init was {fun_form}, (m,n)=({kwargs['ms'][0]},{kwargs['ns'][0]})",
                                )
                                ax = plt.gca()
                                ax.set_aspect("equal", adjustable="box")

                                plt.figure(key + "_" + form + "-form_e1e3 " + mapping[0])
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
                                    f"component {c + 1}, init was {fun_form}, (m,n)=({kwargs['ms'][0]},{kwargs['ns'][0]})",
                                )
                                ax = plt.gca()
                                ax.set_aspect("equal", adjustable="box")

    if do_plot and rank == 0:
        plt.show()


if __name__ == "__main__":
    # mapping = ['Colella', {'Lx': 4., 'Ly': 5., 'alpha': .07, 'Lz': 6.}]
    mapping = ["HollowCylinder", {"a1": 0.1}]
    # mapping = ['Cuboid', {'l1': 0., 'r1': 4., 'l2': 0., 'r2': 5., 'l3': 0., 'r3': 6.}]
    test_init_modes([16, 16, 16], [2, 3, 4], [False, True, True], mapping, combine_comps=None, do_plot=False)
    # mapping = ["HollowTorus", {"tor_period": 1}]
    # test_init_modes([16, 14, 14], [2, 3, 4], [False, True, True], mapping, combine_comps=None, do_plot=True)
