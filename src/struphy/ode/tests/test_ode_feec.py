from typing import get_args

import pytest

from struphy.ode.utils import OptsButcher


@pytest.mark.parametrize(
    "spaces",
    [
        ("0",),
        ("1",),
        ("3",),
        ("2", "v"),
        ("1", "0", "2"),
    ],
)
@pytest.mark.parametrize("algo", get_args(OptsButcher))
def test_exp_growth(spaces, algo, show_plots=False):
    """Solve dy/dt = omega*y for different feec variables y and with all available solvers
    from the ButcherTableau."""

    from matplotlib import pyplot as plt
    from psydac.ddm.mpi import mpi as MPI
    from psydac.linalg.block import BlockVector
    from psydac.linalg.stencil import StencilVector

    from struphy.feec.psydac_derham import Derham
    from struphy.ode.solvers import ODEsolverFEEC
    from struphy.ode.utils import ButcherTableau
    from struphy.utils.arrays import xp

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    Nel = [1, 8, 9]
    p = [1, 2, 3]
    spl_kind = [True] * 3
    derham = Derham(Nel, p, spl_kind, comm=comm)

    c0 = 1.2
    omega = 2.3
    y_exact = lambda t: c0 * xp.exp(omega * t)

    vector_field = {}
    for i, space in enumerate(spaces):
        var = derham.Vh[space].zeros()
        if isinstance(var, StencilVector):
            var[:] = c0
        elif isinstance(var, BlockVector):
            for b in var.blocks:
                b[:] = c0
        var.update_ghost_regions()

        out = var.space.zeros()
        if len(spaces) == 1:

            def f(t, y1, out=out):
                out *= 0.0
                out += omega * y1
                out.update_ghost_regions()
                return out
        elif len(spaces) == 2:
            if i == 0:

                def f(t, y1, y2, out=out):
                    out *= 0.0
                    out += omega * y1
                    out.update_ghost_regions()
                    return out
            elif i == 1:

                def f(t, y1, y2, out=out):
                    out *= 0.0
                    out += omega * y2
                    out.update_ghost_regions()
                    return out
        elif len(spaces) == 3:
            if i == 0:

                def f(t, y1, y2, y3, out=out):
                    out *= 0.0
                    out += omega * y1
                    out.update_ghost_regions()
                    return out
            elif i == 1:

                def f(t, y1, y2, y3, out=out):
                    out *= 0.0
                    out += omega * y2
                    out.update_ghost_regions()
                    return out
            elif i == 2:

                def f(t, y1, y2, y3, out=out):
                    out *= 0.0
                    out += omega * y3
                    out.update_ghost_regions()
                    return out

        vector_field[var] = f

    print(f"{vector_field = }")
    butcher = ButcherTableau(algo=algo)
    print(f"{butcher = }")

    solver = ODEsolverFEEC(vector_field, butcher=butcher)

    hs = [0.1]
    n_hs = 6
    for i in range(n_hs - 1):
        hs += [hs[-1] / 2]
    Tend = 2

    if rank == 0:
        plt.figure(figsize=(12, 8))
    errors = {}
    for i, h in enumerate(hs):
        errors[h] = {}
        time = xp.linspace(0, Tend, int(Tend / h) + 1)
        print(f"{h = }, {time.size = }")
        yvec = y_exact(time)
        ymax = {}
        for var in vector_field:
            var *= 0.0
            if isinstance(var, StencilVector):
                var[:] = c0
            elif isinstance(var, BlockVector):
                for b in var.blocks:
                    b[:] = c0
            var.update_ghost_regions()
            ymax[var] = c0 * xp.ones_like(time)
        for n in range(time.size - 1):
            tn = h * n
            solver(tn, h)
            for var in vector_field:
                ymax[var][n + 1] = xp.max(var.toarray())

        # checks
        for var in vector_field:
            errors[h][var] = h * xp.sum(xp.abs(yvec - ymax[var])) / (h * xp.sum(xp.abs(yvec)))
            print(f"{errors[h][var] = }")
            assert errors[h][var] < 0.31

        if rank == 0:
            plt.subplot(n_hs // 2, 2, i + 1)
            plt.plot(time, yvec, label="exact")
            for j, var in enumerate(vector_field):
                plt.plot(time, ymax[var], "--", label=f"{spaces[j]}-space")
            plt.xlabel("time")
            plt.ylabel("y")
            plt.legend()

    # convergence checks
    if rank == 0:
        plt.figure(figsize=(12, 8))
    for j, var in enumerate(vector_field):
        h_vec = []
        err_vec = []
        for h, dct in errors.items():
            h_vec += [h]
            err_vec += [dct[var]]

        m, _ = xp.polyfit(xp.log(h_vec), xp.log(err_vec), deg=1)
        print(f"{spaces[j]}-space, fitted convergence rate = {m} for {algo = } with {solver.butcher.conv_rate = }")
        assert xp.abs(m - solver.butcher.conv_rate) < 0.1
        print(f"Convergence check passed on {rank = }.")

        if rank == 0:
            plt.loglog(h_vec, h_vec, "--", label=f"h")
            plt.loglog(h_vec, [h**2 for h in h_vec], "--", label=f"h^2")
            plt.loglog(h_vec, [h**3 for h in h_vec], "--", label=f"h^3")
            plt.loglog(h_vec, [h**4 for h in h_vec], "--", label=f"h^4")
            plt.loglog(h_vec, err_vec, "o-k", label=f"{spaces[j]}-space, {algo}")
    if rank == 0:
        plt.xlabel("log(h)")
        plt.ylabel("log(error)")
        plt.legend()

    if show_plots and rank == 0:
        plt.show()


if __name__ == "__main__":
    # test_one_variable('0', 'rk2', show_plots=True)
    test_exp_growth(("0", "1", "2"), "rk2", show_plots=True)
