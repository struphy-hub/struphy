.. _quickstart:

Quickstart
==========

Struphy is a Python API for solving PDEs with structure-preserving discretizations.
This quickstart shows how to solve a simple problem with minimal input: a 1D Poisson solve.

For interactive tutorials (no local install), use `mybinder <https://mybinder.org/v2/gh/struphy-hub/struphy-tutorials/main>`_.
For more examples, see :ref:`userguide` and the :ref:`tutorial collection <tutorials>`.

Solve Poisson In A Few Steps
----------------------------

Make sure that Struphy is installed and compiled (see :ref:`install_modes`).

We search for a potential :math:`\phi(x)` satisfying the Poisson equation

.. math::

    -\Delta \phi = \rho

for given source term :math:`\rho(x)` on a periodic 1D domain.

1. Import the API and choose a model.

.. code-block:: python

    from struphy import Simulation, domains, grids, perturbations
    from struphy.models import Poisson

2. Create the :class:`~struphy.models.poisson.Poisson` model.

.. code-block:: python

    model = Poisson()

3. This model features the Propagator :class:`~struphy.propagators.poisson_solve.PoissonSolve` under ``propagators.poisson``. 
Connect the ``source`` variable of species ``em_fields`` to the propagator and stabilize via ``options``.

.. code-block:: python

    stab_eps = 1e-8

    model.propagators.poisson.options = model.propagators.poisson.Options(
        rho=model.em_fields.source,
        stab_eps=stab_eps,
    )

4. Add a manufactured term :math:`\rho(x) = (k^2 + \epsilon)\cos(kx)` to the ``source`` variable.

.. code-block:: python

    import numpy as np

    Lx = 2.0 * np.pi
    mode = 2
    k = mode * 2.0 * np.pi / Lx
    source_amp = k**2 + stab_eps

    fun = perturbations.ModesCos(ls=(mode,), amps=(source_amp,))

    model.em_fields.source.add_perturbation(fun)

5. Build domain and grid, then instantiate a simulation.

.. code-block:: python

    domain = domains.Cuboid(l1=0.0, r1=Lx)
    grid = grids.TensorProductGrid(num_elements=(64, 1, 1))

    sim = Simulation(
        model=model,
        domain=domain,
        grid=grid,
    )

6. Run one step (enough for this stationary solve).

.. code-block:: python

    sim.run(one_time_step=True)

7. Post-process and load plotting data.

.. code-block:: python

    sim.pproc()
    sim.load_plotting_data()

8. Compare to the exact solution, and save the figure.

.. code-block:: python

    import matplotlib.pyplot as plt

    x = sim.grids_phy[0][:, 0, 0]
    t_last = max(sim.spline_values.em_fields.phi_log.data)
    phi_num = sim.spline_values.em_fields.phi_log.data[t_last][0][:, 0, 0]
    phi_exact = np.cos(k * x)
    err_max = np.max(np.abs(phi_num - phi_exact))

    plt.figure(figsize=(7, 3.8))
    plt.plot(x, phi_exact, "k--", lw=1.8, label="exact")
    plt.plot(x, phi_num, "o", ms=3.5, label="Struphy")
    plt.xlabel("x")
    plt.ylabel("phi")
    plt.title("Struphy quickstart: Poisson solution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("quickstart_poisson_phi.png", dpi=150)
    plt.show()

    print(f"max error = {err_max:.3e}")

.. figure:: ../pics/quickstart_poisson_phi.png
    :figwidth: 85%
    :alt: Poisson quickstart comparison of exact and numerical solution

    Exact (dashed) and Struphy (markers) solutions from Step 6.

Full copy-paste script:

.. code-block:: python

    import numpy as np
    from struphy import Simulation, domains, grids, perturbations
    from struphy.models import Poisson

    model = Poisson()

    stab_eps = 1e-8
    
    model.propagators.poisson.options = model.propagators.poisson.Options(
        rho=model.em_fields.source,
        stab_eps=stab_eps,
    )

    Lx = 2.0 * np.pi
    mode = 2
    k = mode * 2.0 * np.pi / Lx
    source_amp = k**2 + stab_eps

    fun = perturbations.ModesCos(ls=(mode,), amps=(source_amp,))

    model.em_fields.source.add_perturbation(fun)

    domain = domains.Cuboid(l1=0.0, r1=Lx)
    grid = grids.TensorProductGrid(num_elements=(64, 1, 1))

    sim = Simulation(model=model, domain=domain, grid=grid)
    sim.run(one_time_step=True)

    sim.pproc()
    sim.load_plotting_data()

    x = sim.grids_phy[0][:, 0, 0]
    t_last = max(sim.spline_values.em_fields.phi_log.data)
    phi_num = sim.spline_values.em_fields.phi_log.data[t_last][0][:, 0, 0]
    phi_exact = np.cos(k * x)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 3.8))
    plt.plot(x, phi_exact, "k--", lw=1.8, label="exact")
    plt.plot(x, phi_num, "o", ms=3.5, label="Struphy")
    plt.xlabel("x")
    plt.ylabel("phi")
    plt.title("Struphy quickstart: Poisson solution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("quickstart_poisson_phi.png", dpi=150)
    plt.show()

Same Workflow For All Models
----------------------------

The same Simulation API is reused across models. For example, replace :class:`~struphy.models.poisson.Poisson` with :class:`~struphy.models.maxwell.Maxwell`:

.. code-block:: python

    from struphy import Simulation, perturbations
    from struphy.models import Maxwell

    model = Maxwell()
    model.em_fields.e_field.add_perturbation(
        perturbations.ModesCos(ls=(1,), amps=(1e-2,), comp=1)
    )

    sim = Simulation(model=model)
    sim.run()

Check :ref:`models` for more models and their specific options.

Generate A Default Parameter File
---------------------------------

You can generate a ready-to-edit parameter file for any model from the CLI:

.. code-block:: bash

    struphy params Poisson

This writes ``params_Poisson.py`` in your current directory. You can open and edit it, then run with:

.. code-block:: bash

    python params_Poisson.py

As all data structures in Struphy are written for MPI, you can run the same script with ``mpirun`` to use multiple processes:

.. code-block:: bash

    mpirun -n 4 python params_Poisson.py

            
