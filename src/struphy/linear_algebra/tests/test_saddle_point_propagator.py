import pytest


@pytest.mark.mpi_skip
@pytest.mark.parametrize("Nel", [[16, 1, 1], [32, 1, 1]])
@pytest.mark.parametrize("p", [[1, 1, 1], [2, 1, 1]])
@pytest.mark.parametrize("spl_kind", [[True, True, True]])
@pytest.mark.parametrize("dirichlet_bc", [[[False, False], [False, False], [False, False]]])
@pytest.mark.parametrize("mapping", [["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}]])
@pytest.mark.parametrize("epsilon", [0.000000001])
@pytest.mark.parametrize("dt", [0.001])
def test_propagator1D(Nel, p, spl_kind, dirichlet_bc, mapping, epsilon, dt):
    """Test saddle-point-solver by propagator TwoFluidQuasiNeutralFull. Use manufactured solutions from perturbations to verify h- and p-convergence when model TwoFluidQuasiNeutralToy calculates solution with SaddlePointSolver."""

    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.basis_projection_ops import BasisProjectionOperators
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays
    from struphy.fields_background.equils import HomogenSlab
    from struphy.geometry import domains
    from struphy.propagators.propagators_fields import TwoFluidQuasiNeutralFull
    from struphy.models.variables import FEECVariable
    from struphy.initial import perturbations

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    mpi_comm.Barrier()

    dims_mask = [True, False, False]
    nq_el = [2, 2, 1]
    nq_pr = [2, 2, 1]
    polar_ck = -1

    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])
    # derham object
    derham = Derham(
        Nel,
        p,
        spl_kind,
        comm=mpi_comm,
        dirichlet_bc=dirichlet_bc,
        local_projectors=False,
        mpi_dims_mask=dims_mask,
        nquads=nq_el,
        nq_pr=nq_pr,
        polar_ck=polar_ck,
        domain=domain,
    )
    # Mhd equilibirum (slab)
    mhd_equil_params = {"B0x": 0.0, "B0y": 0.0, "B0z": 1.0, "beta": 0.1, "n0": 1.0}
    eq_mhd = HomogenSlab(**mhd_equil_params)
    eq_mhd.domain = domain

    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    bas_ops = BasisProjectionOperators(derham, domain, eq_mhd=eq_mhd)

    # Manufactured solutions
    uvec = FEECVariable("u", "Hdiv")
    u_evec = FEECVariable("u_e", "Hdiv")
    potentialvec = FEECVariable("potential", "L2")
    uinitial = FEECVariable("u", "Hdiv")

    pp_u = perturbations.ManufacturedSolutionVelocity()
    pp_ue = perturbations.ManufacturedSolutionVelocity(species="Electrons")
    pp_potential = perturbations.ManufacturedSolutionPotential()

    # pp_u = {
    #     "ManufacturedSolutionVelocity": {
    #         "given_in_basis": ["physical", None, None],
    #         "species": "Ions",
    #         "comp": "0",
    #         "dimension": "1D",
    #     }
    # }
    # pp_ue = {
    #     "ManufacturedSolutionVelocity": {
    #         "given_in_basis": ["physical", None, None],
    #         "species": "Electrons",
    #         "comp": "0",
    #         "dimension": "1D",
    #     }
    # }
    # pp_potential = {
    #     "ManufacturedSolutionPotential": {
    #         "given_in_basis": "physical",
    #         "dimension": "1D",
    #     }
    # }

    uvec.add_perturbation(pp_u)
    uvec.allocate(derham, domain, eq_mhd)
    
    u_evec.add_perturbation(pp_ue)
    u_evec.allocate(derham, domain, eq_mhd)
    
    potentialvec.add_perturbation(pp_potential)
    potentialvec.allocate(derham, domain, eq_mhd)
    
    uinitial.allocate(derham, domain, eq_mhd)

    # uvec.initialize_coeffs(domain=domain, pert_params=pp_u)
    # u_evec.initialize_coeffs(domain=domain, pert_params=pp_ue)
    # potentialvec.initialize_coeffs(domain=domain, pert_params=pp_potential)

    # Save manufactured solution to compare it later with the outcome of the propagator
    uvec_initial = uvec.spline.vector.copy()
    u_evec_initial = u_evec.spline.vector.copy()
    potentialvec_initial = potentialvec.spline.vector.copy()

    solver = {}
    solver["type"] = ["gmres", None]
    solver["tol"] = 1.0e-8
    solver["maxiter"] = 3000
    solver["info"] = True
    solver["verbose"] = True
    solver["recycle"] = True

    TwoFluidQuasiNeutralFull.derham = derham
    TwoFluidQuasiNeutralFull.domain = domain
    TwoFluidQuasiNeutralFull.mass_ops = mass_ops
    TwoFluidQuasiNeutralFull.basis_ops = bas_ops

    # Starting with initial condition u=0 and ue and phi start with manufactured solution
    prop = TwoFluidQuasiNeutralFull(
        uinitial.spline.vector,
        u_evec.spline.vector,
        potentialvec.spline.vector,
        stab_sigma=epsilon,
        D1_dt=dt,
        variant="Uzawa",
        dimension="1D",
        nu=10.0,
        nu_e=1.0,
        solver=solver,
        method_to_solve="DirectNPInverse",
        preconditioner=False,
        spectralanalysis=False,
        B0=1.0,
    )

    # Only one step in time to compare different Nel and p at dt
    Tend = dt
    time = 0.0
    while time < Tend:
        # advance in time
        prop(dt)
        time += dt
    if Nel[0] == 16:
        if p[0] == 1:
            compare_arrays(uinitial.vector, uvec_initial.toarray(), mpi_rank, atol=1e-2)
            compare_arrays(u_evec.vector, u_evec_initial.toarray(), mpi_rank, atol=1e-2)
            compare_arrays(potentialvec.vector, potentialvec_initial.toarray(), mpi_rank, atol=1e-2)
        elif p[0] == 2:
            compare_arrays(uinitial.vector, uvec_initial.toarray(), mpi_rank, atol=1e-4)
            compare_arrays(u_evec.vector, u_evec_initial.toarray(), mpi_rank, atol=1e-4)
            compare_arrays(potentialvec.vector, potentialvec_initial.toarray(), mpi_rank, atol=1e-4)
    elif Nel[0] == 32:
        if p[0] == 1:
            compare_arrays(uinitial.vector, uvec_initial.toarray(), mpi_rank, atol=1e-2)
            compare_arrays(u_evec.vector, u_evec_initial.toarray(), mpi_rank, atol=1e-2)
            compare_arrays(potentialvec.vector, potentialvec_initial.toarray(), mpi_rank, atol=1e-3)
        elif p[0] == 2:
            compare_arrays(uinitial.vector, uvec_initial.toarray(), mpi_rank, atol=1e-5)
            compare_arrays(u_evec.vector, u_evec_initial.toarray(), mpi_rank, atol=1e-7)
            compare_arrays(potentialvec.vector, potentialvec_initial.toarray(), mpi_rank, atol=1e-6)


if __name__ == "__main__":
    test_propagator1D(
        [16, 1, 1],
        [1, 1, 1],
        [True, True, True],
        [[False, False], [False, False], [False, False]],
        ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
        0.000000001,
        0.001,
    )
    test_propagator1D(
        [16, 1, 1],
        [2, 1, 1],
        [True, True, True],
        [[False, False], [False, False], [False, False]],
        ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
        0.000000001,
        0.001,
    )
    test_propagator1D(
        [32, 1, 1],
        [2, 1, 1],
        [True, True, True],
        [[False, False], [False, False], [False, False]],
        ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
        0.000000001,
        0.001,
    )
    test_propagator1D(
        [32, 1, 1],
        [1, 1, 1],
        [True, True, True],
        [[False, False], [False, False], [False, False]],
        ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
        0.000000001,
        0.001,
    )


import pytest


@pytest.mark.mpi_skip
@pytest.mark.parametrize("Nel", [[16, 16, 1], [32, 32, 1]])
@pytest.mark.parametrize("p", [[1, 1, 1], [2, 2, 1]])
@pytest.mark.parametrize("spl_kind", [[True, True, True]])
@pytest.mark.parametrize("dirichlet_bc", [[[False, False], [False, False], [False, False]]])
@pytest.mark.parametrize("mapping", [["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}]])
@pytest.mark.parametrize("epsilon", [0.001])
@pytest.mark.parametrize("dt", [0.01])
def test_propagator2D(Nel, p, spl_kind, dirichlet_bc, mapping, epsilon, dt):
    """Test saddle-point-solver by propagator TwoFluidQuasiNeutralFull. Use manufactured solutions from perturbations to verify h- and p-convergence when model TwoFluidQuasiNeutralToy calculates solution with SaddlePointSolver. Allow a certain error after one time step, save this solution and compare the follwing timesteps with this solution but with less tolerance. Shows that the solver can stay in a steady state solution."""

    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.basis_projection_ops import BasisProjectionOperators
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays
    from struphy.fields_background.equils import HomogenSlab
    from struphy.geometry import domains
    from struphy.propagators import TwoFluidQuasiNeutralFull

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    mpi_comm.Barrier()

    dims_mask = [True, False, False]
    nq_el = [2, 2, 1]
    nq_pr = [2, 2, 1]
    polar_ck = -1

    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])
    # derham object
    derham = Derham(
        Nel,
        p,
        spl_kind,
        comm=mpi_comm,
        dirichlet_bc=dirichlet_bc,
        local_projectors=False,
        mpi_dims_mask=dims_mask,
        nquads=nq_el,
        nq_pr=nq_pr,
        polar_ck=polar_ck,
        domain=domain,
    )
    # Mhd equilibirum (slab)
    mhd_equil_params = {"B0x": 0.0, "B0y": 0.0, "B0z": 1.0, "beta": 0.1, "n0": 1.0}
    eq_mhd = HomogenSlab(**mhd_equil_params)
    eq_mhd.domain = domain

    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    bas_ops = BasisProjectionOperators(derham, domain, eq_mhd=eq_mhd)

    # Manufactured solutions
    uvec = FEECVariable("u", "Hdiv")
    u_evec = FEECVariable("u_e", "Hdiv")
    potentialvec = FEECVariable("potential", "L2")

    pp_u = {
        "ManufacturedSolutionVelocity": {
            "given_in_basis": ["physical", None, None],
            "comp": "0",
            "species": "Ions",
            "dimension": "2D",
        },
        "ManufacturedSolutionVelocity_2": {
            "given_in_basis": [None, "physical", None],
            "comp": "1",
            "species": "Ions",
            "dimension": "2D",
        },
    }
    pp_u_e = {
        "ManufacturedSolutionVelocity": {
            "given_in_basis": ["physical", None, None],
            "comp": "0",
            "species": "Electrons",
            "dimension": "2D",
        },
        "ManufacturedSolutionVelocity_2": {
            "given_in_basis": [None, "physical", None],
            "comp": "1",
            "species": "Electrons",
            "dimension": "2D",
        },
    }
    pp_potential = {
        "ManufacturedSolutionPotential": {
            "given_in_basis": "physical",
            "dimension": "2D",
        }
    }

    uvec.initialize_coeffs(domain=domain, pert_params=pp_u)
    u_evec.initialize_coeffs(domain=domain, pert_params=pp_u_e)
    potentialvec.initialize_coeffs(domain=domain, pert_params=pp_potential)

    solver = {}
    solver["type"] = ["gmres", None]
    solver["tol"] = 1.0e-8
    solver["maxiter"] = 3000
    solver["info"] = True
    solver["verbose"] = True
    solver["recycle"] = True

    TwoFluidQuasiNeutralFull.derham = derham
    TwoFluidQuasiNeutralFull.domain = domain
    TwoFluidQuasiNeutralFull.mass_ops = mass_ops
    TwoFluidQuasiNeutralFull.basis_ops = bas_ops

    # Starting with initial condition u=0 and ue and phi start with manufactured solution
    prop = TwoFluidQuasiNeutralFull(
        uvec.vector,
        u_evec.vector,
        potentialvec.vector,
        stab_sigma=epsilon,
        D1_dt=dt,
        eps_norm=1.0,
        variant="Uzawa",
        dimension="2D",
        nu=10.0,
        nu_e=1.0,
        solver=solver,
        method_to_solve="DirectNPInverse",
        preconditioner=False,
        spectralanalysis=False,
        B0=1.0,
    )

    uvec_initial = uvec.vector.copy().toarray()
    ue_vec_initial = u_evec.vector.copy().toarray()
    potentialvec_initial = potentialvec.vector.copy().toarray()

    Tend = 10 * dt
    time = 0.0
    # first time step
    prop(dt)
    time += dt
    # Compare to initial condition, which is also the solution
    if Nel[0] == 16:
        if p[0] == 1:
            compare_arrays(uvec.vector, uvec_initial, mpi_rank, atol=1e-2)
            compare_arrays(u_evec.vector, ue_vec_initial, mpi_rank, atol=1e-1)
            compare_arrays(potentialvec.vector, potentialvec_initial, mpi_rank, atol=1e-1)
        elif p[0] == 2:
            compare_arrays(uvec.vector, uvec_initial, mpi_rank, atol=1e-3)
            compare_arrays(u_evec.vector, ue_vec_initial, mpi_rank, atol=1e-2)
            compare_arrays(potentialvec.vector, potentialvec_initial, mpi_rank, atol=1e-4)
    elif Nel[0] == 32:
        if p[0] == 1:
            compare_arrays(uvec.vector, uvec_initial, mpi_rank, atol=1e-2)
            compare_arrays(u_evec.vector, ue_vec_initial, mpi_rank, atol=1e-2)
            compare_arrays(potentialvec.vector, potentialvec_initial, mpi_rank, atol=1e-2)
        elif p[0] == 2:
            compare_arrays(uvec.vector, uvec_initial, mpi_rank, atol=1e-3)
            compare_arrays(u_evec.vector, ue_vec_initial, mpi_rank, atol=1e-3)
            compare_arrays(potentialvec.vector, potentialvec_initial, mpi_rank, atol=1e-5)

    # Save results after first timestep
    uvec_1step = uvec.vector.copy().toarray()
    ue_vec_1step = u_evec.vector.copy().toarray()
    potentialvec_1step = potentialvec.vector.copy().toarray()

    while time < Tend:
        # advance in time
        prop(dt)
        time += dt

        # Compare to solution after one step in time, but with less tolerance
        if Nel[0] == 16:
            if p[0] == 1:
                compare_arrays(uvec.vector, uvec_1step, mpi_rank, atol=1e-3)
                compare_arrays(u_evec.vector, ue_vec_1step, mpi_rank, atol=1e-3)
                compare_arrays(potentialvec.vector, potentialvec_1step, mpi_rank, atol=1e-3)
            elif p[0] == 2:
                compare_arrays(uvec.vector, uvec_1step, mpi_rank, atol=1e-4)
                compare_arrays(u_evec.vector, ue_vec_1step, mpi_rank, atol=1e-6)
                compare_arrays(potentialvec.vector, potentialvec_1step, mpi_rank, atol=1e-6)
        elif Nel[0] == 32:
            if p[0] == 1:
                compare_arrays(uvec.vector, uvec_1step, mpi_rank, atol=1e-3)
                compare_arrays(u_evec.vector, ue_vec_1step, mpi_rank, atol=1e-3)
                compare_arrays(potentialvec.vector, potentialvec_1step, mpi_rank, atol=1e-4)
            elif p[0] == 2:
                compare_arrays(uvec.vector, uvec_1step, mpi_rank, atol=1e-4)
                compare_arrays(u_evec.vector, ue_vec_1step, mpi_rank, atol=1e-7)
                compare_arrays(potentialvec.vector, potentialvec_1step, mpi_rank, atol=1e-7)


if __name__ == "__main__":
    test_propagator1D(
        [16, 1, 1],
        [2, 2, 1],
        [True, True, True],
        [[False, False], [False, False], [False, False]],
        ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
        0.001,
        0.01,
    )
    # test_propagator2D(
    #     [16, 16, 1],
    #     [1, 1, 1],
    #     [True, True, True],
    #     [[False, False], [False, False], [False, False]],
    #     ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
    #     0.001,
    #     0.01,
    # )
    # test_propagator2D(
    #     [16, 16, 1],
    #     [2, 2, 1],
    #     [True, True, True],
    #     [[False, False], [False, False], [False, False]],
    #     ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
    #     0.001,
    #     0.01,
    # )
    # test_propagator2D(
    #     [32, 32, 1],
    #     [2, 2, 1],
    #     [True, True, True],
    #     [[False, False], [False, False], [False, False]],
    #     ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
    #     0.001,
    #     0.01,
    # )
    # test_propagator2D(
    #     [32, 32, 1],
    #     [1, 1, 1],
    #     [True, True, True],
    #     [[False, False], [False, False], [False, False]],
    #     ["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
    #     0.001,
    #     0.01,
    # )
