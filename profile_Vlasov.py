import subprocess
import os

for Np in range(1000000, 5000000, 1000000):
    for cores in [1, 2, 4, 8]:
        # #1 create new Vlasov parameter file in ./src/struphy/io/inp (ALREADY DONE)

        # string = f"""
        # grid:
        #     Nel: [12, 14, 1]
        #     p: [2, 3, 1]
        #     spl_kind: [false, true, true]
        #     dirichlet_bc: null
        #     dims_mask: [true, true, true]
        #     nq_el: [2, 2, 1]
        #     nq_pr: [2, 2, 1]
        #     polar_ck: -1
        #     local_projectors: false

        # time: {{dt: 0.015, Tend: 0.15, split_algo: LieTrotter}}

        # units: {{x: 1.0, B: 1.0, n: 1.0, kBT: 1.0}}

        # geometry:
        #     type: Cuboid
        #     Cuboid: {{}}

        # fluid_background:
        #     HomogenSlab: {{}}

        # kinetic:
        #     ions:
        #         phys_params: {{A: 1, Z: 1}}
        #         markers:
        #             Np: {Np}
        #             ppc: null
        #             ppb: null
        #             bc: [periodic, periodic, periodic]
        #             bufsize: 1.0
        #             loading: pseudo_random
        #             loading_params: {{seed: 1234, spatial: uniform, dir_particles: path_to_particles}}
        #             control_variate: false
        #         weights: {{reject_weights: false, threshold: 0.0}}
        #         boxes_per_dim: [16, 16, 1]
        #         dims_mask: [true, true, true]
        #         save_data:
        #             n_markers: 0
        #             f:
        #                 slices: [e1, e1_e2]
        #                 n_bins:
        #                 - [32]
        #                 - [32, 32]
        #                 ranges:
        #                 -   - [0.0, 1.0]
        #                 -   - [0.0, 1.0]
        #                     - [-5.0, 5.0]
        #             n_sph:
        #                 plot_pts:
        #                 - [32, 1, 1]
        #                 - [1, 16, 1]
        #         perturbation:
        #             n:
        #                 TorusModesCos:
        #                     given_in_basis: '0'
        #                     ms: [1, 3]
        #         options:
        #             PushVxB: {{algo: analytic}}
        #             PushEta: {{algo: rk4}}
        #             PushVinEfield: {{use_e_field: true}}
        #         background:
        #             Maxwellian3D: {{n: 0.05}}

        # model: Vlasov
        # """

        # with open(f"./src/struphy/io/inp/params_Vlasov_{Np}.yml", "w") as file:
        #     file.write(string)
            
        environment_dict = os.environ
        
        id = environment_dict["SLURM_ARRAY_TASK_ID"]
        
        Np = id%10
            
        #2 run command

        subprocess.run(f"struphy run Vlasov --time-trace --cprofile -i params_Vlasov_{Np}.yml -o struphy_parallel_{cores}_{Np}_2nd_time --mpi {cores} | tee struphy_parallel_{cores}_{Np}_2nd_time.txt", shell=True, check=True)
        
        print("\n\n\n\n")
        
        subprocess.run(f"struphy pproc struphy_parallel_{cores}_{Np}_2nd_time --time-trace", shell=True, check=True)
        
        print("\n\n\n\n")
        
        subprocess.run(f"struphy run Vlasov --time-trace --cprofile -i params_Vlasov_{Np}.yml -o amrex_parallel_{cores}_{Np}_2nd_time --mpi {cores} --amrex | tee amrex_parallel_{cores}_{Np}_2nd_time.txt", shell=True, check=True)
        
        print("\n\n\n\n")
        
        subprocess.run(f"struphy pproc amrex_parallel_{cores}_{Np}_2nd_time --time-trace", shell=True, check=True)
        
        print("\n\n\n\n")
    
        subprocess.run('find . -name "*.hdf5" -type f -delete', shell=True, check=True)