#!/usr/bin/env python3


def execute(file_in, path_out, comm, restart=False, verbose=False):
    '''Executes the code inverse_mass_test.

    Parameters
    ----------

    file_in : str
        Absolute path to input parameters file (.yml).

    path_out : str
        Absolute path to output folder.

    comm : mpi communicator

    restart : boolean
        Restart ('True') or new simulation ('False').

    verbose : boolean
        Print more solver info.
    '''
     
    import yaml
    import numpy as np

    from struphy.geometry.domain_3d import Domain
    from struphy.feec.psydac_derham import Derham_build
    from struphy.models.substeps.push_inverse_mass import InvertMassMatrices

    from psydac.linalg.stencil import StencilVector
    from psydac.linalg.block import BlockVector

    # mpi communicator
    MPI_COMM = comm
    mpi_rank = MPI_COMM.Get_rank()
    if mpi_rank == 0:
        print(f'\nMPI communicator initialized with {MPI_COMM.Get_size()} process(es).\n')

    code_name = '"inverse_mass_test"'
    
    if mpi_rank == 0:
        print('Starting code ' + code_name +  '...\n')
        print(f'file_in : {file_in}')
        print(f'path_out: {path_out}\n')
    
    # load simulation VARIABLES
    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # ========================================================================================= 
    # DOMAIN object
    # =========================================================================================
    dom_type   = params['geometry']['type']
    dom_params = params['geometry']['params_' + dom_type]

    DOMAIN     = Domain(dom_type, dom_params)
    F_psy      = DOMAIN.Psydac_mapping('F', **dom_params) # create psydac mapping for mass matrices only
    
    if mpi_rank == 0:
        print(f'domain type: {dom_type}')
        print(f'domain parameters: {dom_params}')
        print(f'DOMAIN of type "' + dom_type + '" set.')
        print()
    
    # ========================================================================================= 
    # DERHAM sequence (Psydac)
    # =========================================================================================
    # Grid parameters
    Nel             = params['grid']['Nel']             # Number of grid cells
    p               = params['grid']['p']               # spline degree
    spl_kind        = params['grid']['spl_kind']        # Spline type

    DR=Derham_build(Nel, p, spl_kind, F = F_psy, comm = MPI_COMM)   

    if mpi_rank == 0:
        print('GRID parameters:')
        print(f'Nel     : {Nel}')
        print(f'p       : {p}')
        print(f'spl_kind: {spl_kind}\n')
        print('Discrete Derham set (polar=' + str(params['grid']['polar']) + ').')
        print()

    # Assemble necessary mass matrices 
    DR.assemble_M0()
    if verbose: print(f'Rank: {mpi_rank} | Assembly of M0 done.')
    DR.assemble_M1()
    if verbose: print(f'Rank: {mpi_rank} | Assembly of M1 done.')
    DR.assemble_M2()
    if verbose: print(f'Rank: {mpi_rank} | Assembly of M2 done.\n')
    DR.assemble_M3()
    if verbose: print(f'Rank: {mpi_rank} | Assembly of M3 done.')

    # ========================================================================================= 
    # Initialize mass inverting function
    # =========================================================================================
    # Initialize splitting substeps
    update_mass = InvertMassMatrices(DR, params['solvers']['step_mass'])  

    # Random stencil vectors
    v0 = StencilVector(DR.V0.vector_space)
    v0._data = np.random.rand(*v0._data.shape)

    v1 = BlockVector(DR.V1.vector_space)
    for v1i in v1:
        v1i._data = np.random.rand(*v1i._data.shape)

    v2 = BlockVector(DR.V2.vector_space)
    for v2i in v2:
        v2i._data = np.random.rand(*v2i._data.shape)

    v3 = StencilVector(DR.V3.vector_space)
    v3._data = np.random.rand(*v3._data.shape)

    # Define update function
    def update():
        update_mass(v0, v1, v2, v3)

    if mpi_rank == 0:
        print('Update function set.')
        print()
        print('Start inverting... ')
        print()
            
    v0_old = v0.copy()
    v1_old = v1.copy()
    v2_old = v2.copy()
    v3_old = v3.copy()

    # call update function for time stepping
    update() 

    # Verify results:
    d0 = np.max(np.abs(DR.M0.dot(v0).toarray() - v0_old.toarray()))
    d1 = np.max(np.abs(DR.M1.dot(v1).toarray() - v1_old.toarray()))
    d2 = np.max(np.abs(DR.M2.dot(v2).toarray() - v2_old.toarray()))
    d3 = np.max(np.abs(DR.M3.dot(v3).toarray() - v3_old.toarray()))

    if mpi_rank == 0:
        print(f'Maxdiff v0: {d0}')
        print(f'Maxdiff v1: {d1}')
        print(f'Maxdiff v2: {d2}')
        print(f'Maxdiff v3: {d3}')


