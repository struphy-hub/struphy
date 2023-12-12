def test_pproc_codes(model=None):
    '''Tests the post processing of runs in test_codes.py'''
    
    import os
    import struphy
    import inspect
    from struphy.models import fluid, kinetic, hybrid, toy
    from struphy.post_processing import pproc_struphy
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    
    libpath = struphy.__path__[0]
    
    list_fluid = []
    for name, obj in inspect.getmembers(fluid):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', }:
                list_fluid += [name]

    list_kinetic = []
    for name, obj in inspect.getmembers(kinetic):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', }:
                list_kinetic += [name]

    list_hybrid = []
    for name, obj in inspect.getmembers(hybrid):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', }:
                list_hybrid += [name]

    list_toy = []
    for name, obj in inspect.getmembers(toy):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', }:
                list_toy += [name]

    list_models = list_fluid + list_kinetic + list_hybrid + list_toy
    
    if comm.Get_rank() == 0:
        if model is None:
            for model in list_models:
                
                # TODO: remove if-clause
                if 'VlasovMasslessElectrons' in model:
                    print(
                        f'Model {model} is currently excluded from tests.')
                    continue
                
                path_out = os.path.join(libpath, 'io/out/test_' + model)
                pproc_struphy.main(path_out)
        else:
            
            # TODO: remove if-clause
            if 'VlasovMasslessElectrons' in model:
                print(
                    f'Model {model} is currently excluded from tests.')
                exit()
            
            path_out = os.path.join(libpath, 'io/out/test_' + model)
            pproc_struphy.main(path_out)
        
        
if __name__ == '__main__':
    test_pproc_codes()
        