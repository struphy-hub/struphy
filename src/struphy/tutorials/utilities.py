def print_all_attr(obj):
    '''Print all object's attributes that do not start with "_" to screen.'''
    import numpy as np

    for k in dir(obj):
        if k[0] != '_':
            v = getattr(obj, k)
            if isinstance(v, np.ndarray):
                v = f'{type(getattr(obj, k))} of shape {v.shape}'
            if 'proj_' in k or 'quad_grid_' in k:
                v = '(arrays not displayed)'
            print(k.ljust(26), v)