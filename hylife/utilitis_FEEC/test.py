from pyccel.decorators import types
import ..geometry.equilibrium as eq
import ..geometry.mappings_analytical as mapping

# ==========================================================================================
@types('double','double','double','int','int','double[:]')        
def fun(xi1, xi2, xi3, kind_fun, kind_map, params):
    
    value = 0.
    
    value = eq.p_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 11)
    
    return value