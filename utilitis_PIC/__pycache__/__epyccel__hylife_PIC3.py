from pyccel.decorators import f2py_compatible
from f2py_hylife_pic3 import f2py_hylife_pic3

@f2py_compatible
def matrix_step3(*args):
    return f2py_hylife_pic3.f2py_matrix_step3(*args)
