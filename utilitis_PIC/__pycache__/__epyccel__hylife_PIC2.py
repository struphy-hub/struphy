from pyccel.decorators import f2py_compatible
from f2py_hylife_pic2 import f2py_hylife_pic2

@f2py_compatible
def matrix_step1(*args):
    return f2py_hylife_pic2.f2py_matrix_step1(*args)
