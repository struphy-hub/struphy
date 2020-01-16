from pyccel.decorators import f2py_compatible
from f2py_echo_accumulation import f2py_echo_accumulation

@f2py_compatible
def vector_step3(*args):
    return f2py_echo_accumulation.f2py_vector_step3(*args)


@f2py_compatible
def matrix_step3(*args):
    return f2py_echo_accumulation.f2py_matrix_step3(*args)


@f2py_compatible
def matrix_step1(*args):
    return f2py_echo_accumulation.f2py_matrix_step1(*args)
