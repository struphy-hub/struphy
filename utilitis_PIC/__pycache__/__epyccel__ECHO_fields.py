from pyccel.decorators import f2py_compatible
from f2py_echo_fields import f2py_echo_fields

@f2py_compatible
def evaluate_2form(*args):
    return f2py_echo_fields.f2py_evaluate_2form(*args)


@f2py_compatible
def evaluate_1form(*args):
    return f2py_echo_fields.f2py_evaluate_1form(*args)
