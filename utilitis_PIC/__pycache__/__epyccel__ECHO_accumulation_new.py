from pyccel.decorators import f2py_compatible
from f2py_echo_accumulation_new import f2py_echo_accumulation_new

@f2py_compatible
def accumulation_step3(*args):
    return f2py_echo_accumulation_new.f2py_accumulation_step3(*args)


@f2py_compatible
def accumulation_step1(*args):
    return f2py_echo_accumulation_new.f2py_accumulation_step1(*args)
