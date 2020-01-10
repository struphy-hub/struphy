from pyccel.decorators import f2py_compatible
from f2py_hylife_pic import f2py_hylife_pic

@f2py_compatible
def pusher_step5(*args):
    return f2py_hylife_pic.f2py_pusher_step5(*args)


@f2py_compatible
def pusher_step4(*args):
    return f2py_hylife_pic.f2py_pusher_step4(*args)
