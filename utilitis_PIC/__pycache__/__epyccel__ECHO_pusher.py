from pyccel.decorators import f2py_compatible
from f2py_echo_pusher import f2py_echo_pusher

@f2py_compatible
def pusher_step5(*args):
    return f2py_echo_pusher.f2py_pusher_step5(*args)


@f2py_compatible
def pusher_step4(*args):
    return f2py_echo_pusher.f2py_pusher_step4(*args)


@f2py_compatible
def pusher_step3(*args):
    return f2py_echo_pusher.f2py_pusher_step3(*args)
