from pyccel.decorators import inline

# @inline
def f() -> float:
    _tmp_g = g()
    return _tmp_g

@inline
def g() -> float:
    _tmp_h = h()
    for i in range(3):
        _tmp_h = h()
    return _tmp_h

@inline
def h() -> float:
    _s1 = 1.0
    _s2 = 2.0
    return _s1 * _s1
