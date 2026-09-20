from struphy.models.scalars import Scalar, Scalars


class _Constant(Scalar):
    """A scalar whose value is set from outside, to follow updates without a simulation."""

    def __init__(self, value):
        super().__init__()
        self.current = value

    def _local_update(self):
        self.local_value[0] = self.current

    def _mpi_sum(self):
        self.value[0] = self.local_value[0]


def test_nested_sum_follows_its_summands():
    """`a + b + c` is SumOfScalars(SumOfScalars(a, b), c); the inner sum must be recomputed at every update."""
    a, b, c = _Constant(1.0), _Constant(2.0), _Constant(3.0)
    scalars = Scalars(a=a, b=b, c=c, total=a + b + c)

    scalars.update()
    assert scalars.dct["total"].value[0] == 6.0

    a.current, b.current, c.current = 10.0, 20.0, 30.0
    scalars.update()
    assert scalars.dct["total"].value[0] == 60.0
