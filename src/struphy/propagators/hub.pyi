from struphy.propagators import propagators_fields
from struphy.propagators.base import Pro

class Propagator:
    
    @property
    def vars(self):
        ...

    @property
    def init_kernels(self):
        ...

    @property
    def eval_kernels(self):
        ...

class Propagators:
    
    @property
    def all(self) -> dict: ...
    
    def add(self, prop: Propagator, *vars): ...
    
    @property
    def Maxwell(self) -> propagators_fields.Maxwell: ...
    
    