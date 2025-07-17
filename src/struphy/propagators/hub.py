from struphy.propagators.base import Propagator


class Propagators:
    """Handels the Propagators of a StruphyModel."""
    def __init__(self):
        pass

    @property
    def all(self):
        return self.__dict__

    def add(self, prop: Propagator, *vars):
        # print(f'{prop = }')
        # print(f'{prop.__name__ = }')
        # for var in vars:
        #     print(var)
        setattr(self, prop.__name__, prop(*vars))
        
    def set_options(
        self,
        name: str,
        **opts,
    ):
        print(f'{self.all = }')
        assert name in self.all, f"Propagator {name} is not part of model propagators {self.all.keys()}"
        prop = getattr(self, name)
        assert isinstance(prop, Propagator)
        prop.set_options(**opts)