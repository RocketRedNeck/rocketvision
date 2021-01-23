# Standard Imports
from functools import wraps

# 3rd Party Imports

# Local Imports

def freeze_it(cls):
    ''' Implements a decorator that can be use to "freeze" or "finalize" a class so that
        no new attributes can be inadvertently added by user due to a typo

        Uses a combination of techniques found at https://stackoverflow.com/questions/3603502/prevent-creating-new-attributes-outside-init
    '''
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and key not in dir(self):    # NOTE was self.__frozen and not hasattr(self, key):... but that will cause conflict with getattr usage
            raise TypeError(f'{self} is a frozen class. Check spelling. Cannot create-set {key} = {value}')

        object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


# Self test on import
@freeze_it
class Foo():
    def __init__(self):
        self.bar = 10

foo = Foo()
foo.bar = 42    # Should be ok
works = False
try:
    foo.foobar = "no way"
except TypeError as e:
    works = True

if not works:
    raise AttributeError('@freeze_it decorator is not functioning')
