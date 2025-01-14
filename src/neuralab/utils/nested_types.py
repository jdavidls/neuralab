# %%
import sys
from functools import cached_property, reduce
from typing import ClassVar, Self, Sequence

## AUTO SUBCLASS NESTED TYPES


def isnested(obj, nest: type):
    assert isinstance(nest, type)

    if not isinstance(obj, type):
        return False
    
    # if not issubclass(obj, nested_class):
    #     return False

    if obj.__module__ != nest.__module__:
        return False

    if not obj.__qualname__.startswith(nest.__qualname__):
        return False

    return True


def subclass_nested_types(cls: type):
    def collect_mro_nested_types(cum, base):
        cum.update(
            {
                k: v
                for k, v in base.__dict__.items()
                if isnested(v, base) and v not in cls.__dict__
            }
        )
        return cum
    
    to_subclass = reduce(collect_mro_nested_types, reversed(cls.__mro__), {})

    def auto_subclass(mro_nested_type):
        class Subclass(mro_nested_type):
            pass

        Subclass.__doc__ = mro_nested_type.__doc__
        Subclass.__name__ = mro_nested_type.__name__
        Subclass.__qualname__ = f"{cls.__qualname__}.{mro_nested_type.__name__}"
        Subclass.__module__ = cls.__module__

        return Subclass

    for k, v in to_subclass.items():
        setattr(cls, k, auto_subclass(v))

    return cls

def get_global(module: str, path: str | Sequence[str]):
    if isinstance(path, str):
        path = path.split('.')
    return reduce(getattr, path, sys.modules[module])

def get_nest(cls: type) -> type:
    nest_and_nested = cls.__qualname__.rsplit('.', 1)
    if len(nest_and_nested) == 1:
        raise TypeError(f"{cls} is not nested within a class")
    result = get_global(cls.__module__, nest_and_nested[0])
    assert isinstance(result, type)
    return result

class class_property(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class cached_class_property(class_property):
    def __get__(self, cls, owner):
        value = self.fget.__get__(None, owner)(owner)
        setattr(owner, self.fget.__name__, value)
        return value

class nested_class:

    @cached_class_property
    #@classmethod
    def __nest__(cls) -> type:
        return get_nest(cls)

    def __init_subclass__(cls) -> None:
        nest_and_nested = cls.__qualname__.rsplit('.', 1)
        if len(nest_and_nested) == 1:
            raise TypeError(f"Nested class {cls} must be nested within a class")

class nest_class:
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        subclass_nested_types(cls)


class A[T](nest_class):
    type Z = int

    class B[M](nest_class): ...

    b: B[Self] = B()

class Alpha(A): ...



# %%
