from __future__ import annotations
from abc import abstractmethod, ABC
from functools import wraps


class Neighborhood(ABC):

    def __init_subclass__(cls, **kwargs):

        original_eq = cls.__eq__

        @wraps(original_eq)
        def wrapper(self, other: Neighborhood):
            if type(self) is not type(other):
                return False
            return original_eq(self, other)

        cls.__eq__ = wrapper

    @abstractmethod
    def __eq__(self, other: Neighborhood):
        pass


class RectangularNeighborhood(Neighborhood):

    def __init__(self, ):
        pass
    def left(self):
        pass
    def right(self):
        pass
    def above(self):
        pass
    def below(self):
        pass
    def neighbors(self, next=0, inclusive=True):
        pass

class HexagonalNeighborhood(Neighborhood):
    pass
