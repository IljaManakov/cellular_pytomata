from __future__ import annotations
import itertools
from enum import Enum
from typing import Callable, Generator

import numpy as np

Rule = Callable[[np.ndarray], float | None]


class RetrievalMode(Enum):
    WRAPPING: str = 'wrapping'
    PADDED: str = 'padded'


class Cell:

    __slots__ = ('state', 'position', 'neighborhood')

    def __init__(self,
                 state: list[float, ...],
                 position: tuple[int, ...],
                 neighbors: set[Cell] | None = None) -> None:

        #: State of the cell that can change over time.
        self.state = state

        #: Position of the cell in the grid represented by a tuple of indices in row-major order
        self.position = position

        #: Set of neighboring cells.
        #:
        #: ..note:
        #:
        #:   Although sets are insertion-ordered in Python, one should not rely on this representing an ordering of neighbors here
        #:   e.g. top-left to bottom right.
        #:   This is because cells can be arranged into arbitrary grids.
        #:   Instead, if one is interested in a specific subset of neighbors (e.g. all neighbors left of the cell),
        #:   one should utilize the :attr:`~cellular_pytomata.Cell.position` attribute.
        self.neighborhood = neighbors or []


class Grid:

    def __init__(self, shape: tuple[int, ...]):
        pass


class Engine:

    def __init__(self,
                 grid: np.ndarray | Callable[[], np.ndarray],
                 rules: list[Rule],
                 *,
                 neighbourhood_shape: tuple[int, ...] = (3, 3),
                 retrieval_mode: RetrievalMode = RetrievalMode.WRAPPING):

        self._grid_generator = (lambda: grid) if isinstance(grid, np.ndarray) else grid
        self.grid = self._grid_generator()
        self.rules = rules
        self.neighbourhood_shape = neighbourhood_shape
        self.retrieval_mode = retrieval_mode
        self.neighbourhood_indices = np.indices(self.neighbourhood_shape)
        self.index_cap_per_dim = np.array(self.grid.shape)
        self.index_cap_per_dim = self.index_cap_per_dim[..., *([None] * self.grid.ndim)]

    def __iter__(self) -> Generator[tuple[int, np.ndarray], None, None]:
        iterator = itertools.count()

        for step in iterator:
            updated_grid = np.zeros_like(self.grid)
            for index in itertools.product(*[range(dim) for dim in self.grid.shape]):
                neighbourhood = self._get_neighbourhood(index)
                for rule in self.rules:
                    if (result := rule(neighbourhood)) is not None:
                        updated_grid[*index] = result
                        break
                else:
                    updated_grid[*index] = self.grid[*index]

            self.grid = updated_grid
            yield step, self.grid

    def reset_grid(self):
        self.grid = self._grid_generator()

    def _get_neighbourhood(self, index: tuple[int, ...]) -> np.ndarray:

        offsets = (np.array(index) - np.array(self.neighbourhood_shape) // 2)

        # we need to add dimensions to allow proper broadcasting
        offsets = offsets[..., *([None] * self.grid.ndim)]
        neighbourhood = self.grid[*((self.neighbourhood_indices + offsets) % self.index_cap_per_dim)]

        if self.retrieval_mode is not RetrievalMode.WRAPPING:
            neighbourhood[self.neighbourhood_indices < 0 | self.neighbourhood_indices > self.index_cap_per_dim] = 0

        return neighbourhood
