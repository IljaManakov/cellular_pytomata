import itertools
from enum import Enum
from typing import Callable, Generator, Protocol

import numpy as np


class Rules(Protocol):

    def __call__(self, neighborhood: np.ndarray, current_state: int) -> int: ...


class RetrievalMode(Enum):
    WRAPPING: str = 'wrapping'
    PADDED: str = 'padded'


class Engine:

    def __init__(self,
                 grid: np.ndarray | Callable[[], np.ndarray],
                 rules: Rules,
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
                current_state = self.grid[*index]
                neighbourhood = self._get_neighbourhood(index)
                updated_grid[*index] = self.rules(neighbourhood, current_state)

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
