import itertools
from enum import Enum
from typing import Callable, Generator, Protocol

import numba
import numpy as np


@numba.jit
def _get_neighborhood(grid: np.ndarray, neighborhood_indices: np.ndarray, index_cap_per_dim: np.ndarray, index: np.ndarray) -> np.ndarray:
    shape = neighborhood_indices.shape[1:]
    neighborhood_indices = (neighborhood_indices + index) % index_cap_per_dim
    rows, columns = neighborhood_indices.reshape(2, -1)
    neighborhood = grid.ravel()[columns + rows*grid.shape[1]].reshape(shape)

    # if retrieval_mode is not RetrievalMode.WRAPPING:
    #     neighborhood[neighborhood_indices < 0 | neighborhood_indices > index_cap_per_dim] = 0

    return neighborhood


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
                 neighborhood_shape: tuple[int, ...] = (3, 3),
                 retrieval_mode: RetrievalMode = RetrievalMode.WRAPPING):

        self._grid_generator = (lambda: grid) if isinstance(grid, np.ndarray) else grid
        self.grid = self._grid_generator()
        self.rules = rules
        self.retrieval_mode = retrieval_mode
        self.neighborhood_indices = np.indices(neighborhood_shape)
        self.neighborhood_indices -= np.array(neighborhood_shape)[..., *((None,)*len(neighborhood_shape))] // 2
        self.index_cap_per_dim = np.array(self.grid.shape)
        self.index_cap_per_dim = self.index_cap_per_dim[..., *([None] * self.grid.ndim)]

    def __iter__(self) -> Generator[tuple[int, np.ndarray], None, None]:
        iterator = itertools.count()

        for step in iterator:
            updated_grid = np.zeros_like(self.grid)
            for index in itertools.product(*[range(dim) for dim in self.grid.shape]):
                current_state = self.grid[*index]
                shape = self.neighborhood_indices.shape[1:]
                neighborhood = _get_neighborhood(self.grid, self.neighborhood_indices, self.index_cap_per_dim, np.array(index)[..., *((None,)*len(shape))])
                updated_grid[*index] = self.rules(neighborhood, current_state)

            self.grid = updated_grid
            yield step, self.grid

    def reset_grid(self):
        self.grid = self._grid_generator()
