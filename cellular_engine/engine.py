import itertools
from enum import Enum
from typing import Callable, Generator

import numpy as np

Rule = Callable[[np.ndarray], float | None]


class RetrievalMode(Enum):
    WRAPPING: str = 'wrapping'
    PADDED: str = 'padded'


class Engine:

    def __init__(self,
                 grid: np.ndarray,
                 rules: list[Rule],
                 *,
                 neighbourhood_shape: tuple[int, ...] = (3, 3),
                 retrieval_mode: RetrievalMode = RetrievalMode.WRAPPING):

        self.grid = grid
        self.rules = rules
        self.neighbourhood_shape = neighbourhood_shape
        self.retrieval_mode = retrieval_mode

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

    def _get_neighbourhood(self, index: tuple[int, ...]) -> np.ndarray:

        dimensionality = self.grid.ndim
        indices = np.indices(self.neighbourhood_shape)
        offsets = (np.array(index) - np.array(self.neighbourhood_shape) // 2)
        index_cap_per_dim = np.array(self.grid.shape)

        # we need to add dimensions to allow proper broadcasting
        offsets = offsets[..., *([None] * dimensionality)]
        index_cap_per_dim = index_cap_per_dim[..., *([None] * dimensionality)]

        neighbourhood = self.grid[*((indices + offsets) % index_cap_per_dim)]

        if self.retrieval_mode is not RetrievalMode.WRAPPING:
            neighbourhood[indices < 0 | indices > index_cap_per_dim] = 0

        return neighbourhood

    def start(self):
        pass
