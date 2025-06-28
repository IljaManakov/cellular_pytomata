from dataclasses import dataclass
import numpy as np
import numba


@numba.jit
def update_state(neighborhood: np.ndarray, current_state: int, overpopulation: int, underpopulation: int,
                 reproduction: int) -> int:
    sum_neighbors = np.sum(neighborhood) - current_state

    if sum_neighbors > overpopulation or sum_neighbors < underpopulation:
        return 0
    if sum_neighbors == reproduction:
        return 1
    return current_state

@dataclass
class GameOfLife:
    overpopulation: int = 3
    underpopulation: int = 2
    reproduction: int = 3

    def __call__(self, neighborhood: np.ndarray, current_state: int) -> int:
        return update_state(neighborhood, current_state, self.overpopulation, self.underpopulation, self.reproduction)


@dataclass
class Maze(GameOfLife):
    overpopulation: int = 5
    underpopulation: int = 1
    reproduction: int = 3
