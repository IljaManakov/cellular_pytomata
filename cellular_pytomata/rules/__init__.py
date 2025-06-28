from dataclasses import dataclass
import numpy as np

@dataclass
class GameOfLife:
    overpopulation: int = 3
    underpopulation: int = 2
    reproduction: int = 3

    def __call__(self, neighborhood: np.ndarray, current_state: int) -> int:
        sum_neighbors = np.sum(neighborhood) - current_state

        if sum_neighbors > self.overpopulation or sum_neighbors < self.underpopulation:
            return 0
        if sum_neighbors == self.reproduction:
            return 1
        return current_state


@dataclass
class Maze(GameOfLife):
    overpopulation: int = 5
    underpopulation: int = 1
    reproduction: int = 3
