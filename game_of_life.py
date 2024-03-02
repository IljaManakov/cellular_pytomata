import numpy as np

from cellular_pytomata import Engine
from cellular_pytomata.renderer import Renderer


def overpopulation(neighbourhood: np.ndarray) -> float | None:
    return 0. if neighbourhood.sum() > 3 else None


def reproduction(neighbourhood: np.ndarray) -> float | None:
    return 1. if neighbourhood.sum() == 3 else None


def underpopulation(neighbourhood: np.ndarray) -> float | None:
    return 0. if neighbourhood.sum() < 2 else None


engine = Engine(np.random.randint(0, 2, (64, 64)), [overpopulation, reproduction, underpopulation])
renderer = Renderer(engine)
renderer.start()
