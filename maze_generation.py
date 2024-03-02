import numpy as np

from cellular_pytomata import Engine
from cellular_pytomata.renderer import Renderer


def activate(neighbourhood: np.ndarray) -> float | None:
    return 1. if neighbourhood.sum() == 3 else None


def deactivate(neighbourhood: np.ndarray) -> float | None:

    return 0. if (sum := neighbourhood.sum()) < 1 or sum > 5 else None


engine = Engine(lambda: np.random.rand(64, 64) > 0.95, [activate, deactivate])
renderer = Renderer(engine)
renderer.start()
