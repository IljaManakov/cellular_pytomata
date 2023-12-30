import numpy as np

from cellular_engine import Engine
from cellular_engine.renderer import Renderer


def activate(neighbourhood: np.ndarray) -> float | None:
    return 1. if neighbourhood.sum() == 3 else None


def deactivate(neighbourhood: np.ndarray) -> float | None:

    return 0. if (sum := neighbourhood.sum()) < 1 or sum > 5 else None


engine = Engine(np.random.randint(0, 2, (64, 64)), [activate, deactivate])
renderer = Renderer(engine)
renderer.start()
