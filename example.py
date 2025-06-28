import numpy as np

from cellular_pytomata import Engine
from cellular_pytomata.renderer import SDLRenderer as Renderer
from cellular_pytomata.rules import Maze

WINDOW_SIZE = 128
engine = Engine(np.random.randint(0, 101, (int(WINDOW_SIZE * 16/9), WINDOW_SIZE)) // 95, Maze())
print(engine.grid.shape)
renderer = Renderer(engine)
renderer.start()
