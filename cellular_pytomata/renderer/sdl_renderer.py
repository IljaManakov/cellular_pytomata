from time import perf_counter

import numpy as np
import sdl2.ext

FULLSCREEN = (1920, 1080)
PIXEL_COLOR = (199, 170, 45)
BACKGROUND = (0, 0, 0)


class SDLRenderer:

    def __init__(self, engine):
        sdl2.ext.init()
        self.engine = engine
        self.engine_iter = iter(engine)
        self.window = sdl2.ext.Window("Cellular Pytomata", size=FULLSCREEN)
        self.window.show()
        self.canvas = sdl2.ext.Renderer(self.window, logical_size=self.engine.grid.shape)
        self.paused = True
        self._fps = 0

    def _handle_events(self):
        events = sdl2.ext.get_events()
        if sdl2.ext.key_pressed(events, sdl2.SDLK_SPACE):
            self.paused = not self.paused
        if sdl2.ext.key_pressed(events, sdl2.SDLK_q):
            self.running = False
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                self.running = False

    def _count_fps(self, _last_call_time = [perf_counter()], _time_elapsed = [0]):
        _time_elapsed[0] += perf_counter() - _last_call_time[0]
        _last_call_time[0] = perf_counter()
        self._fps += 1
        if  _time_elapsed[0] >= 1:
            self.window.title = f"Cellular Pytomata - {self._fps} FPS"
            self._fps = 0
            _time_elapsed[0] = 0


    def _draw(self):
        grid = self.engine.grid
        self.canvas.clear(BACKGROUND)
        self.canvas.draw_point([(x, y) for x, y in zip(*np.where(grid == 1))], PIXEL_COLOR)
        self.canvas.present()

    def _main_loop(self):
        self._count_fps()
        self._handle_events()
        self._draw()
        if not self.paused:
            next(self.engine_iter)
        self.window.refresh()

    def start(self):
        self.running = True
        while self.running:
            self._main_loop()
