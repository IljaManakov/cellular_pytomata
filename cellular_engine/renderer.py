import matplotlib.pyplot as plt
import numpy as np


class Renderer:

    def __init__(self, engine):
        self.engine = engine
        self.engine_iterator = iter(engine)
        self.paused = True
        self.figure, self.axes = plt.subplots(1, 1, figsize=(6, 6))
        self.canvas = self.figure.canvas
        self.timer = self.canvas.new_timer(interval=1)
        self.background = None

        grid_shape = engine.grid.shape
        x, y = np.meshgrid(range(grid_shape[0]), range(grid_shape[1]))
        self.points = self.axes.scatter(x, y, marker='s')
        self.points.set_array(engine.grid.flat)

        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        self.axes.set_facecolor('black')
        self.figure.tight_layout()

        self.timer.add_callback(self._run_step)
        self.canvas.mpl_connect('key_press_event', self._handle_key_press)
        self.canvas.mpl_connect('draw_event', lambda event: setattr(self, 'background', None))

    def start(self):
        self.timer.start()
        plt.show()

    def _run_step(self):
        if self.paused:
            return

        self.canvas.restore_region(self.canvas.copy_from_bbox(self.axes.bbox))
        step, grid = next(self.engine_iterator)
        self.points.set_array(grid.flat)
        self.axes.draw_artist(self.points)
        self.figure.canvas.blit(self.axes.bbox)
        self.canvas.flush_events()

    def _handle_key_press(self, event):
        if event.key == ' ':
            self.paused = not self.paused

        if event.key == 'q':
            plt.close()
        if event.key == 'r':
            self.engine.reset_grid()
