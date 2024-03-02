import matplotlib.pyplot as plt
import numpy as np


class Renderer:

    def __init__(self, engine):
        self.engine = engine
        self.engine_iterator = iter(engine)
        self.paused = True
        self.adding_cells = False
        self._last_index = None
        self.figure, self.axes = plt.subplots(1, 1, figsize=(6, 6))
        self.canvas = self.figure.canvas
        self.timer = self.canvas.new_timer(interval=1)
        self.background = None

        grid_shape = engine.grid.shape
        x, y = np.meshgrid(range(grid_shape[0]), range(grid_shape[1]))
        self.points = self.axes.scatter(x, y, marker='s')
        self.points.set_array(engine.grid.flat)
        self.points.set_cmap('inferno')

        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        self.axes.set_facecolor('black')
        self.figure.tight_layout()

        self.timer.add_callback(self._run_step)
        self.canvas.mpl_connect('key_press_event', self._handle_key_press)
        self.canvas.mpl_connect('button_press_event', lambda event: setattr(self, 'adding_cells', True))
        self.canvas.mpl_connect('button_release_event', lambda event: setattr(self, 'adding_cells', False))
        self.canvas.mpl_connect('motion_notify_event', self._handle_mouse_interaction)
        self.canvas.mpl_connect('draw_event', lambda event: setattr(self, 'background', None))

    def start(self):
        self.timer.start()
        plt.show()
        self.canvas.mpl_connect('close_event', plt.close())

    def _run_step(self):
        if self.paused:
            return

        self.canvas.restore_region(self.canvas.copy_from_bbox(self.axes.bbox))
        step, grid = next(self.engine_iterator)
        self.points.set_array(grid.flat / grid.max())
        self.draw()

    def draw(self):
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
            self.points.set_array(self.engine.grid.flat)
            self.draw()
        if event.key == 'right':
            if self.paused:
                self.paused = False
                self._run_step()
                self.paused = True

    def _handle_mouse_interaction(self, event):

        if not self.paused and not (self.adding_cells or self.adding_cells):
            return

        if not event.inaxes:
            return
        x, y = int(event.xdata), int(event.ydata)
        index = y * self.engine.grid.shape[0] + x
        if index == self._last_index:
            return

        if self.adding_cells:
            grid = self.engine.grid.ravel()
            grid[index] = 1
            self.engine.grid = grid.reshape(self.engine.grid.shape)
            self.points.set_array(grid / grid.max())
            self.draw()
