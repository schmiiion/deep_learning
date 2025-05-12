import numpy as np
import matplotlib.pyplot as plt

class Checker:

    def __init__(self,resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):

        def square(n, color):
            return np.full((n, n), fill_value=color, dtype=np.uint8)

        def checker_tile(single_tile_n):
            first_row = np.hstack([square(single_tile_n, 0), square(single_tile_n, 1)])
            second_row = np.hstack([square(single_tile_n, 1), square(single_tile_n, 0)])
            return np.vstack([first_row, second_row])

        if self.resolution % (2* self.tile_size) != 0:
            return

        checker_tile_repetitions = int(self.resolution / (2 * self.tile_size))
        ct = checker_tile(self.tile_size)
        self.output = np.tile(ct, (checker_tile_repetitions, checker_tile_repetitions))
        return self.output.copy()


    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
        return

class Circle:

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        x,y = self.position
        x_scale = np.arange(-x, self.resolution - x)
        y_scale = np.arange(-y, self.resolution - y)

        cutoff_range = self.radius ** 2

        xx, yy = np.meshgrid(x_scale, y_scale)
        f = (xx **2 + yy ** 2 < cutoff_range).astype(int)
        self.output = f
        return f.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
        return

class Spectrum:

    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        r_channel_row = np.linspace(0, 1, self.resolution)
        r_channel = np.tile(r_channel_row, (self.resolution, 1))

        g_channel_row = np.linspace(0, 1, self.resolution)
        g_channel_col = g_channel_row[:, np.newaxis]
        g_channel = np.tile(g_channel_col, (1, self.resolution))

        b_channel_row = np.linspace(1, 0, self.resolution)
        b_channel = np.tile(b_channel_row, (self.resolution, 1))

        self.output = np.stack((r_channel, g_channel, b_channel), axis=-1)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()
        return