from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import shapes
from bz_reaction import BZReaction
from LifelikeAutomaton.bz_reaction import BZReaction


def get_color(state):
    return [int(255 * v * state[3] / 1.2) for v in state[:3]]


class SnapshotWindow(pyglet.window.Window):
    def __init__(self, data, generations):
        super().__init__(width=500, height=500)
        self.data = data
        self.generations = generations
        self.batches = []
        self.labels = []
        self.images = []
        self.current_image = 0

        for i in range(len(self.data)):
            self.images.append([])
            self.batches.append(pyglet.graphics.Batch())
            for y in range(len(self.data[i])):
                for x in range(len(self.data[i][y])):
                    self.images[-1].append(self.batches[i].add(4, pyglet.gl.GL_QUADS, None,
                                                               ('v2i', (
                                                                   x * 5, y * 5,
                                                                   x * 5 + 5, y * 5,
                                                                   x * 5 + 5, y * 5 + 5,
                                                                   x * 5, y * 5 + 5)),
                                                               ('c3B', get_color(self.data[i][y][x]) * 4)))
            self.labels.append(pyglet.text.Label(f"{self.generations[i]}", x=10, y=10))

    def on_draw(self):
        self.clear()
        self.batches[self.current_image].draw()
        self.labels[self.current_image].draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.RIGHT:
            self.current_image = (self.current_image + 1) % len(self.images)
        elif symbol == pyglet.window.key.LEFT:
            self.current_image = (self.current_image - 1) % len(self.images)


def main():
    data = np.load("data.npz")
    plt.plot(data["xvalues"], data["entropy"], label="No Light")
    data_light = np.load("data_light.npz")
    plt.plot(data_light["xvalues"], data_light["entropy"], label="Light Rectangle")
    data_light_gradient = np.load("data_light_gradient.npz")
    plt.plot(data_light_gradient["xvalues"], data_light_gradient["entropy"], label="Light Gradient")
    plt.title("Average Local Entropy")
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.show(block=False)

    window = SnapshotWindow(np.reshape(data["states"], (-1, 100, 100, 4)), data["state_steps"])
    window_light = SnapshotWindow(np.reshape(data_light["states"], (-1, 100, 100, 4)), data["state_steps"])
    window_light_grad = SnapshotWindow(np.reshape(data_light_gradient["states"], (-1, 100, 100, 4)), data["state_steps"])
    pyglet.app.run()


if __name__ == "__main__":
    main()
