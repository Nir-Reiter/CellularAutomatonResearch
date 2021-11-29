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
    def __init__(self, name, data, generations):
        super().__init__(width=500, height=500)
        self.data = data
        self.generations = generations
        self.batches = []
        self.labels = []
        self.images = []
        self.current_image = 0
        self.name = pyglet.text.Label(name, x=10, y=460)

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
        self.name.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.RIGHT:
            self.current_image = (self.current_image + 1) % len(self.images)
        elif symbol == pyglet.window.key.LEFT:
            self.current_image = (self.current_image - 1) % len(self.images)


def main():
    files = ["newdata.npz", "data.npz"]
    labels = [[1.2, 1.0, 0.8, 0.6, 0.4], "Full Light"]
    data = [np.load(f) for f in files]
    windows = []
    for i in range(len(data)):
        plt.plot(data[i]["xvalues"], data[i]["entropy"], label=labels[i])
        windows.append(SnapshotWindow(files[i], np.reshape(data[i]["states"], (-1, 100, 100, 4)),
                                      data[i]["state_steps"]))
    plt.title("Average Local Entropy")
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.show(block=False)
    pyglet.app.run()


if __name__ == "__main__":
    main()
