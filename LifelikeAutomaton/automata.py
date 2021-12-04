import time
import numpy as np
import pandas as pd
import anndata
import scipy.stats as st
import pygame
import matplotlib.pyplot as plt
from cellular_automaton import CAWindow
from bz_reaction import *
# from LifelikeAutomaton.bz_reaction import *

GRID_SIZE = 100
LIGHT_LAYERS = 5


class CustomCAWindow(CAWindow):
    LOCAL_SCALE = 5

    def __init__(self, cellular_automaton: CellularAutomaton, *args, **kwargs):
        self.coefficients = kwargs.pop("coefficients")
        self.measure_gradient = kwargs.pop("entropy_gradient")
        print(self.measure_gradient)
        super().__init__(cellular_automaton, *args, **kwargs)
        self.saved_states = []
        self.entropy_data = []

    def run(self, evolutions_per_second=0, evolutions_per_draw=1, draws_per_calculation=1,
            draws_per_save=100, last_evolution_step=0):

        self.calculate_stats(self.LOCAL_SCALE)
        self.save_state()

        frequency = draws_per_calculation * evolutions_per_draw
        draws = 0
        while self._is_not_user_terminated() and self._not_at_the_end(last_evolution_step):
            keys = pygame.key.get_pressed()

            time_ca_start = time.time()
            self._cellular_automaton.evolve(evolutions_per_draw)
            time_ca_end = time.time()
            draws += 1
            if draws % draws_per_calculation == 0:
                self.calculate_stats(self.LOCAL_SCALE)

                current_step = self._cellular_automaton.get_evolution_step()
                xx = list(range(0, current_step + 1, frequency))
                plt.clf()  # clear figure
                plt.plot(xx, self.entropy_data, label=["1.2", "1.0", "0.8", "0.6", "0.4"])
                plt.title(f"Entropy at A={self.coefficients[0]} B={self.coefficients[1]} C={self.coefficients[2]}")
                plt.ylim(0.5, 1.0)
                plt.xlim(-10, xx[-1] + 90)
                plt.legend()
                plt.show(block=False)
                plt.pause(0.05)
            if draws % draws_per_save == 0:
                self.save_state()

            self._redraw_dirty_cells()
            time_ds_end = time.time()

            self.print_process_info(evolve_duration=(time_ca_end - time_ca_start),
                                    draw_duration=(time_ds_end - time_ca_end),
                                    evolution_step=self._cellular_automaton.evolution_step)
            self._sleep_to_keep_rate(time.time() - time_ca_start, evolutions_per_second)

        return self.saved_states, self.entropy_data

    def save_state(self):
        self.saved_states.append([c.state for c in self._cellular_automaton.get_cells().values()])

    def calculate_stats(self, size):
        # cells = self._cellular_automaton.get_cells()
        # states = [c.state for c in cells.values()]

        entropy = list()
        if self.measure_gradient:
            h = self._cellular_automaton.dimension[1] // LIGHT_LAYERS
            for i in range(LIGHT_LAYERS):
                entropy.append(list())
                for y in range(i * h, (i+1) * h, size):
                    for x in range(0, self._cellular_automaton.dimension[0], size):
                        data = self.get_region_data(y, x, 5)
                        entropy[i].append(st.entropy(data, axis=0) / np.log(size * size))
            result = np.average(entropy, axis=(1, 2))
        else:
            for y in range(0, self._cellular_automaton.dimension[1], size):
                for x in range(0, self._cellular_automaton.dimension[0], size):
                    data = self.get_region_data(y, x, 5)
                    entropy.append(st.entropy(data, axis=0) / np.log(size * size))
            result = np.average(entropy, axis=(0, 1))

        self.entropy_data.append(result)

        # print(cells[0, 0].state)  # this is how you do it

    def get_region_data(self, x, y, size):
        total = []
        (w, h) = self._cellular_automaton.dimension
        for j in range(y, y + size):
            for i in range(x, x + size):
                total.append(np.clip(self._cellular_automaton.get_cells()[i % w, j % h].state, 0.0001, 1))
        return np.delete(total, 3, 1)


def main():
    draw_rate = 10
    calculation_rate = 10
    save_rate = 100
    steps = 3000
    coefficients = [1.0, 1.0, 1.0]

    max_light = 1.2
    gradient = GRID_SIZE // LIGHT_LAYERS

    light_grids = [np.full((GRID_SIZE, GRID_SIZE), max_light - i / LIGHT_LAYERS) for i in range(LIGHT_LAYERS)]
    light_grids.append(np.ones((GRID_SIZE, GRID_SIZE)))
    for i in range(LIGHT_LAYERS):
        light_grids[-1][i * gradient:(i + 1) * gradient, :] = max_light - i / LIGHT_LAYERS

    for lg in light_grids:
        c = CustomCAWindow(cellular_automaton=BZReaction(GRID_SIZE, *coefficients, lg),
                           window_size=(1080, 720),
                           state_to_color_cb=BZReaction.draw_combined,
                           coefficients=coefficients,
                           entropy_gradient=(lg is light_grids[-1]))
        saved_states, entropy = c.run(evolutions_per_draw=draw_rate,
                                      draws_per_calculation=calculation_rate // draw_rate,
                                      draws_per_save=save_rate // draw_rate,
                                      last_evolution_step=steps)

        filename = time.asctime().replace(":", "_")
        np.savez(filename,
                 states=saved_states,
                 entropy=entropy,
                 xvalues=np.array(range(0, steps + 1, calculation_rate)),
                 state_steps=np.array(range(0, steps + 1, save_rate)))


if __name__ == "__main__":
    main()
