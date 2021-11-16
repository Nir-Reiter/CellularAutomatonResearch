import time
import numpy as np
import pandas as pd
import anndata
import scipy.stats as st
import pygame
import matplotlib.pyplot as plt
from cellular_automaton import CAWindow

from LifelikeAutomaton.bz_reaction import *


saved_data = pd.DataFrame()


class CustomCAWindow(CAWindow):
    STEPS_PER_CALCULATION = 10
    LOCAL_SCALE = 5

    def __init__(self, cellular_automaton: CellularAutomaton, *args, **kwargs):
        self.coefficients = kwargs.pop("coefficients")
        super().__init__(cellular_automaton, *args, **kwargs)
        self.data = [[], [], [], []]

        self.fig, self.ax = plt.subplots()
        self.lines = [
            self.ax.plot([], self.data[0], label="Average Local Entropy")[0],
            self.ax.plot([], self.data[1], label="Global Entropy")[0],
            self.ax.plot([], self.data[2], label="Average Local STD")[0],
            self.ax.plot([], self.data[3], label="Global STD")[0]
        ]
        self.ax.legend()
        self.ax.set_xlim([0, 80])
        self.ax.set_ylim([0, 1.5])
        plt.ion()
        self.fig.show()

    def run(self,
            evolutions_per_second=0,
            evolutions_per_draw=1,
            last_evolution_step=0, ):

        self.calculate_stats(5)

        while self._is_not_user_terminated() and self._not_at_the_end(last_evolution_step):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.save_state()

            time_ca_start = time.time()
            for i in range(0, evolutions_per_draw, CustomCAWindow.STEPS_PER_CALCULATION):
                self._cellular_automaton.evolve(CustomCAWindow.STEPS_PER_CALCULATION)
                self.calculate_stats(self.LOCAL_SCALE)
            time_ca_end = time.time()
            self._redraw_dirty_cells()
            time_ds_end = time.time()

            self.print_process_info(evolve_duration=(time_ca_end - time_ca_start),
                                    draw_duration=(time_ds_end - time_ca_end),
                                    evolution_step=self._cellular_automaton.evolution_step)
            self._sleep_to_keep_rate(time.time() - time_ca_start, evolutions_per_second)

        saved_data['Local Entropy at A:{}, B:{}, C:{}'.format(*self.coefficients)] = self.data[0]

    def save_state(self):
        print("Saving state...")
        for i in range(3):
            np.savetxt("{}.csv".format(i), self._cellular_automaton.start_state[i].tolist())

    def calculate_stats(self, size):
        cells = self._cellular_automaton.get_cells()
        states = [c.state for c in cells.values()]
        entropy = []
        std = []
        for y in range(0, self._cellular_automaton.dimension[1], size):
            for x in range(0, self._cellular_automaton.dimension[0], size):
                data = self.get_region_data(x, y, 5)
                entropy.append(list(st.entropy(data, axis=0) / np.log(size * size)))
                std.append(list(np.std(data, axis=0)))

        # comparing average local and global entropy with local and global deviation
        result = [np.average(entropy, axis=0),
                  st.entropy(states, axis=0) / np.log(len(states)),
                  np.average(std, axis=0),
                  np.std(states, axis=0)]

        result = np.average(result, axis=1)
        self.data = np.c_[self.data, result]
        xx = list(range(0, len(self.data[0])*10, 10))
        for i in range(4):
            self.lines[i].set_data(xx, self.data[i])
            self.ax.set_xlim([-10, xx[-1]+90])

        # print(cells[0, 0].state)  # this is how you do it

    def get_region_data(self, x, y, size):
        total = []
        (w, h) = self._cellular_automaton.dimension
        for j in range(y, y + size):
            for i in range(x, x + size):
                total.append(np.clip(self._cellular_automaton.get_cells()[i % w, j % h].state, 0.0001, 1))
        return total


if __name__ == "__main__":
    coefficients = [
        [1.2, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.8, 1.0, 1.0],
        [0.8, 1.0, 1.2]
    ]
    for i in range(len(coefficients)):
        CustomCAWindow(cellular_automaton=BZReaction(*coefficients[i]),
                       window_size=(1080, 720),
                       state_to_color_cb=BZReaction.draw_combined,
                       coefficients=coefficients[i]) \
            .run(evolutions_per_draw=100, last_evolution_step=2000)

    annotated = anndata.AnnData(saved_data)
    annotated.write("entropy_data.h5ad")
