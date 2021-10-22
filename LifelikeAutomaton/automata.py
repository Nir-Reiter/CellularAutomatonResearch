import time
import pygame
import numpy as np
import scipy.stats as st
from bz_reaction import BZReaction
from cellular_automaton import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule


class CustomCAWindow(CAWindow):
    def run(self,
            evolutions_per_second=0,
            evolutions_per_draw=1,
            last_evolution_step=0, ):

        while self._is_not_user_terminated() and self._not_at_the_end(last_evolution_step):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.save_state()
            
            time_ca_start = time.time()
            self._cellular_automaton.evolve(evolutions_per_draw)
            if self._cellular_automaton.evolution_step % 20 == 0:
                self.calculate_stats()
            time_ca_end = time.time()
            self._redraw_dirty_cells()
            time_ds_end = time.time()
            self.print_process_info(evolve_duration=(time_ca_end - time_ca_start),
                                    draw_duration=(time_ds_end - time_ca_end),
                                    evolution_step=self._cellular_automaton.evolution_step)
            self._sleep_to_keep_rate(time.time() - time_ca_start, evolutions_per_second)

    def save_state(self):
        print("Saving state...")
        for i in range(3):
            np.savetxt("{}.txt".format(i), self._cellular_automaton.start_state[i].tolist())

    def calculate_stats(self):
        cells = self._cellular_automaton.get_cells()
        total = 0
        count = 0
        for y in range(0, self._cellular_automaton.dimension[1]):
            for x in range(0, self._cellular_automaton.dimension[0]-2):
                total += CustomCAWindow.calculate3x3(x, y, cells)
                count += 1

        # print(total/count)
        input()
        # print(cells[0, 0].state)  # this is how you do it

    @staticmethod
    def calculate3x3(x, y, cells):
        total = []
        for j in range(y, y+3):
            for i in range(x, x+3):
                total.append(cells[i, j].state)
        print(total)
        print(st.entropy(total, axis=0))
        input()
        return 1


if __name__ == "__main__":
    CustomCAWindow(cellular_automaton=BZReaction(),
                   window_size=(1080, 720),
                   state_to_color_cb=BZReaction.draw_combined)\
        .run(evolutions_per_draw=1)
