import time
import pygame
import numpy as np
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


if __name__ == "__main__":
    CustomCAWindow(cellular_automaton=BZReaction(),
                   window_size=(1080, 720),
                   state_to_color_cb=BZReaction.draw_combined)\
        .run(evolutions_per_draw=1)
