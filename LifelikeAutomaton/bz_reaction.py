import random
import sys
import os
from typing import Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule

ALPHA = 1.2
BETA = 1.0
GAMMA = 1.0


# states represented as [a, b, c]
class BZReaction(CellularAutomaton):
    def __init__(self):
        super().__init__(dimension=[100, 100],
                         neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:
        return [random.random() for _ in range(3)]

    def evolve_rule(self, last_cell_state, neighbors_last_states: Sequence) -> Sequence:
        average = last_cell_state[:]
        for cell in neighbors_last_states:
            average[0] += cell[0]
            average[1] += cell[1]
            average[2] += cell[2]
        average = [v/9.0 for v in average]

        new_state = [
            last_cell_state[0] + average[0]*(ALPHA*average[1] - GAMMA*average[2]),
            last_cell_state[1] + average[1]*(BETA*average[2] - ALPHA*average[0]),
            last_cell_state[2] + average[2]*(GAMMA*average[0] - BETA*average[1])
        ]
        return [max(0, min(v, 1)) for v in new_state]

    @staticmethod
    def draw_combined(current_state: Sequence) -> Sequence:
        return [255*v for v in current_state]

    @staticmethod
    def draw_highest(current_state: Sequence) -> Sequence:
        if current_state[0] >= current_state[1] and current_state[0] >= current_state[2]:
            return [255, 0, 0]
        if current_state[1] >= current_state[0] and current_state[1] >= current_state[2]:
            return [0, 255, 0]
        return [0, 0, 255]


if __name__ == "__main__":
    CAWindow(cellular_automaton=BZReaction(),
             window_size=(1080, 720),
             state_to_color_cb=BZReaction.draw_combined).run()
