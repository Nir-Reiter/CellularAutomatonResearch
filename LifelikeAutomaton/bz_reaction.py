import random
import sys
import os
import numpy as np
from typing import Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule

ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0

load = input("Load from saved state? (y/n):") == "y"
if load:
    load_state = [np.loadtxt("{}.txt".format(i)) for i in range(3)]


def clamp(a, x, b):
    return max(a, min(x, b))


# states represented as [a, b, c]
class BZReaction(CellularAutomaton):
    def __init__(self):
        if load:
            self.start_state = np.swapaxes(load_state, 0, 2)
        else:
            self.start_state = np.random.rand(100, 100, 3)
        super().__init__(dimension=[100, 100],
                         neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

    # maybe override this to give the array all at once
    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:
        return self.start_state[cell_coordinate[0], cell_coordinate[1]]

    def evolve_rule(self, last_cell_state, neighbors_last_states: Sequence) -> Sequence:
        print(last_cell_state)
        average = [sum(x)/9.0 for x in zip(last_cell_state, *neighbors_last_states)]
        new_state = [
            last_cell_state[0] + average[0]*(ALPHA*average[1] - GAMMA*average[2]),
            last_cell_state[1] + average[1]*(BETA*average[2] - ALPHA*average[0]),
            last_cell_state[2] + average[2]*(GAMMA*average[0] - BETA*average[1])
        ]
        return [clamp(0, v, 1) for v in new_state]

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
