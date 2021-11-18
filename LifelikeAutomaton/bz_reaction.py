import numpy as np
from typing import Sequence

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton import CellularAutomaton, MooreNeighborhood, EdgeRule


def clamp(a, x, b):
    return max(a, min(x, b))


# states represented as [a, b, c]
class BZReaction(CellularAutomaton):
    def __init__(self, alpha, beta, gamma):
        load = False  # input("Load from saved state? (y/n):") == "y"
        # LOADING NEEDS TO BE FIXED WITH NEW 4TH LAYER
        if load:
            pass
        else:
            self.start_state = np.random.rand(100, 100, 4)
            self.start_state[0:100, 0:100, 3] = 0.75
            self.start_state[30:70, 30:70, 3] = 1
        super().__init__(dimension=[100, 100],
                         neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

        self.ALPHA = alpha
        self.BETA = beta
        self.GAMMA = gamma

    # maybe override this to give the array all at once
    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:
        return list(self.start_state[cell_coordinate[0], cell_coordinate[1]])

    def evolve_rule(self, state, neighbor_states: Sequence) -> Sequence:
        average = [sum(x) / 9.0 for x in zip(state, *neighbor_states)]
        return [clamp(0, state[0] + average[0] * (self.ALPHA * average[1] - self.GAMMA * average[2]) * state[3], 1),
                clamp(0, state[1] + average[1] * (self.BETA * average[2] - self.ALPHA * average[0]) * state[3], 1),
                clamp(0, state[2] + average[2] * (self.GAMMA * average[0] - self.BETA * average[1]) * state[3], 1),
                state[3]]

    @staticmethod
    def draw_combined(current_state: Sequence) -> Sequence:
        return [255 * v for v in current_state]

    @staticmethod
    def draw_highest(current_state: Sequence) -> Sequence:
        if current_state[0] >= current_state[1] and current_state[0] >= current_state[2]:
            return [255, 0, 0]
        if current_state[1] >= current_state[0] and current_state[1] >= current_state[2]:
            return [0, 255, 0]
        return [0, 0, 255]

    @staticmethod
    def draw_energy(current_state: Sequence) -> Sequence:
        return [255 * current_state[3], 255 * current_state[3], 0]
