#!/usr/bin/env python3
"""
Copyright 2019 Richard Feistenauer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=wrong-import-position
# pylint: disable=missing-function-docstring
# pylint: disable=no-self-use

import random
import sys
import os
from typing import Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule

ALIVE = [1.0]
DEAD = [0.0]


class TestAutomaton(CellularAutomaton):

    def __init__(self):
        super().__init__(dimension=[100, 100],
                         neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

    def init_cell_state(self, coords) -> Sequence:
        return [random.randint(0, 1)]

    def evolve_rule(self, last_cell_state, neighbors_last_states: Sequence) -> Sequence:
        alive_count = 0
        for cell in neighbors_last_states:
            if cell == ALIVE:
                alive_count += 1

        if alive_count == 3 and last_cell_state[0] < ALIVE[0]:
            return ALIVE
        if alive_count > 3 or alive_count < 2 or last_cell_state[0] < ALIVE[0]:
            return [max(0.0, last_cell_state[0] - 0.1)]
        return ALIVE

    @staticmethod
    def state_to_color(current_state: Sequence) -> Sequence:
        if current_state == ALIVE:
            return 255, 255, 255
        else:
            return [127 * current_state[0]] * 3


if __name__ == "__main__":
    CAWindow(cellular_automaton=TestAutomaton(),
             window_size=(1000, 830),
             state_to_color_cb=TestAutomaton.state_to_color).run()
