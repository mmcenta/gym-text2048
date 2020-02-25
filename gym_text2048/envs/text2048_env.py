from contextlib import closing
from itertools import product, tee
import logging
import sys

import gym
from gym import error, spaces
from gym.utils import colorize, seeding
import numpy as np
from six import StringIO, b

logger = logging.getLogger(__name__)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

TILE_FORMAT = {
    1: ('white', True),
    2: ('cyan', False),
    3: ('cyan', True),
    4: ('blue', False),
    5: ('blue', True),
    6: ('magenta', False),
    7: ('magenta', True),
    8: ('red', False),
    9: ('red', True),
    10: ('yellow', False),
    11: ('yellow', True),
}

class Text2048Env(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size=4):
        self.size = size

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([32] * size * size)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _add_random_tile(self):
        empty_tiles = [t for t in product(*tee(range(self.size)))
                      if self.board[t] == 0]
        k = self.np_random.randint(len(empty_tiles))
        value = 1 if self.np_random.random() < 0.9 else 2
        self.board[empty_tiles[k]] = value

    def _compress(self, view):
        changed = False
        for col in range(self.size):
            top, bottom = 0, self.size - 1
            while top < bottom:
                if view[top][col] != 0:
                    top += 1
                elif view[bottom][col] == 0:
                    bottom -= 1
                else:
                    view[top][col], view[bottom][col] = view[bottom][col], view[top][col]
                    top, bottom = top + 1, bottom - 1
                    changed = True
        return changed

    def _merge(self, view):
        reward = 0
        for col in range(self.size):
            for i in range(self.size - 1):
                if view[i][col] == view[i + 1][col] and view[i][col] != 0:
                    view[i][col] += 1
                    view[i + 1][col] = 0
                    reward += (2 ** view[i][col])
        return reward

    def step(self, action):
        assert self.action_space.contains(action)

        view = np.rot90(self.board, k=action)
        changed = self.compress(view)
        reward = self._mergeE(view)
        self.compress(view)
        done = (not changed and score == 0)

        self.score += reward
        self._add_random_tile()

        return np.ravel(self.board), reward, done, {'score': self.score}

    def reset(self):
        self.score = 0
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self._add_random_tile()
        self._add_random_tile()

    def render(self, mode='human'):
        out = StringIO if mode == 'ansi' else sys.stdout

        scoreline = "\n SCORE: " + str(self.score) + "\n"
        hline = "\n|" + "|".join(["-----"] * len(board)) + "|\n"

        def tile_to_symbol(tile):
            if tile == 0:
                return "     "
            elif tile in TILE_FORMAT:
                return colorize("{:>5}".format(2 ** tile), *TILE_FORMAT[tile])
            return colorize("{:>5}".format(2 ** tile), "gray", False)

        symbols = [[tile_to_symbol(t) for t in line] for line in board]
        out.write(scoreline +
                  hline +
                  hline.join('|' + '|'.join(line) + '|' for line in symbols) +
                  hline)

        if mode != 'human':
            with closing(out):
                return out.getvalue()