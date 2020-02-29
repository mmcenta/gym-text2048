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

    def __init__(self, size=4, use_heuristic=False, merge_weight=0.,
                 empty_weight=0., monotonicity_params=None, sum_params=None):

        def get_params(params):
            if params is not None:
                if isinstance(params, (int, float)):
                    params =  {"weight": params, "exp": 1.}
                if isinstance(params, (list, tuple, np.ndarray)):
                    assert len(params) == 2
                    params = {"weight": params[0], "exp": params[1]}
            return params

        self.size = size
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.merge_weight = merge_weight
            self.empty_weight = empty_weight
            self.monotonicity_params = get_params(monotonicity_params)
            self.sum_params = get_params(sum_params)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([size * size + 2] * size * size)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _add_random_tile(self):
        empty_tiles = [t for t in product(*tee(range(self.size)))
                      if self.board[t] == 0]
        if len(empty_tiles) > 0:
            k = self.np_random.randint(len(empty_tiles))
            value = 1 if self.np_random.random() < 0.9 else 2
            self.board[empty_tiles[k]] = value

    def _compress(self, view):
        changed = False
        for j in range(self.size):
            count = 0
            for i in range(self.size):
                if view[i][j] != 0:
                    view[count][j], view[i][j] = view[i][j], view[count][j]
                    if count != i:
                        changed = True
                    count += 1
        return changed

    def _merge(self, view):
        reward = 0
        for j in range(self.size):
            for i in range(self.size - 1):
                if view[i][j] == view[i + 1][j] and view[i][j] != 0:
                    view[i][j] += 1
                    view[i + 1][j] = 0
                    reward += (2 ** view[i][j])
        return reward

    def _is_done(self):
        def can_merge(i, j):
            return any([self.board[i][j] == self.board[i + di][j + dj]
                        for di, dj in ((-1, 0), (0, -1), (0, 1), (1, 0))
                        if (i + di < self.size and i + di >= 0) and
                           (j + dj < self.size and j + dj >= 0)])

        for i, j in product(*tee(range(self.size))):
            if self.board[i][j] == 0 or can_merge(i, j):
                return False
        return True

    def _calculate_state_value(self):
        tile_sum = np.sum(np.power(self.board, self.sum_params["exp"]))
        empty = self.size * self.size - np.count_nonzero(self.board)

        def count_merges(line):
            count, merge, prev = 0, 0, 0
            for value in line:
                if value != 0:
                    if value == prev:
                        count += 1
                    elif count > 1:
                        merge += 1 + count
                        count = 0
            return merges

        # Maybe try taking the maximum of the lines and columns instead
        merges = sum([count_merges(self.board[i]) for i in range(self.size)])
        merges += sum([count_merges(self.board[:][j]) for j in range(self.size)])

        def score_monotonicity(line):
            left, right = 0., 0.
            for i in range(self.size):
                if line[i-1] > line[i]:
                    left += (pow(line[i-1], self.monotonicity_params["exp"]) -
                             pow(line[i], self.monotonicity_params["exp"]))
                else:
                    right += (pow(line[i], self.monotonicity_params["exp"]) -
                                           pow(line[i-1], self.monotonicity_params["exp"]))
            return min(left, right)

        monotonicity = sum([score_monotonicity(self.board[i]) for i in range(self.size)])
        monotonicity += sum([score_monotonicity(self.board[:][j]) for j in range(self.size)])

        return (self.empty_weight * empty +
                self.merge_weight * merges -
                self.monotonicity_params["weight"] * monotonicity -
                self.sum_params["weight"] * tile_sum)


    def step(self, action):
        assert self.action_space.contains(action)

        view = np.rot90(self.board, k=action)
        changed = self._compress(view)
        reward = self._merge(view)
        if changed or reward> 0:
            self._compress(view)
            self._add_random_tile()

        # Overwrite standard reward if using heuristics
        if self.use_heuristic:
            curr_value = self._calculate_state_value()
            reward = curr_value - self.last_state_value
            self.last_state_value = curr_value

        done = self._is_done()
        self.lastaction = action
        self.score += reward

        return np.ravel(self.board), reward, done, {'score': self.score}

    def reset(self):
        self.score = 0
        self.last_action = None
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self._add_random_tile()
        self._add_random_tile()
        if self.use_heuristic:
            self.last_state_value = self._calculate_state_value()
        return np.ravel(self.board)

    def render(self, mode='human'):
        out = StringIO if mode == 'ansi' else sys.stdout

        if self.lastaction is not None:
            out.write("  ({})\n".format(["Up", "Right", "Down", "Left"][self.lastaction]))

        def tile_to_symbol(tile):
            if tile == 0:
                return "     "
            elif tile in TILE_FORMAT:
                return colorize("{:>5}".format(2 ** tile), *TILE_FORMAT[tile])
            return colorize("{:>5}".format(2 ** tile), "gray", False)

        hline = "\n|" + "|".join(["-----"] * len(self.board)) + "|\n"
        symbols = [[tile_to_symbol(t) for t in line] for line in self.board]
        out.write(hline +
                  hline.join('|' + '|'.join(line) + '|' for line in symbols) +
                  hline)

        if mode != 'human':
            with closing(out):
                return out.getvalue()
