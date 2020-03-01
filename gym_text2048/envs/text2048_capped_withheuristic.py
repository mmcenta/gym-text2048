import logging

import numpy as np

from gym_text2048.envs import Text2048WithHeuristicEnv


logger = logging.getLogger(__name__)


class Text2048CappedWithHeuristicEnv(Text2048WithHeuristicEnv):
    def __init__(self, size=4, goal_tile=11):
        super(Text2048CappedWithHeuristicEnv, self).__init__(size=size)
        self.goal_tile = goal_tile

    def _is_done(self):
        max_tile = np.max(self.board)
        return (max_tile >= self.goal_tile or
                super(Text2048CappedWithHeuristicEnv, self)._is_done())
