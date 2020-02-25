import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Text2048-v0',
    entry_point='gym_text2048.envs:Text2048Env',
    nondetermistic=True,
)