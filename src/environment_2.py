import random

import gymnasium
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.utils import set_random_seed
import tensorflow as tf

from PantheonRL.pantheonrl.common.observation import Observation


class OvercookedWrapper:
    def __init__(self, env: gymnasium.Env):
        self.env = env

    def _seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        sb3.common.utils.set_random_seed(seed)

    def reset(self, seed):
        self._seed(seed)
        obs = self.env.reset()
        return (Observation(obs), self.env.env.base_env.state), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return (Observation(obs), self.env.env.base_env.state), reward, done, done, info
