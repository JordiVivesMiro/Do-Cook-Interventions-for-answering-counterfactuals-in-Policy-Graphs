from typing import Optional, Tuple, Any

import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

from PantheonRL.pantheonrl.common.agents import StaticPolicyAgent
from PantheonRL.pantheonrl.common.observation import Observation
from pgeon import Agent
from stable_baselines3 import PPO


class OvercookedAgent(Agent):
    def __init__(self, ckpt_path: str, **kwargs):
        self.agent = StaticPolicyAgent(
            PPO.load(ckpt_path, **kwargs).policy
        )

    def act(self,
            state: Tuple[Observation, OvercookedState]
            ) -> np.ndarray:
        obs, _ = state
        return self.agent.get_action(obs)


