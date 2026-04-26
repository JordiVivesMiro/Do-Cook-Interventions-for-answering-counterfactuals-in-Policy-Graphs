from typing import Literal, Tuple, Optional, Dict, Any

import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

import pantheonrl.common.agents as prl
from pantheonrl.common.multiagentenv import SimultaneousEnv
from pantheonrl.common.observation import Observation


class OvercookedTrainingPantheonRL(SimultaneousEnv):
    def __init__(self, mdp):
        super(OvercookedTrainingPantheonRL, self).__init__()

        self.env: OvercookedEnv = OvercookedEnv.from_mdp(mdp, horizon=400)
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.INDEX_TO_ACTION))

        self._actions1 = []
        self._actions2 = []
        self._rewards = []

    def _setup_observation_space(self):
        dummy_mdp = self.env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.env.featurize_state_mdp(dummy_state)[0].shape
        # high = np.ones(obs_shape) * max(dummy_mdp.soup_cooking_time, dummy_mdp.num_items_for_soup, 5)
        return gym.spaces.Box(- np.ones(obs_shape) * np.inf, np.ones(obs_shape) * np.inf, dtype=np.float32)

    def multi_step(self,
                   ego_action: np.ndarray,
                   alt_action: np.ndarray
                   ) -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                              Tuple[float, float],
                              bool,
                              Dict]:
        self._actions1.append(int(ego_action))
        self._actions2.append(int(alt_action))
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]

        state, _, done, info = self.env.step((ego_action, alt_action))
        state1, state2 = self.env.featurize_state_mdp(state)
        reward = sum(info['sparse_r_by_agent']) + sum(info['shaped_r_by_agent'])
        self._rewards.append(reward)

        return (state1, state2), (reward, reward), done, info

    def multi_reset(self) -> Tuple[np.ndarray, np.ndarray]:
        print(self._rewards)
        print(self._actions1)
        print(self._actions2)
        if self._rewards:
            print(max(self._rewards))
        self._rewards = []
        self._actions1 = []
        self._actions2 = []
        self.env.reset()
        state = self.env.state

        state1, state2 = self.env.featurize_state_mdp(state)

        return state1, state2

    def render(self, mode="human"):
        pass


class OvercookedPantheonRL(SimultaneousEnv):
    def __init__(self, mdp):
        super(OvercookedPantheonRL, self).__init__()

        self.env: OvercookedEnv = OvercookedEnv.from_mdp(mdp, horizon=400)
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.INDEX_TO_ACTION))

        self._actions1 = []
        self._actions2 = []
        self._rewards = []

    def _setup_observation_space(self):
        dummy_mdp = self.env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.env.featurize_state_mdp(dummy_state)[0].shape
        # high = np.ones(obs_shape) * max(dummy_mdp.soup_cooking_time, dummy_mdp.num_items_for_soup, 5)
        return gym.spaces.Box(- np.ones(obs_shape) * np.inf, np.ones(obs_shape) * np.inf, dtype=np.float32)

    def multi_reset(self
                    ) -> Tuple[Observation, Observation]:
        # if self._rewards:
        #     print(max(self._rewards))
        self._rewards = []
        self._actions1 = []
        self._actions2 = []
        self.env.reset()
        state = self.env.state

        state1, state2 = self.env.featurize_state_mdp(state)

        return Observation(state1), Observation(state2)

    def multi_step(self,
                   ego_action: np.ndarray,
                   alt_action: np.ndarray
                   ) -> Tuple[Tuple[Observation, Observation],
                              Tuple[float, float],
                              bool,
                              Dict]:
        self._actions1.append(int(ego_action))
        self._actions2.append(int(alt_action))
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]

        state, _, done, info = self.env.step((ego_action, alt_action))
        state1, state2 = self.env.featurize_state_mdp(state)
        reward = sum(info['sparse_r_by_agent']) + sum(info['shaped_r_by_agent'])
        self._rewards.append(reward)

        return (Observation(state1), Observation(state2)), (reward, reward), done, info

    def render(self, mode="human"):
        ...


class OvercookedPantheonRLSingleAgent(OvercookedPantheonRL):
    def __init__(self, mdp, alt: prl.Agent, pov: Literal['player', 'partner']):
        super(OvercookedPantheonRLSingleAgent, self).__init__(mdp)
        self.alt = alt
        self.pov = pov

        self._stateAlt = None

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None,
              ) -> Tuple[Tuple[Observation, OvercookedState],
                         Dict]:
        if self.pov == 'player':
            state, self._stateAlt = super().multi_reset()
        else:
            self._stateAlt, state = super().multi_reset()
        return (state, self.env.state), {}

    def step(self,
             action
             ) -> Tuple[Tuple[Observation, OvercookedState],
                        Tuple[float, float],
                        bool,
                        bool,
                        Dict
                        ]:
        actionAlt = self.alt.get_action(self._stateAlt)

        if self.pov == 'player':
            (statePov, self._stateAlt), reward, done, _ = super().multi_step(action, actionAlt)
        else:
            (self._stateAlt, statePov), reward, done, _ = super().multi_step(actionAlt, action)
        return (statePov, self.env.state), reward, done, done, {}
