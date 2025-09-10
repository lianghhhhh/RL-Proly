import numpy as np
import gymnasium as gym
from gymnasium import spaces
from actionProcessor import ActionProcessor
from observationProcessor import ObservationProcessor


class EnvWrapper(gym.Env):
    def __init__(self, observation_structure, action_space_info):
        super().__init__()
        self.observation_processor = ObservationProcessor(observation_structure)
        self.action_processor = ActionProcessor(action_space_info)

        obs_size = self.observation_processor.get_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = self.action_processor.get_gym_action_space()
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return dummy_obs, {}

    def step(self, action):
        self.step_count += 1
        dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return dummy_obs, reward, terminated, truncated, info