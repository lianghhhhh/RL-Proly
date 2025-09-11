import os
import torch
from envWrapper import EnvWrapper
from stable_baselines3 import PPO

class MLPlay():
    def __init__(self, observation_structure, action_space_info, model_path):
        model_path = model_path
        if(os.path.exists(model_path) == False):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path in config.json.")
        self.model = PPO.load(model_path)
        self.env_wrapper = EnvWrapper(observation_structure, action_space_info)

    def reset(self):
        pass

    def update(self, observations, done, info, keyboard=set(), *args, **kwargs):
        observation = observations["flattened"]
        action, log_prob, value = self._predict_with_info(observation)
        return self.env_wrapper.action_processor.create_action(action)
    
    def _predict_with_info(self, obs):
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, value, log_prob = self.model.policy(obs_tensor)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value.cpu().numpy().flatten()
