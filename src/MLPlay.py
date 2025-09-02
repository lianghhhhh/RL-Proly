import os
import time
import torch
import numpy as np
from RLPlay import RLPlay
from envWrapper import EnvWrapper
from stable_baselines3 import PPO
from MLPlayArgsSaver import MLPlayArgsSaver
from stable_baselines3.common.utils import safe_mean
from RlplayRewardCalculator import RlplayRewardCalculator

class MLPlay:
    def __init__(self, observation_structure, action_space_info, name, *args, **kwargs):
        self.mlplayArgs = MLPlayArgsSaver()
        self.mlplayArgs.name = name
        self.mlplayArgs.init_kwargs = kwargs
        self.rlplayRewardCalculator = RlplayRewardCalculator()
        self.rlplayRewardCalculator.reset()
        self.RLPlay = RLPlay()

        self.env_wrapper = EnvWrapper(observation_structure, action_space_info)
        self.config = {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "clip_range": 0.2,
            "gamma": 0.99,
            "ent_coef": 0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "tensorboard_log": os.path.join(os.path.dirname(__file__), "tensorboard"),
            "policy_kwargs": {
                "net_arch": [64, 64],
                "activation_fn": torch.nn.Tanh
            }
        }
        self.prev_observation = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.episode_rewards = []
        self.total_steps = 0
        self.episode_count = 1
        self.update_count = 0
        self.start_time = time.strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = os.path.join(os.path.dirname(__file__), "models", self.start_time)
        self.model_path = os.path.join(os.path.dirname(__file__), 'model' + ".zip")

        os.makedirs(self.model_save_dir, exist_ok=True)

        self._initialize_model()
        print(f"PPO initialized in training mode")

    def reset(self):
        if self.episode_rewards:
                total_reward = sum(self.episode_rewards)
                print(f"Episode {self.episode_count}: Total Reward = {total_reward:.2f}, Steps = {len(self.episode_rewards)}")
                self.episode_rewards = []

        self._update_policy()

        self.prev_observation = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.episode_count += 1

        self.rlplayRewardCalculator.reset()
        self.RLPlay.reset()

    def update(self, observations, done, info, keyboard=set(), *args, **kwargs):
        self.mlplayArgs.observations = observations
        self.mlplayArgs.keyboard = keyboard
        self.rlplayRewardCalculator.update(observations)
        observation = observations["flattened"]

        reward, not_used_for_training = self.RLPlay.update()
        action, log_prob, value = self._predict_with_info(observation)

        if self.prev_observation is not None:
            self.episode_rewards.append(reward)

            if not not_used_for_training and not self.model.rollout_buffer.full:
                self._add_to_rollout_buffer(
                    obs=self.prev_observation,
                    action=self.prev_action,
                    reward=reward,
                    done=done,
                    value=self.prev_value,
                    log_prob=self.prev_log_prob
                )
                if self.model.rollout_buffer.full:
                    done_tensor = np.array([done])
                    value_tensor = torch.as_tensor(value).unsqueeze(0) if value.ndim == 0 else torch.as_tensor(value)
                    self.model.rollout_buffer.compute_returns_and_advantage(last_values=value_tensor, dones=done_tensor)

        self.prev_observation = observation
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.prev_value = value
        self.total_steps += 1

        return self.env_wrapper.action_processor.create_action(action)

    def _initialize_model(self):
        print(f"Initializing PPO model...")
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path, env=self.env_wrapper, **self.config, verbose=1)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                print("Creating new model...")
                self.model = PPO("MlpPolicy", env=self.env_wrapper, **self.config, verbose=1)
        else:
            print(f"No pre-trained model found at {self.model_path}. Creating new model...")
            self.model = PPO("MlpPolicy", env=self.env_wrapper, **self.config, verbose=1)
        self.model.learn(total_timesteps=0, tb_log_name=f"PPO_{self.start_time}")

    def _save_model(self):
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")

            update_path = f"{self.model_save_dir}/ppo_model_{self.update_count}.zip"
            self.model.save(update_path)
            print(f"Model saved to {update_path}")

    def _predict_with_info(self, obs):
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, value, log_prob = self.model.policy(obs_tensor)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value.cpu().numpy().flatten()

    def _add_to_rollout_buffer(self, obs, action, reward, done, value, log_prob):
        if not self.model.rollout_buffer.full:
            self.model.rollout_buffer.add(
                obs=torch.as_tensor(obs).unsqueeze(0),
                action=torch.as_tensor(action).unsqueeze(0),
                reward=torch.as_tensor([reward]),
                episode_start=torch.as_tensor([done]),
                value=torch.as_tensor(value).unsqueeze(0) if value.ndim == 0 else torch.as_tensor(value),
                log_prob=torch.as_tensor(log_prob).unsqueeze(0) if log_prob.ndim == 0 else torch.as_tensor(log_prob)
            )

    def _update_policy(self):
        if self.model.rollout_buffer.size() == 0 or not self.model.rollout_buffer.full:
            return

        print(f"Updating PPO policy with {self.model.rollout_buffer.size()} experiences...")

        self.model.num_timesteps += self.model.rollout_buffer.size()
        self.model.train()
        self.update_count += 1

        self.model.logger.record("train/mean_reward", safe_mean(self.model.rollout_buffer.rewards))
        self.model.logger.record("param/n_steps", self.model.n_steps)
        self.model.logger.record("param/batch_size", self.model.batch_size)
        self.model.logger.record("param/n_epochs", self.model.n_epochs)
        self.model.logger.record("param/gamma", self.model.gamma)
        self.model.logger.record("param/gae_lambda", self.model.gae_lambda)
        self.model.logger.record("param/ent_coef", self.model.ent_coef)
        self.model.logger.record("param/vf_coef", self.model.vf_coef)
        self.model.logger.record("param/max_grad_norm", self.model.max_grad_norm)
        self.model._dump_logs(self.update_count)

        self.model.rollout_buffer.reset()
        print("PPO policy updated successfully")

        self._save_model()
