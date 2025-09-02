import numpy as np
from gymnasium import spaces

class ActionProcessor:
    def __init__(self, action_space_info):
        self.action_space_info = action_space_info

        if action_space_info.is_continuous():
            self.action_type = "continuous"
            self.action_size = action_space_info.continuous_size
        elif action_space_info.is_discrete():
            self.action_type = "discrete"
            self.action_size = sum(action_space_info.discrete_branches)
            self.discrete_branches = action_space_info.discrete_branches
        else:
            self.action_type = "hybrid"
            self.continuous_size = action_space_info.continuous_size
            self.discrete_branches = action_space_info.discrete_branches
            self.discrete_size = sum(action_space_info.discrete_branches)
            self.action_size = self.continuous_size + self.discrete_size

        print(f"Action space detected: {self.action_type}")
        if self.action_type == "continuous":
            print(f"  Continuous size: {self.action_size}")
        elif self.action_type == "discrete":
            print(f"  Discrete branches: {self.discrete_branches}")
        else:
            print(f"  Continuous size: {self.continuous_size}")
            print(f"  Discrete branches: {self.discrete_branches}")
            print(f"  Unified Box space size: {self.action_size}")

    def create_action(self, network_output):
        if self.action_type == "continuous":
            return network_output
        elif self.action_type == "discrete":
            return self._process_discrete_action(network_output)
        else:
            return self._process_hybrid_action(network_output)

    def action_to_network_output(self, action):
        if self.action_type == "continuous":
            return action
        elif self.action_type == "discrete":
            return self._process_discrete_to_network_output(action)
        else:
            return self._process_hybrid_to_network_output(action)

    def get_size(self):
        return self.action_size

    def get_gym_action_space(self):
        if self.action_type == "continuous":
            return spaces.Box(low=-1.0, high=1.0, shape=(self.action_size,), dtype=np.float32)
        elif self.action_type == "discrete":
            if len(self.discrete_branches) == 1:
                return spaces.Discrete(self.discrete_branches[0])
            else:
                return spaces.MultiDiscrete(self.discrete_branches)
        else:
            return spaces.Box(low=-1.0, high=1.0, shape=(self.action_size,), dtype=np.float32)

    def _process_discrete_action(self, network_output):
        if isinstance(network_output, np.ndarray):
            if len(self.discrete_branches) == 1:
                return np.array([network_output], dtype=np.int32)
            else:
                return network_output.astype(np.int32)
        else:
            return np.array([network_output], dtype=np.int32)

    def _process_hybrid_action(self, network_output):
        continuous_part = network_output[:self.continuous_size]
        discrete_part = network_output[self.continuous_size:]

        continuous_action = continuous_part
        discrete_action = self._continuous_to_discrete(discrete_part)

        return (continuous_action, discrete_action)

    def _continuous_to_discrete(self, continuous_values):
        discrete_actions = []
        value_idx = 0

        for branch_size in self.discrete_branches:
                discrete_action = 0
                max_continuous_val = float("-inf")
                for i in range(branch_size):
                        if value_idx + i < len(continuous_values):
                                continuous_val = continuous_values[value_idx + i]
                                if continuous_val > max_continuous_val:
                                        discrete_action = i
                                        max_continuous_val = continuous_val
                discrete_actions.append(discrete_action)
                value_idx += branch_size

        return np.array(discrete_actions, dtype=np.int32)

    def _process_discrete_to_network_output(self, action):
        if isinstance(action, np.ndarray) and len(self.discrete_branches) == 1 and len(action) == 1:
            return action[0]
        return action

    def _process_hybrid_to_network_output(self, action):
        continuous_action, discrete_action = action
        discrete_continuous = self._discrete_to_continuous(discrete_action)
        return np.concatenate([continuous_action, discrete_continuous])

    def _discrete_to_continuous(self, discrete_values):
        continuous_actions = []
        value_idx = 0

        for branch_size in self.discrete_branches:
            if value_idx < len(discrete_values):
                discrete_val = discrete_values[value_idx]
                for i in range(branch_size):
                    if i == discrete_val:
                        continuous_actions.append(1.0)
                    else:
                        continuous_actions.append(-1.0)
            value_idx += 1

        return np.array(continuous_actions, dtype=np.float32)