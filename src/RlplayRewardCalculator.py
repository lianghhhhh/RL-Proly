import numpy as np

class RlplayRewardCalculator:
    def __init__(self):
        self.prev_observation = None
        self.observation = None

    def update(self, observation):
        self.prev_observation = self.observation
        self.observation = observation

    def reset(self):
        self.prev_observation = None
        self.observation = None

    def calculate_checkpoint_reward(self, weight):
        if self.prev_observation is None or self.observation is None:
            return 0.0
        if self.prev_observation["last_checkpoint_index"] < self.observation["last_checkpoint_index"]:
            return weight * (self.observation["last_checkpoint_index"] - self.prev_observation["last_checkpoint_index"])
        return 0.0

    def calculate_distance_reward(self, close_weight, leave_weight):
        if self.prev_observation is None or self.observation is None:
            return 0.0
        if self.prev_observation["last_checkpoint_index"] != self.observation["last_checkpoint_index"]:
            return 0.0
        prev_distance = np.linalg.norm(np.array([
            self.prev_observation["target_position"][0],
            self.prev_observation["target_position"][1]
        ]))
        current_distance = np.linalg.norm(np.array([
            self.observation["target_position"][0],
            self.observation["target_position"][1]
        ]))
        if current_distance <= prev_distance:
            return close_weight * (prev_distance - current_distance)
        else:
            return leave_weight * (current_distance - prev_distance)

    def calculate_health_reward(self, death_weight, increase_weight, decrease_weight):
        if self.prev_observation is None or self.observation is None:
            return 0.0
        if self.prev_observation["agent_health"] <= 0.0:
            return 0.0
        if self.observation["agent_health"] <= 0.0:
            return death_weight
        if self.observation["agent_health"] >= self.prev_observation["agent_health"]:
            return increase_weight * (self.observation["agent_health"] - self.prev_observation["agent_health"])
        else:
            return decrease_weight * (self.prev_observation["agent_health"] - self.observation["agent_health"])

    def calculate_mud_reward(self, threshold, leave_weight, close_weight):
        if self.prev_observation is None or self.observation is None:
            return 0.0
        prev_nearby_obects = self.prev_observation["nearby_map_objects"]
        nearby_obects = self.observation["nearby_map_objects"]
        prev_muds = [obj for obj in prev_nearby_obects if obj["object_type"] == 1]
        muds = [obj for obj in nearby_obects if obj["object_type"] == 1]
        if not prev_muds or not muds:
            return 0.0
        prev_nearest_mud = min(prev_muds, key=lambda x: np.linalg.norm(np.array(x["relative_position"])))
        nearest_mud = min(muds, key=lambda x: np.linalg.norm(np.array(x["relative_position"])))
        prev_distance = np.linalg.norm(np.array(prev_nearest_mud["relative_position"]))
        distance = np.linalg.norm(np.array(nearest_mud["relative_position"]))
        if prev_distance > threshold or distance > threshold:
            return 0.0
        if distance >= prev_distance:
            return leave_weight * (distance - prev_distance)
        else:
            return close_weight * (prev_distance - distance)

    def calculate_time_reward(self, time_penalty):
        return time_penalty
    