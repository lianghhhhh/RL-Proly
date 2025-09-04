class RLPlay:
    def __init__(self, reward_calculator):
        self.reward_calculator = reward_calculator
        self.step_count = 0

    def update(self):
        self.step_count += 1
        reward = 0.0
        reward += self.reward_calculator.calculate_checkpoint_reward(weight=80.0)
        reward += self.reward_calculator.calculate_distance_reward(close_weight=1.0, leave_weight=-1.0)
        reward += self.reward_calculator.calculate_health_reward(death_weight=-100.0, increase_weight=0, decrease_weight=0)
        # reward += self.reward_calculator.calculate_mud_reward(threshold=2.0, leave_weight=30.0, close_weight=-30.0)
        # reward += self.reward_calculator.calculate_time_reward(time_penalty=-5)
        not_used_for_training = (self.step_count < 10)
        return reward, not_used_for_training

    def reset(self):
        self.step_count = 0