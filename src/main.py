import json
from mlgame3d.game_runner import GameRunner
from mlgame3d.game_env import GameEnvironment
from mlgame3d.mlplay_loader import create_mlplay_from_file

# Load configuration from JSON file
with open("C:\\Users\\selen\\OneDrive\\Desktop\\RL-Proly\\config.json", "r") as f:
    config = json.load(f)

# Create the environment with controlled players
env = GameEnvironment(
    file_name=config["game_path"],  # Or None to connect to a running Unity editor
    worker_id=0,
    no_graphics=False,
    fps=60,  # Target frames per second for rendering
    time_scale=1.0,  # Time scale factor for simulation speed
    decision_period=5,  # Number of FixedUpdate steps between AI decisions
    controlled_players=[0],  # Control P1
    control_modes=["mlplay"],  # Control P1 with MLPlay
    game_parameters=[("checkpoint", 10), ("items", 0), ("mud_pit", 0)],  # Game parameters
    result_output_file="results.csv"  # Save result data to this CSV file
)

behavior_name = env.behavior_names[0]
observation_structure = env.get_observation_structure(behavior_name)
action_space_info = env.get_action_space_info(behavior_name)
mlplay_to_behavior_map = { 0: behavior_name }


mlplay = create_mlplay_from_file(
    file_path=config["mlplay_path"],
    observation_structure=observation_structure,
    action_space_info=action_space_info,
    name="MyCustomMLPlay",
    game_parameters={"checkpoint": 10, "items": 0, "mud_pit": 0}
)


# Create a game runner
runner = GameRunner(
    env=env,
    mlplays=[mlplay],
    max_episodes=5,
    render=True,
    mlplay_timeout=0.1,  # Timeout for MLPlay actions in seconds
    game_parameters={"checkpoint": 10, "items": 0, "mud_pit": 0},
    mlplay_to_behavior_map=mlplay_to_behavior_map
)

# Run the game
runner.run()

# Close the environment
env.close()