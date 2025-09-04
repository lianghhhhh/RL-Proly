from utils import get_config
from mlgame3d.game_runner import GameRunner
from mlgame3d.game_env import GameEnvironment
from mlgame3d.mlplay_loader import create_mlplay_from_file

# Load configuration from JSON file
config = get_config()

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
    game_parameters=[("checkpoint", 10), ("items", 0), ("mud_pit", 0), ("map", 4)],  # Game parameters
    result_output_file=config["result"]  # Save result data to this CSV file
)

behavior_name = env.behavior_names[0]
observation_structure = env.get_observation_structure(behavior_name)
action_space_info = env.get_action_space_info(behavior_name)
mlplay_to_behavior_map = { 0: behavior_name }

if config["mode"] == "train":
    file_path = config["mlplay_path"]
elif config["mode"] == "inference":
    file_path = config["inference_path"]
else:
    raise ValueError("Invalid mode in config.json. Use 'train' or 'inference'.")

mlplay = create_mlplay_from_file(
    file_path=file_path,
    observation_structure=observation_structure,
    action_space_info=action_space_info,
    name="MyCustomMLPlay",
    game_parameters={"checkpoint": 10, "items": 0, "mud_pit": 0, "map": 4}
)


# Create a game runner
runner = GameRunner(
    env=env,
    mlplays=[mlplay],
    max_episodes=1000,
    render=True,
    mlplay_timeout=0.1,  # Timeout for MLPlay actions in seconds
    game_parameters={"checkpoint": 10, "items": 0, "mud_pit": 0, "map": 4},
    mlplay_to_behavior_map=mlplay_to_behavior_map
)

# Run the game
runner.run()

# Close the environment
env.close()