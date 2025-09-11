from mlgame3d.game_runner import GameRunner
from mlgame3d.game_env import GameEnvironment
from utils import get_config, dict_to_tuple_list
from mlgame3d.mlplay_loader import create_mlplay_from_file

def create_env(game_path, game_parameters: dict):
    # Create the environment with controlled players
    env = GameEnvironment(
        file_name=game_path,  # Or None to connect to a running Unity editor
        worker_id=0,
        no_graphics=False,
        fps=60,  # Target frames per second for rendering
        time_scale=1.0,  # Time scale factor for simulation speed
        decision_period=5,  # Number of FixedUpdate steps between AI decisions
        controlled_players=[0],  # Control P1
        control_modes=["mlplay"],  # Control P1 with MLPlay
        game_parameters=dict_to_tuple_list(game_parameters),  # Game parameters
    )
    return env

def get_mlplay(mode, observation_structure, action_space_info, config):
    if mode == "train":
        file_path = config["mlplay_path"]
    elif mode == "inference":
        file_path = config["inference_path"]

    mlplay = create_mlplay_from_file(
        file_path=file_path,
        observation_structure=observation_structure,
        action_space_info=action_space_info,
        game_parameters=config["game_parameters"]
    )
    return mlplay

def create_runner(mlplay, env, mlplay_to_behavior_map, game_parameters):
    # Create a game runner
    runner = GameRunner(
        env=env,
        mlplays=[mlplay],
        max_episodes=3000,
        render=True,
        mlplay_timeout=0.1,  # Timeout for MLPlay actions in seconds
        game_parameters=game_parameters,
        mlplay_to_behavior_map=mlplay_to_behavior_map
    )
    return runner

def choose_mode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    choice = input("Enter choice (1 or 2): ")
    if choice == '1':
        return "train"
    elif choice == '2':
        return "inference"
    else:
        raise KeyError("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    config = get_config()
    mode = choose_mode()
    env = create_env(config["game_path"], config["game_parameters"])
    behavior_name = env.behavior_names[0]
    observation_structure = env.get_observation_structure(behavior_name)
    action_space_info = env.get_action_space_info(behavior_name)
    mlplay_to_behavior_map = {0: behavior_name}
    mlplay = get_mlplay(mode, observation_structure, action_space_info, config)

    runner = create_runner(mlplay, env, mlplay_to_behavior_map, config["game_parameters"])
    runner.run() # Run the game
    env.close() # Close the environment
