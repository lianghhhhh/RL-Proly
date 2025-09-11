from mlPlay import MLPlay
from mlgame3d.game_runner import GameRunner
from mlgame3d.game_env import GameEnvironment
from utils import get_config, dict_to_tuple_list

def create_env(game_path, game_parameters: dict, count: int):
    # Create the environment with controlled players
    controlled_players = list(range(count))
    control_modes = ["mlplay"] * count
    env = GameEnvironment(
        file_name=game_path,  # Or None to connect to a running Unity editor
        worker_id=0,
        no_graphics=False,
        fps=60,  # Target frames per second for rendering
        time_scale=1.0,  # Time scale factor for simulation speed
        decision_period=5,  # Number of FixedUpdate steps between AI decisions
        controlled_players=controlled_players,  # Control P1 to P{count}
        control_modes=control_modes,  # Control P1 to P{count} with MLPlay
        game_parameters=dict_to_tuple_list(game_parameters),  # Game parameters
    )
    return env

def get_mlplay(index, observation_structure, action_space_info, config):
    # Create an MLPlay instance for the given index
    mlplay = MLPlay(
        observation_structure=observation_structure,
        action_space_info=action_space_info,
        model_path=config[f"model_{index}"]
    )

    return mlplay

def create_runner(mlplays: list, env, mlplay_to_behavior_map, game_parameters):
    # Create a game runner
    runner = GameRunner(
        env=env,
        mlplays=mlplays,
        max_episodes=3000,
        render=True,
        mlplay_timeout=0.1,  # Timeout for MLPlay actions in seconds
        game_parameters=game_parameters,
        mlplay_to_behavior_map=mlplay_to_behavior_map
    )
    return runner


def choose_runner():
    count = input("Choose how many runners(1-4): ")
    if count in ['1', '2', '3', '4']:
        return int(count)
    else:
        raise KeyError("Invalid choice. Please enter number 1-4.")

if __name__ == "__main__":
    config = get_config()
    count = choose_runner()
    env = create_env(config["game_path"], config["game_parameters"], count)
    mlplay_to_behavior_map = {}
    mlplays = []
    for i in range(count):
        behavior_name = env.behavior_names[i]
        observation_structure = env.get_observation_structure(behavior_name)
        action_space_info = env.get_action_space_info(behavior_name)
        mlplay_to_behavior_map[i] = behavior_name
        mlplay = get_mlplay(i + 1, observation_structure, action_space_info, config)
        mlplays.append(mlplay)

    runner = create_runner(mlplays, env, mlplay_to_behavior_map, config["game_parameters"])
    runner.run() # Run the game
    env.close() # Close the environment
