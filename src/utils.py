import os
import json


def get_config():
    curr_path = os.path.dirname(__file__)
    parent_path = os.path.dirname(curr_path)
    config_path = os.path.join(parent_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config