import os
import json


def get_config():
    curr_path = os.path.dirname(__file__)
    parent_path = os.path.dirname(curr_path)
    config_path = os.path.join(parent_path, "racing_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def dict_to_tuple_list(d):
    return [(k, v) for k, v in d.items()]