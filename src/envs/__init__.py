import os
import sys
import numpy as np
import itertools as it
from collections import OrderedDict
from .car_racing import CarRacingV1, extract_track_name, extract_tire_name
from .gym import GymEnv, gym

gym_types = ["classic_control", "box2d"]
env_grps = OrderedDict(
	gym_cct = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in [gym_types[0]])]),
	gym_b2d = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in [gym_types[1]])]),
	gym = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in gym_types[:2])]),
	car = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)!=str]),
)

def get_names(groups):
	return list(it.chain(*[env_grps.get(group, []) for group in groups]))

all_envs = get_names(["gym", "car"])

def get_env(env_name, **kwargs):
	if env_name == "CarRacing-v1": return CarRacingV1(**kwargs)
	if env_name.startswith("CarRacing"): 
		if env_name.endswith("v1"): return CarRacingV1(env_name)
	return GymEnv(env_name, **kwargs)
