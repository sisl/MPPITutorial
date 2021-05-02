import os
import numpy as np
import matplotlib.pyplot as plt

try: from .Gym import gym
except ImportError as e: print(e)

class GymEnv(gym.Wrapper):
	def __init__(self, env_name, **kwargs):
		super().__init__(gym.make(env_name, **kwargs))
		self.unwrapped.verbose = 0

	def reset(self, **kwargs):
		self.time = 0
		state = self.env.reset(**kwargs)
		return state

	def step(self, action, train=False, **kwargs):
		self.time += 1
		state, reward, done, info = super().step(action)
		return state, reward, done, info

	def set_state(self, state, device=None):
		self.reset()
		self.env.env.set_state(state)