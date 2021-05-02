import torch
import pickle
import numpy as np
from src.utils.multiprocess import get_client, get_server
from src.envs.Gym.gym.spaces import MultiDiscrete, Discrete, Box, Dict

NUM_ENVS = 16					# The default number of environments to simultaneously train the a3c in parallel

def get_space_size(space):
	if isinstance(space, MultiDiscrete): return [*space.shape, space.nvec[0]]
	if isinstance(space, Discrete): return [space.n]
	if isinstance(space, Box): return space.shape
	if isinstance(space, list): return [get_space_size(sp) for sp in space]
	if isinstance(space, Dict): return (np.sum([get_space_size(sp)[-1] for sp in space.spaces.values()]),)
	raise ValueError()

class EnsembleEnv():
	def __init__(self, make_env, num_envs=NUM_ENVS, max_steps=None):
		self.num_envs = len(num_envs) if type(num_envs)==list else num_envs
		self.env = make_env()
		self.envs = [make_env() for _ in range(max(self.num_envs, 0))]
		self.test_envs = [make_env() for _ in range(max(self.num_envs, 0))]
		self.state_size = get_space_size(self.env.observation_space)
		self.action_size = get_space_size(self.env.action_space)
		self.action_space = self.env.action_space
		self.max_steps = self.env.spec.max_episode_steps if self.env.spec.max_episode_steps else max_steps

	def reset(self, train=False, **kwargs):
		envs = [self.env] if self.num_envs<=0 else self.envs if train else self.test_envs
		obs = [env.reset(train=train, **kwargs) for env in envs]
		return np.stack(obs)

	def step(self, actions, train=False, render=False):
		results = []
		envs = [self.env] if self.num_envs<=0 else self.envs if train else self.test_envs
		for env,action in zip(envs, actions):
			state, rew, done, info = env.step(action)
			state = env.reset(train=train) if train and done else state
			results.append((state, rew, done, info))
			if render: env.render()
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos

	def render(self, train=False, **kwargs):
		envs = [self.env] if self.num_envs<=0 else self.envs if train else self.test_envs
		envs[0].render(**kwargs)

	def close(self, **kwargs):
		self.env.close(**kwargs)
		for env in self.envs: env.close()
		for env in self.test_envs: env.close()

	def __del__(self):
		self.close()

class EnvWorker():
	def __init__(self, make_env, root=0):
		self.env = [make_env(), make_env()]
		self.conn = get_server(root)

	def start(self):
		step = 0
		rewards = [None, None]
		while True:
			data = self.conn.recv()
			train = data.get("train", False)
			env = self.env[int(train)]
			if data["cmd"] == "RESET":
				message = env.reset(train=train)
				rewards[int(train)] = None
			elif data["cmd"] == "STEP":
				state, reward, done, info = env.step(data["action"])
				state = env.reset(train=train) if train and done else state
				rewards[int(train)] = np.array(reward) if rewards[int(train)] is None else rewards[int(train)] + np.array(reward)
				message = (state, reward, done, info)
				step += int(train)
				if np.all(done): 
					print(f"{'Train' if train else 'Test'} Step: {step}, Reward: {rewards[int(train)]}")
					rewards[int(train)] = None
			elif data["cmd"] == "RENDER":
				env.render()
				continue
			elif data["cmd"] == "CLOSE":
				[env.close() for env in self.env]
				return
			self.conn.send(message)

class EnvManager():
	def __init__(self, make_env, server_ports, max_steps=None):
		self.env = make_env()
		self.state_size = get_space_size(self.env.observation_space)
		self.action_size = get_space_size(self.env.action_space)
		self.action_space = self.env.action_space
		self.server_ports = sorted(server_ports)
		self.conn = get_client(server_ports)
		self.num_envs = len(server_ports)
		self.max_steps = self.env.spec.max_episode_steps if self.env.spec.max_episode_steps else max_steps

	def reset(self, train=False, **kwargs):
		self.conn.broadcast([{"cmd": "RESET", "action": [0.0], "train": train} for _ in self.server_ports])
		obs = self.conn.gather()
		return np.stack(obs)

	def step(self, actions, train=False, render=False):
		self.conn.broadcast([{"cmd": "STEP", "action": action, "render": render, "train": train} for action in actions])
		results = self.conn.gather()
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos

	def render(self, num=1, train=False):
		self.conn.broadcast([{"cmd": "RENDER", "train": train} for _ in self.server_ports[:num]])

	def close(self):
		self.env.close()
		if hasattr(self, "conn"): self.conn.broadcast([{"cmd": "CLOSE", "action": [0.0]} for _ in self.server_ports])

	def __del__(self):
		self.close()
