import torch
import numpy as np
import torchvision as tv
from collections import deque
from src.utils.rand import RandomAgent
from src.utils.misc import resize, rgb2gray, IMG_DIM
from src.envs.car_racing import extract_track_name, extract_pos, rotate_path
from src.envs.CarRacing.ref import RefDriver
from src.agents.pytorch.icm import ICMNetwork
from src.agents.rllib.base import RayEnv

FRAME_STACK = 2					# The number of consecutive image states to combine for training a3c on raw images

class RawState():
	def __init__(self, state_size, load="", gpu=True):
		self.state_size = state_size

	def reset(self):
		pass

	def get_state(self, state):
		return state

class ImgStack(RawState):
	def __init__(self, state_size, stack_len=FRAME_STACK, load="", gpu=True):
		super().__init__(state_size)
		self.transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.Grayscale(), tv.transforms.Resize((IMG_DIM, IMG_DIM)), tv.transforms.ToTensor()])
		self.process = lambda x: self.transform(x.astype(np.uint8)).unsqueeze(0).numpy()
		self.state_size = [*self.process(np.zeros(state_size)).shape[-2:], stack_len]
		self.stack_len = stack_len
		self.reset()

	def reset(self):
		self.stack = deque(maxlen=self.stack_len)

	def get_state(self, state):
		state = np.concatenate([self.process(s) for s in state]) if len(state.shape)>3 else self.process(state)
		while len(self.stack) < self.stack_len: self.stack.append(state)
		self.stack.append(state)
		return np.concatenate(self.stack, axis=1)

class RefStack(RawState):
	def __init__(self, state_size, config, load="", gpu=True):
		super().__init__(state_size)
		self.stack_len = config.REF.SEQ_LEN
		self.ref = RefDriver(extract_track_name(config.env_name))
		refsequence = self.ref.get_sequence(0,1+self.stack_len)
		self.state_size = (refsequence.shape[-2], refsequence.shape[-1])
		self.reset()

	def get_state(self, state, time=None):
		if time is None: time = self.ref.get_time(extract_pos(state))
		refstates = self.ref.get_sequence(time, self.stack_len)
		refstates = rotate_path(state, refstates)
		state = state[...,None,:self.state_size[-1]]
		states = np.concatenate([state, refstates],-2)
		return states

class ParallelAgent(RandomAgent):
	def __init__(self, state_size, action_size, agent_cls, config, load="", gpu=True, **kwargs):
		self.stack = (ImgStack if len(state_size)==3 else RawState)(state_size, load=load, gpu=gpu)
		self.agent = agent_cls(self.stack.state_size, action_size, config, load=load, gpu=gpu)
		self.network = getattr(self.agent, "network", None)
		super().__init__(self.stack.state_size, action_size, config)

	def get_env_action(self, env, state, eps=None, sample=True):
		state = self.stack.get_state(state)
		env_action, action = self.agent.get_env_action(env, state, eps, sample)
		return env_action, action, state

	def train(self, state, action, next_state, reward, done):
		next_state = self.stack.get_state(next_state)
		self.agent.train(state, action, next_state, reward, done)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		if self.network is not None: 
			return self.network.save_model(dirname, name, net)

	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		if self.network is not None: 
			self.network.load_model(dirname, name, net)
		return self

	@property
	def eps(self):
		return self.agent.eps if hasattr(self, "agent") else 0

	@eps.setter
	def eps(self, value):
		if hasattr(self, "agent"): self.agent.eps = value 

	def get_stats(self):
		return {**super().get_stats(), **self.agent.get_stats()}

class RefAgent(ParallelAgent):
	def __init__(self, state_size, action_size, agent_cls, config, load="", gpu=True, **kwargs):
		self.stack = RefStack(state_size, config, load=load, gpu=gpu)
		self.agent = agent_cls(self.stack.state_size, action_size, config, load=load, gpu=gpu)
		self.network = getattr(self.agent, "network", None)
		self.refname = config.ref
		self.times = np.zeros(())
		RandomAgent.__init__(self, state_size, action_size, config)

	def get_env_action(self, env, state, eps=None, sample=True):
		batch = state.shape[:-len(self.state_size)]
		if self.times.shape != batch: self.times = np.zeros(batch)
		states = self.stack.get_state(state)
		env_action, action = self.agent.get_env_action(env, states, eps, sample)
		return env_action, action, states

	def train(self, state, action, next_state, reward, done):
		self.times = np.where(done, self.times*0, self.times+1)
		next_state = self.stack.get_state(next_state)
		self.agent.train(state, action, next_state, reward, done)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		if self.network is not None: 
			return self.network.save_model(dirname, f"{name}_ref{self.refname}", net)

	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		if self.network is not None: 
			self.network.load_model(dirname, f"{name}_ref{self.refname}", net)
		return self

class MetaAgent(ParallelAgent):
	def __init__(self, state_size, action_size, agent_cls, config, load="", gpu=True, **kwargs):
		self.agent = agent_cls(state_size, action_size, config, load=load, gpu=gpu)
		self.network = getattr(self.agent, "network", None)
		self.times = np.zeros(())
		RandomAgent.__init__(self, state_size, action_size, config)

	def get_env_action(self, env, state, eps=None, sample=True):
		env_action, action = self.agent.get_env_action(env, state, eps, sample)
		return env_action, action, state

	def train(self, state, action, next_state, reward, done):
		self.agent.train(state, action, next_state, reward, done)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		if self.network is not None: 
			return self.network.save_model(dirname, f"{name}_ref{self.refname}", net)

	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		if self.network is not None: 
			self.network.load_model(dirname, f"{name}_ref{self.refname}", net)
		return self

class RayAgent(RandomAgent):
	def __init__(self, state_size, action_size, model, config, gpu=True):
		self.agent = model(state_size, action_size, config)
		super().__init__(state_size, action_size, config)

	def get_env_action(self, env, state, eps=None, sample=True):
		env_action, action = self.agent.get_env_action(env, state, eps, sample)
		return env_action, action, state

	def train(self, *wargs):
		return self.agent.train()

	@property
	def eps(self):
		return self.agent.eps

	@eps.setter
	def eps(self, value):
		if hasattr(self, "agent"): self.agent.eps = value 
