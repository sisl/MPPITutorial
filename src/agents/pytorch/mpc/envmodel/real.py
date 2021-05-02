import torch
import numpy as np
from src.envs import get_env
from src.agents.pytorch.rl.base import one_hot_from_logits
from src.utils.rand import RandomAgent

class RealEnv():
	def __init__(self, state_size, action_size, config, gpu=True, load="", name="real"):
		self.env = get_env(config.get("DYN_ENV_NAME", config.env_name))
		self.discrete = type(action_size) != tuple
		self.state_size = state_size
		self.action_size = action_size
		self.dyn_index = config.get("dynamics_size", state_size[0]) 

	def value(self, action, state, next_state):
		return np.zeros([*action.shape[:-1], 1]) 

	def rollout(self, actions, state, train=False, times=None):
		if self.discrete: actions = np.argmax(actions, -1) 
		action_size = () if self.discrete else self.action_size
		batch = state.shape[:-len(self.state_size)]
		state = np.reshape(state.detach().cpu().numpy(), [np.prod(batch), *self.state_size])
		actions = np.split(actions, actions.shape[0], 0)
		actions = [np.reshape(action, [np.prod(batch), *action_size]) for action in actions]
		states = [np.zeros_like(state) for action in actions]
		next_states = [np.zeros_like(state) for action in actions]
		rewards = [np.zeros([np.prod(batch)]) for action in actions]
		for i in range(state.shape[0]):
			s = state[i]
			d = False
			self.env.set_state(s)
			for t in range(len(actions)):
				states[t][i] = s
				a = actions[t][i]
				s, r, d, _ = self.env.step(a)
				next_states[t][i] = s
				rewards[t][i] = r
				if d: break 
		actions = np.stack([np.reshape(action, [*batch, *action_size]) for action in actions])
		states = np.stack([np.reshape(state, [*batch, *self.state_size]) for state in states])
		next_states = np.stack([np.reshape(state, [*batch, *self.state_size]) for state in next_states])
		rewards = np.reshape(np.stack(rewards), [len(rewards), *batch, 1])
		return actions, states, next_states, rewards 

	def optimize(self, states, actions, next_states, rewards, dones, mask=None):
		return 0,0 

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		return ""
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		pass

	def get_stats(self):
		return {}

class BatchRealEnv(RealEnv):
	def __init__(self, state_size, action_size, config, gpu=True, load="", name="batchreal"):
		super().__init__(state_size, action_size, config, gpu=gpu, load=load, name=name)
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

	def rollout(self, actions, state, train=False, times=None):
		if self.discrete: actions = np.argmax(actions, -1) 
		states, next_states, rewards = [], [], []
		self.env.set_state(state, device=self.device)
		for action in actions:
			states.append(state)
			state, reward, done, _ = self.env.step(action, device=self.device, info=False)
			next_states.append(state)
			rewards.append(reward[...,None])
		states = np.stack(states)
		next_states = np.stack(next_states)
		rewards = np.stack(rewards)
		return actions, states, next_states, rewards