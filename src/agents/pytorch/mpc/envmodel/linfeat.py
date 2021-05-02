import os
import torch
import numpy as np
from src.utils.misc import load_module
from src.agents.pytorch.rl.base import PTNetwork, one_hot_from_logits

class TransitionModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.config = config
		self.reg_coeff = 1e-5
		self.somask = torch.nn.Parameter(torch.tensor(self.config.get("dynamics_somask", 1)), requires_grad=False)
		self.feature_size = self.features(torch.zeros(action_size), torch.zeros(state_size)).shape[0]
		self.weight = torch.nn.Parameter(torch.zeros((self.feature_size, state_size[0])), requires_grad=False)
		self.eye = torch.nn.Parameter(torch.eye(self.feature_size, dtype=torch.float32), requires_grad=False)

	def features(self, action, state):
		ones = torch.ones([*action.shape[:-1], 1]).to(action.device)
		features = torch.cat([ones, action, state*self.somask, (state*self.somask)**2], -1)
		return features

	def forward(self, action, state):
		features = self.features(action, state)
		features = features.view(-1, features.shape[-1])
		state_dot = torch.matmul(features, self.weight)
		state_dot = state_dot.view(*state.shape[:-1], -1)
		return state + state_dot

	def fit(self, action, state, next_state, mask):
		features = self.features(action, state)
		features = features.reshape(-1, features.shape[-1])
		state_dot = next_state - state
		state_dot = state_dot.reshape(-1, state_dot.shape[-1])
		mask = torch.nonzero(mask.flatten()).flatten()
		features = features[mask]
		state_dot = state_dot[mask]
		XT_y = torch.matmul(features.t(), state_dot)
		XT_X = torch.matmul(features.t(), features)
		reg_coeff = self.reg_coeff
		for _ in range(5):
			coeffs, _ = torch.lstsq(XT_y, XT_X + reg_coeff*self.eye)
			if not (torch.isnan(coeffs).any() or torch.isinf(coeffs).any()): break
			reg_coeff *= 10
		else:
			raise RuntimeError("Unable to solve the normal equations")
		self.weight.data.copy_(coeffs)
		return torch.norm(self.forward(action, state) - next_state)

class RewardModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.cost = load_module(config.REWARD_MODEL)(config=config) if config.get("REWARD_MODEL") else None
		self.dyn_spec = load_module(config.DYNAMICS_SPEC) if config.get("DYNAMICS_SPEC") else None
		self.reg_coeff = 1e-5
		self.feature_size = self.features(torch.zeros(action_size), torch.zeros(state_size), torch.zeros(state_size)).shape[0]
		self.weight = torch.nn.Parameter(torch.zeros((self.feature_size, 1)), requires_grad=False)
		self.eye = torch.nn.Parameter(torch.eye(self.feature_size, dtype=torch.float32), requires_grad=False)

	def features(self, action, state, next_state):
		ones = torch.ones([*action.shape[:-1], 1]).to(action.device)
		features = torch.cat([ones, action, state, next_state], -1)
		return features

	def forward(self, action, state, next_state, grad=False, model=False, times=None):
		if self.cost and self.dyn_spec and not grad and not model:
			next_state, state = [x.cpu().numpy() for x in [next_state, state]]
			ns_spec, s_spec = map(self.dyn_spec.observation_spec, [next_state, state])
			reward = -torch.FloatTensor(self.cost.get_cost(ns_spec, s_spec, times)).unsqueeze(-1)
		else:
			features = self.features(action, state, next_state)
			reward = torch.matmul(features.view(-1, features.shape[-1]), self.weight)
			reward = reward.view(*state.shape[:-1], -1)
		return reward

	def fit(self, action, state, next_state, reward, mask):
		features = self.features(action, state, next_state)
		features = features.view(-1, features.shape[-1])
		reward = reward.reshape(-1, 1)
		mask = torch.nonzero(mask.flatten()).flatten()
		features = features[mask]
		reward = reward[mask]
		XT_y = torch.matmul(features.t(), reward)
		XT_X = torch.matmul(features.t(), features)
		reg_coeff = self.reg_coeff
		for _ in range(5):
			coeffs, _ = torch.lstsq(XT_y, XT_X + reg_coeff*self.eye)
			if not (torch.isnan(coeffs).any() or torch.isinf(coeffs).any()): break
			reg_coeff *= 10
		else:
			raise RuntimeError("Unable to solve the normal equations")
		self.weight.copy_(coeffs)
		return torch.norm(torch.matmul(features, self.weight) - reward)

class LinearFeatureEnv(PTNetwork):
	def __init__(self, state_size, action_size, config, load="", gpu=True, name="linfeat"):
		super().__init__(config, gpu, name)
		self.state_size = state_size
		self.action_size = action_size
		self.discrete = type(self.action_size) != tuple
		self.dyn_index = config.get("dynamics_size", state_size[-1])
		self.reward = RewardModel([self.dyn_index], action_size, config)
		self.dynamics = TransitionModel([self.dyn_index], action_size, config)
		self.to(self.device)
		if load: self.load_model(load)

	def value(self, action, state, next_state, train=False, times=None):
		state, next_state = [x[...,:self.dyn_index] for x in [state, next_state]]
		reward = self.reward(action, state, next_state, grad=train, model=False, times=times)
		return reward

	def rollout(self, actions, state, timedim=0, train=False, times=None):
		state = self.to_tensor(state[...,:self.dyn_index])
		buffer = []
		for action in self.to_tensor(actions):
			if self.discrete: action = one_hot_from_logits(action)
			next_state = self.dynamics(action, state)
			buffer.append((action, state, next_state))
			state = next_state
		actions, states, next_states = map(torch.stack, zip(*buffer))
		time_buffer = np.stack([t+times for t in range(actions.shape[0])]) if times is not None else None
		rewards = self.value(actions, states, next_states, train=train, times=time_buffer)
		if train==False: actions, states, next_states, rewards = map(lambda x: x.cpu().numpy(), [actions, states, next_states, rewards])
		return actions, states, next_states, rewards

	def get_loss(self, states, actions, next_states, rewards, dones, mask=1, train=True):
		s, a, ns, r, mask = map(self.to_tensor, [states, actions, next_states, rewards, mask])
		s, ns = [x[...,:self.dyn_index] for x in [s, ns]]
		rewards_hat = self.reward(a, s, ns)
		rew_loss = ((rewards_hat - r.unsqueeze(-1)).pow(2) * mask).mean()
		next_states_hat = self.dynamics(a, s)
		dyn_loss = ((next_states_hat - ns).pow(2) * mask).sum(-1).mean()
		return rew_loss, dyn_loss

	def optimize(self, states, actions, next_states, rewards, dones, mask=1):
		s, a, ns, r, mask = map(self.to_tensor, [states, actions, next_states, rewards, mask])
		s, ns = [x[...,:self.dyn_index] for x in [s, ns]]
		dyn_loss = self.dynamics.fit(a, s, ns, mask)
		rew_loss = self.reward.fit(a, s, ns, r, mask)
		self.stats.mean(dyn_loss=dyn_loss, rew_loss=rew_loss)
		return [loss.item() for loss in [rew_loss, dyn_loss]]

	def schedule(self, test_loss):
		pass

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, net_path = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.reward.state_dict(), filepath.replace(".pth", "_r.pth"))
		torch.save(self.dynamics.state_dict(), filepath.replace(".pth", "_d.pth"))
		return net_path
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath.replace(".pth", "_d.pth")):
			try:
				self.reward.load_state_dict(torch.load(filepath.replace(".pth", "_r.pth"), map_location=self.device))
				self.dynamics.load_state_dict(torch.load(filepath.replace(".pth", "_d.pth"), map_location=self.device))
				print(f"Loaded LINFEAT model at {filepath}")
			except Exception as e:
				print(f"Error loading LINFEAT model at {filepath}")
		return self