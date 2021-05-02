import os
import torch
import numpy as np
from src.utils.misc import load_module
from src.agents.pytorch.rl.base import PTNetwork, one_hot_from_logits

class TransitionModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.config = config
		self.gru = torch.nn.GRUCell(action_size[-1] + 2*state_size[-1], config.DYN.TRANSITION_HIDDEN)
		self.linear1 = torch.nn.Linear(config.DYN.TRANSITION_HIDDEN, config.DYN.TRANSITION_HIDDEN)
		self.drop1 = torch.nn.Dropout(p=0.5)
		self.linear2 = torch.nn.Linear(config.DYN.TRANSITION_HIDDEN, config.DYN.TRANSITION_HIDDEN)
		self.drop2 = torch.nn.Dropout(p=0.5)
		self.state_ddot = torch.nn.Linear(config.DYN.TRANSITION_HIDDEN, state_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, action, state, state_dot, grad=False):
		with torch.enable_grad() if grad else torch.no_grad():
			input_dim = action.shape[:-1]
			hidden = self.hidden.view(np.prod(input_dim),-1)
			inputs = torch.cat([action, state*self.second_order_mask, state_dot],-1)
			hidden = self.gru(inputs.view(np.prod(input_dim),-1), hidden).view(*input_dim,-1)
			linear1 = self.linear1(hidden).relu() + hidden
			linear1 = self.drop1(linear1)
			linear2 = self.linear2(linear1).relu() + linear1
			linear2 = self.drop2(linear2)
			state_ddot = self.state_ddot(linear2)
			state_dot = state_dot + state_ddot
			next_state = state + state_dot
			self.hidden = hidden
			return next_state, state_dot

	def init_hidden(self, batch_size, device, train=False):
		self.train() if train else self.eval()
		if batch_size is None: batch_size = self.hidden[0].shape[1:2] if hasattr(self, "hidden") else [1]
		self.hidden = torch.zeros(*batch_size, self.config.DYN.TRANSITION_HIDDEN, device=device)
		self.second_order_mask = torch.tensor(self.config.get("dynamics_somask", 1), device=device)

class RewardModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.cost = load_module(config.REWARD_MODEL)(config=config) if config.get("REWARD_MODEL") else None
		self.dyn_spec = load_module(config.DYNAMICS_SPEC) if config.get("DYNAMICS_SPEC") else None
		self.linear1 = torch.nn.Linear(action_size[-1] + 2*state_size[-1], config.DYN.REWARD_HIDDEN)
		self.drop1 = torch.nn.Dropout(p=0.5)
		self.linear2 = torch.nn.Linear(config.DYN.REWARD_HIDDEN, config.DYN.REWARD_HIDDEN)
		self.drop2 = torch.nn.Dropout(p=0.5)
		self.linear3 = torch.nn.Linear(config.DYN.REWARD_HIDDEN, config.DYN.REWARD_HIDDEN)
		self.linear4 = torch.nn.Linear(config.DYN.REWARD_HIDDEN, 1)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, action, state, next_state, grad=False, model=False, times=None):
		if self.cost and self.dyn_spec and not grad and not model:
			next_state, state = [x.cpu().numpy() for x in [next_state, state]]
			ns_spec, s_spec = map(self.dyn_spec.observation_spec, [next_state, state])
			reward = -torch.FloatTensor(self.cost.get_cost(ns_spec, s_spec, times)).unsqueeze(-1)
		else:
			with torch.enable_grad() if grad else torch.no_grad():
				inputs = torch.cat([action, state, next_state],-1)
				layer1 = self.linear1(inputs).relu()
				layer1 = self.drop1(layer1)
				layer2 = self.linear2(layer1).tanh() + layer1
				layer2 = self.drop2(layer2)
				layer3 = self.linear3(layer2).tanh() + layer1
				reward = self.linear4(layer3)
		return reward

class DifferentialEnv(PTNetwork):
	def __init__(self, state_size, action_size, config, load="", gpu=True, name="dfrntl"):
		super().__init__(config, gpu, name)
		self.state_size = state_size
		self.action_size = action_size
		self.discrete = type(self.action_size) != tuple
		self.dyn_index = config.get("dynamics_size", state_size[-1])
		self.reward = RewardModel([self.dyn_index], action_size, config)
		self.dynamics = TransitionModel([self.dyn_index], action_size, config)
		self.dynamics_norm = torch.from_numpy(config.get("dynamics_norm", np.ones(state_size))).to(self.device)
		self.reward_optimizer = torch.optim.Adam(self.reward.parameters(), lr=config.DYN.LEARN_RATE, weight_decay=config.DYN.REG_LAMBDA)
		self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=config.DYN.LEARN_RATE, weight_decay=config.DYN.REG_LAMBDA)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.dynamics_optimizer, factor=config.DYN.FACTOR, patience=config.DYN.PATIENCE)
		self.to(self.device)
		if load: self.load_model(load)

	def value(self, action, state, next_state, train=False, times=None):
		state, next_state = [x[...,:self.dyn_index] for x in [state, next_state]]
		reward = self.reward(action, state, next_state, grad=train, model=False, times=times)
		return reward

	def rollout(self, actions, state, timedim=0, train=False, times=None):
		state = self.to_tensor(state[...,:self.dyn_index])/self.dynamics_norm
		state_dot = torch.zeros_like(state)
		self.dynamics.init_hidden(state.shape[:-len(self.state_size)], self.device, train=train)
		buffer = []
		for action in self.to_tensor(actions):
			if self.discrete: action = one_hot_from_logits(action)
			next_state, state_dot = self.dynamics(action, state, state_dot, grad=train)
			buffer.append((action, state*self.dynamics_norm, next_state*self.dynamics_norm))
			state = next_state
		actions, states, next_states = map(torch.stack, zip(*buffer))
		time_buffer = np.stack([times+t for t in range(actions.shape[0])]) if times is not None else None
		rewards = self.value(actions, states, next_states, train=train, times=time_buffer)
		if train==False: actions, states, next_states, rewards = map(lambda x: x.cpu().numpy(), [actions, states, next_states, rewards])
		return actions, states, next_states, rewards

	def get_loss(self, states, actions, next_states, rewards, dones, mask=1, train=True):
		s, a, ns, r, mask = map(self.to_tensor, [states, actions, next_states, rewards, mask])
		s, ns = [x[...,:self.dyn_index] for x in [s, ns]]
		rewards_hat = self.reward(a, s, ns, grad=train)
		rew_loss = ((rewards_hat - r.unsqueeze(-1)).pow(2) * mask).mean()
		state = s[0]/self.dynamics_norm
		state_dot = torch.zeros_like(state)
		self.dynamics.init_hidden(state.shape[:-len(self.state_size)], self.device, train=train)
		next_states_hat = []
		for action in torch.split(a, 1):
			state, state_dot = self.dynamics(action.squeeze(0), state, state_dot, grad=train)
			next_states_hat.append(state)
		next_states_hat = torch.stack(next_states_hat)
		dyn_loss = ((next_states_hat - ns/self.dynamics_norm).pow(2) * mask).sum(-1).mean()
		return rew_loss, dyn_loss

	def optimize(self, states, actions, next_states, rewards, dones, mask=1):
		rew_loss, dyn_loss = self.get_loss(states, actions, next_states, rewards, dones, mask=mask)
		super().step(self.reward_optimizer, rew_loss)
		super().step(self.dynamics_optimizer, dyn_loss)
		self.stats.mean(dyn_loss=dyn_loss, rew_loss=rew_loss)
		return [loss.item() for loss in [rew_loss, dyn_loss]]

	def schedule(self, test_loss):
		self.scheduler.step(test_loss)

	def get_stats(self):
		return {**super().get_stats(), "lr": self.dynamics_optimizer.param_groups[0]["lr"] if self.dynamics_optimizer else None}

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
				print(f"Loaded DFRNTL model at {filepath}")
			except Exception as e:
				print(f"Error loading DFRNTL model at {filepath}")
		return self