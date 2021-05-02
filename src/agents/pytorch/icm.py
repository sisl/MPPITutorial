import os
import torch
import numpy as np
from .rl.base import PTNetwork

class ICM(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__(self)
		self.state_features = torch.nn.Linear(state_size[-1], config.ACTOR_HIDDEN)
		self.forward_state_pred = torch.nn.Linear(config.ACTOR_HIDDEN+action_size[-1], config.ACTOR_HIDDEN)
		self.inverse_action_pred = torch.nn.Linear(2*config.ACTOR_HIDDEN, action_size[-1])
		self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARN_RATE, weight_decay=config.REG_LAMBDA)

	def train(self, state, action, next_state, eta=0.1, beta=0.5):
		state_features = self.state_features(state)
		next_state_features = self.state_features(next_state)
		next_state_pred = self.forward_state_pred(torch.cat([state_features, action], dim=-1))
		action_pred = self.inverse_action_pred(torch.cat([state_features, next_state_features], dim=-1))
		forward_error = (next_state_features - next_state_pred).pow(2).mean(-1)
		inverse_error = (action - action_pred).pow(2).mean(-1)
		self.step(self.optimizer, (forward_error+inverse_error).mean(), self.parameters())
		intrinsic = eta*(beta*forward_error + (1-beta)*inverse_error)
		return intrinsic.cpu().detach().numpy()

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.state_dict(), filepath)
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath):
			try: self.load_state_dict(torch.load(filepath, map_location=self.device))
			except: print(f"WARN: Error loading model from {filepath}")

class ICMNetwork(PTNetwork):
	def __init__(self, state_size, action_size, config, gpu=True):
		super().__init__(config, gpu=gpu, name="icm")
		self.model = ICM(state_size, action_size, config).to(self.device)

	def get_reward(self, state, action, next_state, reward, done):
		state, action, next_state = [torch.from_numpy(x).float().to(self.device) for x in [state, action, next_state]]
		intrinsic = self.model.train(state, action, next_state)
		self.stats.sum(r_i=intrinsic)
		return reward + intrinsic*(1-done)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		self.model.save_model(dirname,name,net)
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		self.model.load_model(dirname,name,net)