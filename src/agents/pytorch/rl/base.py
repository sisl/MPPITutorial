import os
import math
import torch
import random
import inspect
import numpy as np
from src.utils.logger import Stats, LOG_DIR
from src.utils.rand import RandomAgent, ReplayBuffer
from ..network import Conv, gsoftmax, one_hot_from_logits, one_hot_from_indices

class PTActor(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		input_layer, actor_hidden = config.INPUT_LAYER, config.ACTOR_HIDDEN
		self.state_fc1 = torch.nn.Linear(state_size[-1], input_layer) if len(state_size)!=3 else Conv(state_size, input_layer)
		self.state_fc2 = torch.nn.Linear(input_layer, actor_hidden)
		self.state_fc3 = torch.nn.Linear(actor_hidden, actor_hidden)
		self.action_mu = torch.nn.Linear(actor_hidden, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		state = self.state_fc1(state).relu() 
		state = self.state_fc2(state).relu() 
		state = self.state_fc3(state).relu() 
		action_mu = self.action_mu(state)
		return action_mu

class PTCritic(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		input_layer, critic_hidden = config.INPUT_LAYER, config.CRITIC_HIDDEN
		self.state_fc1 = torch.nn.Linear(state_size[-1], input_layer) if len(state_size)!=3 else Conv(state_size, input_layer)
		self.state_fc2 = torch.nn.Linear(input_layer, critic_hidden)
		self.state_fc3 = torch.nn.Linear(critic_hidden, critic_hidden)
		self.value = torch.nn.Linear(critic_hidden, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, action=None):
		state = self.state_fc1(state).relu()
		state = self.state_fc2(state).relu()
		state = self.state_fc3(state).relu()
		value = self.value(state)
		return value

class PTNetwork(torch.nn.Module):
	def __init__(self, config, gpu=True, name="pt"): 
		super().__init__()
		self.tau = config.get("TARGET_UPDATE_RATE",0)
		self.name = name
		self.stats = Stats()
		self.config = config
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

	def init_weights(self, model=None):
		model = self if model is None else model
		model.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def to_tensor(self, x):
		return x.float().to(self.device) if isinstance(x,torch.Tensor) else self.to_tensor(torch.tensor(x)) if x is not None else x
		
	def step(self, optimizer, loss, param_norm=None, retain=False, norm=0.5):
		optimizer.zero_grad()
		loss.backward(retain_graph=retain)
		if param_norm is not None: torch.nn.utils.clip_grad_norm_(param_norm, norm)
		optimizer.step()

	def soft_copy(self, local, target, tau=None):
		tau = self.tau if tau is None else tau
		for t,l in zip(target.parameters(), local.parameters()):
			t.data.copy_(t.data + tau*(l.data - t.data))

	def get_stats(self):
		return self.stats.get_stats()

	def get_checkpoint_path(self, dirname="pytorch", name="checkpoint", net=None):
		net_path = os.path.join(LOG_DIR, "models", self.name if net is None else net, dirname)
		filepath = os.path.join(net_path, f"{name}.pth")
		return filepath, net_path

class PTQNetwork(PTNetwork):
	def __init__(self, state_size, action_size, config, critic=PTCritic, gpu=True, load="", name="ptq"): 
		super().__init__(config, gpu=gpu, name=name)
		self.src = inspect.getsource(critic) 
		self.critic_local = critic(state_size, action_size, config).to(self.device)
		self.critic_target = critic(state_size, action_size, config).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=config.LEARN_RATE, weight_decay=config.REG_LAMBDA)
		if load: self.load_model(load)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, net_path = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.critic_local.state_dict(), filepath)
		with open(filepath.replace(".pth", ".txt"), "w") as f:
			f.write(f"{str(self.config)}\n\n{self.src}")
		return net_path
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath):
			try:
				self.critic_local.load_state_dict(torch.load(filepath, map_location=self.device))
				self.critic_target.load_state_dict(torch.load(filepath, map_location=self.device))
			except:
				print(f"WARN: Error loading model from {filepath}")

class PTACNetwork(PTNetwork):
	def __init__(self, state_size, action_size, config, actor=PTActor, critic=PTCritic, gpu=True, load="", name="ptac"): 
		super().__init__(config, gpu=gpu, name=name)
		self.src = [inspect.getsource(model) for model in [actor, critic]]
		self.actor_local = actor(state_size, action_size, config).to(self.device)
		self.actor_target = actor(state_size, action_size, config).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=config.LEARN_RATE, weight_decay=config.REG_LAMBDA)
		
		self.critic_local = critic(state_size, action_size, config).to(self.device)
		self.critic_target = critic(state_size, action_size, config).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=config.LEARN_RATE, weight_decay=config.REG_LAMBDA)
		if load: self.load_model(load)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, net_path = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.actor_local.state_dict(), filepath.replace(".pth", "_a.pth"))
		torch.save(self.critic_local.state_dict(), filepath.replace(".pth", "_c.pth"))
		with open(filepath.replace(".pth", ".txt"), "w") as f:
			f.write("\n".join([str(self.config)+"\n"]+self.src))
		return net_path
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath.replace(".pth", "_a.pth")):
			try:
				self.actor_local.load_state_dict(torch.load(filepath.replace(".pth", "_a.pth"), map_location=self.device))
				self.actor_target.load_state_dict(torch.load(filepath.replace(".pth", "_a.pth"), map_location=self.device))
				self.critic_local.load_state_dict(torch.load(filepath.replace(".pth", "_c.pth"), map_location=self.device))
				self.critic_target.load_state_dict(torch.load(filepath.replace(".pth", "_c.pth"), map_location=self.device))
				print(f"Loaded model from {filepath}")
			except Exception as e:
				print(f"WARN: Error loading model from {filepath}: {e}")

class PTAgent(RandomAgent):
	def __init__(self, state_size, action_size, config, network=PTACNetwork, gpu=True, load=None):
		super().__init__(state_size, action_size, config)
		self.network = network(state_size, action_size, config, gpu=gpu, load=load)
		self.replay_buffer = ReplayBuffer(config.MAX_BUFFER_SIZE)
		self.buffer = []

	def to_tensor(self, arr):
		if isinstance(arr, torch.Tensor): return arr.to(self.network.device)
		if isinstance(arr, np.ndarray): return tuple(self.to_tensor(a) for a in arr) if arr.dtype=='O' else torch.tensor(arr, requires_grad=False).float().to(self.network.device)
		return self.to_tensor(np.array(arr))

	def compute_gae(self, last_value, rewards, dones, values):
		last_value, rewards, dones, values = map(self.to_tensor, [last_value, rewards, dones, values])
		with torch.no_grad():
			gae = 0
			gamma, lamda = self.config.DISCOUNT_RATE, self.config.ADVANTAGE_DECAY
			targets = torch.zeros_like(values, device=values.device)
			values = torch.cat([values, last_value.unsqueeze(0)])
			for step in reversed(range(len(rewards))):
				delta = rewards[step] + gamma * values[step + 1] * (1-dones[step]) - values[step]
				gae = delta + gamma * lamda * (1-dones[step]) * gae
				targets[step] = gae + values[step]
			advantages = targets - values[:-1]
			return targets, advantages
		
	def train(self, state, action, next_state, reward, done):
		pass

	def get_stats(self):
		return {**super().get_stats(), **self.network.get_stats()}

