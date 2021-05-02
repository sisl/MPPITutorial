import torch
import random
import numpy as np
from .base import PTQNetwork, PTAgent

class DDQNetwork(PTQNetwork):
	def __init__(self, state_size, action_size, config, gpu=True, load=None, name="ddqn"): 
		super().__init__(state_size, action_size, config, gpu=gpu, load=load, name=name)

	def get_action(self, state, use_target=False, numpy=True, sample=True):
		with torch.no_grad():
			q_values = self.critic_local(state) if not use_target else self.critic_target(state)
			return q_values.softmax(-1).cpu().numpy() if numpy else q_values.softmax(-1)

	def get_q_value(self, state, action, use_target=False, grad=False, numpy=False):
		with torch.enable_grad() if grad else torch.no_grad():
			q_values = self.critic_local(state) if not use_target else self.critic_target(state)
			q_selected = torch.gather(q_values, dim=-1, index=action.argmax(-1, keepdims=True))
			return q_selected.cpu().numpy() if numpy else q_selected
	
	def optimize(self, states, actions, q_targets):
		q_values = self.get_q_value(states, actions, grad=True)
		critic_loss = (q_values - q_targets.detach()).pow(2)
		self.step(self.critic_optimizer, critic_loss.mean())
		self.soft_copy(self.critic_local, self.critic_target)

class DDQNAgent(PTAgent):
	def __init__(self, state_size, action_size, config, gpu=True, load=None):
		super().__init__(state_size, action_size, config, DDQNetwork, gpu=gpu, load=load)

	def get_action(self, state, eps=None, sample=True, e_greedy=True):
		eps = self.eps if eps is None else eps
		action_random = super().get_action(state, eps)
		if e_greedy and random.random() < eps: return action_random
		action_greedy = self.network.get_action(self.to_tensor(state), sample=sample)
		return action_greedy
		
	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, action, reward, done))
		if np.any(done[0]) or len(self.buffer) >= self.config.NUM_STEPS:
			states, actions, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()	
			states = torch.cat([states, self.to_tensor(next_state).unsqueeze(0)], dim=0)
			actions = torch.cat([actions, self.network.get_action(states[-1], use_target=True, numpy=False).unsqueeze(0)], dim=0)
			values = self.network.get_q_value(states, actions, use_target=True, numpy=False)
			targets = self.compute_gae(values[-1], rewards.unsqueeze(-1), dones.unsqueeze(-1), values[:-1])[0]
			states, actions, targets = [x.view(x.size(0)*x.size(1), *x.size()[2:]).cpu().numpy() for x in (states[:-1], actions[:-1], targets)]
			self.replay_buffer.extend(list(zip(states, actions, targets)), shuffle=True)	
		if len(self.replay_buffer) > self.config.REPLAY_BATCH_SIZE:
			states, actions, targets = self.replay_buffer.sample(self.config.REPLAY_BATCH_SIZE, dtype=self.to_tensor)[0]
			self.network.optimize(states, actions, targets)
			if np.any(done[0]): self.eps = max(self.eps * self.config.EPS_DECAY,  self.config.EPS_MIN)
