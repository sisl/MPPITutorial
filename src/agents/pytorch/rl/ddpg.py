import torch
import random
import numpy as np
from src.agents.pytorch.ref import get_ref
from src.agents.pytorch.rl.base import PTACNetwork, PTAgent, PTCritic, Conv, gsoftmax, one_hot_from_logits
from src.utils.rand import RandomAgent, PrioritizedReplayBuffer, ReplayBuffer

class DDPGActor(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		input_layer, actor_hidden = config.INPUT_LAYER, config.ACTOR_HIDDEN
		self.discrete = type(action_size) != tuple
		self.layer1 = torch.nn.Linear(state_size[-1], input_layer) if len(state_size)==1 else Conv(state_size, input_layer) if len(state_size)==3 else get_ref(state_size, input_layer, config)
		self.layer2 = torch.nn.Linear(input_layer, actor_hidden)
		self.layer3 = torch.nn.Linear(actor_hidden, actor_hidden)
		self.action_mu = torch.nn.Linear(actor_hidden, action_size[-1])
		self.action_sig = torch.nn.Linear(actor_hidden, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, sample=True):
		state = self.layer1(state).relu() 
		state = self.layer2(state).relu() 
		state = self.layer3(state).relu() 
		action_mu = self.action_mu(state)
		action_sig = self.action_sig(state).clamp(-5,0).exp()
		epsilon = torch.randn_like(action_sig)
		action = action_mu + epsilon.mul(action_sig) if sample else action_mu
		return action.tanh() if not self.discrete else gsoftmax(action)
	
class DDPGCritic(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		input_layer, critic_hidden = config.INPUT_LAYER, config.CRITIC_HIDDEN
		self.net_state = torch.nn.Linear(state_size[-1], input_layer) if len(state_size)==1 else Conv(state_size, input_layer) if len(state_size)==3 else get_ref(state_size, input_layer, config)
		self.net_action = torch.nn.Linear(action_size[-1], input_layer)
		self.net_layer1 = torch.nn.Linear(2*input_layer, critic_hidden)
		self.net_layer2 = torch.nn.Linear(critic_hidden, critic_hidden)
		self.q_value = torch.nn.Linear(critic_hidden, 1)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, action):
		state = self.net_state(state).relu()
		net_action = self.net_action(action).relu()
		net_layer = torch.cat([state, net_action], dim=-1)
		net_layer = self.net_layer1(net_layer).relu()
		net_layer = self.net_layer2(net_layer).relu()
		q_value = self.q_value(net_layer)
		return q_value

class DDPGNetwork(PTACNetwork):
	def __init__(self, state_size, action_size, config, actor=DDPGActor, critic=DDPGCritic, gpu=True, load=None, name="ddpg"): 
		self.discrete = type(action_size)!=tuple
		super().__init__(state_size, action_size, config, actor, critic if not self.discrete else lambda s,a,c: PTCritic(s,a,c), gpu=gpu, load=load, name=name)

	def get_action(self, state, use_target=False, grad=False, numpy=False, sample=True):
		with torch.enable_grad() if grad else torch.no_grad():
			actor = self.actor_local if not use_target else self.actor_target
			return actor(state, sample).cpu().numpy() if numpy else actor(state, sample)

	def get_q_value(self, state, action, use_target=False, grad=False, numpy=False, probs=False):
		with torch.enable_grad() if grad else torch.no_grad():
			critic = self.critic_local if not use_target else self.critic_target
			q_value = critic(state) if self.discrete else critic(state, action)
			q_value = q_value.gather(-1, action.argmax(-1, keepdim=True)) if self.discrete and not probs else q_value
			return q_value.cpu().numpy() if numpy else q_value
	
	def optimize(self, states, actions, q_targets):
		actions = one_hot_from_logits(actions) if self.actor_local.discrete else actions
		q_values = self.get_q_value(states, actions, grad=True, probs=False)
		critic_loss = (q_values - q_targets.detach()).pow(2).mean()
		self.step(self.critic_optimizer, critic_loss)
		self.soft_copy(self.critic_local, self.critic_target)

		actor_action = self.actor_local(states)
		q_actions = self.get_q_value(states, actor_action, grad=True, probs=True)
		q_actions = (actor_action*q_actions).sum(-1) if self.discrete else q_actions
		q_baseline = q_targets if self.discrete else q_values
		actor_loss = -(q_actions - q_baseline.detach()).mean()
		self.step(self.actor_optimizer, actor_loss, self.actor_local.parameters())
		self.soft_copy(self.actor_local, self.actor_target)
		self.stats.mean(critic_loss=critic_loss, actor_loss=actor_loss)
		
class DDPGAgent(PTAgent):
	def __init__(self, state_size, action_size, config, gpu=True, load=None):
		super().__init__(state_size, action_size, config, DDPGNetwork, gpu=gpu, load=load)

	def get_action(self, state, eps=None, sample=True, ref=None):
		eps = self.eps if eps is None else eps
		action_random = super().get_action(state, eps)
		if self.discrete and random.random() < eps: return action_random
		action_greedy = self.network.get_action(self.to_tensor(state), numpy=True, sample=sample)
		action = np.clip((1-eps)*action_greedy + eps*action_random, -1, 1)
		return action
		
	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, action, reward, done))
		if np.any(done[0]) or len(self.buffer) >= self.config.NUM_STEPS:
			states, actions, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()	
			states = torch.cat([states, self.to_tensor(next_state).unsqueeze(0)], dim=0)
			actions = torch.cat([actions, self.network.get_action(states[-1], use_target=True).unsqueeze(0)], dim=0)
			values = self.network.get_q_value(states, actions, use_target=True)
			targets = self.compute_gae(values[-1], rewards.unsqueeze(-1), dones.unsqueeze(-1), values[:-1])[0]
			states, actions, targets = [x.view(x.size(0)*x.size(1), *x.size()[2:]).cpu().numpy() for x in (states[:-1], actions[:-1], targets)]
			self.replay_buffer.extend(list(zip(states, actions, targets)), shuffle=False)	
		if len(self.replay_buffer) > self.config.REPLAY_BATCH_SIZE:
			states, actions, targets = self.replay_buffer.sample(self.config.REPLAY_BATCH_SIZE, dtype=self.to_tensor)[0]
			self.network.optimize(states, actions, targets)
			if np.any(done[0]): self.eps = max(self.eps * self.config.EPS_DECAY, self.config.EPS_MIN)
