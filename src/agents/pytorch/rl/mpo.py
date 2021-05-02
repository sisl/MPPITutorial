import torch
import numpy as np
from .base import PTACNetwork, PTAgent, PTCritic, Conv, gsoftmax
from src.utils.rand import ReplayBuffer
from scipy.optimize import minimize

class MPOActor(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		input_layer, actor_hidden = config.INPUT_LAYER, config.ACTOR_HIDDEN
		self.discrete = type(action_size) != tuple
		self.layer1 = torch.nn.Linear(state_size[-1], input_layer) if len(state_size)!=3 else Conv(state_size, input_layer)
		self.layer2 = torch.nn.Linear(input_layer, actor_hidden)
		self.layer3 = torch.nn.Linear(actor_hidden, actor_hidden)
		self.action_mu = torch.nn.Linear(actor_hidden, action_size[-1])
		self.action_cho = torch.nn.Linear(actor_hidden, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		
	def forward(self, state, detach=False, sample=True):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu()
		action_mu = self.action_mu(state)
		action_cho = self.action_cho(state).exp().diag_embed()
		action_mu, action_cho = [x.detach() if detach else x for x in [action_mu, action_cho]]
		dist = torch.distributions.MultivariateNormal(action_mu, scale_tril=action_cho)
		action = dist.rsample() if sample else action_mu
		return action.tanh(), (action_mu, action_cho)

class MPOCritic(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		input_layer, critic_hidden = config.INPUT_LAYER, config.CRITIC_HIDDEN
		self.net_state = torch.nn.Linear(state_size[-1], input_layer) if len(state_size)!=3 else Conv(state_size, input_layer)
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

class MPONetwork(PTACNetwork):
	def __init__(self, state_size, action_size, config, actor=MPOActor, critic=MPOCritic, gpu=True, load=None, name="mpo"):
		super().__init__(state_size, action_size, config, actor, critic, gpu=gpu, load=load, name=name)
		self.neta_mu, self.neta_cho, self.neta = (1.0, 1.0, 1.0)
		self.eps_mu, self.eps_cho, self.eps_dual = (1e-4, 1e-4, 1.0)

	def get_action_dist(self, state, detach=False, use_target=False, grad=False, numpy=False, sample=True):
		with torch.enable_grad() if grad else torch.no_grad():
			actor = self.actor_local if not use_target else self.actor_target
			action, dist = actor(state.to(self.device), detach, sample)
			return (action.cpu().numpy() if numpy else action), dist

	def get_q_value(self, state, action, use_target=False, grad=False, numpy=False):
		with torch.enable_grad() if grad else torch.no_grad():
			critic = self.critic_local if not use_target else self.critic_target
			q_value = critic(state, action)
			return q_value.cpu().numpy() if numpy else q_value
	
	def optimize(self, states, actions, q_targets, config):
		q_values = self.get_q_value(states, actions, grad=True)
		critic_loss = (q_values - q_targets.detach()).pow(2)
		self.step(self.critic_optimizer, critic_loss.mean(), self.critic_local.parameters())

		target_mu, target_cho = self.get_action_dist(states, detach=True)[1]
		target_dist = torch.distributions.MultivariateNormal(target_mu, scale_tril=target_cho)
		action_set = target_dist.sample([64]).transpose(0,1).tanh()
		target_value_set = self.get_q_value(states.unsqueeze(1).repeat_interleave(64,1), action_set)
		Q_np = target_value_set.detach().cpu().numpy()
		Q_np_max = np.max(Q_np, axis=1, keepdims=True)
		dual = lambda x: self.eps_dual*x + x*np.mean(Q_np_max/x + np.log(np.mean(np.exp((Q_np-Q_np_max)/x),1)))
		self.neta = minimize(dual, np.array([self.neta]), method="SLSQP", bounds=[(1e-6,10)]).x[0]
		advantage = target_value_set - torch.max(target_value_set,1,keepdim=True)[0]
		weights = torch.exp(advantage/self.neta).squeeze(-1)
		weights_norm = weights / torch.sum(weights,1,keepdim=True)
		local_mu, local_cho = self.get_action_dist(states, grad=True)[1]
		local_dist = torch.distributions.MultivariateNormal(local_mu, scale_tril=local_cho)
		log_probs = local_dist.log_prob(action_set.transpose(0,1)).transpose(0,1)
		weighted_log_probs = weights_norm * log_probs
		mu_dist = torch.distributions.MultivariateNormal(local_mu, scale_tril=local_cho.detach())
		mu_kl = torch.distributions.kl_divergence(target_dist, mu_dist).mean()
		cho_dist = torch.distributions.MultivariateNormal(local_mu.detach(), scale_tril=local_cho)
		cho_kl = torch.distributions.kl_divergence(target_dist, cho_dist).mean()
		self.neta_mu = max(self.neta_mu-config.LEARN_RATE*(self.eps_mu - mu_kl).cpu().detach().numpy(), 0)
		self.neta_cho = max(self.neta_cho-config.LEARN_RATE*(self.eps_cho - cho_kl).cpu().detach().numpy(), 0)
		actor_loss = -(weighted_log_probs.sum() + self.neta_mu*(self.eps_mu-mu_kl) + self.neta_cho*(self.eps_cho-cho_kl) + local_dist.entropy())
		self.step(self.actor_optimizer, actor_loss.mean(), self.actor_local.parameters())

class MPOAgent(PTAgent):
	def __init__(self, state_size, action_size, config, gpu=True, load=None):
		super().__init__(state_size, action_size, config, MPONetwork, gpu=gpu, load=load)

	def get_action(self, state, eps=None, sample=True, e_greedy=False):
		action = self.network.get_action_dist(self.to_tensor(state), numpy=True, sample=sample)[0]
		return action
		
	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, action, reward, done))
		if np.any(done[0]) or len(self.buffer) >= self.config.NUM_STEPS:
			states, actions, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()	
			states = torch.cat([states, self.to_tensor(next_state).unsqueeze(0)], dim=0)
			actions = torch.cat([actions, self.network.get_action_dist(states[-1])[0].unsqueeze(0)], dim=0)
			values = self.network.get_q_value(states, actions, use_target=False)
			targets = self.compute_gae(values[-1], rewards.unsqueeze(-1), dones.unsqueeze(-1), values[:-1])[0]
			states, actions, targets = [x.view(x.size(0)*x.size(1), *x.size()[2:]).cpu().numpy() for x in (states[:-1], actions[:-1], targets)]
			self.replay_buffer.extend(list(zip(states, actions, targets)), shuffle=False)	
		if len(self.replay_buffer) > self.config.REPLAY_BATCH_SIZE:
			states, actions, targets = self.replay_buffer.sample(self.config.REPLAY_BATCH_SIZE, dtype=self.to_tensor)[0]
			self.network.optimize(states, actions, targets, config=self.config)
