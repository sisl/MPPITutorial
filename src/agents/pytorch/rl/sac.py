import torch
import numpy as np
from src.agents.pytorch.ref import get_ref
from src.agents.pytorch.rl.base import PTACNetwork, PTAgent, PTCritic, Conv, gsoftmax
from src.utils.rand import ReplayBuffer

class SACActor(torch.nn.Module):
	def __init__(self, state_size, action_size, config, use_discrete=False):
		super().__init__()
		input_layer, actor_hidden = config.INPUT_LAYER, config.ACTOR_HIDDEN
		self.discrete = use_discrete and type(action_size) != tuple
		self.layer1 = torch.nn.Linear(state_size[-1], input_layer) if len(state_size)==1 else Conv(state_size, input_layer) if len(state_size)==3 else get_ref(state_size, input_layer, config)
		self.layer2 = torch.nn.Linear(input_layer, actor_hidden)
		self.layer3 = torch.nn.Linear(actor_hidden, actor_hidden)
		self.action_mu = torch.nn.Linear(actor_hidden, action_size[-1])
		self.action_sig = torch.nn.Linear(actor_hidden, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		self.dist = lambda m,s: torch.distributions.Categorical(m.softmax(-1)) if self.discrete else torch.distributions.Normal(m,s)
		
	def forward(self, state, action=None, sample=True):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu()
		action_mu = self.action_mu(state)
		action_sig = self.action_sig(state).clamp(-5,0).exp()
		dist = torch.distributions.Normal(action_mu, action_sig)
		action = dist.rsample() if sample else action_mu
		action_out = gsoftmax(action_mu, hard=False) if self.discrete else action.tanh()
		log_prob = torch.log(action_out+1e-6) if self.discrete else dist.log_prob(action)-torch.log(1-action_out.pow(2)+1e-6)
		return action_out, log_prob

class SACCritic(torch.nn.Module):
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

class SACNetwork(PTACNetwork):
	def __init__(self, state_size, action_size, config, actor=SACActor, critic=SACCritic, gpu=True, load=None, name="sac", use_discrete=False):
		self.discrete = use_discrete and critic==SACCritic and type(action_size)!=tuple
		super().__init__(state_size, action_size, config, actor, critic if not self.discrete else lambda s,a,c: PTCritic(s,a,c), gpu=gpu, load=load, name=name)
		self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True).to(self.device))
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.LEARN_RATE)
		self.target_entropy = -np.product(action_size)

	def get_action_probs(self, state, action_in=None, grad=False, numpy=False, sample=True):
		with torch.enable_grad() if grad else torch.no_grad():
			action, log_prob = self.actor_local(state.to(self.device), action_in, sample)
			return [x.cpu().numpy() if numpy else x for x in [action, log_prob]]

	def get_q_value(self, state, action, use_target=False, grad=False, numpy=False, probs=False):
		with torch.enable_grad() if grad else torch.no_grad():
			critic = self.critic_local if not use_target else self.critic_target
			q_value = critic(state) if self.discrete else critic(state, action)
			return q_value.cpu().numpy() if numpy else q_value
	
	def optimize(self, states, actions, targets, next_log_probs, dones, config):
		alpha = self.log_alpha.clamp(-5, 0).detach().exp()
		if not self.discrete: next_log_probs = next_log_probs.sum(-1, keepdim=True)
		q_targets = targets - config.DISCOUNT_RATE*alpha*next_log_probs*(1-dones.view(-1,*[1]*(len(targets.shape)-1)))
		q_targets = (actions*q_targets).mean(-1, keepdim=True) if self.discrete else q_targets

		q_values = self.get_q_value(states, actions, grad=True)
		q_values = q_values.gather(-1, actions.argmax(-1, keepdim=True)) if self.discrete else q_values
		critic_loss = (q_values - q_targets.detach()).pow(2).mean()
		self.step(self.critic_optimizer, critic_loss, self.critic_local.parameters())
		self.soft_copy(self.critic_local, self.critic_target)

		actor_action, log_prob = self.actor_local(states)
		q_actions = self.get_q_value(states, actor_action, grad=True)
		q_baseline = q_targets if self.discrete else q_values
		actor_loss = alpha*log_prob - (q_actions - q_baseline.detach())
		actor_loss = actor_action*actor_loss if self.discrete else actor_loss
		self.step(self.actor_optimizer, actor_loss.mean(), self.actor_local.parameters())
		
		log_prob = (actor_action*log_prob).sum(-1) if self.discrete else log_prob
		alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
		self.step(self.alpha_optimizer, alpha_loss, [self.log_alpha])
		self.stats.mean(critic_loss=critic_loss, actor_loss=actor_loss.mean(), alpha_loss=alpha_loss)

class SACAgent(PTAgent):
	def __init__(self, state_size, action_size, config, gpu=True, load=None):
		super().__init__(state_size, action_size, config, SACNetwork, gpu=gpu, load=load)

	def get_action(self, state, eps=None, sample=True, e_greedy=False):
		action, self.log_prob = self.network.get_action_probs(self.to_tensor(state), numpy=True, sample=sample)
		return action
		
	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, action, self.log_prob, reward, done))
		if np.any(done[0]) or len(self.buffer) >= self.config.NUM_STEPS:
			states, actions, log_probs, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()	
			states = torch.cat([states, self.to_tensor(next_state).unsqueeze(0)], dim=0)
			next_action, next_log_prob = self.network.get_action_probs(states[-1])
			actions = torch.cat([actions, next_action.unsqueeze(0)], dim=0)
			log_probs = torch.cat([log_probs, next_log_prob.unsqueeze(0)], dim=0)
			values = self.network.get_q_value(states, actions, use_target=True)
			targets = self.compute_gae(values[-1], rewards.unsqueeze(-1), dones.unsqueeze(-1), values[:-1])[0]
			states, actions, targets, next_log_probs, dones = [x.view(x.size(0)*x.size(1), *x.size()[2:]).cpu().numpy() for x in (states[:-1], actions[:-1], targets, log_probs[1:], dones)]
			self.replay_buffer.extend(list(zip(states, actions, targets, next_log_probs, dones)), shuffle=False)	
		if len(self.replay_buffer) > self.config.REPLAY_BATCH_SIZE:
			states, actions, targets, next_log_probs, dones = self.replay_buffer.sample(self.config.REPLAY_BATCH_SIZE, dtype=self.to_tensor)[0]
			self.network.optimize(states, actions, targets, next_log_probs, dones, config=self.config)
