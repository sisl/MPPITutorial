import tqdm
import torch
import random
import numpy as np
import scipy as sp
from collections import deque
from scipy.stats import multivariate_normal
from src.agents.pytorch.mpc.envmodel import get_envmodel
from src.agents.pytorch.rl.base import PTNetwork, PTAgent, Conv, one_hot_from_indices
from src.utils.rand import RandomAgent, ReplayBuffer
from src.utils.misc import load_module, pad

class MPPIController(PTNetwork):
	def __init__(self, state_size, action_size, config, load="", gpu=True, name="mppi"):
		super().__init__(config, gpu=gpu, name=name)
		self.envmodel = get_envmodel(config)(state_size, action_size, config, load=load, gpu=gpu)
		self.discrete = type(action_size)!=tuple
		self.mu = np.zeros(action_size)
		self.cov = np.diag(np.ones(action_size))*config.MPC.COV*(1+4*int(self.discrete))
		self.icov = np.linalg.inv(self.cov)
		self.lamda = config.MPC.LAMBDA
		self.horizon = config.MPC.HORIZON
		self.nsamples = config.MPC.NSAMPLES
		self.state_size = state_size
		self.action_size = action_size
		self.config = config
		self.init_control()

	def init_control(self, batch_size=[1]):
		self.control = np.random.uniform(-1, 1, size=[self.horizon, *batch_size, *self.action_size])
		self.discount = np.power(self.config.DISCOUNT_RATE, np.arange(self.horizon))[:,None,None]
		self.noise = np.random.multivariate_normal(self.mu, self.cov, size=[self.horizon, *batch_size, self.nsamples])
		controls = self.control[:,:,None,None,:]
		icov = self.icov[None,None,None,:,:]
		noise = self.noise[:,:,:,:,None]
		self.init_cost = np.sum(controls @ icov @ noise, axis=(0,3,4))/self.horizon

	def get_action(self, state, eps=None, sample=True, times=None):
		eps = 0 if eps is None else eps
		batch = state.shape[:-len(self.state_size)]
		horizon = max(int((1-eps)*self.horizon),1) if eps else self.horizon
		if self.control.shape[1:-len(self.action_size)] != batch: self.init_control(batch)
		times = np.repeat(np.expand_dims(times, -1), self.nsamples, -1)
		noise = self.noise[:horizon] * max(eps, 0.1)
		discount = self.discount[:horizon]
		controls = np.clip(self.control[:horizon,:,None] + noise, -1, 1)
		# x = torch.Tensor(state).view(*batch, 1,-1).repeat_interleave(self.nsamples, -2)
		x = np.reshape(state, (*batch,1,-1)).repeat(self.nsamples, -2)
		self.actions, self.states, self.next_states, self.rewards = self.envmodel.rollout(controls, x, train=False, times=times)
		costs = -np.sum(np.squeeze(self.rewards,-1)*discount,0) + self.lamda * self.init_cost
		beta = np.min(costs, -1, keepdims=True)
		costs_norm = (costs - beta)/self.lamda*np.exp(-2*eps)
		weights = sp.special.softmax(-costs_norm, axis=-1)
		weighted_noise = np.sum(weights[None,:,:,None]*noise, -len(self.action_size)-1)
		self.control[:horizon] += weighted_noise
		action = self.control[0]
		self.control = np.roll(self.control, -1, axis=0)
		self.control[-1] = 0
		return action

	def optimize(self, states, actions, next_states, rewards, dones, **kwargs):
		return self.envmodel.optimize(states, actions, next_states, rewards, dones, **kwargs)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		return self.envmodel.save_model(dirname, name, net)
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		return self.envmodel.load_model(dirname, name, net)

	def get_stats(self):
		return {**super().get_stats(), **self.envmodel.get_stats()}

class MPPIAgent(PTAgent):
	def __init__(self, state_size, action_size, config, gpu=True, load=None):
		super().__init__(state_size, action_size, config, MPPIController, gpu=gpu, load=load)
		self.ep_lens = deque(maxlen=config.MAX_BUFFER_SIZE)
		self.times = np.zeros(())-1

	def get_action(self, state, eps=None, sample=True):
		batch = state.shape[:-len(self.state_size)]
		if self.times.shape != batch: self.times = np.zeros(batch)-1
		self.times += 1
		eps = self.eps if eps is None else eps
		action_random = super().get_action(state)
		if eps >= 1: return action_random
		action_greedy = self.network.get_action(np.array(state), eps=eps, sample=sample, times=self.times)
		action = np.clip((1-eps)*action_greedy + eps*action_random, -1, 1)
		return action

	def train(self, state, action, next_state, reward, done):
		self.step = getattr(self, "step", 0) + 1
		if not hasattr(self, "buffers"): self.buffers = [[] for _ in done]
		for i,(buffer, s, a, ns, r, d) in enumerate(zip(self.buffers, state, action, next_state, reward, done)):
			buffer.append((s, a, s if d else ns, r, d))
			if not d: continue
			self.times[i] = 0
			self.ep_lens.append(len(buffer))
			states, actions, next_states, rewards, dones = map(lambda x: self.to_tensor(x)[None], zip(*buffer))
			buffer.clear()
			mask = torch.ones_like(rewards).unsqueeze(-1)
			values = self.network.envmodel.value(actions, states, next_states)[0]
			rewards = self.compute_gae(0*values[-1], rewards.transpose(0,1), dones.transpose(0,1), values)[0].transpose(0,1)
			states, actions, next_states, rewards, dones, mask = map(lambda x: x.cpu().numpy(), [states, actions, next_states, rewards, dones, mask])
			states, actions, next_states, rewards, dones, mask = map(lambda x: pad(x[0], self.config.NUM_STEPS), [states, actions, next_states, rewards, dones, mask])
			self.replay_buffer.extend(list(zip(states, actions, next_states, rewards, dones, mask)), shuffle=False)
		if len(self.replay_buffer) > self.config.REPLAY_BATCH_SIZE and self.step % self.config.TRAIN_EVERY == 0:
			states, actions, next_states, rewards, dones, mask = self.replay_buffer.sample(self.config.REPLAY_BATCH_SIZE, dtype=lambda x: self.to_tensor(x).transpose(0,1))[0]
			self.network.optimize(states, actions, next_states, rewards, dones, mask=mask)
			self.eps = (self.step/int(np.mean(self.ep_lens)+1))%1

	def get_stats(self):
		return {**super().get_stats(), "len":len(self.replay_buffer), "ep_len":np.mean(self.ep_lens) if len(self.ep_lens) else 0}
