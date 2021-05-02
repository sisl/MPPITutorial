import os
import torch
import numpy as np
from ...rl.base import PTNetwork, one_hot_from_logits

class MDRNNEnv(PTNetwork):
	def __init__(self, state_size, action_size, config, load="", gpu=True, name="mdrnn"):
		super().__init__(config, gpu, name)
		self.state_size = state_size
		self.action_size = action_size
		self.n_gauss = config.DYN.NGAUSS
		self.stride = self.n_gauss*state_size[-1]
		self.splits = (self.stride, self.stride, self.n_gauss, 1, 1)
		self.discrete = type(self.action_size) != tuple
		self.lstm = torch.nn.LSTM(action_size[-1] + state_size[-1], config.DYN.HIDDEN_SIZE, batch_first=True)
		self.gmm = torch.nn.Linear(config.DYN.HIDDEN_SIZE, (2*state_size[-1]+1)*self.n_gauss + 2)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=config.DYN.LEARN_RATE)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=config.DYN.FACTOR, patience=config.DYN.PATIENCE)
		self.to(self.device)
		if load: self.load_model(load)

	def forward(self, actions, states):
		if self.discrete: actions = one_hot_from_logits(actions)
		lstm_inputs = torch.cat([actions, states], dim=-1)
		lstm_outs, self.hidden = self.lstm(lstm_inputs, self.hidden)
		gmm_outputs = self.gmm(lstm_outs)
		mus, sigs, pi, rs, ds = torch.split(gmm_outputs, self.splits, -1)
		mus = mus.view(mus.size(0), mus.size(1), self.n_gauss, *self.state_size)
		sigs = sigs.view(sigs.size(0), sigs.size(1), self.n_gauss, *self.state_size).exp()
		logpi = pi.view(pi.size(0), pi.size(1), self.n_gauss).log_softmax(dim=-1)
		return mus, sigs, logpi, rs.squeeze(-1), ds.squeeze(-1).sigmoid()

	def reset(self, batch_size=None, **kwargs):
		if batch_size is None:
			batch_size = self.hidden[0].shape[1] if hasattr(self, "hidden") else 1
		self.hidden = [torch.zeros([1, batch_size, self.config.DYN.HIDDEN_SIZE], device=self.device) for _ in range(2)]

	def step(self, action, state):
		with torch.no_grad():
			states, actions = map(self.to_tensor, [state, action])
			if len(states.shape)<3: states, actions = [x.view(1, 1, -1) for x in [states, actions]]
			mus, sigs, logpi, rs, ds = self.forward(actions, states)
			dist = torch.distributions.categorical.Categorical(logpi.exp())
			indices = dist.sample().unsqueeze(-1).unsqueeze(-1).repeat_interleave(state.shape[-1], -1)
			mu = mus.gather(2, indices).view(state.shape)
			sig = sigs.gather(2, indices).view(state.shape)
			next_states = mu + torch.randn_like(sig).mul(sig)
			return next_states, rs.squeeze(-1).cpu().numpy()

	def get_gmm_loss(self, mus, sigs, logpi, next_states):
		dist = torch.distributions.normal.Normal(mus, sigs)
		log_probs = dist.log_prob(next_states.unsqueeze(-2))
		log_probs = logpi + torch.sum(log_probs, dim=-1)
		max_log_probs = torch.max(log_probs, dim=-1, keepdim=True)[0]
		g_log_probs = log_probs - max_log_probs
		g_probs = torch.sum(torch.exp(g_log_probs), dim=-1)
		log_prob = max_log_probs.squeeze() + torch.log(g_probs)
		return -torch.mean(log_prob)

	def get_loss(self, states, actions, next_states, rewards, dones):
		self.reset(batch_size=states.shape[0])
		s, a, ns, r, d = map(self.to_tensor, (states, actions, next_states, rewards, dones))
		mus, sigs, logpi, rs, ds = self.forward(a, s)
		mse = torch.nn.functional.mse_loss(rs, r)
		bce = torch.nn.functional.binary_cross_entropy_with_logits(ds, d)
		gmm = self.get_gmm_loss(mus, sigs, logpi, ns)
		self.stats.mean(mse=mse, bce=bce, gmm=gmm)
		return (gmm + mse + bce) / (self.state_size[-1] + 2)

	def optimize(self, states, actions, next_states, rewards, dones):
		loss = self.get_loss(states, actions, next_states, rewards, dones)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss

	def schedule(self, test_loss):
		self.scheduler.step(test_loss)

	def save_model(self, dirname="pytorch", name="best", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.state_dict(), filepath)
		
	def load_model(self, dirname="pytorch", name="best", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath):
			self.load_state_dict(torch.load(filepath, map_location=self.device))
			print(f"Loaded MDRNN model at {filepath}")
		return self

class MDRNNCell(torch.nn.Module):
	def __init__(self, state_size, action_size, config, load="", gpu=True):
		super().__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.n_gauss = config.DYN.NGAUSS
		self.discrete = type(self.action_size) == list
		self.lstm = torch.nn.LSTMCell(action_size[-1] + state_size, state_size)
		self.gmm = torch.nn.Linear(state_size, (2*state_size+1)*self.n_gauss + 2)
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
		self.to(self.device)
		if load: self.load_model(load)

	def forward(self, actions, states, hiddens):
		with torch.no_grad():
			actions, states = [x.to(self.device) for x in (torch.from_numpy(actions), states)]
			lstm_inputs = torch.cat([actions, states], dim=-1)
			lstm_hidden = self.lstm(lstm_inputs, hiddens)
			return lstm_hidden

	def step(self, hiddens):
		with torch.no_grad():
			gmm_out = self.gmm(hiddens)
			stride = self.n_gauss*self.state_size
			mus = gmm_out[:,:stride]
			sigs = gmm_out[:,stride:2*stride].exp()
			pi = gmm_out[:,2*stride:2*stride+self.n_gauss].softmax(dim=-1)
			rs = gmm_out[:,2*stride+self.n_gauss]
			ds = gmm_out[:,2*stride+self.n_gauss+1].sigmoid()
			mus = mus.view(-1, self.n_gauss, self.state_size)
			sigs = sigs.view(-1, self.n_gauss, self.state_size)
			dist = torch.distributions.categorical.Categorical(pi)
			indices = dist.sample()
			mus = mus[:,indices,:].squeeze(1)
			sigs = sigs[:,indices,:].squeeze(1)
			next_states = mus + torch.randn_like(sigs).mul(sigs)
			return next_states, rs

	def reset(self, batch_size=1):
		return [torch.zeros(batch_size, self.state_size).to(self.device) for _ in range(2)]

	def load_model(self, dirname="pytorch", name="best"):
		filepath = get_checkpoint_path(dirname, name)
		if os.path.exists(filepath):
			self.load_state_dict({k.replace("_l0",""):v for k,v in torch.load(filepath, map_location=self.device).items()})
			print(f"Loaded MDRNNCell model at {filepath}")
		return self

def get_checkpoint_path(self, dirname="pytorch", name="checkpoint", net=None):
	net_path = os.path.join("./logging/saved_models", self.name if net is None else net, dirname)
	filepath = os.path.join(net_path, f"{name}.pth")
	return filepath, net_path