import os
import torch
import numpy as np
from src.agents.pytorch.rl.base import PTNetwork
from src.agents.pytorch.network import get_output_size, AttentionEncoder

class RefLinear(torch.nn.Module):
	def __init__(self, state_size, output_size, config):
		super().__init__()
		self.flatten = torch.nn.Flatten(-2)
		self.output = torch.nn.Linear(np.prod(state_size), output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, states):
		flat = self.flatten(states)
		output = self.output(flat)
		return output

class RefRNN(torch.nn.Module):
	def __init__(self, state_size, output_size, config):
		super().__init__()
		self.rnn = torch.nn.LSTM(state_size[-1], config.REF.LATENT_SIZE, batch_first=True)
		self.state = torch.nn.Linear(state_size[-1], config.REF.LATENT_SIZE)
		self.output = torch.nn.Linear(2*config.REF.LATENT_SIZE, output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, states):
		refstates, state = states, states[...,0,:]
		batch = refstates.shape[:-2]
		recurrent, (h, c) = self.rnn(refstates.view(np.prod(batch), *refstates.shape[-2:]))
		final = h.view(*batch, *h.shape[-1:])
		state = self.state(state)
		netstate = torch.cat([state, final], -1)
		output = self.output(netstate)
		return output

class RefCNN(torch.nn.Module):
	def __init__(self, state_size, output_size, config):
		super().__init__()
		self.conv1 = torch.nn.Conv1d(state_size[-1], config.REF.LATENT_SIZE//2, 5)
		self.conv2 = torch.nn.Conv1d(get_output_size([self.conv1], state_size[::-1])[0], config.REF.LATENT_SIZE//4, 5)
		self.flatten = torch.nn.Flatten(-2)
		self.linear = torch.nn.Linear(get_output_size([self.conv1, self.conv2, self.flatten], state_size[::-1])[-1], config.REF.LATENT_SIZE)
		self.state = torch.nn.Linear(state_size[-1], config.REF.LATENT_SIZE)
		self.output = torch.nn.Linear(2*config.REF.LATENT_SIZE, output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, states):
		refstates, state = states, states[...,0,:]
		batch = refstates.shape[:-2]
		refstates = refstates.transpose(-1, -2)
		refstates = refstates.view(np.prod(batch), *refstates.shape[-2:])
		conv1 = self.conv1(refstates)
		conv2 = self.conv2(conv1)
		conv = conv2.view(*batch, *conv2.shape[-2:])
		conv = conv.transpose(-1, -2)
		flat = self.flatten(conv)
		linear = self.linear(flat)
		state = self.state(state)
		netstate = torch.cat([state, linear], -1)
		output = self.output(netstate)
		return output

class RefATTN(torch.nn.Module):
	def __init__(self, state_size, output_size, config):
		super().__init__()
		self.attn = AttentionEncoder([state_size[-1], state_size[0]], 1)
		self.flatten = torch.nn.Flatten(-2)
		self.linear = torch.nn.Linear(get_output_size([self.flatten], self.attn.output_size)[-1], config.REF.LATENT_SIZE)
		self.state = torch.nn.Linear(state_size[-1], config.REF.LATENT_SIZE)
		self.output = torch.nn.Linear(2*config.REF.LATENT_SIZE, output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, states):
		refstates, state = states, states[...,0,:]
		batch = refstates.shape[:-2]
		refstates = refstates.transpose(-1, -2)
		refstates = refstates.view(np.prod(batch), *refstates.shape[-2:])
		attn = self.attn(refstates).view(*batch, *self.attn.output_size)
		attn = attn.transpose(-1, -2)
		flat = self.flatten(attn)
		linear = self.linear(flat)
		state = self.state(state)
		netstate = torch.cat([state, linear], -1)
		output = self.output(netstate)
		return output

all_refs = {
	"linear": RefLinear,
	"rnn": RefRNN,
	"cnn": RefCNN,
	"attn": RefATTN,
}

def get_ref(state_size, action_size, config):
	ref_cls = all_refs[config.ref]
	return ref_cls(state_size, action_size, config)
