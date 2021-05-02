import os
import torch
import random
import numpy as np

class Conv(torch.nn.Module):
	def __init__(self, state_size, output_size):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(state_size[-1], 32, kernel_size=4, stride=2)
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)
		self.linear1 = torch.nn.Linear(self.get_conv_output(state_size), output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		out_dims = state.size()[:-3]
		state = state.view(-1, *state.size()[-3:])
		state = self.conv1(state).tanh()
		state = self.conv2(state).tanh() 
		state = self.conv3(state).tanh() 
		state = self.conv4(state).tanh() 
		state = state.view(state.size(0), -1)
		state = self.linear1(state).tanh()
		state = state.view(*out_dims, -1)
		return state

	def get_conv_output(self, state_size):
		inputs = torch.randn(1, state_size[-1], *state_size[:-1])
		output = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
		return np.prod(output.size())

class Linear(torch.nn.Module):
	def __init__(self, input_size, output_size, nlayers=4):
		super().__init__()
		sizes = np.linspace(input_size, output_size, nlayers).astype(np.int32)
		self.layers = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
		self.output = torch.nn.Linear(sizes[-1], output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x).relu()
		output = self.output(x)
		return output

class NoisyLinear(torch.nn.Linear):
	def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
		super().__init__(in_features, out_features, bias=bias)
		self.sigma_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
		self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
		if bias:
			self.sigma_bias = torch.nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
			self.register_buffer("epsilon_bias", torch.zeros(out_features))
		self.reset_parameters()

	def reset_parameters(self):
		std = math.sqrt(3 / self.in_features)
		torch.nn.init.uniform_(self.weight, -std, std)
		torch.nn.init.uniform_(self.bias, -std, std)

	def forward(self, input):
		torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
		bias = self.bias
		if bias is not None:
			torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
			bias = bias + self.sigma_bias * torch.autograd.Variable(self.epsilon_bias)
		weight = self.weight + self.sigma_weight * torch.autograd.Variable(self.epsilon_weight)
		return torch.nn.functional.linear(input, weight, bias)

class PositionalEncoding(torch.nn.Module):
	def __init__(self, input_shape):
		super().__init__()
		self.d_model, self.seq_len = input_shape
		encoding = torch.zeros(1, self.seq_len, self.d_model)
		for pos in range(self.seq_len):
			for i in range(0, self.d_model):
				function = np.sin if i%2==0 else np.cos
				encoding[0,pos,i] = function(pos/(10000**(2*i/self.d_model)))
		self.register_buffer("encoding", encoding)

	def forward(self, inputs):
		encoding = torch.autograd.Variable(self.encoding, requires_grad=False).to(inputs.device)
		outputs = inputs + encoding / np.sqrt(self.d_model)
		return outputs

class MultiHeadedAttention(torch.nn.Module):
	def __init__(self, d_model, n_heads=8):
		super().__init__()
		self.q_linear = torch.nn.Linear(d_model, d_model)
		self.k_linear = torch.nn.Linear(d_model, d_model)
		self.v_linear = torch.nn.Linear(d_model, d_model)
		self.out = torch.nn.Linear(d_model, d_model)
		self.d_model = d_model
		self.n_heads = n_heads

	def forward(self, q, k, v):
		q = self.q_linear(q).view(q.shape[0], -1, self.n_heads, self.d_model//self.n_heads).transpose(1,2)
		k = self.k_linear(k).view(k.shape[0], -1, self.n_heads, self.d_model//self.n_heads).transpose(1,2)
		v = self.v_linear(v).view(v.shape[0], -1, self.n_heads, self.d_model//self.n_heads).transpose(1,2)
		attention = self.attention(q, k, v)
		concat = attention.transpose(1,2).contiguous().view(q.shape[0], -1, self.d_model)
		out = self.out(concat)
		return out

	def attention(self, q, k, v):
		qk = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_model//self.n_heads)
		qk = qk.softmax(-1)
		qkv = torch.matmul(qk, v)
		return qkv

class AttentionLayer(torch.nn.Module):
	def __init__(self, d_model, n_heads):
		super().__init__()
		self.attention = MultiHeadedAttention(d_model, n_heads)
		self.norm1 = torch.nn.LayerNorm(d_model)
		self.feedforward = torch.nn.Linear(d_model, d_model)
		self.norm2 = torch.nn.LayerNorm(d_model)

	def forward(self, inputs):
		state = self.attention(inputs, inputs, inputs) + inputs
		state = self.norm1(state)
		state = self.feedforward(state) + state
		state = self.norm2(state)
		return state

class AttentionEncoder(torch.nn.Module):
	def __init__(self, input_shape, n_heads, nlayers=6):
		super().__init__()
		self.d_model, self.seq_len = input_shape
		self.position = PositionalEncoding(input_shape)
		self.norm = torch.nn.LayerNorm(self.d_model)
		self.layers = torch.nn.ModuleList([AttentionLayer(self.d_model, n_heads) for _ in range(nlayers)])
		self.output_size = get_output_size([self], input_shape)

	def forward(self, inputs):
		state = inputs.transpose(-2, -1)
		state = self.position(state)
		state = self.norm(state)
		for layer in self.layers:
			state = layer(state)
		state = state.transpose(-2, -1)
		return state

class TransformerEncoder(torch.nn.Module):
	def __init__(self, input_shape, n_heads, nlayers=4):
		super().__init__()
		self.d_model, self.seq_len = input_shape
		norm = torch.nn.LayerNorm(self.d_model)
		layer = torch.nn.modules.TransformerEncoderLayer(self.d_model, n_heads, dim_feedforward=256)
		self.encoder = torch.nn.modules.TransformerEncoder(layer, num_layers=nlayers, norm=norm)
		self.output_size = get_output_size([self], input_shape)

	def forward(self, inputs):
		state = inputs.transpose(-2, -1)
		state = self.encoder(state)
		state = state.transpose(-2, -1)
		return state

def gsoftmax(logits, temperature=1.0, eps=1e-20, hard=True):
	U = torch.autograd.Variable(torch.FloatTensor(logits.shape).uniform_(), requires_grad=False)
	sample = -torch.log(-torch.log(U + eps) + eps)
	glogits = logits + sample.to(logits.device)
	softmax = torch.nn.functional.softmax(glogits / temperature, dim=-1)
	if hard:
		y_hard = one_hot_from_logits(softmax)
		softmax = (y_hard - softmax).detach() + softmax
	return softmax

def one_hot_from_logits(logits):
	return (logits == logits.max(-1, keepdim=True)[0]).float().to(logits.device)

def one_hot_from_indices(indices, depth, keepdims=False):
	y_onehot = torch.zeros([*indices.shape, depth]).to(indices.device)
	y_onehot.scatter_(-1, indices.unsqueeze(-1).long(), 1)
	return y_onehot.float() if keepdims else y_onehot.squeeze(-2).float()

def get_output_size(nets, input_size):
	state = torch.randn(1, *input_size)
	for net in nets: state = net(state)
	return state.shape[1:]