import os
import re
import sys
import glob
import torch
import bisect
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from mpl_toolkits import mplot3d
# sys.path.append(os.path.abspath("./"))
from src.utils.misc import seed_all
from src.agents import all_models, get_config
from src.envs import all_envs

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/../../../data")

class RolloutCollector():
	def __init__(self, save_path):
		assert os.path.exists(save_path)
		self.save_path = save_path
		self.reset_rollout()

	def reset_rollout(self):
		self.a_rollout = []
		self.s_rollout = []
		self.r_rollout = []
		self.d_rollout = []

	def step(self, env_action, next_state, reward, done, number=None):
		self.a_rollout.append(env_action)
		self.s_rollout.append(next_state)
		self.r_rollout.append(reward)
		self.d_rollout.append(done)
		if done: self.save_rollout(number=number)

	def save_rollout(self, number=None):
		if len(self.a_rollout) + len(self.s_rollout) + len(self.r_rollout) + len(self.d_rollout) == 0: return
		if number is None: number = len([n for n in os.listdir(self.save_path)])
		a = np.array(self.a_rollout)
		s = np.array(self.s_rollout)
		r = np.array(self.r_rollout)
		d = np.array(self.d_rollout)
		path = os.path.join(self.save_path, f"rollout_{number}.npz")
		np.savez(path, actions=a, states=s, rewards=r, dones=d)
		print(f"Saving {path}, Reward: {np.sum(r):8.2f}, Len: {len(r):5d}")
		self.reset_rollout()

class RolloutDataset(torch.utils.data.Dataset): 
	def __init__(self, config, data_dir, buffer_size=1000000, train=True): 
		self._files = sorted([os.path.join(data_dir, f) for f in glob.glob(f"{data_dir}/**/*.npz", recursive=True)])
		self._files = train_test_split(self._files, train_size=config.train_prop, shuffle=True, random_state=config.SEED)[1-int(train)]
		self._cum_size = None
		self._buffer = None
		self._buffer_fnames = None
		self._buffer_index = 0
		self._buffer_size = buffer_size
		self.load_next_buffer()

	def load_next_buffer(self):
		self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
		self._buffer_index += len(self._buffer_fnames)
		self._buffer_index = self._buffer_index % len(self._files)
		self._buffer = []
		self._cum_size = [0]
		for f in self._buffer_fnames:
			with np.load(f) as data:
				self._buffer += [{k: np.copy(v) for k, v in data.items()}]
				self._cum_size += [self._cum_size[-1] + self._data_per_sequence(data['rewards'].shape[0])]

	def __len__(self):
		return self._cum_size[-1]

	def __getitem__(self, i):
		file_index = bisect.bisect(self._cum_size, i) - 1
		seq_index = i - self._cum_size[file_index]
		data = self._buffer[file_index]
		return self._get_data(data, seq_index)

	def _get_data(self, data, seq_index):
		pass

	def _data_per_sequence(self, data_length):
		pass

class RolloutSequenceDataset(RolloutDataset): 
	def __init__(self, config, data_dir, buffer_size=1000000, train=True): 
		self._seq_len = config.seq_len
		super().__init__(config, data_dir, buffer_size, train)

	def _get_data(self, data, seq_index):
		states_data = data['states'][seq_index:seq_index + self._seq_len + 1].astype(np.float32)
		states, next_states = states_data[:-1], states_data[1:]
		actions = data['actions'][seq_index:seq_index + self._seq_len].astype(np.float32)
		rewards = data['rewards'][seq_index:seq_index + self._seq_len].astype(np.float32)
		dones = data['dones'][seq_index:seq_index + self._seq_len].astype(np.float32)
		return states, actions, next_states, rewards, dones

	def _data_per_sequence(self, data_length):
		return data_length - self._seq_len + 1

def sample_single(arg):
	eps, config, data_dir = arg
	make_env, model, _ = get_config(config.env_name, config.model)
	rollout = RolloutCollector(data_dir)
	env = make_env()
	state_size = env.observation_space.shape
	action_size = [env.action_space.n] if hasattr(env.action_space, 'n') else env.action_space.shape
	agent = model(state_size, action_size, config, load=config.env_name, gpu=not config.no_gpu)
	for ep in eps:
		seed_all(env, ep)
		state = env.reset()
		total_reward = 0
		done = False
		epsilon = 0
		rollout.s_rollout.append(state)
		scale = np.random.uniform(0.5,0.8)
		while not done:
			epsilon = np.tanh(epsilon + np.random.random())
			env_action, action = agent.get_env_action(env, state, eps=scale*epsilon)
			state, reward, done, _ = env.step(env_action)
			rollout.step(action, state, reward, done, number=ep)
			total_reward += reward
	env.close()

def sample(make_env, model, config, data_dir, number=None, render=False):
	args = []
	existing = {int(re.match(r"rollout_(.*).npz",f).groups()[0]) for f in os.listdir(data_dir)}
	eps = [k for k in np.arange(config.nsamples) if k not in existing]
	groups = np.split(eps, np.arange(0, len(eps), len(eps)//config.nworkers)[1:-1]) if config.nworkers > 0 else [eps]
	for group in groups:
		arg = (group, config, data_dir)
		args.append(arg)
	if config.nworkers==0: 
		return [sample_single(arg) for arg in args]
	with Pool(config.nworkers) as p:
		p.map(sample_single, args)

def visualize(dirname):
	dirname = os.path.join(ROOT, dirname)
	if os.path.exists(dirname):
		states = []
		actions = []
		rewards = []
		files = [os.path.join(dirname, n) for n in sorted(os.listdir(dirname), key=lambda x: str(len(x))+x)]
		for i,f in list(enumerate(files)):
			with np.load(f) as data:
				states.append(data["states"])
				actions.append(data["actions"])
				rewards.append(sum(data["rewards"]))
				if i%100==0: print(i, rewards[-1], data["states"].shape, data["actions"].shape, data["rewards"].shape)

def get_data_dir(env_name, model=None):
	if model is None: model = ""
	return os.path.expanduser(os.path.join(ROOT, env_name, model))

def save_rollouts(data_dir, states, actions, next_states, rewards):
	os.makedirs(data_dir, exist_ok=True)
	dones = np.zeros_like(rewards)
	states = np.concatenate([states, next_states[-1:]])
	states, actions, rewards, dones = [x.reshape(x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]) for x in [states, actions, rewards, dones]]
	start = len([n for n in os.listdir(data_dir)])
	for i in range(states.shape[1]):
		path = os.path.join(data_dir, f"rollout_{start+i}.npz")
		np.savez(path, actions=actions[:,i], states=states[:,i], rewards=rewards[:,i,0], dones=dones[:,i,0])

def parse_args(envs, models):
	parser = argparse.ArgumentParser(description="Rollout Generator")
	parser.add_argument("env_name", type=str, choices=envs, help="Name of the environment to use. Allowed values are:\n"+', '.join(envs), metavar="env_name")
	parser.add_argument("--model", type=str, default="rand", choices=models, help="Which RL algorithm to use as the agent. Allowed values are:\n"+', '.join(models), metavar="model")
	parser.add_argument("--nsamples", type=int, default=0, help="How many rollouts to save")
	parser.add_argument("--nworkers", type=int, default=0, help="Number of workers to use to load dataloader")
	parser.add_argument("--no_gpu", action="store_true", help="Whether to use gpu for rollouts")
	parser.add_argument("--render", action="store_true", help="Whether to render rollouts")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args(all_envs, list(all_models.values())[0].keys())
	make_env, model, config = get_config(args.env_name, args.model)
	data_dir = get_data_dir(args.env_name, args.model)
	os.makedirs(data_dir, exist_ok=True)
	config.update(**args.__dict__)
	if args.nsamples > 0:
		sample(make_env, model, config, data_dir, render=args.render)
