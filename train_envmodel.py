import os
import sys
import tqdm
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.envs import all_envs
from src.agents import all_envmodels, all_agents, all_frameworks, get_config, get_envmodel_cls
from src.utils.data import RolloutSequenceDataset, get_data_dir
from src.utils.logger import Logger

class EnvmodelTrainer():
	def __init__(self, make_env, envmodel_cls, config):
		self.data_dir = get_data_dir(config.env_name)
		self.dataset_train = RolloutSequenceDataset(config, self.data_dir, train=True)
		self.dataset_test = RolloutSequenceDataset(config, self.data_dir, train=False)
		self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.nworkers)
		self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.nworkers)
		self.envmodel = envmodel_cls(config.state_size, config.action_size, config, gpu=True)
		self.logger = Logger(make_env(), self.envmodel, config)
		self.config = config

	def start(self):
		ep_train_losses = []
		ep_test_losses = []
		for ep in range(self.config.epochs):
			train_loss = self.train_loop(ep)
			test_loss = self.test_loop(ep)
			ep_train_losses.append(train_loss)
			ep_test_losses.append(test_loss)
			self.envmodel.schedule(test_loss)
			self.envmodel.save_model(self.config.env_name)
			if ep_test_losses[-1] <= np.min(ep_test_losses): self.envmodel.save_model(self.config.env_name, "best")
			self.logger.log(f"Ep: {ep:7d}, Reward: {ep_test_losses[-1]:9.3f} [{ep_train_losses[-1]:8.3f}], Avg: {np.mean(ep_test_losses, axis=0):9.3f} ({1.0:.3f})", self.envmodel.get_stats())

	def train_loop(self, ep, update=1):
		batch_losses = []
		self.envmodel.train()
		with tqdm.tqdm(total=len(self.dataset_train)) as pbar:
			pbar.set_description_str(f"Train Ep: {ep}, ")
			for i,data in enumerate(self.train_loader):
				states, actions, next_states, rewards, dones = map(lambda x: x.transpose(0,1), data)
				rloss, dloss = self.envmodel.optimize(states, actions, next_states, rewards, dones)
				if i%update == 0:
					pbar.set_postfix_str(f"Loss: {rloss+dloss:.4f}")
					pbar.update(states.shape[1]*update)
				batch_losses.append(rloss+dloss)
		return np.mean(batch_losses)

	def test_loop(self, ep):
		batch_losses = []
		self.envmodel.eval()
		with torch.no_grad():
			for data in self.test_loader:
				states, actions, next_states, rewards, dones = map(lambda x: x.transpose(0,1), data)
				rloss, dloss = self.envmodel.get_loss(states, actions, next_states, rewards, dones)
				batch_losses.append(rloss.item()+dloss.item())
		return np.mean(batch_losses)

def parse_args(envs, agents, envmodels, frameworks):
	parser = argparse.ArgumentParser(description="Envmodel Trainer")
	parser.add_argument("env_name", type=str, choices=envs, help="Name of the environment to use. Allowed values are:\n"+', '.join(envs), metavar="env_name")
	parser.add_argument("envmodel", type=str, choices=envmodels, help="Which model to use as the dynamics. Allowed values are:\n"+', '.join(envmodels), metavar="envmodels")
	parser.add_argument("--framework", type=str, default=frameworks[0], choices=frameworks, help="Which framework to use for training. Allowed values are:\n"+', '.join(frameworks), metavar="framework")
	parser.add_argument("--model", type=str, default=None, choices=agents, help="Which RL algorithm to use as the agent. Allowed values are:\n"+', '.join(agents), metavar="model")
	parser.add_argument("--nworkers", type=int, default=0, help="Number of workers to use to load dataloader")
	parser.add_argument("--epochs", type=int, default=250, help="Number of epochs to train the envmodel")
	parser.add_argument("--seq_len", type=int, default=150, help="Length of sequence to train RNN")
	parser.add_argument("--batch_size", type=int, default=256, help="Size of batch to train RNN")
	parser.add_argument("--train_prop", type=float, default=0.8, help="Proportion of trajectories to use for training")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args(all_envs, all_agents, all_envmodels, all_frameworks)
	make_env, agent_cls, config = get_config(args.env_name, args.model)
	config.update(**args.__dict__)
	envmodel_cls = get_envmodel_cls(config, make_env)
	EnvmodelTrainer(make_env, envmodel_cls, config=config).start()
		