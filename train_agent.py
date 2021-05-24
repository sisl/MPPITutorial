import argparse
import numpy as np
from src import utils
from src.envs import all_envs
from src.envs.wrappers import EnsembleEnv
from src.agents import all_agents, all_frameworks, all_refs, get_config
from src.agents.wrappers import ParallelAgent, RayAgent, RayEnv, RefAgent
from src.utils.multiprocess import set_rank_size, get_server, get_client
from src.utils.misc import rollout, SEED, rmdir
from src.utils.rand import RandomAgent
from src.utils.logger import Logger
np.set_printoptions(precision=3)

class GymTrainer():
	def __init__(self, make_env, agent_cls, config):
		self.envs = EnsembleEnv(make_env, config.num_envs, config.TRIAL_AT)
		self.agent = (RefAgent if config.ref else ParallelAgent)(self.envs.state_size, self.envs.action_size, agent_cls, config, gpu=not config.no_gpu)
		self.logger = Logger(self.envs, self.agent, config) if not config.trial else None
		self.checkpoint = config.env_name
		self.config = config
		self.total_rewards = []
		
	def start(self):
		if self.config.trial: self.trial(agent=self.agent.load_model(self.checkpoint))
		else: self.train()
	
	def trial(self, agent, step=0, eps=0, stats={}, time_sleep=0.0):
		rollouts = rollout(self.envs, agent, eps=eps, render=self.config.render, time_sleep=time_sleep, print_action=False)
		self.total_rewards.append(np.mean(rollouts, axis=-1))
		if self.config.trial: return print(f"Reward: {self.total_rewards[-1]} [{np.std(rollouts)}]")
		if len(self.total_rewards) % self.config.SAVE_AT==0: agent.save_model(self.checkpoint)
		if self.total_rewards[-1] >= max(self.total_rewards): agent.save_model(self.checkpoint, "best")
		if self.logger: self.logger.log(f"Step: {step:7d}, Reward: {self.total_rewards[-1]:9.3f} [{np.std(rollouts):8.3f}], Avg: {np.mean(self.total_rewards, axis=0):9.3f} ({stats.get('eps',eps):.3f})", {**stats, **{f"{k}_e":v for k,v in agent.get_stats().items()}})

	def train(self):
		states = self.envs.reset(train=True)
		for s in range(self.config.nsteps+1):
			env_actions, actions, states = self.agent.get_env_action(self.envs.env, states)
			next_states, rewards, dones, _ = self.envs.step(env_actions, train=True)
			self.agent.train(states, actions, next_states, rewards, dones)
			if s%self.config.TRIAL_AT==0: self.trial(self.agent, s, self.config.EPS_MIN, self.agent.get_stats())
			states = next_states

	def __del__(self):
		self.envs.close()

class RayTrainer():
	def __init__(self, make_env, model, config):
		import ray
		self.ray = ray
		self.envs = RayEnv({"env_name":config.env_name})
		self.agent = RayAgent(self.envs.state_size, self.envs.action_size, model, config, gpu=True)
		self.logger = Logger(self.envs, self.agent, config, **self.agent.agent.config)
		self.config = config
		self.total_rewards = []

	def start(self):
		for s in range(0, self.config.nsteps+1, self.config.TRIAL_AT):
			rollouts = self.agent.train()
			self.total_rewards.append(np.mean(rollouts))
			if self.logger: self.logger.log(f"Step: {s:7d}, Reward: {self.total_rewards[-1]:9.3f} [{np.std(rollouts):8.3f}], Avg: {np.mean(self.total_rewards, axis=0):9.3f} ({self.agent.eps:.4f})", {})
			else: print(f"Reward: {self.total_rewards[-1]}")

	def __del__(self):
		self.envs.close()
		self.ray.shutdown()

def parse_args(envs, agents, frameworks):
	parser = argparse.ArgumentParser(description="Gym Trainer")
	parser.add_argument("env_name", type=str, choices=envs, help="Name of the environment to use. Allowed values are:\n"+', '.join(envs), metavar="env_name")
	parser.add_argument("agent_name", type=str, choices=agents, help="Which RL algorithm to use as the agent. Allowed values are:\n"+', '.join(agents), metavar="agent_name")
	parser.add_argument("--framework", type=str, default=frameworks[0], choices=frameworks, help="Which framework to use for training. Allowed values are:\n"+', '.join(frameworks), metavar="framework")
	parser.add_argument("--num_envs", type=int, default=1, help="How many parallel envs to train the agent in")
	parser.add_argument("--nsteps", type=int, default=10000, help="Number of steps to train the agent")
	parser.add_argument("--ref", type=str, default=None, choices=all_refs, help="Which reference processing network to use")
	parser.add_argument("--render", action="store_true", help="Whether to render an environment rollout")
	parser.add_argument("--trial", action="store_true", help="Whether to trial run from saved checkpoint")
	parser.add_argument("--no_gpu", action="store_true", help="Whether to force using only cpu")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args(all_envs, all_agents, all_frameworks)
	make_env, agent_cls, config = get_config(args.env_name, args.agent_name, args.framework, args.render)
	config.update(**args.__dict__)
	GymTrainer(make_env=make_env, agent_cls=agent_cls, config=config).start()
