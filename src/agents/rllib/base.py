import numpy as np
from src.utils.rand import RandomAgent
from src.utils.misc import dict_update
from src.envs.wrappers import get_space_size
from src.envs.Gym import gym

class RayEnv(gym.Wrapper):
	def __init__(self, config):
		env = gym.make(config["env_name"])
		super().__init__(env)
		self.non_discrete = isinstance(env.action_space, gym.spaces.Discrete) and not config.get("allow_discrete",True)
		if self.non_discrete: self.action_space = gym.spaces.Box(-1,1,(env.action_space.n,))
		self.state_size = get_space_size(self.observation_space)
		self.action_size = get_space_size(self.action_space)

	def step(self, action):
		if self.non_discrete: action = np.argmax(action)
		return super().step(action)

class RllibAgent(RandomAgent):
	def __init__(self, state_size, action_size, config, model_config, trainer):
		import ray
		ray.init()
		rllib_config = {
			"lr": config.LEARN_RATE,
			"gamma": config.DISCOUNT_RATE,
			"num_workers": config.num_envs, 
			"env_config": {"env_name":config.env_name, "allow_discrete":False},
			"train_batch_size": config.num_envs*config.TRIAL_AT,
			"evaluation_num_episodes": config.num_envs,
			"evaluation_num_workers": config.num_envs,
			"evaluation_config": {"explore": True},
			"evaluation_interval": 1,
		}
		self.config = dict_update(rllib_config, model_config)
		self.network = trainer(env=RayEnv, config=self.config)
		eps_max = [v for k,v in self.config["exploration_config"].items() if "initial" in k][0] if "exploration_config" in self.config else 1
		eps_min = [v for k,v in self.config["exploration_config"].items() if "final" in k][0] if "exploration_config" in self.config else 1
		eps_time = [v for k,v in self.config["exploration_config"].items() if "timesteps" in k][0] if "exploration_config" in self.config else 1
		self.get_eps = lambda t: max(eps_min + (1-(t/eps_time))*(eps_max-eps_min), eps_min)
		super().__init__(state_size, action_size)

	def get_action(self, state, eps=None, sample=True):
		return self.network.compute_action(state, explore=False)

	def train(self):
		res = self.network.train()
		self.eps = self.get_eps(res["timesteps_total"])
		return res["evaluation"]["hist_stats"]["episode_reward"]
		