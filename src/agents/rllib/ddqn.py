import numpy as np
from ray.rllib.agents.dqn import DQNTrainer
from .base import RllibAgent, gym

class rDDQNAgent(RllibAgent):
	def __init__(self, state_size, action_size, config):
		ep_length = gym.envs.registration.registry.env_specs[config.env_name].max_episode_steps
		eps_steps = ep_length*np.log(config.EPS_MIN/config.EPS_MAX)/np.log(config.EPS_DECAY)*config.num_envs
		ddpg_config = {
			"hiddens": [config.INPUT_LAYER, config.CRITIC_HIDDEN, config.CRITIC_HIDDEN],
			"timesteps_per_iteration": config.num_envs*config.TRIAL_AT,
			"prioritized_replay": False,
			"train_batch_size": config.REPLAY_BATCH_SIZE,
			"buffer_size": config.MAX_BUFFER_SIZE,
			"n_step": config.NUM_STEPS,
			"target_network_update_freq": 1/config.TARGET_UPDATE_RATE,
			"env_config": {"allow_discrete":True},
			"exploration_config": {"initial_epsilon": config.EPS_MAX, "final_epsilon": config.EPS_MIN, "epsilon_timesteps": eps_steps}
		}
		super().__init__(state_size, action_size, config, ddpg_config, DQNTrainer)
