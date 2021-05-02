import numpy as np
from ray.rllib.agents.ddpg import DDPGTrainer
from .base import RllibAgent, gym

class rDDPGAgent(RllibAgent):
	def __init__(self, state_size, action_size, config):
		ep_length = gym.envs.registration.registry.env_specs[config.env_name].max_episode_steps
		eps_steps = ep_length*np.log(config.EPS_MIN/config.EPS_MAX)/np.log(config.EPS_DECAY)*config.num_envs
		ddpg_config = {
			"critic_hiddens": [config.INPUT_LAYER, config.CRITIC_HIDDEN, config.CRITIC_HIDDEN],
			"actor_hiddens": [config.INPUT_LAYER, config.ACTOR_HIDDEN, config.ACTOR_HIDDEN],
			"timesteps_per_iteration": config.num_envs*config.TRIAL_AT,
			"use_state_preprocessor": len(state_size)>1,
			"prioritized_replay": False,
			"train_batch_size": config.REPLAY_BATCH_SIZE,
			"buffer_size": config.MAX_BUFFER_SIZE,
			"critic_lr": config.LEARN_RATE,
			"actor_lr": config.LEARN_RATE,
			"n_step": config.NUM_STEPS,
			"tau": config.TARGET_UPDATE_RATE,
			"l2_reg": config.REG_LAMBDA,
			"env_config": {"allow_discrete":False},
			"exploration_config": {"initial_scale": config.EPS_MAX, "final_scale": config.EPS_MIN, "scale_timesteps": eps_steps}
		}
		super().__init__(state_size, action_size, config, ddpg_config, DDPGTrainer)
		