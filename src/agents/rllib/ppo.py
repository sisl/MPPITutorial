from ray.rllib.agents.ppo import PPOTrainer
from .base import RllibAgent

class rPPOAgent(RllibAgent):
	def __init__(self, state_size, action_size, config):
		ppo_config = {
			"lambda": config.ADVANTAGE_DECAY,
			"sgd_minibatch_size": config.BATCH_SIZE,
			"num_sgd_iter": config.PPO_EPOCHS,
			"entropy_coeff": config.ENTROPY_WEIGHT,
			"clip_param": config.CLIP_PARAM,
		}
		super().__init__(state_size, action_size, config, ppo_config, PPOTrainer)
