from ray.rllib.agents.sac import SACTrainer
from .base import RllibAgent

class rSACAgent(RllibAgent):
	def __init__(self, state_size, action_size, config):
		ddpg_config = {
			"Q_model": {"hidden_layer_sizes":[config.INPUT_LAYER, config.ACTOR_HIDDEN, config.ACTOR_HIDDEN]},
			"policy_model": {"hidden_layer_sizes":[config.INPUT_LAYER, config.CRITIC_HIDDEN, config.CRITIC_HIDDEN]},
			"optimization": {f"{k}_learning_rate":config.LEARN_RATE for k in ["actor", "critic", "entropy"]},
			"timesteps_per_iteration": config.num_envs*config.TRIAL_AT,
			"use_state_preprocessor": len(state_size)>1,
			"grad_norm_clipping": 0.5,
			"normalize_actions": False,
			"train_batch_size": config.REPLAY_BATCH_SIZE,
			"buffer_size": config.MAX_BUFFER_SIZE,
			"n_step": 1,#config.NUM_STEPS, (not implemented on rllib)
			"tau": config.TARGET_UPDATE_RATE,
			"env_config": {"allow_discrete":True},
		}
		super().__init__(state_size, action_size, config, ddpg_config, SACTrainer)
		