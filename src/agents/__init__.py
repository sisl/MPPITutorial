import inspect
import numpy as np
from src.envs import get_env, all_envs, env_grps
from src.envs.wrappers import get_space_size
from src.envs.car_racing import CostModel, CarRacingV1
from src.agents.rllib import rPPOAgent, rSACAgent, rDDPGAgent, rDDQNAgent
from src.agents.pytorch import PPOAgent, SACAgent, DDQNAgent, DDPGAgent, MPOAgent, MPPIAgent, MPPIController
from src.agents.pytorch.mpc.envmodel import all_envmodels, get_envmodel
from src.agents.pytorch.ref import all_refs
from src.utils.rand import RandomAgent
from src.utils.config import Config

all_models = {
	"pt": {
		"ddpg":DDPGAgent, 
		"ppo":PPOAgent, 
		"sac":SACAgent, 
		"ddqn":DDQNAgent, 
		"rand":RandomAgent,
		"mpo":MPOAgent,
		"mppi":MPPIAgent
	},
}

all_agents = list(all_models.values())[0].keys()
all_frameworks = list(all_models.keys())

all_controllers = {
	"mppi":MPPIController
}

if not None in [rPPOAgent, rSACAgent, rDDPGAgent, rDDQNAgent]:
	all_models.update({"rl": {
		"ppo":rPPOAgent, 
		"sac":rSACAgent,
		"ddpg":rDDPGAgent,
		"ddqn":rDDQNAgent
	}})

net_config = Config(
	REG_LAMBDA = 1e-6,             	# Penalty multiplier to apply for the size of the network weights
	LEARN_RATE = 0.0001,           	# Sets how much we want to update the network weights at each training step
	DISCOUNT_RATE = 0.99,			# The discount rate to use in the Bellman Equation
	ADVANTAGE_DECAY = 0.95,			# The discount factor for the cumulative GAE calculation
	INPUT_LAYER = 512,				# The number of output nodes from the first layer to Actor and Critic networks
	ACTOR_HIDDEN = 256,				# The number of nodes in the hidden layers of the Actor network
	CRITIC_HIDDEN = 1024,			# The number of nodes in the hidden layers of the Critic networks

	EPS_MAX = 1.0,                 	# The starting proportion of random to greedy actions to take
	EPS_MIN = 0.1,               	# The lower limit proportion of random to greedy actions to take
	EPS_DECAY = 0.998,             	# The rate at which eps decays from EPS_MAX to EPS_MIN
	NUM_STEPS = 500,				# The number of steps to collect experience in sequence for each GAE calculation
	MAX_BUFFER_SIZE = 1000000,    	# Sets the maximum length of the replay buffer
	REPLAY_BATCH_SIZE = 32,        	# How many experience tuples to sample from the buffer for each train step
	TARGET_UPDATE_RATE = 0.0004,   	# How frequently we want to copy the local network to the target network (for double DQNs)
	REF = Config(
		LATENT_SIZE = 64,
		SEQ_LEN = 10,
	)
)

agent_configs = {
	"ppo": net_config.clone(
		BATCH_SIZE = 32,				# Number of samples to train on for each train step
		PPO_EPOCHS = 2,					# Number of iterations to sample batches for training
		ENTROPY_WEIGHT = 0.01,			# The weight for the entropy term of the Actor loss
		CLIP_PARAM = 0.05,				# The limit of the ratio of new action probabilities to old probabilities
	),
	"ddpg": net_config.clone(
		EPS_DECAY = 0.998,             	# The rate at which eps decays from EPS_MAX to EPS_MIN
	),
	"mppi": net_config.clone(
		MAX_BUFFER_SIZE = 1000000,    	# Sets the maximum length of the replay buffer
		REPLAY_BATCH_SIZE = 32,  		# How many experience tuples to sample from the buffer for each train step
		TRAIN_EVERY = 1,   				# Number of iterations to sample batches for training
		NUM_STEPS = 20,  				# The number of steps to collect experience in sequence for each GAE calculation
		ENV_MODEL = "batchreal",
		MPC = Config(
			NSAMPLES = 100, 
			HORIZON = 20, 
			LAMBDA = 0.1,
			COV = 1,
		)
	)
}

env_agent_configs = {
	env_grps["gym_cct"]: {
		"ddpg": Config(
			EPS_DECAY = 0.98,             	# The rate at which eps decays from EPS_MAX to EPS_MIN
			LEARN_RATE = 0.0001,           	# Sets how much we want to update the network weights at each training step
			TARGET_UPDATE_RATE = 0.0004,   	# How frequently we want to copy the local network to the target network (for double DQNs)
		),
		"ddqn": Config(),
		"sac": Config(),
		"mppi": Config(),
	},
	env_grps["gym_b2d"]: {
		"ddpg": Config(
			EPS_DECAY = 0.99,             	# The rate at which eps decays from EPS_MAX to EPS_MIN
		),
		"ddqn": Config(),
		"sac": Config(),
		"mppi": Config(),
	},
	env_grps["car"]: {
		"mppi": Config(
			REWARD_MODEL = f"{inspect.getmodule(CostModel).__name__}:{CostModel.__name__}",
			DYNAMICS_SPEC = f"{inspect.getmodule(CarRacingV1).__name__}:{CarRacingV1.__name__}",
			# DYN_ENV_NAME = "CarRacing-sebring-v1",
			MAX_BUFFER_SIZE = 100000,    	# Sets the maximum length of the replay buffer
			REPLAY_BATCH_SIZE = 32,  		# How many experience tuples to sample from the buffer for each train step
			TRAIN_EVERY = 1,    			# Number of iterations to sample batches for training
			NUM_STEPS = 20,  				# The number of steps to collect experience in sequence for each GAE calculation
			ENV_MODEL = "batchreal",
			MPC = Config(
				NSAMPLES = 100, 			# The number of sample trajectories to rollout when calculating the control update
				HORIZON = 20,  			# How many time steps into the future to rollout each trajectory sample
				LAMBDA = 0.1, 				# The degree of spread over the samples when combining weights. Lower means high weights dominate low weights more
				COV = 10,					# The value for the diagonal of the cov matrix for degree of randomness in the noise samples
			)
		)
	}
}

train_config = Config(
	TRIAL_AT = 1000,					# Number of steps between each evaluation rollout
	SAVE_AT = 1, 						# Number of evaluation rollouts between each save weights
	SEED = 0,
)

env_configs = {
	None: train_config,
	env_grps["gym"]: train_config,
}

envmodel_config = Config(
	REG_LAMBDA = 1e-6,             	# Penalty multiplier to apply for the size of the network weights
	FACTOR = 0.97,
	PATIENCE = 10,
	LEARN_RATE = 0.0001,
)

envmodel_configs = {
	"dfrntl": envmodel_config.clone(
		TRANSITION_HIDDEN = 512,
		REWARD_HIDDEN = 256,
		DROPOUT = 0.5,
		BETA_DYN = 1,
		BETA_DOT = 0,
		BETA_DDOT = 0,
	),
	"mdp": envmodel_config.clone(
		TRANSITION_HIDDEN = 512,
		REWARD_HIDDEN = 256,
		DROPOUT = 0.5,
	)
}

def add_envmodel_config(config, make_env):
	envmodel = config.get("ENV_MODEL", None)
	if envmodel is None: return config
	env = make_env()
	config.state_size = get_space_size(env.observation_space)
	config.action_size = get_space_size(env.action_space)
	config.dynamics_size = getattr(env, "dynamics_size", config.state_size[-1])
	config.dynamics_norm = getattr(env, "dynamics_norm", np.ones(config.dynamics_size)).astype(np.float32)
	config.dynamics_somask = getattr(env, "dynamics_somask", np.ones(config.dynamics_size)).astype(np.float32)
	env.close()
	dyn_config = envmodel_configs.get(envmodel, envmodel_config)
	return config.update(DYN=dyn_config)

def get_envmodel_cls(config, make_env):
	config.ENV_MODEL = config.envmodel
	add_envmodel_config(config, make_env)
	envmodel_cls = None
	if config.framework == "pt": envmodel_cls = get_envmodel(config)
	return envmodel_cls

def get_config(env_name, agent_name, framework="pt", render=False):
	assert env_name in all_envs, "Env name not found"
	env_grp = [x for x in env_grps.values() if env_name in x][0]
	env_config = env_configs.get(env_grp, train_config)
	agent_cls = all_models[framework].get(agent_name, None)
	agent_config = agent_configs.get(agent_name, net_config)
	env_agent_config = env_agent_configs.get(env_grp, agent_configs).get(agent_name, agent_configs.get(agent_name, Config()))
	agent_config.merge(env_agent_config)
	env_config.merge(agent_config)
	make_env = lambda **kwargs: get_env(env_name, **kwargs)
	config = add_envmodel_config(env_config, make_env).update(env_name=env_name)
	return make_env, agent_cls, config
