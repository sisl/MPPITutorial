import os
import argparse
import numpy as np
try: import keyboard as kbd
except: print("No module keyboard")
from src.agents import get_config, all_refs, all_agents, all_envmodels
from src.agents.wrappers import ParallelAgent, RefAgent
from src.envs import all_envs
from src.envs.wrappers import EnsembleEnv, EnvManager, EnvWorker
from src.utils.data import get_data_dir, save_rollouts
from src.utils.rand import RandomAgent
from src.utils.logger import Logger
from src.utils.config import Config
from src.utils.misc import rollout
np.set_printoptions(precision=3, sign=" ")
	
class InputController(RandomAgent):
	def __init__(self, state_size, action_size, config=None, **kwargs):
		self.state_size = state_size
		self.action_size = action_size

	def get_action(self, state, eps=0.0, sample=False):
		shape = state.shape[:-len(self.state_size)]
		action = np.zeros([*shape, *self.action_size])
		try:
			if kbd.is_pressed("left"):
				action[...,0] += 1
			if kbd.is_pressed("right"):
				action[...,0] -= 1
			if kbd.is_pressed(kbd.KEY_UP):
				action[...,1] += 1
			if kbd.is_pressed(kbd.KEY_DOWN):
				action[...,1] -= 1
		except Exception as e:
			print(e)
		return action

def test_mppi(args, nsteps, log=False, save_video=False, show_trajectories=True):
	model = "mppi"
	env_name = "CarRacing-curve-v1"
	make_env, agent_cls, config = get_config(env_name, model, "pt", args.envmodel)
	envs = EnsembleEnv(lambda: make_env(withtrack=False), 0)
	load_env = config.get("DYN_ENV_NAME", env_name)
	agent = ParallelAgent(envs.state_size, envs.action_size, agent_cls, config, gpu=True).load_model(load_env)
	logger = Logger(envs, agent, config, log_type="runs")
	state = envs.reset()
	total_reward = None
	done = False
	for step in range(0,nsteps):
		env_action, action, state = agent.get_env_action(envs.env, state, 0.0)
		state, reward, done, info = envs.step(env_action)
		spec = envs.env.dynamics.observation_spec(state[0])
		total_reward = reward  if total_reward is None else total_reward + reward
		log_string = f"Step: {step:8d}, Reward: {f'{reward[0]:5.3f}'.rjust(8,' ')}, Action: {np.array2string(env_action[0], separator=',')}, Done: {done[0]}"
		logger.log(log_string, info[0]) if log else print(log_string)
		trajectory_pos = None
		if show_trajectories and hasattr(agent.agent.network, "states"):
			if args.sample: save_rollouts(get_data_dir(env_name, model), agent.agent.network.states, agent.agent.network.actions, agent.agent.network.next_states, agent.agent.network.rewards)
			samples = np.squeeze(agent.agent.network.states)
			state_spec = envs.env.dynamics.observation_spec(samples) if hasattr(envs.env, "dynamics") else None
			trajectory_pos = np.stack([state_spec.X.T, state_spec.Y.T],-1)
		envs.render(mode="video" if save_video else "human", trajectories=trajectory_pos)
		if done: break
	print(f"Reward: {total_reward}")
	envs.close(path=logger.log_path.replace("runs","videos").replace(".txt",".mp4"))

def test_rl(args, nsteps, log=False, save_video=False):
	make_env, agent_cls, config = get_config(args.env_name, args.agent_name, "pt")
	config.update(**args.__dict__)
	envs = EnsembleEnv(make_env, 0)
	checkpoint = args.env_name
	agent = (RefAgent if config.ref else ParallelAgent)(envs.state_size, envs.action_size, agent_cls, config, gpu=True).load_model(checkpoint, "checkpoint")
	logger = Logger(envs, agent, config, log_type="runs") if log else None
	state = envs.reset(sample_track=False)
	total_reward = None
	done = False
	for step in range(0,nsteps):
		env_action, action, state = agent.get_env_action(envs.env, state, 0.0)
		state, reward, done, info = envs.step(env_action)
		total_reward = reward[0] if total_reward is None else total_reward + reward[0]
		log_string = f"Step: {step:8d}, Reward: {f'{reward[0]:5.3f}'.rjust(8,' ')}, Action: {np.array2string(env_action[0], separator=',')}, Done: {done[0]}"
		logger.log(log_string, info[0]) if logger else print(log_string)
		envs.render(mode="video" if save_video else "human")
		if done: break
	print(f"Reward: {total_reward}")
	envs.close(path=logger.log_path.replace("runs","videos").replace(".txt",".mp4") if logger else None)

def test_input(args, nsteps, log=False, save_video=False, new_track=None):
	make_env, agent_cls, config = get_config("CarRacing-curve-v1", "rand", "pt")
	envs = EnsembleEnv(make_env, 0)
	agent = InputController(envs.state_size, envs.action_size, config, gpu=True)
	state = envs.reset()
	total_reward = None
	done = False
	for step in range(0,nsteps):
		env_action, action = agent.get_env_action(envs.env, state, 0.0)
		state, reward, done, info = envs.step(env_action)
		spec = envs.env.dynamics.observation_spec(state[0]) if hasattr(envs.env, "dynamics") else None
		props = spec.print() if spec is not None else ""
		total_reward = reward if total_reward is None else total_reward + reward
		log_string = f"Step: {step:8d}, Reward: {reward[0]:5.3f}, Action: {np.array2string(env_action[0], separator=',')}, Done: {done[0]}, {props}"
		if log: log_ref(envs.env.track.track_name, spec, step*envs.env.delta_t)
		if new_track: log_track(new_track, spec, step*envs.env.delta_t)
		if done: break#envs.reset()
		print(log_string)
		envs.render()
	print(f"Reward: {total_reward}")
	envs.close()

def log_ref(ref_name, spec=None, time=-1):
	with open(os.path.join(os.path.dirname(__file__), "src", "envs", "CarRacing", "spec", "refs", f"{ref_name}.csv"), "w" if time==0 else "a+") as ref:
		time = np.round(time, 2)
		if time==0.0: 
			ref.write("Time,Dist,Curv,PathX,PathY,Speed,PathHeading,LatGap,YawDev,LongAcc,LatAcc,SideSlip,Vx,Vy,YawVel,SlipAngleFront,SlipAngleRear,LatForceFront,LatForceRear,LongForceFront,LongForceRear,VertForceFront,VertForceRear,Steer,Throttle,Brake,Gear,Clutch\n")
			ref.write("sec,m,1/m,m,m,m/s,rad,m,deg,g,g,deg,m/s,m/s,rad/s,deg,deg,N,N,N,N,N,N,deg,%,N,#,0-1\n")
		Dist = np.round(spec.S*1000, 4)
		Curv = 0.0
		PathX = np.round(spec.X, 4)
		PathY = np.round(spec.Y, 4)
		Speed = np.round(np.sqrt(spec.Vx**2 + spec.Vy**2), 4)
		PathHeading = np.round(spec.ψ, 4)
		LatGap = -1
		YawDev = -1
		LongAcc = -1
		LatAcc = -1
		SideSlip = np.round(spec.β, 4)
		Vx = np.round(spec.Vx, 4)
		Vy = np.round(spec.Vy, 4)
		YawVel = np.round(spec.ψ̇ , 4)
		SlipAngleFront = np.round(spec.αf, 4)
		SlipAngleRear = np.round(spec.αr, 4)
		LatForceFront = np.round(spec.FyF*1000, 4)
		LatForceRear = np.round(spec.FyR*1000, 4)
		LongForceFront = np.round(spec.FxF*1000, 4)
		LongForceRear = np.round(spec.FxR*1000, 4)
		VertForceFront = np.round(spec.FzF*1000, 4)
		VertForceRear = np.round(spec.FzR*1000, 4)
		Steer = np.round(spec.δ, 4)
		Throttle = np.maximum(spec.pedals, 0)
		Brake = np.minimum(spec.pedals, 0)
		Gear = 0
		Clutch = 0
		ref.write(f"{time},{Dist},{Curv},{PathX},{PathY},{Speed},{PathHeading},{LatGap},{YawDev},{LongAcc},{LatAcc},{SideSlip},{Vx},{Vy},{YawVel},{SlipAngleFront},{SlipAngleRear},{LatForceFront},{LatForceRear},{LongForceFront},{LongForceRear},{VertForceFront},{VertForceRear},{Steer},{Throttle},{Brake},{Gear},{Clutch}\n")

def log_track(track_name, spec=None, time=-1):
	log_ref(track_name, spec, time)
	with open(os.path.join(os.path.dirname(__file__), "src", "envs", "CarRacing", "spec", "tracks", f"{track_name}.csv"), "w" if time==0 else "a+") as track:
		PathX = np.round(spec.X, 4)
		PathY = np.round(spec.Y, 4)
		track.write(f"{PathX},{PathY}\n")

def parse_args(envmodels):
	parser = argparse.ArgumentParser(description="MPC Tester")
	parser.add_argument("--nsteps", type=int, default=7000, help="Number of steps to train the agent")
	parser.add_argument("--save_run", action="store_true", help="Whether to log each time step's state in a run txt file")
	parser.add_argument("--save_video", action="store_true", help="Whether to save the simulation run as video instead of rendering")
	parser.add_argument("--input", action="store_true", help="Whether to use keyboard as input")
	parser.add_argument("--sample", action="store_true", help="Whether to save the sampled mppi trajectories")
	parser.add_argument("--env_name", type=str, default="CarRacing-sebring-v1", choices=all_envs, help="Which env to use")
	parser.add_argument("--agent_name", type=str, default=None, choices=all_agents, help="Which agent network to use")
	parser.add_argument("--ref", type=str, default=None, choices=all_refs, help="Which reference processing network to use")
	parser.add_argument("--envmodel", type=str, default=None, choices=envmodels, help="Which model to use as the dynamics. Allowed values are:\n"+', '.join(envmodels), metavar="envmodels")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args(list(all_envmodels.keys()))
	function = test_input if args.input else test_rl if args.agent_name is not None else test_mppi
	function(args, args.nsteps, args.save_run, args.save_video)