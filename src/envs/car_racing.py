import os
import sys
import torch
import inspect
import numpy as np
import itertools as it
import pyquaternion as pyq
import importlib.util as il
from collections import OrderedDict
from src.envs.CarRacing.dynamics import X1Dynamics, CarState, DELTA_T, USE_DELTA, extract_pos
from src.envs.CarRacing.viewer import PathAnimator
from src.envs.CarRacing.track import Track, track_name
from src.envs.CarRacing.cost import CostModel, USE_TEMP
from src.envs.CarRacing.ref import RefDriver, rotate_path
from src.envs.Gym import gym
np.set_printoptions(precision=3, sign=" ")

tracks = {"curve": ["curve1","curve2","curve3","curve4","curve5"],"cubic": ["cubic1","cubic2","cubic3","cubic4","cubic5"]}

def extract_track_name(env_name):
	fields = env_name.split("-")
	names = fields[1:-1]
	return names[0] if names else track_name

def extract_tire_name(env_name):
	fields = env_name.split("-")
	names = fields[1:-1]
	return names[1] if len(names)>1 else None

class EnvMeta(type):
	def __new__(meta, name, bases, class_dict):
		cls = super().__new__(meta, name, bases, class_dict)
		gym.register(cls.name, entry_point=cls)
		for track in os.listdir(os.path.join(os.path.dirname(__file__), "CarRacing", "spec", "tracks")):
			track_name = os.path.splitext(track)[0]
			fields = cls.name.split("-")
			fields.insert(1, track_name)
			gym.register("-".join(fields), entry_point=lambda: cls("-".join(fields)))
			fields.insert(2, "")
		return cls

class CarRacingV1(gym.Env, metaclass=EnvMeta):
	name = "CarRacing-v1"
	dynamics_norm = np.concatenate([X1Dynamics.dynamics_norm, [100]])
	dynamics_somask = np.concatenate([X1Dynamics.dynamics_somask, [0]])
	def __init__(self, env_name="CarRacing-v1", max_time=None, delta_t=DELTA_T, withtrack=False):
		self.track_name = extract_track_name(env_name)
		self.tire_name = extract_tire_name(env_name)
		self.dynamics = X1Dynamics()
		self.withtrack = withtrack
		self.use_delta = False
		self.init_track(self.track_name, max_time, delta_t)
		self.action_space = self.dynamics.action_space
		self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.reset(sample_track=False).shape)
		self.spec = gym.envs.registration.EnvSpec(self.name, max_episode_steps=int(self.max_time/self.delta_t))

	def init_track(self, track_name, max_time=None, delta_t=DELTA_T):
		self.ref = RefDriver(track_name)
		self.track = Track(track_name)
		self.delta_t = delta_t
		self.max_time = self.ref.max_time if max_time is None else max_time
		self.cost_model = CostModel(self.track, self.ref, self.max_time, self.delta_t)

	def reset(self, train=True, sample_track=False):
		if sample_track: self.init_track(np.random.choice(tracks.get(self.track_name,[self.track_name])))
		self.time = 0
		self.realtime = 0.0
		self.action = np.zeros(self.action_space.shape)
		self.dynamics.reset(self.ref.start_pos, self.ref.start_vel)
		self.state_spec, state = self.observation()
		self.info = {"ref":{}, "car":{}}
		self.done = False
		return state

	def step(self, action, device=None, info=True, temp=USE_TEMP):
		self.time += 1
		self.realtime = self.time * self.delta_t
		success = self.dynamics.step(action, dt=self.delta_t, use_delta=self.use_delta)
		next_state_spec, next_state = self.observation()
		pos = np.stack([next_state_spec.X, next_state_spec.Y], -1)
		reward = -self.cost_model.get_cost(next_state_spec, self.state_spec, self.time, temp)
		ind, trackdist = self.track.get_nearest(pos)
		done = np.logical_or(not success, self.done)
		done = np.logical_or(trackdist > 40.0, done)
		if temp: 
			done = np.logical_or(self.ref.get_time(pos, next_state_spec.S) >= self.max_time, done)
			done = np.logical_or(next_state_spec.Vx < 4.0, done)
		self.done = np.logical_or(self.realtime >= self.max_time, done)
		self.info = self.get_info(reward, action) if info else {"ref":{}, "car":{}}
		self.state_spec = next_state_spec
		return next_state, reward, self.done, self.info

	def render(self, mode="human", **kwargs):
		if not hasattr(self, "viewer"): self.viewer = PathAnimator(interactive=mode!="video", dt=self.delta_t)
		ref_spec = self.ref.state(self.realtime)
		pos = np.stack([self.state_spec.X, self.state_spec.Y], -1)
		refpos = np.stack([ref_spec.X, ref_spec.Y], -1)
		car = np.stack([pos, pos + np.array([np.cos(self.state_spec.ψ ), np.sin(self.state_spec.ψ)])])
		ref = np.stack([refpos, refpos + np.array([np.cos(ref_spec.ψ), np.sin(ref_spec.ψ)])])
		if self.withtrack:
			state = self.observation()[1][...,:self.dynamics_size]*self.dynamics_norm
			relrefs = self.ref.get_state_sequence(state, 10, CarRacingV1.observation_spec, dt=self.delta_t)
			# path = extract_pos(relrefs)
			path = self.track.get_state_sequence(state, 10, CarRacingV1.observation_spec)
			kwargs["path"] = path + pos
		return self.viewer.animate_path(self.track, pos, [car, ref], info=self.info, **kwargs)

	def observation(self, carstate=None):
		dyn_state = self.dynamics.observation(carstate)
		realtime = np.expand_dims(self.realtime, axis=-1)
		state = np.concatenate([dyn_state, realtime], axis=-1)
		self.dynamics_size = state.shape[-1]
		spec = self.observation_spec(state)
		if self.withtrack:
			refpath = self.ref.get_state_sequence(state, 10, CarRacingV1.observation_spec, dt=self.delta_t)
			veldiff = refpath[...,3]
			yawdiff = refpath[...,2]
			timediff = refpath[...,-1]
			# path = extract_pos(refpath)
			path = self.track.get_state_sequence(state, 10, CarRacingV1.observation_spec)
			path = np.reshape(path, [*path.shape[:-2], -1]) 
			state = np.concatenate([state/self.dynamics_norm, path], -1)
			# state = np.concatenate([state/self.dynamics_norm, path, veldiff, yawdiff, timediff], -1)
		return spec, state

	@staticmethod
	def observation_spec(observation):
		dyn_state = observation[...,:-1]
		dyn_spec = CarState.observation_spec(dyn_state)
		realtime = observation[...,-1]
		dyn_spec.realtime = realtime
		return dyn_spec

	def set_state(self, state, device=None, times=None):
		if isinstance(state, torch.Tensor): state = state.cpu().numpy()
		dyn_state = state[...,:-1]
		self.dynamics.set_state(dyn_state, device=device)
		self.realtime = state[...,-1] if times is None else times*self.delta_t 
		self.state_spec = self.observation()[0]
		self.time = self.realtime / self.delta_t

	def get_info(self, reward, action):
		dynspec = self.dynamics.state
		refspec = self.ref.state(self.realtime)
		refaction = self.ref.action(self.realtime, self.delta_t)
		reftime = self.ref.get_time(np.stack([dynspec.X, dynspec.Y], -1), dynspec.S)
		carinfo = info_stats(dynspec, reftime, reward, action)
		refinfo = info_stats(refspec, self.realtime, 0, refaction)
		info = {"ref": refinfo, "car": carinfo}
		return info

	def close(self, path=None):
		if hasattr(self, "viewer"): self.viewer.close(path)
		self.closed = True

def info_stats(state, realtime, reward, action):
	turn_rate = action[...,0]
	pedal_rate = action[...,1]
	info = {
		"Time": f"{realtime:7.2f}",
		"Pos": f"{{'X':{justify(state.X)}, 'Y':{justify(state.Y)}}}",
		"Vel": f"{{'X':{justify(state.Vx)}, 'Y':{justify(state.Vy)}}}",
		"Speed": np.round(state.info.get("V",0), 4),
		"Dist": np.round(state.S, 4),
		"Yaw angle": np.round(state.ψ, 4),
		"Yaw vel": np.round(state.ψ̇, 4),
		"Beta": np.round(state.info.get("β",0), 4),
		"Alpha": f"{{'F':{justify(state.info.get('αf',0))}, 'R':{justify(state.info.get('αr',0))}}}",
		"Fz": f"{{'F':{justify(state.info.get('FzF',0))}, 'R':{justify(state.info.get('FzR',0))}}}",
		"Fy": f"{{'F':{justify(state.info.get('FyF',0))}, 'R':{justify(state.info.get('FyR',0))}}}",
		"Fx": f"{{'F':{justify(state.info.get('FxF',0))}, 'R':{justify(state.info.get('FxR',0))}}}",
		"Steer angle": np.round(state.δ, 4),
		"Pedals": np.round(state.pedals, 4),
		"Reward": np.round(reward, 4),
		"Action": f"{{'Trn':{justify(turn_rate)}, 'ped':{justify(pedal_rate)}}}"
	}
	return info

def justify(num): return str(np.round(num, 3)).rjust(10,' ')
