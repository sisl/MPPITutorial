import os
import torch
import numpy as np
import itertools as it
from multiprocessing import Pool
try: 
	from src.envs.CarRacing.track import Track, track_name
	from src.envs.CarRacing.ref import RefDriver
	from src.envs.CarRacing.dynamics import DELTA_T
except ImportError as e: 
	from track import Track
	from ref import RefDriver
	from dynamics import DELTA_T

USE_TEMP = False

def extract_track_name(env_name):
	fields = env_name.split("-")
	names = fields[1:-1]
	return names[0] if names else track_name

class CostModel():
	def __init__(self, track=None, ref=None, max_time=np.inf, delta_t=DELTA_T, config=None):
		self.track = track if track else Track(extract_track_name(config.env_name)) if config else Track()
		self.ref = RefDriver(self.track.track_name) if ref is None else ref
		self.max_time = max_time if ref is None else ref.max_time
		self.vmin, self.vmax = (25,70) if self.track.track_name=="sebring" else (5,35)
		self.delta_t = delta_t

	def __call__(self, action, state, next_state):
		cost = self.get_cost(next_state, state)
		return -cost

	def get_cost(self, state, prevstate=None, times=None, temp=USE_TEMP):
		if prevstate is None: prevstate = state
		if times is not None: times = times*self.delta_t
		cost = self.get_temporal_cost(state, prevstate, times) if temp else self.get_ref_cost(state, times)
		return cost

	def get_temporal_cost(self, state, prevstate, realtime=None):
		prevpos = np.stack([prevstate.X, prevstate.Y], -1)
		pos = np.stack([state.X, state.Y], -1)
		progress = self.track.get_progress(prevpos, pos)
		dist = self.get_point_cost(pos, transform=False)
		# reward = progress - np.tanh(dist/10)**2 + np.tanh(state.Vx/self.vmin) - np.power(1-state.Vx/self.vmin,2)
		# reward = 2*progress + 9-(dist/10)**2 - np.logical_or(state.Vx<self.vmin, state.Vx>self.vmax) + 1-(state.Vx/self.vmin)*np.abs(state.δ)
		reward = np.tanh(progress) + 1-np.tanh(dist/20)**2 + 1-(state.Vx/self.vmin)*np.abs(state.δ) + 1-np.logical_or(state.Vx<self.vmin, state.Vx>self.vmax)
		return -reward

	def get_point_cost(self, pos, transform=True):
		idx, dist = self.track.get_nearest(pos) 
		return np.tanh(dist/20)**2 if transform else dist

	def get_ref_cost(self, state, realtime=None):
		pos = np.stack([state.X, state.Y], -1)
		yaw = state.ψ
		vel = np.sqrt(state.Vx**2 + state.Vy**2)
		reftime = self.ref.get_time(pos, state.S)
		reftime = np.minimum(reftime, self.ref.max_time-self.delta_t)
		refposstate = self.ref.state(reftime)
		refpos = np.stack([refposstate.X, refposstate.Y], -1)
		refyaw = refposstate.ψ
		refvel = np.sqrt(refposstate.Vx**2 + refposstate.Vy**2)
		yawdiff = np.arctan2(np.sin(refyaw-yaw), np.cos(refyaw-yaw))
		refdist = np.linalg.norm(refpos-pos, axis=-1)
		veldiff = vel-refvel
		cost = (refdist/10)**2 + (veldiff/refvel)**2 + np.abs(yawdiff/np.pi)
		# cost = (refdist/20)**2 + (veldiff/10)**2 + np.abs(yawdiff/np.pi)
		if realtime is not None:
			realtime = np.minimum(realtime, self.ref.max_time-self.delta_t)
			timediff = np.minimum(np.abs(realtime - reftime), 2)
			# timediff = np.tanh(np.abs(realtime - reftime))
			cost += timediff
		return cost

	def get_reftemp_cost(self, state, prevstate, realtime=None):
		times = state.realtime if realtime is None else realtime
		prevtimes = prevstate.realtime if realtime is None else realtime-self.delta_t
		refcost = self.get_ref_cost(state, times)
		prevrefcost = self.get_ref_cost(prevstate, prevtimes)
		reward = (prevrefcost-refcost) - refcost + 2
		return -reward