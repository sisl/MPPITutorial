import os
import numpy as np
import pandas as pd
import itertools as it
from operator import itemgetter
from multiprocessing import Pool
from collections import OrderedDict
try: 
	from src.envs.CarRacing.dynamics import CarState, DELTA_T, TURN_SCALE, PEDAL_SCALE, extract_pos
	from src.envs.CarRacing.track import track_name
except ImportError as e: 
	from track import track_name
	from dynamics import CarState, DELTA_T, TURN_SCALE, PEDAL_SCALE, extract_pos

root = os.path.dirname(os.path.abspath(__file__))
get_ref_file = lambda ref_name: os.path.join(root, "spec", "refs", f"{ref_name}.csv")
get_map_file = lambda ref_name: os.path.join(root, "spec", "point_maps", f"{ref_name}_ref.npz")
loaded_refs = {}
loaded_point_maps = {}

class RefDriver():
	def __init__(self, track_name=track_name, dt=0.01):
		self.ref_name = track_name
		self.ref, self.ref_dt = load_ref(self.ref_name)
		self.load_point_map(self.ref_name, dt=dt)
		self.start_vel = self.ref.Speed.values[0,0]
		self.start_pos = (self.ref.PathX.values[0,0], self.ref.PathY.values[0,0], self.ref.PathHeading.values[0,0])
		self.max_time = min(self.ref.Time.values[-1,0], np.max(self.time_map))
		self.state_size = self.state(0.0).observation().shape

	def state(self, time):
		index = (time % self.max_time) / self.ref_dt
		index = index.astype(np.int32) if isinstance(index, np.ndarray) else int(index)
		X = self.ref.PathX.values[index][...,0]
		Y = self.ref.PathY.values[index][...,0]
		ψ = self.ref.PathHeading.values[index][...,0]
		V = self.ref.Speed.values[index][...,0]							#Speed, m/s
		Vx = self.ref.Vx.values[index][...,0]							#Longitudinal Speed, m/s
		Vy = self.ref.Vy.values[index][...,0]							#Lateral Speed, m/s
		S = self.ref.Dist.values[index][...,0]/1000.0					#Distance travelled, km
		ψ̇ = self.ref.YawVel.values[index][...,0]						 #Yaw rate, rad/s
		β = self.ref.SideSlip.values[index][...,0]*np.pi/180			#SideSlip, rad
		δ = self.ref.Steer.values[index][...,0]*np.pi/180				#Steer, rad
		αf = self.ref.SlipAngleFront.values[index][...,0]*np.pi/180		#Front tire slip angle, rad
		αr = self.ref.SlipAngleRear.values[index][...,0]*np.pi/180		#Rear tire slip anlge, rad
		FxF = self.ref.LongForceFront.values[index][...,0]/1000.0		#Front tire long force, N
		FxR = self.ref.LongForceRear.values[index][...,0]/1000.0		#Rear tire long force, N
		FyF = self.ref.LatForceFront.values[index][...,0]/1000.0		#Front tire lat force, N
		FyR = self.ref.LatForceRear.values[index][...,0]/1000.0			#Rear tire lat force, N
		FzF = self.ref.VertForceFront.values[index][...,0]/1000.0		#Front tire vert force, N
		FzR = self.ref.VertForceRear.values[index][...,0]/1000.0		#Rear tire vert force, N
		curvature = self.ref.Curv.values[index][...,0]					#Track curvature, 1/m
		throttle = self.ref.Throttle.values[index][...,0]/100.0			#Throttle, %
		brake = -1*self.ref.Brake.values[index][...,0]/2000.0			#Brake, %
		pedals = (throttle + brake)
		ψ = np.arctan2(np.sin(ψ), np.cos(ψ))
		info = OrderedDict(pedals=pedals, curv=curvature, time=self.ref.Time.values[index][...,0],
			V=V, β=β, αf=αf, αr=αr, FxF=FxF, FxR=FxR, FyF=FyF, FyR=FyR, FzF=FzF, FzR=FzR)
		state = CarState(X,Y,ψ,Vx,Vy,S,ψ̇,δ,pedals,info)
		return state

	def action(self, time, dt):
		states = self.state(time)
		prevstates = self.state(time - dt)
		turn_rate = (states.δ - prevstates.δ)/(TURN_SCALE*dt)
		pedal_rate = (states.pedals - prevstates.pedals)/(PEDAL_SCALE*dt)
		action = np.stack([turn_rate, pedal_rate], -1)
		return action

	def get_sequence(self, time, n, stride=4, dt=DELTA_T):
		times = np.minimum((np.array(time).reshape(-1,1) + np.arange(0, n*stride, stride)[None])*dt, self.max_time-dt)
		state_spec = self.state(times)
		state = state_spec.observation()
		return np.concatenate([state, np.expand_dims(times,-1)], -1)

	def get_state_sequence(self, state, n, obs_spec_fn, stride=4, dt=DELTA_T):
		pos = extract_pos(state)
		reftime = self.get_time(pos)/dt
		refstates = self.get_sequence(reftime, n, stride=stride, dt=dt)
		refpath = rotate_path(state, refstates, obs_spec_fn)[0]
		return refpath

	def get_time(self, point, s=0):
		point = np.array(point)
		shape = list(point.shape)
		minref = self.min_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		maxref = self.max_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		point = np.clip(point, minref, maxref)
		dist = np.clip(s, self.Smap[0], self.Smap[-1])
		dist_ind = np.round((dist-self.Smap[0])/self.sres).astype(np.int32)
		index = np.round((point-minref)/self.res).astype(np.int32)
		times = self.time_map[index[...,0],index[...,1],dist_ind]
		return times

	def load_point_map(self, ref_name, res=1, buffer=50, nthreads=1, ngroups=30, dt=0.01, smaps=5):
		map_file = get_map_file(ref_name)
		if not os.path.exists(map_file):
			self.dt = max(dt, self.ref_dt)
			stride = int(np.round(self.dt/self.ref_dt))
			x = self.ref.PathX.values[::stride]
			y = self.ref.PathY.values[::stride]
			s = self.ref.Dist.values[::stride]/1000.0 if hasattr(self.ref, "Dist") else np.zeros_like(x)
			self.path = np.concatenate([x,y,s], axis=-1)
			x_min, y_min, s_min = map(np.min, [x,y,s])
			x_max, y_max, s_max = map(np.max, [x,y,s])
			X = np.arange(x_min-buffer, x_max+buffer, res).astype(np.float32)
			Y = np.arange(y_min-buffer, y_max+buffer, res).astype(np.float32)
			S = np.linspace(s_min-0.001, s_max+0.001, smaps).astype(np.float32) if smaps>1 and s_min!=s_max else [0]
			points = np.array(list(it.product(X, Y, S)))
			groups = np.split(points, np.arange(0,len(points),len(points)//(ngroups*smaps))[1:])
			if nthreads > 1:
				with Pool(nthreads) as p: times = p.map(self.nearest_point, groups)
			else:
				times = [self.nearest_point(group) for group in groups]
			time = np.concatenate(times, 0)
			time = time.reshape(len(X), len(Y), len(S))
			os.makedirs(os.path.dirname(map_file), exist_ok=True)
			np.savez(map_file, X=X, Y=Y, S=S, time=time, res=res, sres=S[1]-S[0] if len(S)>1 else 1.0, buffer=buffer, dt=self.dt)
		if ref_name not in loaded_point_maps: loaded_point_maps[ref_name] = np.load(map_file)
		data = loaded_point_maps[ref_name]
		self.Xmap = data["X"]
		self.Ymap = data["Y"]
		self.Smap = data["S"]
		self.time_map = data["time"]
		self.sres = data["sres"]
		self.res = data["res"]
		self.dt = data["dt"]
		self.min_point = np.array([self.Xmap[0], self.Ymap[0]])
		self.max_point = np.array([self.Xmap[-1], self.Ymap[-1]])

	def nearest_point(self, point):
		print(f"Computing {point.shape[0]}")
		points = np.expand_dims(np.array(point),1)
		path = np.expand_dims(self.path, 0)
		dists = np.linalg.norm(path-points, axis=-1)
		idx = np.argmin(dists, -1)
		time = idx * self.dt
		return time

	def __len__(self):
		return len(self.path)

def rotate_path(state, refstates, obs_spec_fn):
	refstates[...,4:-1] = 0
	s = obs_spec_fn(state[...,None,:])
	rs = obs_spec_fn(refstates)
	refrot = np.pi/2 - s.ψ
	X = np.cos(refrot)*(rs.X-s.X) - np.sin(refrot)*(rs.Y-s.Y)
	Y = np.sin(refrot)*(rs.X-s.X) + np.cos(refrot)*(rs.Y-s.Y)
	ψ = np.arctan2(np.sin(rs.ψ-s.ψ), np.cos(rs.ψ-s.ψ))
	Vx = rs.Vx - s.Vx
	timediff = rs.realtime - s.realtime
	relref = CarState(X=X, Y=Y, ψ=ψ, Vx=Vx).observation()
	relref = np.concatenate([relref, timediff[...,None]],-1)
	return np.concatenate([relref, refstates[...,relref.shape[-1]:]], -1)

def load_ref(ref_name):
	ref_file = get_ref_file(ref_name)
	if ref_file not in loaded_refs: loaded_refs[ref_file] = pd.read_csv(ref_file, header=[0,1], dtype=np.float32)
	df = loaded_refs[ref_file]
	ref_dt = np.diff(df.Time.values[0:2,0])[0]
	return df, ref_dt

if __name__ == "__main__":
	ref = RefDriver(track_name)