import os
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from multiprocessing import Pool

root = os.path.dirname(os.path.abspath(__file__))
get_track_file = lambda track_name: os.path.join(root, "spec", "tracks", f"{track_name}.csv")
get_map_file = lambda track_name: os.path.join(root, "spec", "point_maps", f"{track_name}.npz")
loaded_tracks = {}
loaded_point_maps = {}
track_name = "curve"

class Track():
	def __init__(self, track_name=track_name, stride=1):
		self.stride = stride
		self.track_name = track_name
		self.track = load_track(track_name, stride)
		self.boundaries = compute_boundaries(self.track, 15)
		self.X, self.Y = self.track[:,0], self.track[:,1]
		self.load_point_map(self.track_name)

	def get_nearest(self, point):
		point = np.array(point)
		shape = list(point.shape)
		minref = self.min_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		maxref = self.max_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		point = np.clip(point, minref, maxref)
		index = np.round((point-minref)/self.res).astype(np.int32)
		indices = self.idx_map[index[...,0],index[...,1]]
		min_dists = self.dist_map[index[...,0],index[...,1]]
		return indices, min_dists

	def get_progress(self, src, dst):
		start, _ = self.get_nearest(src)
		fin, _ = self.get_nearest(dst)
		offset = int(0.8*len(self.track))
		progress = (offset + fin - start)%len(self.track) - offset
		return np.clip(progress, -5, 5)

	def load_point_map(self, track_name, res=1, buffer=50, nthreads=1, ngroups=24):
		map_file = get_map_file(track_name)
		if not os.path.exists(map_file):
			X, Y = self.X, self.Y
			x_min, x_max = np.min(X), np.max(X)
			y_min, y_max = np.min(Y), np.max(Y)
			X = np.arange(x_min-buffer, x_max+buffer, res).astype(np.float32)
			Y = np.arange(y_min-buffer, y_max+buffer, res).astype(np.float32)
			points = np.array(list(it.product(X, Y)))
			groups = np.split(points, np.arange(0,len(points),len(points)//ngroups)[1:])
			if nthreads > 1:
				with Pool(nthreads) as p: nearests = p.map(self.compute_nearest, groups)
			else:
				nearests = [self.compute_nearest(group) for group in groups]
			idxs, min_dists = map(list, zip(*nearests))
			idx = np.concatenate(idxs, 0)
			min_dist = np.concatenate(min_dists, 0)
			idx = idx.reshape(len(X), len(Y))
			min_dist = min_dist.reshape(len(X), len(Y))
			os.makedirs(os.path.dirname(map_file), exist_ok=True)
			np.savez(map_file, X=X, Y=Y, idx=idx, min_dist=min_dist, res=res, buffer=buffer, stride=self.stride)
		if track_name not in loaded_point_maps: loaded_point_maps[track_name] = np.load(map_file)
		data = loaded_point_maps[track_name]
		self.Xmap = data["X"]
		self.Ymap = data["Y"]
		self.idx_map = data["idx"]
		self.dist_map = data["min_dist"]
		self.res = data["res"]
		self.stride = data["stride"]
		self.min_point = np.array([self.Xmap[0], self.Ymap[0]])
		self.max_point = np.array([self.Xmap[-1], self.Ymap[-1]])

	def compute_nearest(self, point):
		print(f"Computing {point.shape[0]}")
		points = np.expand_dims(np.array(point),1)
		track = np.expand_dims(self.track, 0)
		dists = np.linalg.norm(track - points, axis=-1)
		idx = np.argmin(dists, -1)
		min_dist = dists[np.arange(dists.shape[0]),idx]
		return idx, min_dist

	def get_path(self, point, length=10, step:int=2, heading=None):
		nearest, _ = self.get_nearest(point)
		sequence = np.arange(0,length*step,step)
		sequence = np.reshape(sequence, [*np.ones(len(nearest.shape), dtype=np.int32), *sequence.shape])
		ipath = (nearest[...,None]+sequence) % len(self.track)
		path = self.track[ipath]
		if heading is not None:
			path = np.array(path)
			grad = path[...,length//2,:]-path[...,0,:]
			refdirn = np.arctan2(grad[...,1],grad[...,0])
			dirn = heading - np.pi/2
			dirn = np.reshape(dirn, [*dirn.shape, *np.ones(len(grad.shape)-len(dirn.shape), dtype=np.int32)])
			relpath = path - path[...,0:1,:]
			path = np.copy(relpath)
			path[...,0] = relpath[...,0]*np.cos(dirn) + relpath[...,1]*np.sin(dirn)
			path[...,1] = np.abs(relpath[...,0]*np.sin(dirn) + relpath[...,1]*np.cos(dirn))
		self.path = path
		return np.reshape(path, [*path.shape[:-2], -1]) #np.concatenate([*path])

	def get_state_sequence(self, state, n, obs_spec_fn, stride=15, dt=None):
		pos = state[...,[0,1]]
		stride *= self.res
		nearest, _ = self.get_nearest(pos)
		sequence = np.arange(0,n*stride,stride)
		sequence = np.reshape(sequence, [*np.ones(len(nearest.shape), dtype=np.int32), *sequence.shape])
		ipath = (nearest[...,None]+sequence) % len(self.track)
		path = self.track[ipath]
		refstates = np.concatenate([path, np.zeros([n,state.shape[0]-2])],-1)
		s = obs_spec_fn(state[...,None,:])
		rs = obs_spec_fn(refstates)
		refrot = np.pi/2 - s.ψ
		X = np.cos(refrot)*(rs.X-s.X) - np.sin(refrot)*(rs.Y-s.Y)
		Y = np.sin(refrot)*(rs.X-s.X) + np.cos(refrot)*(rs.Y-s.Y)
		self.path = np.stack([X, Y], -1)
		return self.path

	def __len__(self):
		return len(self.track)

def load_track(track_name, stride=1):
	if track_name in loaded_tracks: return loaded_tracks[track_name]
	track_file = get_track_file(track_name)
	df = pd.read_csv(track_file, header=None)
	track = df.values[::stride]
	loaded_tracks[track_name] = track.astype(np.float32)
	return track.astype(np.float32)

def compute_boundaries(points, lane_width=10):
	boundaries = []
	grad = points[1] - points[0]
	normal = np.array([-grad[1], grad[0]]) / np.linalg.norm(grad)
	left = points[0] + lane_width*normal
	right = points[0] - lane_width*normal
	boundaries.append(np.stack([left, right]))
	for i in range(1, points.shape[0]):
		grad = points[i] - points[i-1]
		normal = np.array([-grad[1], grad[0]]) / np.linalg.norm(grad)
		left = points[i] + lane_width*normal
		right = points[i] - lane_width*normal
		boundaries.append(np.stack([left, right]))
	boundaries = np.stack(boundaries)
	return boundaries

if __name__ == "__main__":
	track = Track(track_name)