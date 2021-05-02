import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from cost import CostModel
from track import Track, track_name
from ref import RefDriver

root = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(root, "spec", "plots")
os.makedirs(plot_dir, exist_ok=True)

def plot_track(track):
	plt.figure()
	plt.style.use('dark_background')
	plt.title("Track Positions")
	plt.xlabel("X (m)")
	plt.ylabel("Y (m)")
	plt.scatter(track.X,track.Y,s=0.1)
	plt.savefig(f"{plot_dir}/Track2D_{track.track_name}", dpi=500, bounding_box="tight")

def plot_track_map(track):
	plt.figure()
	ax = plt.axes(projection='3d')
	X, Y = track.X, track.Y
	grid = np.array(list(zip(X, Y)))
	ax.scatter(X, Y, track.get_nearest(grid)[0], s=1)
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlim3d(0, 1.1*len(track))

def plot_ref(ref, track, save=True):
	plt.figure()
	plt.style.use('dark_background')
	plt.title("Reference Driver Positions")
	plt.xlabel("X (m)")
	plt.ylabel("Y (m)")
	plt.scatter(track.X, track.Y, s=10, c="grey")
	plt.scatter(ref.ref.PathX.values[0:1], ref.ref.PathY.values[0:1], s=4, c="red")
	plt.scatter(ref.ref.PathX.values[4000::4000], ref.ref.PathY.values[4000::4000], s=4, c="yellow")
	plt.plot(ref.ref.PathX.values, ref.ref.PathY.values, linewidth=0.5, c="blue")
	plt.savefig(f"{plot_dir}/Ref2D_{track.track_name}", dpi=500, bounding_box="tight")

def plot_ref_map(ref, track, s=0):
	plt.figure()
	plt.style.use('dark_background')
	XX,YY = np.meshgrid(ref.Xmap, ref.Ymap)
	grid = np.stack([XX,YY], -1)
	times = ref.get_time(grid, s=ref.Smap[s])
	plt.pcolormesh(XX, YY, times, cmap='RdYlGn_r')
	plt.colorbar()
	plt.title("Position to Ref time Map")
	plt.xlabel("X (m)")
	plt.ylabel("Y (m)")
	filename = f"Ref2Ds{s}_{ref.ref_name}.png"
	plt.scatter(track.X, track.Y, s=10, c="grey")
	plt.scatter(ref.ref.PathX.values[0:1], ref.ref.PathY.values[0:1], s=4, c="red")
	plt.plot(ref.ref.PathX.values, ref.ref.PathY.values, linewidth=0.5, c="blue")
	plt.savefig(f"{plot_dir}/{filename}", dpi=500, bounding_box="tight")

def plot_cost_map(cmodel, transform=False):
	plt.figure()
	plt.style.use('dark_background')
	XX,YY = np.meshgrid(cmodel.track.Xmap, cmodel.track.Ymap)
	grid = np.stack([XX,YY], -1)
	cost = cmodel.get_point_cost(grid, transform=transform)
	plt.pcolormesh(XX, YY, cost, cmap='RdYlGn_r')
	plt.colorbar()
	plt.title("Position to Deviation Cost Map")
	plt.xlabel("X (m)")
	plt.ylabel("Y (m)")
	filename = f"Cost2D_{cmodel.track.track_name}{'_raw' if not transform else ''}.png"
	plt.savefig(f"{plot_dir}/{filename}", dpi=500, bounding_box="tight")

def plot_cost_map3D(cmodel):
	plt.figure()
	ax = plt.axes(projection='3d')
	XX,YY = np.meshgrid(cmodel.track.Xmap, cmodel.track.Ymap)
	grid = np.stack([XX,YY], -1)
	cost = cmodel.get_point_cost(grid, transform=False)
	ax.plot_surface(XX, YY, np.tanh(cost/20)**2, cmap='RdYlGn_r')
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlim3d(0, 2)

def animate_path(track):
	plt.ion()
	plt.figure()
	X, Y = track.X, track.Y
	ax = plt.axes(projection='3d')
	grid = np.array(list(zip(X,Y)))
	for point in grid:
		path = np.array(track.get_path(point))
		xs, ys = zip(*path)
		ax.set_zlim3d(-100, 100)
		ax.plot(X,Y, color="#DDDDDD")
		ax.plot(xs, ys, linewidth=2)
		relpath = track.get_path(point, dirn=True)
		rx, ry = map(np.array, zip(*relpath))
		ax.plot(rx, ry, linewidth=2)
		plt.draw()
		plt.pause(0.01)
		ax.cla()

if __name__ == "__main__":
	track_name = "curve"
	track = Track(track_name)
	ref = RefDriver(track_name)
	cost_model = CostModel(track, ref)
	plot_track(track)
	plot_ref(ref, track)
	plot_ref_map(ref, track)
	plot_cost_map(cost_model, True)
	plot_track_map(track)
	plot_cost_map3D(cost_model)
	# animate_path(track)
	# plt.show()
