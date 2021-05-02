import os
import sys
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits import mplot3d
from PIL import Image, ImageFont, ImageDraw
from src.utils.misc import make_video, resize
np.set_printoptions(precision=2, sign=' ', floatmode="fixed", suppress=True, linewidth=100)

dark_cols = ["#F24346", "#2244DD", "#777777", "#008000", "#FFA500", "#FFA500"]

class PathAnimator():
	def __init__(self, interactive=True, D3=False, dt=0.02):
		if interactive: plt.ion()
		self.interactive = interactive
		self.fig = plt.figure(figsize=(12, 8), dpi=60)
		plt.style.use('dark_background')
		self.fig.patch.set_facecolor("#000000")
		self.ax = plt.axes(projection='3d' if D3 else None)
		self.fig.tight_layout()
		self.renders = []
		self.dt = dt

	def animate_path(self, track, pos, cars=[], trajectories=None, info=None, **kwargs):
		self.ax.cla()
		self.ax.set_facecolor("#5AA05A")
		self.plot(track, pos, cars, [] if trajectories is None else trajectories, ref=kwargs.get("path"), info=info)
		if self.interactive:
			plt.draw()
			plt.pause(0.0000001)
			image = None
		else:
			cvs = FigureCanvasAgg(self.fig)
			cvs.draw()
			w, h = map(int, self.fig.get_size_inches()*self.fig.get_dpi())
			image = np.copy(np.frombuffer(cvs.tostring_rgb(), dtype="uint8").reshape(h,w,3))
			image = cv2.resize(image, tuple(map(int, (0.8*w,0.8*h))), interpolation=cv2.INTER_AREA)
			self.renders.append(image)
			plt.close()
		return image

	def plot(self, track, center, cars, trajectories, view=200, ref=None, info=None):
		for car, col in zip(cars, dark_cols[:len(cars)]):
			self.ax.scatter(car[:,0], car[:,1], s=22, color=col, zorder=1)
		self.ax.plot(track.X,track.Y, linewidth=50, color="#646464", zorder=0)
		self.ax.set_xlim(center[0]-1.5*view, center[0]+1.5*view)
		self.ax.set_ylim(center[1]-view, center[1]+view)
		if info and info.get("car"): self.make_overlay(info["car"], 0.025, 0.975)
		if info and info.get("ref"): self.make_overlay(info["ref"], 0.625, 0.975)
		if ref is not None:
			self.ax.plot(ref[:,0], ref[:,1], color="#111111")
		for path in trajectories:
			xs, ys = path[:,0], path[:,1]
			self.ax.plot(xs, ys, linewidth=0.2)
		self.ax.grid(b=True, which='major', color='#666666', linestyle=':')

	def make_overlay(self, info, x, y):
		text = '\n'.join([f"{k}: {v if isinstance(v,str) else f'{v:.4f}'}" for k,v in info.items()])
		self.ax.text(x, y, text, size=15, color="#000000", horizontalalignment='left', verticalalignment='top', transform=self.ax.transAxes)

	def close(self, path="test.mp4"):
		if len(self.renders) > 0 and path:
			make_video(self.renders, path, fps=int(1/self.dt))
