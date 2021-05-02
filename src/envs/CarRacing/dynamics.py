import os
import sys
import inspect
import numpy as np
import itertools as it
import pyquaternion as pyq
from types import SimpleNamespace
from collections import OrderedDict
try:
	from src.envs.Gym import gym
except:
	import gym

USE_DELTA = True
DELTA_T = 0.1
TURN_SCALE = 0.05
PEDAL_SCALE = 1.0
TURN_LIMIT = 0.025
PEDAL_LIMIT = 1.0

X1constants = SimpleNamespace(
	m = 2000.0, 			# Mass (kg)
	I_zz = 3764.0, 			# Inertia (kg m^2)
	h_cm = 0.3,	 			# Distance from CG to front axle (m)
	l_f = 1.53, 			# Distance from CG to front axle (m)
	l_r = 1.23, 			# Distance from CG to rear axle (m)
	C_D0 = 241, 			# Coefficient of drag
	C_D1 = 25.1, 			# Coefficient of drag
	C_αf = 150000.0, 		# Front tire cornering stiffness (N/rad)
	C_αr = 280000.0, 		# Rear tire cornering stiffness (N/rad)
	μ_f = 1.5, 				# Front tire friction
	μ_r = 1.5, 				# Rear tire friction
)

class CarState():
	def __init__(self, *args, **kwargs):
		self.update(*args, **kwargs)

	def update(self,X=None,Y=None,ψ=None,Vx=None,Vy=None,S=None,ψ̇=None,δ=None,pedals=None,info={}):
		givens = [x for x in [X,Y,ψ,Vx,Vy,S,ψ̇,δ,pedals] if x is not None]
		default = lambda: givens[0]*0 if len(givens) > 0 else 0.0
		self.X = X if X is not None else default()
		self.Y = Y if Y is not None else default()
		self.ψ = ψ if ψ is not None else default()
		self.Vx = Vx if Vx is not None else default()
		self.Vy = Vy if Vy is not None else default()
		self.S = S if S is not None else default()
		self.ψ̇  = ψ̇  if ψ̇  is not None else default()
		self.δ = δ if δ is not None else default()
		self.pedals = pedals if pedals is not None else default()
		self.info = info
		self.shape = getattr(default(), "shape", ())
		return self

	def observation(self):
		pos_x = np.expand_dims(self.X, axis=-1)
		pos_y = np.expand_dims(self.Y, axis=-1)
		rot_f = np.expand_dims(self.ψ, axis=-1)
		vel_f = np.expand_dims(self.Vx, axis=-1)
		vel_s = np.expand_dims(self.Vy, axis=-1)
		dist = np.expand_dims(self.S, axis=-1)
		yaw_dot = np.expand_dims(self.ψ̇, axis=-1)
		steer = np.expand_dims(self.δ, axis=-1)
		pedals = np.expand_dims(self.pedals, axis=-1)
		return np.concatenate([pos_x, pos_y, rot_f, vel_f, vel_s, dist, yaw_dot, steer, pedals], axis=-1)

	@staticmethod
	def observation_spec(state):
		pos_x = state[...,0]
		pos_y = state[...,1]
		rot_f = state[...,2]
		vel_f = state[...,3]
		vel_s = state[...,4]
		dist = state[...,5]
		yaw_dot = state[...,6]
		steer = state[...,7]
		pedals = state[...,8]
		info = OrderedDict(V=0, β=0, αf=0, αr=0, FxF=0, FxR=0, FyF=0, FyR=0, FzF=0, FzR=0)
		state_spec = CarState(pos_x,pos_y,rot_f,vel_f,vel_s,dist,yaw_dot,steer,pedals,info)
		return state_spec

	def print(self):
		return f"X: {self.X:4.3f}, Y: {self.Y:4.3f}, ψ: {self.ψ:4.3f}, Vx: {self.Vx:4.3f}, Vy: {self.Vy:4.3f}, δ: {self.δ:4.3f}, pedals: {self.pedals:4.3f}"

class X1TireModel():
	name = "x1"
	def calc_F_Aero(self, Vx, C0, C1): 
		return C0 + C1*Vx

	def calc_Fz(self, Fx, m, h_cm, l, L): 
		return (m*l*9.81 + h_cm*Fx)/L

	def calc_Fy(self, α, μ, C_α, F_Z, F_X, tan=np.tan, arctan=np.arctan, **kwargs):
		Fy_max = np.sqrt(np.maximum(np.power(μ*F_Z,2) - np.power(F_X,2), 1e-8))
		return np.where(np.abs(α) < arctan(3*Fy_max/C_α), -C_α*tan(α) + C_α**2/(3*Fy_max)*np.abs(tan(α))*tan(α) - C_α**3/(27*Fy_max**2)*(tan(α)**3), -Fy_max*np.sign(α))

class X1Dynamics():
	dynamics_norm = np.array([100, 100, 2*np.pi, 50, 50, 5, 1, 0.1, 1])
	dynamics_somask = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1])
	def __init__(self, tire_name=None, *kwargs):
		self.action_space = gym.spaces.Box(-1.0, 1.0, (2,))
		self.tire_model = X1TireModel()

	def reset(self, start_pos, start_vel):
		self.state = CarState(X=start_pos[0], Y=start_pos[1], ψ=start_pos[2], Vx=start_vel)
		self.turn_scale = TURN_SCALE
		self.pedal_scale = PEDAL_SCALE
		self.turn_limit = TURN_LIMIT
		self.pedal_limit = PEDAL_LIMIT

	def step(self, action, dt=DELTA_T, integration_steps=1, use_delta=USE_DELTA):
		turn_rate = action[...,0]
		pedal_rate = action[...,1]
		dt = dt/integration_steps
		state = self.state
		turn_limit = self.turn_limit*np.minimum(50/state.Vx,2)**2
		Fy_scale = np.minimum(np.abs(state.Vx), 1)
		
		for i in range(integration_steps):
			δ = clamp(state.δ + clamp(turn_rate*self.turn_scale-state.δ, turn_limit) * dt, turn_limit) if use_delta else turn_rate*turn_limit
			αf = np.arctan2((state.Vy + X1constants.l_f * state.ψ̇),state.Vx) - δ
			αr = np.arctan2((state.Vy - X1constants.l_r * state.ψ̇),state.Vx) + 0.0
			
			pedals = clamp(state.pedals + clamp(pedal_rate*self.pedal_scale-state.pedals, self.pedal_limit) * dt, PEDAL_LIMIT) if use_delta else pedal_rate*self.pedal_limit
			accel = 4000 * np.maximum(pedals, 0)
			brake = np.minimum(pedals, 0)*22500*(self.state.Vx > 0)

			F_X_Aero = self.tire_model.calc_F_Aero(state.Vx, X1constants.C_D0, X1constants.C_D1)

			FxF = brake * 0.6
			FxR = accel+brake*0.4
			FzF = self.tire_model.calc_Fz(FxF, X1constants.m, X1constants.h_cm, X1constants.l_f, X1constants.l_f+X1constants.l_r)
			FzR = self.tire_model.calc_Fz(FxR, X1constants.m, X1constants.h_cm, X1constants.l_r, X1constants.l_f+X1constants.l_r)
			FyF = self.tire_model.calc_Fy(αf, X1constants.μ_f, X1constants.C_αf, FzF, FxF, side="F") * Fy_scale
			FyR = self.tire_model.calc_Fy(αr, X1constants.μ_r, X1constants.C_αr, FzR, FxR, side="R") * Fy_scale
			
			ψ̈ = (1/X1constants.I_zz) * ((2*FxF*np.sin(δ) + 2*FyF*np.cos(δ))*X1constants.l_f - 2*FyR*X1constants.l_r)
			V̇x = (1/X1constants.m) * (2*FxF*np.cos(δ) - 2*FyF*np.sin(δ) + 2*FxR - F_X_Aero) + state.ψ̇ * state.Vy
			V̇y = (1/X1constants.m) * (2*FyF*np.cos(δ) + 2*FxF*np.sin(δ) + 2*FyR) - state.ψ̇ * state.Vx
			
			ψ̇ = state.ψ̇ + ψ̈  * dt
			Vx = state.Vx + V̇x * dt
			Vy = state.Vy + V̇y * dt
			V = np.sqrt(Vx**2 + Vy**2)
			
			β = np.arctan2(Vy,Vx)
			ψ = (state.ψ + ψ̇  * dt)
			X = (state.X + (Vx * np.cos(ψ) - Vy * np.sin(ψ)) * dt)
			Y = (state.Y + (Vy * np.cos(ψ) + Vx * np.sin(ψ)) * dt)
			ψ = np.arctan2(np.sin(ψ), np.cos(ψ))
			S = (state.S*1000 + V * dt)/1000

			info = OrderedDict(F_X_Aero=F_X_Aero, yaw_acc=ψ̈ , vx_dot=V̇x, vy_dot=V̇y, V=V, β=β, αf=αf, αr=αr, FxF=FxF/1000, FxR=FxR/1000, FyF=FyF/1000, FyR=FyR/1000, FzF=FzF/1000, FzR=FzR/1000)
			state = state.update(X,Y,ψ,Vx,Vy,S,ψ̇,δ,pedals,info)

		self.state = state
		return True

	def observation(self, state):
		state = self.state if state == None else state
		return state.observation()

	@staticmethod
	def observation_spec(state, device=None):
		return CarState.observation_spec(state)

	def set_state(self, state, device=None):
		self.state = self.observation_spec(state, device=device)

def clamp(x, r): return np.clip(x, -r, r)

def extract_pos(state): return state[...,[0,1]]