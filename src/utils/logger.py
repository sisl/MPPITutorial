import os
import re
import sys
import time
import torch
import inspect
import datetime
import subprocess
import numpy as np
import psutil as psu
import GPUtil as gputil
import platform as pfm
from collections import OrderedDict
np.set_printoptions(precision=3, sign=' ', floatmode="fixed", linewidth=2000)

LOG_DIR = os.path.abspath(f"{os.path.dirname(__file__)}../../../logging")

class Stats():
	def __init__(self):
		self.mean_dict = {}
		self.sum_dict = {}

	def mean(self, **kwargs):
		for k,v in kwargs.items():
			if not k in self.mean_dict: self.mean_dict[k] = []
			if isinstance(v, torch.Tensor): v = v.detach().cpu().mean().numpy()
			self.mean_dict[k].append(v)

	def sum(self, **kwargs):
		for k,v in kwargs.items():
			if not k in self.sum_dict: self.sum_dict[k] = []
			self.sum_dict[k].append(v)

	def get_stats(self):
		mean_stats = {k:np.round(np.mean(v, axis=0), 5) if v is not None else v for k,v in self.mean_dict.items()}
		sum_stats = {k:np.round(np.sum(v, axis=0), 5) if v is not None else v for k,v in self.sum_dict.items()}
		self.mean_dict.clear()
		self.sum_dict.clear()
		return {**mean_stats, **sum_stats}

class Logger():
	def __init__(self, envs, agent, config, force_git=False, log_type="logs", **kwconfig):
		self.git_info = self.get_git_info(force_git)
		agent_model = getattr(agent, "agent", agent)
		self.agent_class = agent_model.__class__
		self.config = {"config":config.print(), "num_envs":config.get("split",1)-1, "envs":self.dict_to_string(envs), "agent":self.dict_to_string(agent), **kwconfig}
		self.env_name = config.env_name
		model_name = getattr(agent_model, "name", self.get_module(agent_model)) 
		log_folder = f"{LOG_DIR}/{log_type}/{config.get('framework', 'pt')}/{model_name}/{self.env_name}"
		os.makedirs(log_folder, exist_ok=True)
		self.run_num = len([n for n in os.listdir(log_folder)]) if log_type != "runs" else datetime.datetime.now().strftime("%y%m%d%H%M")
		self.model_src = [line for line in open(inspect.getabsfile(self.agent_class))]
		self.log_path = os.path.join(log_folder, f"logs_{self.run_num}.txt") 
		self.log_num = 0

	def log(self, string, stats, debug=True):
		with open(self.log_path, "a+", encoding="utf-8") as f:
			if self.log_num == 0: 
				self.start_time = time.time()
				f.write(f"Model: {self.agent_class}, Env: {self.env_name}, Date: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
				f.write(self.get_hardware_info() + "\n")
				f.write(self.git_info + "\n\n")
				if hasattr(self, "config"): f.writelines("\n".join([f"{k}: {v}," for k,v in self.config.items()]) + "\n\n")
				if hasattr(self, "model_src"): f.writelines(self.model_src + ["\n"])
				if hasattr(self, "net_src"): f.writelines(self.net_src + ["\n"])
				if hasattr(self, "trn_src"): f.writelines(self.trn_src + ["\n"])
				f.write("\n")
			f.write(f"{string} <{self.get_time()}> ({pydict_to_string(stats)})\n")
		if debug: print(string)
		self.log_num += 1

	def get_git_info(self, force_git):
		if force_git: 
			git_status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], universal_newlines=True)
			assert len(git_status)==0, "Uncommitted changes need to be committed to log current codebase state"
		git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
		git_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], universal_newlines=True).strip()
		git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], universal_newlines=True).strip()
		return "\n".join([f"{k}: {v}" for v,k in zip([git_url, git_hash, git_branch], ["Git URL", "Hash", "Branch"])])

	def get_time(self):
		delta = time.gmtime(time.time()-self.start_time)
		return f"{delta.tm_mday-1}-{time.strftime('%H:%M:%S', delta)}"

	def get_hardware_info(self):
		cpu_info = f"CPU: {psu.cpu_count(logical=False)} Core, {psu.cpu_freq().max/1000}GHz, {np.round(psu.virtual_memory().total/1024**3,2)} GB, {pfm.platform()}"
		gpu_info = [f"GPU {gpu.id}: {gpu.name}, {np.round(gpu.memoryTotal/1000,2)} GB (Driver: {gpu.driver})" for gpu in gputil.getGPUs()]
		return "\n".join([cpu_info, *gpu_info])

	def get_net_train_src(self):
		self.net_src = [line for line in open(f"utils/network.py") if re.match("^[A-Z]", line)] 
		self.trn_src = [line for line in open(f"train.py")]

	def dict_to_string(self, item):
		if self.get_module(item,0) != self.get_module(self, 0): return f"<list len={len(item)}>" if isinstance(item, list) and len(item)>10 else str(item)
		string = "".join([f"\n{k} = {self.dict_to_string(v)}" for k,v in item.__dict__.items() if not k.startswith("_")])
		return f"{item} {string}".replace('\n','\n\t')

	@staticmethod
	def get_classes(obj):
		return [v for k,v in obj.__dict__.items() if inspect.getmembers(v)[0][0] == "__class__"]

	@staticmethod
	def get_module(obj, index=-1):
		module = inspect.getmodule(obj.__class__).__name__.split(".")
		return module[index]

	@staticmethod
	def sci_form(num):
		if type(num) in [type(None), str]: return num
		return f"{num:10.2e}" if np.abs(num)<1e-4 else f"{num:10.4f}"

def pydict_to_string(item):
	if isinstance(item, (type(None), str)): return item
	if isinstance(item, (dict, OrderedDict)): 
		string_dict = {k:pydict_to_string(v) for k,v in item.items()}
		return "{" + ', '.join([f"'{k}': {v}" for k,v in string_dict.items()]) + "}"
	return f"{item:10.2e}" if np.abs(item)<1e-4 else f"{item:10.3f}".rjust(10,' ')