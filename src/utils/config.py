
class Config(object):
	def __init__(self, **kwargs):
		self.update(**kwargs)

	def props(self):   
		return {k:v for k,v in self.__dict__.items() if k[:1] != '_'}

	def update(self, **kwargs):
		for k,v in kwargs.items():
			if hasattr(self, k) and isinstance(k, Config):
				getattr(self, k).update(**v.props())
			else:
				setattr(self, k, v)
		return self

	def get(self, key, default=None):
		return getattr(self, key, default)

	def merge(self, config):
		self.update(**config.props())

	def clone(self, **kwargs):
		return self.__class__(**self.props()).update(**kwargs)

	def print(self, level=1):
		return "".join([f"\n{'   '*level}{k} = {v.print(level+1) if isinstance(v,Config) else v}" for k,v in self.__dict__.items()])