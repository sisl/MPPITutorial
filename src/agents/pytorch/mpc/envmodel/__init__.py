from .mdrnn import MDRNNEnv
from .dfrntl import DifferentialEnv
from .linfeat import LinearFeatureEnv
from .real import RealEnv, BatchRealEnv

all_envmodels = {
	"real": RealEnv,
	"mdrnn": MDRNNEnv,
	"linfeat": LinearFeatureEnv,
	"dfrntl": DifferentialEnv,
	"batchreal": BatchRealEnv
}

def get_envmodel(config):
	envmodel = config.get("envmodel", config.get("ENV_MODEL"))
	return all_envmodels[envmodel]