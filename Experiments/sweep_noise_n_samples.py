from architectures import IBMGestures
from datajuicer import run, configure
from datajuicer.utils import split
from experiment_utils import *
from datajuicer.visualizers import *

noise_n_samples = [0,100,1000,5000,10000]

class sweep_noise_n_samples:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        grid = configure(grid, {"batch_size": 32, "dt":2000})
        grid = split(grid, "noise_n_samples", noise_n_samples)
        return grid

    @staticmethod
    def visualize():
        grid = sweep_noise_n_samples.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
