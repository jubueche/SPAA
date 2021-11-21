from architectures import IBMGestures
from datajuicer import run, configure
from datajuicer.utils import split
from experiment_utils import *
from datajuicer.visualizers import *

beta_robustness = [0.0,0.01,0.05,0.1,0.2]

class sweep_beta_rob:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        grid = configure(grid, {"batch_size": 16, "boundary_loss": "trades", "epochs":2})
        grid = split(grid, "beta_robustness", beta_robustness)
        return grid

    @staticmethod
    def visualize():
        grid = sweep_beta_rob.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
