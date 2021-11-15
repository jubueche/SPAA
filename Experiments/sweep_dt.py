from architectures import IBMGestures
from datajuicer import run, configure
from datajuicer.utils import split
from experiment_utils import *
from datajuicer.visualizers import *

dts = [1000, 2000, 5000, 10000]

class sweep_dt:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        grid = configure(grid, {"batch_size": 256})
        grid = split(grid, "dt", dts)
        return grid

    @staticmethod
    def visualize():
        grid = sweep_dt.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")