from architectures import IBMGestures
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as np

class TRADES_experiment:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        grid_trades = configure(grid, {"TRADES":True, "beta_robustness":0.1})
        return grid + grid_trades

    @staticmethod
    def visualize():
        grid = TRADES_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")