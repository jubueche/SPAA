""" Example experiment illustrating how to setup experiment """

from architectures import NMNIST
from datajuicer import run, split, configure, query
from experiment_utils import *

class example_experiment:

    @staticmethod
    def train_grid():
        grid = NMNIST.make()
        return grid

    @staticmethod
    def visualize():
        grid = example_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        
        print(grid)
        print(grid[0]["prob_net"])
        print(grid[0]["ann"])