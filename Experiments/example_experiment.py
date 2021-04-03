""" Example experiment illustrating how to setup experiment """

from architectures import NMNIST
from datajuicer import run, split, configure, query
from experiment_utils import *

class example_experiment:

    @staticmethod
    def train_grid():
        grid = [NMNIST.make()]
        return grid

    @staticmethod
    def visualize():
        # - Load the grid
        grid = example_experiment.train_grid()
        # - Load the models in the grid from the database
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        
        # - Experiment
        epsilons = [1.0,1.5,2.0]
        # - split performs a division of the grid: E.g. [{"k1":v1}] -> [{"k1":v1,"eps":1.0},{"k1":v1,"eps":1.5},{"k1":v1,"eps":2.0}] 
        grid = split(grid, "eps", epsilons)
        # - Configure just sets some parameters: E.g. [{"k1":v1}] -> [{"k1":v1,"norm":"np.inf"}]
        grid = configure(grid, {"eps_iter":0.3,"N_pgd":20,"N_MC":10,"norm":"2","rand_minmax":0.01,"limit":10,"N_samples":50})

        """
        Run the experiment. Function that is being run is get_prob_attack_robustness
        defined in experiment_utils. This is a cachable function, meaning that the results
        get saved in a database (the DB keeps references to pickle objects).
        The dependencies in the cachable function determine what is important for the experiment. For example,
        if we would not depend on eps, and we re-run the experiment with a different eps value, we would
        get the cached results from the old eps. This makes no sense since the results are expected to change for
        different epsilons, so we need to include eps in the dependencies. Same goes for other hyperparameters.
        """        
        grid = run(grid, get_prob_attack_robustness, n_threads=1, store_key="prob_attack_robustness")(
            "{*}",
            "{eps}",
            "{eps_iter}",
            "{N_pgd}",
            "{N_MC}",
            "{norm}",
            "{rand_minmax}",
            "{limit}",
            "{N_samples}"
        )