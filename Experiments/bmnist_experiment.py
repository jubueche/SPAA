from architectures import BMNIST
from datajuicer import run, split, configure, query
from experiment_utils import *

class bmnist_experiment:

    @staticmethod
    def train_grid():
        grid = [BMNIST.make()]
        return grid

    @staticmethod
    def visualize():
        grid = bmnist_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        N_pgd = 50
        N_MC = 5
        eps = 1.5
        eps_iter = 0.3
        rand_minmax = 0.01
        norm = 2
        hamming_distance_eps = 50 / 784
        k = 50 # - For Boosted Prob
        thresh = 0.1 # - For SCAR
        early_stopping = True
        verbose = False
        limit = 100
        
        grid = configure(grid, {
            "N_pgd":N_pgd,
            "N_MC":N_MC,
            "eps":eps,
            "eps_iter":eps_iter,
            "rand_minmax":rand_minmax,
            "norm":norm,
            "hamming_distance_eps":hamming_distance_eps,
            "k":k,
            "thresh":thresh,
            "early_stopping":early_stopping,
            "verbose":verbose,
            "limit":limit})

        grid = run(grid, scar_attack_on_test_set, n_threads=1, store_key="scar_attack")(
            "{*}",
            "{hamming_distance_eps}",
            "{thresh}",
            "{early_stopping}",
            "{verbose}",
            "{limit}"
        )

        grid = run(grid, prob_boost_attack_on_test_set, n_threads=1, store_key="prob_boost_attack")(
            "{*}",
            "{N_pgd}",
            "{N_MC}",
            "{eps}",
            "{eps_iter}",
            "{rand_minmax}",
            "{norm}",
            "{k}",
            "{verbose}",
            "{limit}"
        )

        grid = run(grid, prob_attack_on_test_set, n_threads=1, store_key="prob_attack")(
            "{*}",
            "{N_pgd}",
            "{N_MC}",
            "{eps}",
            "{eps_iter}",
            "{rand_minmax}",
            "{norm}",
            "{hamming_distance_eps}",
            "{early_stopping}",
            "{verbose}",
            "{limit}"
        )
