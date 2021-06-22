from architectures import BMNIST
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
from copy import deepcopy
from datajuicer.visualizers import *

def split_attack_grid(grid, attacks):
    grid_tmp = []
    for g in grid:
        for attack in attacks:
            g_tmp = deepcopy(g)
            g_tmp["attack"] = attack
            g_tmp["attack_result"] = g[attack]
            for attack in attacks:
                g_tmp.pop(attack)
            grid_tmp.append(g_tmp)
    return grid_tmp

def make_summary(d):
    network_correct = d["attack_result"]["predicted"] == d["attack_result"]["targets"]
    d["success_rate"] = 100*np.mean(d["attack_result"]["success"][network_correct])
    d["median_elapsed_time"] = np.median(d["attack_result"]["elapsed_time"][np.array(d["attack_result"]["success"], dtype=bool) & network_correct])
    d["median_n_queries"] = np.median(d["attack_result"]["n_queries"][np.array(d["attack_result"]["success"], dtype=bool) & network_correct])
    d["mean_L0"] = np.mean(d["attack_result"]["L0"][np.array(d["attack_result"]["success"], dtype=bool) & network_correct])
    d["median_L0"] = np.median(d["attack_result"]["L0"][np.array(d["attack_result"]["success"], dtype=bool) & network_correct])
    return d

label_dict={"scar":"SCAR",
            "prob_fool":"Prob. PGA",
            "non_prob_fool":"PGA",
            "sparse_fool":"Sparse Fool",
            "frame_based_sparse_fool": "Frame Based Sparse Fool",
            "success_rate": "Success Rate",
            "median_elapsed_time":"Median Elapsed Time",
            "median_n_queries":"Median No. Queries",
            "mean_L0":"Mean L0 distance",
            "median_L0":"Median L0 distance",
            "attack":"Attack"}

class bmnist_comparison_experiment:
    @staticmethod
    def train_grid():
        grid = [BMNIST.make()]
        return grid

    @staticmethod
    def visualize():
        grid = bmnist_comparison_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        N_pgd = 50
        N_MC = 5
        eps = 1.5
        eps_iter = 0.3
        norm = 2
        max_hamming_distance = 200
        thresh = 0.1  # - For SCAR
        early_stopping = True
        boost = False
        verbose = True
        limit = 1000
        lambda_ = 2.0
        rand_minmax = 0.01
        round_fn = "stoch_round"
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        max_iter_deep_fool = 50
        step_size = 0.01

        grid = configure(
            grid,
            {
                "N_pgd": N_pgd,
                "N_MC": N_MC,
                "eps": eps,
                "eps_iter": eps_iter,
                "norm": norm,
                "max_hamming_distance": max_hamming_distance,
                "thresh": thresh,
                "boost": boost,
                "early_stopping": early_stopping,
                "lambda_": lambda_,
                "round_fn": round_fn,
                "verbose": verbose,
                "limit": limit,
                "rand_minmax": rand_minmax,
                "max_iter": max_iter,
                "epsilon": epsilon,
                "overshoot": overshoot,
                "step_size": step_size,
                "max_iter_deep_fool": max_iter_deep_fool
            },
        )

        grid = run(grid, scar_attack_on_test_set, n_threads=1, store_key="scar")(
            "{*}",
            "{max_hamming_distance}",
            "{thresh}",
            "{early_stopping}",
            "{verbose}",
            "{limit}",
        )

        grid = run(
            grid,
            prob_fool_on_test_set,
            n_threads=1,
            store_key="prob_fool",
        )(
            "{*}",
            "{N_pgd}",
            "{N_MC}",
            "{eps}",
            "{eps_iter}",
            "{rand_minmax}",
            "{norm}",
            "{max_hamming_distance}",
            "{boost}",
            "{early_stopping}",
            "{verbose}",
            "{limit}",
        )

        grid = run(grid, non_prob_fool_on_test_set, n_threads=1, store_key="non_prob_fool")(
            "{*}",
            "{N_pgd}",
            "{round_fn}",
            "{eps}",
            "{eps_iter}",
            "{rand_minmax}",
            "{norm}",
            "{max_hamming_distance}",
            "{boost}",
            "{early_stopping}",
            "{verbose}",
            "{limit}",
        )

        grid = run(grid, sparse_fool_on_test_set, n_threads=1, store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            "{lambda_}",
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{early_stopping}",
            "{boost}",
            "{verbose}",
            "{limit}"
        )

        attacks = ["scar","prob_fool","non_prob_fool","sparse_fool"]
        grid = split_attack_grid(grid, attacks)

        grid = run(grid, make_summary, store_key=None)("{*}")
        
        independent_keys = ["attack"]
        dependent_keys = ["success_rate","median_elapsed_time","median_n_queries","mean_L0","median_L0"]
        reduced = reduce_keys(grid, dependent_keys, reduction=lambda x:x[0], group_by=["attack"])

        print(latex(reduced, independent_keys, dependent_keys, label_dict=label_dict))