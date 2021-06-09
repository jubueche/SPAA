from architectures import NMNIST
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as np

class nmnist_experiment:
    @staticmethod
    def train_grid():
        grid = [NMNIST.make()]
        return grid

    @staticmethod
    def visualize():
        grid = nmnist_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        N_pgd = 50
        N_MC = 5
        eps = 1.5
        eps_iter = 0.3
        norm = 2
        max_hamming_distance = 1000
        early_stopping = True
        boost = False
        verbose = False
        limit = 500
        lambda_ = 1.0
        rand_minmax = 0.01
        round_fn = "stoch_round"
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.02
        max_iter_deep_fool = 50

        grid = configure(
            grid,
            {
                "N_pgd": N_pgd,
                "N_MC": N_MC,
                "eps": eps,
                "eps_iter": eps_iter,
                "norm": norm,
                "max_hamming_distance": max_hamming_distance,
                "boost": boost,
                "early_stopping": early_stopping,
                "lambda_": lambda_,
                "round_fn": round_fn,
                "verbose": verbose,
                "limit": limit,
                "rand_minmax": rand_minmax,
                "max_iter":max_iter,
                "epsilon":epsilon,
                "overshoot":overshoot,
                "step_size":step_size,
                "max_iter_deep_fool":max_iter_deep_fool
            },
        )

        # grid = run(
        #     grid,
        #     prob_fool_on_test_set,
        #     n_threads=1,
        #     store_key="prob_fool",
        # )(
        #     "{*}",
        #     "{N_pgd}",
        #     "{N_MC}",
        #     "{eps}",
        #     "{eps_iter}",
        #     "{rand_minmax}",
        #     "{norm}",
        #     "{max_hamming_distance}",
        #     "{boost}",
        #     "{early_stopping}",
        #     "{verbose}",
        #     "{limit}",
        # )

        # grid = run(grid, non_prob_fool_on_test_set, n_threads=1, store_key="non_prob_fool")(
        #     "{*}",
        #     "{N_pgd}",
        #     "{round_fn}",
        #     "{eps}",
        #     "{eps_iter}",
        #     "{rand_minmax}",
        #     "{norm}",
        #     "{max_hamming_distance}",
        #     "{boost}",
        #     "{early_stopping}",
        #     "{verbose}",
        #     "{limit}",
        #     True,
        # )

        grid = run(grid, sparse_fool_on_test_set, n_threads=1, run_mode="normal", store_key="sparse_fool")(
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
            "{limit}",
            True, # - Use SNN
        )

        def print_dict_summary(d):
            network_correct = d["predicted"] == d["targets"]
            sr = np.mean(d["success"][network_correct])
            median_elapsed_time = np.median(d["elapsed_time"][np.array(d["success"], dtype=bool) & network_correct])
            median_n_queries = np.median(d["n_queries"][np.array(d["success"],dtype=bool) & network_correct])
            mean_L0 = np.mean(d["L0"][np.array(d["success"],dtype=bool) & network_correct])
            median_L0 = np.median(d["L0"][np.array(d["success"],dtype=bool) & network_correct])
            print("%.4f \t\t %.2f \t\t %.2f \t\t %.2f" % (sr,median_n_queries,mean_L0,median_L0))

        attacks = ["sparse_fool"] # ,"prob_fool","non_prob_fool"]

        for attack in attacks:
            result_dict = query(grid, attack, where={"boost":False})
            print(attack)
            print_dict_summary(result_dict[0])
