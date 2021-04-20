from architectures import BMNIST
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as np

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
        norm = 2
        max_hamming_distance = 200
        thresh = 0.1  # - For SCAR
        early_stopping = True
        boost = False
        verbose = True
        limit = 100
        lambda_ = 2.0
        rand_minmax = 0.01
        round_fn = "stoch_round"
        max_iter = 20
        epsilon = 0.02
        overshoot = 0.02
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
                "thresh": thresh,
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
                "max_iter_deep_fool":max_iter_deep_fool
            },
        )

        # grid = run(grid, scar_attack_on_test_set, n_threads=1, store_key="scar")(
        #     "{*}",
        #     "{max_hamming_distance}",
        #     "{thresh}",
        #     "{early_stopping}",
        #     "{verbose}",
        #     "{limit}",
        # )

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
        # )

        # grid = run(grid, prob_sparse_fool_on_test_set, n_threads=1, store_key="prob_sparse_fool")(
        #     "{*}",
        #     "{max_hamming_distance}",
        #     "{lambda_}",
        #     "{max_iter}",
        #     "{epsilon}",
        #     "{overshoot}",
        #     "{max_iter_deep_fool}",
        #     "{rand_minmax}",
        #     "{early_stopping}",
        #     "{boost}",
        #     "{verbose}",
        #     "{limit}"
        # )

        grid = run(grid, sparse_fool_on_test_set, n_threads=1, run_mode="force", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            "{lambda_}",
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{max_iter_deep_fool}",
            "{round_fn}",
            "{early_stopping}",
            "{boost}",
            "{verbose}",
            "{limit}"
        )

        def print_dict_summary(d):
            network_correct = d["predicted"] == d["targets"]
            sr = np.mean(d["success"][network_correct])
            median_elapsed_time = np.median(d["elapsed_time"][np.array(d["success"], dtype=bool) & network_correct]) 
            median_n_queries = np.median(d["n_queries"][np.array(d["success"],dtype=bool) & network_correct])
            mean_L0 = np.mean(d["L0"][np.array(d["success"],dtype=bool) & network_correct])
            median_L0 = np.median(d["L0"][np.array(d["success"],dtype=bool) & network_correct])
            print("%.4f \t\t %.2f \t\t %.2f \t\t %.2f" % (sr,median_n_queries,mean_L0,median_L0))

        attacks = ["sparse_fool","scar","prob_fool","non_prob_fool","prob_sparse_fool"]
        attacks = ["sparse_fool"]

        print("No Boost")
        for attack in attacks:
            result_dict = query(grid, attack, where={"boost":False})
            print(attack)
            print_dict_summary(result_dict[0])