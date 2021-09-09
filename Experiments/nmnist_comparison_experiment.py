from architectures import NMNIST
from datajuicer import run, configure, reduce_keys
from experiment_utils import *
from datajuicer.visualizers import *
from Experiments.bmnist_comparison_experiment import split_attack_grid, make_summary, label_dict


class nmnist_comparison_experiment:
    @staticmethod
    def train_grid():
        grid = [NMNIST.make()]
        return grid

    @staticmethod
    def visualize():
        grid = nmnist_comparison_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        N_pgd = 50
        N_MC = 5
        eps = 1.5
        eps_iter = 0.3
        norm = 2
        max_hamming_distance = 1000
        early_stopping = True
        boost = False
        verbose = True
        limit = 10
        lambda_ = 2.0
        rand_minmax = 0.01
        round_fn = "stoch_round"
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.2
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
                "max_iter": max_iter,
                "epsilon": epsilon,
                "overshoot": overshoot,
                "step_size": step_size,
                "max_iter_deep_fool": max_iter_deep_fool
            },
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
            True,
        )

        grid = run(grid, sparse_fool_on_test_set, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            "{lambda_}",
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            True, # - Use SNN
        )

        attacks = ["prob_fool","non_prob_fool","sparse_fool"]
        grid = split_attack_grid(grid, attacks)

        grid = run(grid, make_summary, store_key=None)("{*}")

        independent_keys = ["attack"]
        dependent_keys = ["success_rate","median_elapsed_time","median_n_queries","mean_L0","median_L0"]
        reduced = reduce_keys(grid, dependent_keys, reduction=lambda x:x[0], group_by=["attack"])

        print(latex(reduced, independent_keys, dependent_keys, label_dict=label_dict))
