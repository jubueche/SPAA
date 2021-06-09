from architectures import IBMGestures
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as np

class ibm_gestures_experiment:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = ibm_gestures_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        max_hamming_distance = 500
        early_stopping = True
        boost = False
        verbose = True
        limit = 30
        lambda_ = 3.0
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.05
        max_iter_deep_fool = 50
        n_attack_frames = 1

        grid = configure(
            grid,
            {
                "max_hamming_distance": max_hamming_distance,
                "boost": boost,
                "early_stopping": early_stopping,
                "lambda_": lambda_,
                "verbose": verbose,
                "limit": limit,
                "max_iter":max_iter,
                "epsilon":epsilon,
                "overshoot":overshoot,
                "n_attack_frames":n_attack_frames,
                "step_size":step_size,
                "max_iter_deep_fool":max_iter_deep_fool
            },
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
            "{early_stopping}",
            "{boost}",
            "{verbose}",
            "{limit}",
            True, # - Use SNN
        )

        grid = run(grid, frame_based_sparse_fool_on_test_set, n_threads=1, run_mode="normal", store_key="frame_based_sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            "{lambda_}",
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{n_attack_frames}",
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

        attacks = ["sparse_fool","frame_based_sparse_fool"]

        for attack in attacks:
            result_dict = query(grid, attack, where={"boost":False})
            print(attack)
            print_dict_summary(result_dict[0])