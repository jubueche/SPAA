from architectures import IBMGestures
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
from datajuicer.visualizers import *
from Experiments.bmnist_comparison_experiment import split_attack_grid, make_summary, label_dict
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)
import networkx as nx

from Experiments.visual_ibm_experiment import class_labels
from collections import Counter

class label_graph_experiment:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():

        class_labels = [
            "Hand Clap",
            "RH Wave",
            "LH Wave",
            "RH Clock.",
            "RH Counter\nClockw.",
            "LH Clock.",
            "LH Counter\nClockw.",
            "Arm Roll",
            "Air Drums",
            "Air Guitar",
            "Other",
        ]

        grid = label_graph_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        max_hamming_distance = 2000
        verbose = True
        limit = 1000
        lambda_ = 3.0
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.1
        max_iter_deep_fool = 50
        n_attack_frames = 1

        grid = configure(
            grid,
            {
                "max_hamming_distance": max_hamming_distance,
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
            "{verbose}",
            "{limit}",
            True, # - Use SNN
        )

        from_label = grid[0]["sparse_fool"]["predicted"]
        to_label = grid[0]["sparse_fool"]["predicted_attacked"]

        transition_matrix = np.zeros(shape=(11,11))
        for (f,t) in zip(from_label,to_label):
            transition_matrix[f,t] += 1

        plt.matshow(transition_matrix)
        plt.colorbar()
        plt.ylabel("From label")
        plt.xlabel("To label")

        plt.savefig("Resources/Figures/IBM_connection_graph.pdf")
        # plt.show(block=True)
