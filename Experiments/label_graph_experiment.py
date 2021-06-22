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
        early_stopping = True
        boost = False
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

        from_label = grid[0]["sparse_fool"]["predicted"]
        to_label = grid[0]["sparse_fool"]["predicted_attacked"]

        edges = Counter(list(zip([class_labels[i] for i in from_label],[class_labels[i] for i in to_label])))
        edges = [(e[0],e[1],dict(edges)[e]) for e in dict(edges)]
        M = len(edges)

        fig = plt.figure(figsize=(15,15), constrained_layout=True)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        pos=nx.nx_agraph.graphviz_layout(G)
        options = {
            'node_size': 5000,
            'width': 2,
            'node_color': 'white',
            'node_shape': 's',
            'alpha': 0.5,
        }

        nx.draw_networkx(G, pos=pos, **options)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

        plt.savefig("Resources/Figures/IBM_connection_graph.pdf")
        plt.show(block=True)
