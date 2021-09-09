from architectures import IBMGestures
from datajuicer import run, split, configure, query, run, reduce_keys
from sparsefool import sparsefool, frame_based_sparsefool
from experiment_utils import *
import numpy as np
from datajuicer.visualizers import *

class universal_vs_random_experiment:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = universal_vs_random_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        lambda_ = 2.0
        max_iter = 1 # - Max iter for universal attack (# rounds through batch)
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.1
        n_attack_frames = 1
        use_snn = True
        attack_fn_name = "sparsefool"
        num_samples = 1 # - Number of samples per class label

        max_hamming_distances = [500,750,1000,2000]

        def attack_fn(X,y):
            if attack_fn_name == "frame_based_sparsefool":
                return frame_based_sparsefool(
                    x_0=X,
                    net=grid[0]["snn"] if use_snn else grid[0]["ann"],
                    max_hamming_distance=int(1e6),
                    lambda_=lambda_,
                    epsilon=epsilon,
                    overshoot=overshoot,
                    n_attack_frames=n_attack_frames,
                    step_size=step_size,
                    device=device,
                    verbose=True,
                )
            elif attack_fn_name == "sparsefool":
                return sparsefool(
                    x_0=X,
                    net=grid[0]["snn"] if use_snn else grid[0]["ann"],
                    max_hamming_distance=int(1e6),
                    lambda_=lambda_,
                    device=device,
                    epsilon=epsilon,
                    overshoot=overshoot,
                    verbose=True
                )

        grid = configure(
            grid,
            {
                "attack_fn_name":attack_fn_name,
                "max_iter":max_iter,
                "num_samples":num_samples,
                "use_snn":use_snn,
                "attack_fn":attack_fn,
                "eviction":"Heatmap",
            },
        )

        grid = split(grid, "max_hamming_distance", max_hamming_distances)

        grid_heatmap = run(grid, universal_heatmap_attack_test_acc, n_threads=1, run_mode="normal", store_key="*")(
            "{*}",
            "{attack_fn}",
            "{attack_fn_name}",
            "{num_samples}",
            "{max_hamming_distance}",
            "{use_snn}"
        )

        grid_heatmap = configure(grid_heatmap, {"method":"heatmap"})

        grid_universal = run(grid, universal_attack_test_acc, n_threads=1, run_mode="normal", store_key="*")(
            "{*}",
            "{attack_fn}",
            "{attack_fn_name}",
            "{num_samples}",
            "{max_hamming_distance}",
            "{max_iter}",
            "{eviction}",
            "{use_snn}"
        )
        
        grid_universal = configure(grid_universal, {"method":"standard"})

        grid_random = run(grid, random_universal_test_acc, n_threads=1, run_mode="normal", store_key="*")(
            "{*}",
            "{max_hamming_distance}",
            "{use_snn}"
        )
        
        grid_random = configure(grid_random, {"method":"random"})

        indep = ["max_hamming_distance", "method"]
        dep = ["attacked_test_acc", "test_acc", "L0"]

        print(latex(grid_universal+grid_random + grid_heatmap, indep, dep))
        