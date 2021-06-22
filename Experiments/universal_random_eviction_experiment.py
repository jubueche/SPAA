from architectures import IBMGestures
from datajuicer import run, split, configure, query, run, reduce_keys
from sparsefool import sparsefool, frame_based_sparsefool
from experiment_utils import *
import numpy as np
from datajuicer.visualizers import *

class universal_random_eviction_experiment:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = universal_random_eviction_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        max_hamming_distance = 500
        lambda_ = 2.0
        max_iter = 3 # - Max iter for universal attack (# rounds through batch)
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.1
        max_iter_deep_fool = 50
        n_attack_frames = 1
        use_snn = True
        attack_fn_name = "sparsefool"
        num_samples = 8 # - Number of samples per class label

        def attack_fn(X,y):
            if attack_fn_name == "frame_based_sparsefool":
                return frame_based_sparsefool(
                    x_0=X,
                    net=grid[0]["snn"] if use_snn else grid[0]["ann"],
                    max_hamming_distance=5000,
                    lambda_=lambda_,
                    epsilon=epsilon,
                    overshoot=overshoot,
                    n_attack_frames=n_attack_frames,
                    step_size=step_size,
                    device=device,
                    early_stopping=False,
                    boost=False,
                    verbose=True,
                )
            elif attack_fn_name == "sparsefool":
                return sparsefool(
                    x_0=X,
                    net=grid[0]["snn"] if use_snn else grid[0]["ann"],
                    max_hamming_distance=5000,
                    lambda_=lambda_,
                    device=device,
                    epsilon=epsilon,
                    overshoot=overshoot,
                    early_stopping=True,
                    boost=False,
                    verbose=True
                )

        grid = configure(
            grid,
            {
                "max_hamming_distance":max_hamming_distance,
                "max_iter":max_iter,
                "attack_fn_name":attack_fn_name,
                "num_samples":num_samples,
                "use_snn":use_snn,
                "attack_fn":attack_fn
            },
        )

        grid = split(grid, "eviction", ["Heatmap", "RandomEviction"])

        grid = run(grid, universal_attack_test_acc, n_threads=1, run_mode="normal", store_key="*")(
            "{*}",
            "{attack_fn}",
            "{attack_fn_name}",
            "{num_samples}",
            "{max_hamming_distance}",
            "{max_iter}",
            "{eviction}",
            "{use_snn}"
        )

        indep = ["eviction"]
        dep = ["attacked_test_acc", "test_acc", "L0"]

        print(latex(grid, indep, dep))
        