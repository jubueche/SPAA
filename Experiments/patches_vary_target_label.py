from architectures import IBMGestures
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
from datajuicer.visualizers import *
from Experiments.bmnist_comparison_experiment import label_dict

class patches_vary_target_label:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = patches_vary_target_label.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        # - Hyperparams for adversarial patch
        n_epochs = 5
        patch_type = 'circle'
        input_shape = (20,2,128,128)
        patch_size = 0.05
        target_labels = np.arange(11)
        # target_labels = [2]
        max_iter = 20 # - Number of samples per epoch
        eval_after = -1 # _ Evaluate after X samples
        max_iter_test = 100
        label_conf = 0.75
        max_count = 300

        grid = configure(
            grid,
            {
                "n_epochs": n_epochs,
                "patch_type": patch_type,
                "input_shape": input_shape,
                "patch_size": patch_size,
                "max_iter": max_iter,
                "eval_after": eval_after,
                "max_iter_test":max_iter_test,
                "label_conf":label_conf,
                "max_count":max_count
            },
        )

        grid = split(grid, "target_label", target_labels)

        grid = run(grid, adversarial_patches_exp, n_threads=1, run_mode="normal", store_key="*")(
            "{*}",
            "{n_epochs}",
            "{target_label}",
            "{patch_type}",
            "{input_shape}",
            "{patch_size}",
            "{max_iter}",
            "{eval_after}",
            "{max_iter_test}",
            "{label_conf}",
            "{max_count}",
            True
        )
        