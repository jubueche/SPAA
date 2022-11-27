from architectures import IBMGestures
from datajuicer import run, configure, reduce_keys
from experiment_utils import *
from datajuicer.visualizers import *
from Experiments.bmnist_comparison_experiment import split_attack_grid, make_summary, label_dict


class ibm_gestures_comparison_experiment:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = ibm_gestures_comparison_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        max_hamming_distance = int(1e6)
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

        # - Marchisio
        # - Hyperparams from https://github.com/albertomarchisio/DVS-Attacks/blob/main/DVS128Gesture/DVS128GestureAttacks.ipynb
        frame_sparsity = 479. / 1450.
        n_iter = 5
        lr = 1.

        # - Liang et al. https://arxiv.org/pdf/2001.01587.pdf
        n_iter_liang = 50
        prob_mult = 0.01

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
                "max_iter_deep_fool":max_iter_deep_fool,
                "frame_sparsity": frame_sparsity,
                "lr": lr,
                "n_iter": n_iter,
                "n_iter_liang": n_iter_liang,
                "prob_mult": prob_mult
            },
        )

        rep_grid_liang = [run(
            grid,
            liang_on_test_set_v2,
            n_threads=1,
            store_key="liang"
        )(
            "{*}",
            "{n_iter_liang}",
            "{prob_mult}",
            "{limit}",
            iter
        ) for iter in range(5)]

        rep_grid_march = [run(
            grid,
            marchisio_on_test_set_v2,
            n_threads=1,
            store_key="marchisio"
        )(
            "{*}",
            "{frame_sparsity}",
            "{lr}",
            "{n_iter}",
            "{limit}",
            iter
        ) for iter in range(5)]

        rep_grid_spike_fool_lambda_3 = [run(grid, sparse_fool_on_test_set_v2, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            3.0,
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            True,  # - Use SNN
            iter
        ) for iter in range(5)]

        rep_grid_spike_fool_lambda_2 = [run(grid, sparse_fool_on_test_set_v2, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            2.0,
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            True,  # - Use SNN
            iter
        ) for iter in range(5)]

        rep_grid_spike_fool_lambda_1 = [run(grid, sparse_fool_on_test_set_v2, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            1.0,
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            True,  # - Use SNN
            iter
        ) for iter in range(5)]

        # attacks = ["liang","marchisio","sparse_fool"]
        # grid = split_attack_grid(grid, attacks)

        # grid = run(grid, make_summary, store_key=None)("{*}")

        # independent_keys = ["attack"]
        # dependent_keys = ["success_rate","median_elapsed_time","median_n_queries","mean_L0","median_L0"]
        # reduced = reduce_keys(grid, dependent_keys, reduction=lambda x:x[0], group_by=["attack"])

        # print(latex(reduced, independent_keys, dependent_keys, label_dict=label_dict))