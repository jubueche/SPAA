from architectures import IBMGestures
from dataloader_IBMGestures import IBMGesturesDataLoader
from sparsefool import sparsefool, reset
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from Experiments.visual_ibm_experiment import plot, generate_sample, class_labels

class visual_universal:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = visual_universal.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        net = grid[0]["snn"]

        ibm_gesture_dataloader = IBMGesturesDataLoader()

        data_loader_test = ibm_gesture_dataloader.get_data_loader(dset="test",
                                                                shuffle=False,
                                                                num_workers=4,
                                                                batch_size=1)

        max_hamming_distance = 500
        lambda_ = 2.0
        max_iter = 1 # - Max iter for universal attack (# rounds through batch)
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.1
        n_attack_frames = 1
        use_snn = True
        attack_fn_name = "sparsefool"
        sample_len_ms = 200.
        num_samples = 1 # - Number of samples per class label
        eviction = "Heatmap" # RandomEviction

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
                "max_hamming_distance":max_hamming_distance,
                "max_iter":max_iter,
                "attack_fn_name":attack_fn_name,
                "num_samples":num_samples,
                "use_snn":use_snn,
                "attack_fn":attack_fn,
                "eviction":eviction
            },
        )

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

        pert_total = grid[0]["pert_total"]
        def wrapper_attack_fn(X0):
            X0 = X0.clone()
            return_dict = {}
            reset(net)
            return_dict["predicted"] = torch.argmax(net.forward(X0).data).item()
            X0[:,pert_total] = 1. - X0[:,pert_total]
            reset(net)
            return_dict["predicted_attacked"] = torch.argmax(net.forward(X0).data).item()
            return_dict["X_adv"] = X0
            return return_dict

        source_labels = ["RH Wave","Air Guitar","Hand Clap"]
        target_labels = None

        samples = generate_sample(
            attack_fn=wrapper_attack_fn,
            data_loader=data_loader_test,
            source_label=source_labels,
            target_label=target_labels,
            num=len(source_labels),
            class_labels=class_labels)

        # - Create gridspec
        N_rows = 3
        N_cols = 5
        num_per_sample = int(N_rows*N_cols / len(samples))
        fig = plt.figure(constrained_layout=True, figsize=(10,6))
        spec = mpl.gridspec.GridSpec(ncols=N_cols, nrows=N_rows, figure=fig)
        axes = [fig.add_subplot(spec[i,j]) for i in range(N_rows) for j in range(N_cols)]

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False,
                            right=False,
                            left=False,
                            labelleft=False)

        sub_axes_samples = [(axes[i*num_per_sample:(i+1)*num_per_sample],samples[i],i,class_labels,sample_len_ms) for i in range(len(samples))]
        list(map(plot, sub_axes_samples))

        plt.savefig("Resources/Figures/universal_ibm_gestures.pdf")
        # plt.show(block=False)

        