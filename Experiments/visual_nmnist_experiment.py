from architectures import NMNIST
from dataloader_NMNIST import NMNISTDataLoader
from sparsefool import sparsefool
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['xtick.bottom'] = False
mpl.rcParams['ytick.left'] = False
mpl.rcParams['xtick.labelbottom'] = False
mpl.rcParams['ytick.labelleft'] = False
import matplotlib.pyplot as plt

from Experiments.visual_ibm_experiment import generate_sample, plot

class_labels = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine"
]

class visual_nmnist_experiment:
    @staticmethod
    def train_grid():
        grid = [NMNIST.make()]
        return grid

    @staticmethod
    def visualize():
        grid = visual_nmnist_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        net = grid[0]["snn"]

        nmnist_dataloader = NMNISTDataLoader()

        data_loader_test = nmnist_dataloader.get_data_loader(dset="test",
                                                                mode="snn",
                                                                shuffle=False,
                                                                num_workers=4,
                                                                batch_size=1)

        max_hamming_distance = 1000
        lambda_ = 1.0
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.02
        max_iter_deep_fool = 50
        n_attack_frames = 1

        def attack_fn(X0):
            X0 = X0.squeeze()
            d = sparsefool(
                x_0=X0,
                net=net,
                max_hamming_distance=max_hamming_distance,
                lambda_=lambda_,
                max_iter=max_iter,
                epsilon=epsilon,
                overshoot=overshoot,
                step_size=step_size,
                max_iter_deep_fool=max_iter_deep_fool,
                device=device,
                early_stopping=True,
                boost=False,
                verbose=True
            )
            return d

        source_labels = ["Two","Three","Nine"]
        target_labels = None

        samples = generate_sample(
            attack_fn=attack_fn,
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
        sub_axes_samples = [(axes[i*num_per_sample:(i+1)*num_per_sample],samples[i],i,class_labels) for i in range(len(samples))]
        list(map(plot, sub_axes_samples))

        plt.savefig("Resources/Figures/samples_nmnist.pdf")
        plt.show()

        