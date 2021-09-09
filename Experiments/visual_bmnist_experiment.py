from architectures import BMNIST
from dataloader_BMNIST import BMNISTDataLoader
from sparsefool import sparsefool
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['xtick.bottom'] = False
mpl.rcParams['ytick.left'] = False
mpl.rcParams['xtick.labelbottom'] = False
mpl.rcParams['ytick.labelleft'] = False
import matplotlib.pyplot as plt

from Experiments.visual_ibm_experiment import generate_sample

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

def plot(args):
    ax, sample, idx, class_labels = args
    X0 = sample["X0"].squeeze()
    X_adv = sample["X_adv"].squeeze()
    X_diff = torch.abs(X0-X_adv).cpu().numpy()[::-1]
    X0 = X0.cpu().numpy()[::-1]
    ax.pcolormesh(X0, vmin=0, vmax=2, cmap=plt.cm.Blues)
    ax.pcolormesh(np.ma.masked_array(X_diff,X_diff==0.), vmin=0, vmax=2, cmap=plt.cm.Reds)
    ax.set_ylabel(class_labels[sample["predicted"]] + r"$\rightarrow$" + class_labels[sample["predicted_attacked"]])

class visual_bmnist_experiment:
    @staticmethod
    def train_grid():
        grid = [BMNIST.make()]
        return grid

    @staticmethod
    def visualize():
        grid = visual_bmnist_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        net = grid[0]["ann"]

        bmnist_dataloader = BMNISTDataLoader()

        data_loader_test = bmnist_dataloader.get_data_loader(dset="test",
                                                                shuffle=False,
                                                                num_workers=4,
                                                                batch_size=1)

        max_hamming_distance = int(1e6)
        lambda_ = 1.0
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.01
        max_iter_deep_fool = 50

        def attack_fn(X0):
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
                verbose=True
            )
            return d

        source_labels = class_labels
        target_labels = None

        samples = generate_sample(
            attack_fn=attack_fn,
            data_loader=data_loader_test,
            source_label=source_labels,
            target_label=target_labels,
            num=len(source_labels),
            class_labels=class_labels)

        # - Create gridspec
        N_rows = 2
        N_cols = 5
        fig = plt.figure(constrained_layout=True, figsize=(10,6))
        spec = mpl.gridspec.GridSpec(ncols=N_cols, nrows=N_rows, figure=fig)
        axes = [fig.add_subplot(spec[i,j]) for i in range(N_rows) for j in range(N_cols)]
        sub_axes_samples = [(axes[i],samples[i],i,class_labels) for i in range(len(samples))]
        list(map(plot, sub_axes_samples))

        plt.savefig("Resources/Figures/samples_bmnist.pdf")
        # plt.show(block=False)

        