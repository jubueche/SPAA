from architectures import NMNIST
from dataloader_NMNIST import NMNISTDataLoader
from sparsefool import sparsefool
from datajuicer import run
import matplotlib as mpl
import matplotlib.pyplot as plt

from Experiments.visual_ibm_experiment import generate_sample, plot
from experiment_utils import device


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
        net = grid[0]["snn"].to(device)

        nmnist_dataloader = NMNISTDataLoader()

        data_loader_test = nmnist_dataloader.get_data_loader(dset="test",
                                                             mode="snn",
                                                             shuffle=False,
                                                             num_workers=4,
                                                             batch_size=1)

        # print("Test accuracy", get_test_acc(data_loader_test, net))

        max_hamming_distance = int(1e6)
        lambda_ = 1.0
        max_iter = 5
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.2
        max_iter_deep_fool = 50

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
                verbose=True
            )
            return d

        source_labels = ["Two", "Three", "Nine"]
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
        N_cols = 10
        sample_len_ms = 300.
        num_per_sample = int(N_rows*N_cols / len(samples))
        fig = plt.figure(constrained_layout=True, figsize=(12, 4))
        spec = mpl.gridspec.GridSpec(ncols=N_cols, nrows=N_rows, figure=fig)
        axes = [fig.add_subplot(spec[i, j]) for i in range(N_rows) for j in range(N_cols)]

        for ax in axes:
            # ax.spines['right'].set_visible(False)
            # ax.spines['top'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           top=False,
                           labelbottom=False,
                           right=False,
                           left=False,
                           labelleft=False)

        sub_axes_samples = [(
            axes[i*num_per_sample:(i+1)*num_per_sample],
            samples[i],
            i,
            class_labels,
            sample_len_ms,
            i == N_rows - 1
        ) for i in range(len(samples))]
        list(map(plot, sub_axes_samples))

        plt.savefig("Resources/Figures/samples_nmnist.pdf", bbox_inches='tight')
        # plt.show(block=False)
