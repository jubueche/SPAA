from architectures import IBMGestures
from dataloader_IBMGestures import IBMGesturesDataLoader
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sparsefool import reset
from torch.autograd import Variable

from Experiments.visual_ibm_experiment import plot, generate_sample, class_labels

class visual_patches:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = visual_patches.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        net = grid[0]['snn']

        # - Hyperparams for adversarial patch
        n_epochs = 5
        patch_type = 'circle'
        input_shape = (20,2,128,128)
        patch_size = 0.05
        target_label = 7
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
                "max_count":max_count,
                "target_label":target_label
            },
        )

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

        ibm_gesture_dataloader = IBMGesturesDataLoader()

        data_loader_test = ibm_gesture_dataloader.get_data_loader(dset="test",
                                                                shuffle=False,
                                                                num_workers=4,
                                                                batch_size=1)

        patch = query(grid, "pert_total")[0]
        patch_mask = query(grid, "patch_mask")[0]

        def attack_fn(X):
            X_adv = torch.round(torch.clamp((1. - patch_mask) * X + patch, 0., 1.))
            reset(net)
            adv_label = torch.argmax(net.forward(Variable(X_adv, requires_grad=False)).data).item()
            reset(net)
            pred_label = torch.argmax(net.forward(Variable(X, requires_grad=False)).data).item()
            return {
                "predicted_attacked": adv_label,
                "predicted": pred_label,
                "X0": X,
                "X_adv": X_adv
            }

        source_labels=["RH Wave","Air Guitar","Hand Clap"]

        samples = generate_sample(
                                attack_fn=attack_fn,
                                data_loader=data_loader_test,
                                source_label=source_labels,
                                target_label=[class_labels[target_label]],
                                num=len(source_labels),
                                class_labels=class_labels
                                )

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

        sub_axes_samples = [(axes[i*num_per_sample:(i+1)*num_per_sample],samples[i],i,class_labels) for i in range(len(samples))]
        list(map(plot, sub_axes_samples))

        plt.savefig("Resources/Figures/adversarial_patch.pdf")
        plt.show(block=False)

        