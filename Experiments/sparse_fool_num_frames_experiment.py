from architectures import IBMGestures
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
from datajuicer.visualizers import *
from Experiments.bmnist_comparison_experiment import split_attack_grid, make_summary, label_dict
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['axes.spines.top'] = False
import matplotlib.pyplot as plt

class sparse_fool_num_frames_experiment:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = sparse_fool_num_frames_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        
        config = {
            "max_hamming_distance": 2000,
            "lambda_": 3.0,
            "verbose": True,
            "limit": 1000,
            "max_iter":20,
            "epsilon":0.0,
            "overshoot":0.02,
            "step_size":0.1,
            "max_iter_deep_fool":50,
            "use_snn":True
        }
        grid = configure(grid, config)

        num_frames = [1,2,5,10,20]
        grid = split(grid, "n_attack_frames", num_frames)

        grid = run(grid, frame_based_sparse_fool_on_test_set, n_threads=1, run_mode="normal", store_key="frame_based_sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            "{lambda_}",
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{n_attack_frames}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            "{use_snn}", # - Use SNN
        )

        grid = split_attack_grid(grid, ["frame_based_sparse_fool"])
        grid = run(grid, make_summary, store_key=None)("{*}")


        @visualizer(dim=2)
        def plot(table, axes_dict):
            shape = table.shape()
            for i0 in range(shape[0]):
                label = table.get_label(0,i0)
                ax = axes_dict[label]
                ax.set_xticks(np.arange(0.0,len(num_frames),1.0))
                ax.set_xticklabels(num_frames)
                ax.set_xlabel("Num. frames")
                ax.spines["top"].set_visible(False)
                success_rate = table.get_val(i0,0)
                median_L0 = table.get_val(i0,1)
                ax.set_title(label)
                ax.plot(success_rate, color="b")
                ax.tick_params(axis='y', labelcolor="b")
                ax_twin = ax.twinx()
                ax_twin.spines["top"].set_visible(False)
                ax_twin.plot(median_L0, color="r", linestyle="dashed")
                ax_twin.tick_params(axis='y', labelcolor="r")
                ax_twin.set_ylabel("Median L0")
                ax.set_ylabel("Success rate")

        fig = plt.figure(constrained_layout=True, figsize=(10,4))
        spec = mpl.gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
        axes = [fig.add_subplot(spec[0,i]) for i in range(1)]
        independent_keys = ["architecture"]
        dependent_keys = ["success_rate","median_L0"]
        axes_dict = {"IBMGestures":axes[0]}
        plot(grid, independent_keys=independent_keys,dependent_keys=dependent_keys,label_dict=label_dict, axes_dict=axes_dict)
        plt.savefig("Resources/Figures/vary_num_frames.pdf", dpi=1200)
        plt.show(block=False)