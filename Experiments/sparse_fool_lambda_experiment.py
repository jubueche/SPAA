from architectures import BMNIST, NMNIST, IBMGestures
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
from datajuicer.visualizers import *
from Experiments.bmnist_comparison_experiment import split_attack_grid, make_summary, label_dict
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

class sparse_fool_lambda_experiment:
    @staticmethod
    def train_grid():
        grid = [BMNIST.make(),NMNIST.make(),IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = sparse_fool_lambda_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        # - Split the grids by architecture
        grid_bmnist = [g for g in grid if g["architecture"]=="BMNIST"]
        grid_nmnist = [g for g in grid if g["architecture"]=="NMNIST"]
        grid_ibm = [g for g in grid if g["architecture"]=="IBMGestures"]
        
        lambdas = [1.0,2.0,3.0,4.0,5.0]

        base_config = {
            "early_stopping": True,
            "boost": False,
            "verbose": True,
            "max_iter": 20,
            "epsilon": 0.0,
            "overshoot": 0.02,
            "max_iter_deep_fool": 50
        }

        bmnist_config = {
            "max_hamming_distance":200,
            "limit":1000,
            "step_size":0.01,
            "use_snn": False,
            **base_config}

        nmnist_config = {
            "max_hamming_distance":1000,
            "limit":1000,
            "step_size":0.02,
            "use_snn": True,
            **base_config}

        ibm_config = {
            "max_hamming_distance":2000,
            "limit":1000,
            "step_size":0.1,
            "use_snn": True,
            **base_config}
        
        grid_bmnist = configure(grid_bmnist, bmnist_config)
        grid_nmnist = configure(grid_nmnist, nmnist_config)
        grid_ibm = configure(grid_ibm, ibm_config)
        
        grid = split(grid_bmnist + grid_nmnist + grid_ibm, "lambda_", lambdas)

        grid = run(grid, sparse_fool_on_test_set, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            "{lambda_}",
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{early_stopping}",
            "{boost}",
            "{verbose}",
            "{limit}",
            "{use_snn}",
        )

        grid = split_attack_grid(grid, ["sparse_fool"])
        grid = run(grid, make_summary, store_key=None)("{*}")


        @visualizer(dim=2)
        def plot(table, axes_dict):
            shape = table.shape()
            for i0 in range(shape[0]):
                label = table.get_label(0,i0)
                ax = axes_dict[label]
                ax.spines["top"].set_visible(False)
                ax.set_xticks(np.arange(0.0,len(lambdas),1.0))
                ax.set_xticklabels(lambdas)
                ax.set_xlabel(r"$\lambda$")
                success_rate = table.get_val(i0,0)
                median_L0 = table.get_val(i0,1)
                ax.set_title(label)
                ax.plot(success_rate, color="b")
                ax.tick_params(axis='y', labelcolor="b")
                ax_twin = ax.twinx()
                ax_twin.spines["top"].set_visible(False)
                ax_twin.plot(median_L0, color="r", linestyle="dashed")
                ax_twin.tick_params(axis='y', labelcolor="r")
                if i0 == 1: ax_twin.set_ylabel("Median L0")
                if i0 == 0: ax.set_ylabel("Success rate")

        fig = plt.figure(constrained_layout=True, figsize=(10,4))
        spec = mpl.gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
        axes = [fig.add_subplot(spec[0,i]) for i in range(3)]
        independent_keys = ["architecture"]
        dependent_keys = ["success_rate","median_L0"]
        axes_dict = {"BMNIST":axes[0], "NMNIST":axes[1], "IBMGestures":axes[2]}
        plot(grid, independent_keys=independent_keys,dependent_keys=dependent_keys,label_dict=label_dict, axes_dict=axes_dict)
        plt.savefig("Resources/Figures/vary_lambda.pdf", dpi=1200)
        plt.show(block=False)