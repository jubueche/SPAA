from architectures import IBMGestures
from datajuicer import run, configure
from datajuicer.utils import query, split
from experiment_utils import *
from datajuicer.visualizers import *
import matplotlib.pyplot as plt

beta_robustness = [0.0,0.01,0.05,0.1,0.2]

class sweep_beta_rob:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        grid = configure(grid, {"batch_size": 16, "boundary_loss": "trades", "epochs":2})
        grid = split(grid, "beta_robustness", beta_robustness)
        return grid

    @staticmethod
    def visualize():
        grid = sweep_beta_rob.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        max_hamming_distance = int(1e6)
        verbose = True
        limit = 500
        lambda_ = 3.0
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.1
        max_iter_deep_fool = 50

        grid = configure(
            grid,
            {
                "max_hamming_distance": max_hamming_distance,
                "lambda_": lambda_,
                "verbose": verbose,
                "limit": limit,
                "max_iter":max_iter,
                "epsilon":epsilon,
                "overshoot":overshoot,
                "step_size":step_size,
                "max_iter_deep_fool":max_iter_deep_fool
            },
        )

        def calc_data(grid):
            for g  in grid:
                sf = g["sparse_fool"]
                test_acc = np.mean(np.array(sf["predicted"] == sf["targets"], int))
                consider_index = np.array(sf["predicted"] == sf["targets"], bool)
                success_rate = np.sum(sf["success"][consider_index]) / np.sum(np.array(consider_index,int))
                median_L0 = np.median(sf["L0"][consider_index])
                g["test_acc"] = test_acc
                g["success_rate"] = success_rate
                g["median_L0"] = median_L0
            
            return grid

        grid = run(grid, sparse_fool_on_test_set, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            "{lambda_}",
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            True,  # - Use SNN
        )

        grid = calc_data(grid)

        fig = plt.figure(figsize=(5, 2), constrained_layout=True)
        ax = plt.gca()

        color = 'tab:red'
        ax.set_xlabel(r"TRADES $\beta_{rob}$")
        ax.set_ylabel("Median L0", color=color)
        median_L0s = []
        success_rates = []
        for beta_rob in beta_robustness:
            median_L0 = query(grid, "median_L0", where={"beta_robustness":beta_rob})
            success_rate = query(grid, "success_rate", where={"beta_robustness":beta_rob})
            median_L0s.append(median_L0)
            success_rates.append(success_rate)


        ax.plot(beta_robustness, median_L0s, color=color)
        # ax.set_xticklabels([str(b) for b in beta_robustness])
        ax.set_xticks(beta_robustness)
        ax.tick_params(axis='y', labelcolor=color)

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylim([0.95,1.0])
        color = 'tab:blue'
        # ax2.set_xticklabels([str(b) for b in beta_robustness])
        ax2.set_ylabel('Success rate (%)', color=color)  # we already handled the x-label with ax1
        ax2.plot(beta_robustness, success_rates, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig("tmp.pdf", dpi=1200)
        # plt.show()
        
        