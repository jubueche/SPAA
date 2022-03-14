from architectures import NMNIST
from datajuicer import run, configure, reduce_keys, split
from experiment_utils import *
from datajuicer.visualizers import *
from Experiments.bmnist_comparison_experiment import split_attack_grid, make_summary, label_dict


class sweep_pga_nmnist:
    @staticmethod
    def train_grid():
        grid = [NMNIST.make()]
        return grid

    @staticmethod
    def visualize():
        grid = sweep_pga_nmnist.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        eps = [0.2,0.6,1.0,1.4]
        eps_iter = [0.1,0.2,0.3]

        # grid = split(grid, "eps", eps)
        # grid = split(grid, "eps_iter", eps_iter)
        # grid = split(grid, "norm", ["np.inf","2"])
        # grid = split(grid, "N_pgd", [50])

        grid = split(grid, "eps", [0.2])
        grid = split(grid, "eps_iter", [0.1])
        grid = split(grid, "norm", ["np.inf"])
        grid = split(grid, "N_pgd", [50])

        grid_ = []
        for g in grid:
            if g["eps"] > g["eps_iter"]:
                grid_.append(g)
        grid = grid_

        N_MC = 5
        max_hamming_distance = 5000
        early_stopping = True
        boost = False
        verbose = True
        limit = 1000
        rand_minmax = 0.01
        round_fn = "round"
        batch_size = 100

        grid = configure(
            grid,
            {
                "N_MC": N_MC,
                "max_hamming_distance": max_hamming_distance,
                "boost": boost,
                "early_stopping": early_stopping,
                "round_fn": round_fn,
                "verbose": verbose,
                "limit": limit,
                "rand_minmax": rand_minmax
            },
        )

        grid = run(
            grid,
            prob_fool_on_test_set,
            n_threads=1,
            store_key="prob_fool",
        )(
            "{*}",
            "{N_pgd}",
            "{N_MC}",
            "{eps}",
            "{eps_iter}",
            "{rand_minmax}",
            "{norm}",
            "{max_hamming_distance}",
            "{boost}",
            "{early_stopping}",
            "{verbose}",
            "{limit}",
        )

        # grid = run(grid, non_prob_fool_on_test_set, n_threads=1, store_key="non_prob_fool")(
        #     "{*}",
        #     "{N_pgd}",
        #     "{round_fn}",
        #     "{eps}",
        #     "{eps_iter}",
        #     "{rand_minmax}",
        #     "{norm}",
        #     "{max_hamming_distance}",
        #     "{boost}",
        #     "{early_stopping}",
        #     "{verbose}",
        #     "{limit}",
        #     True,
        #     batch_size
        # )

        grid = split_attack_grid(grid, attacks = ["prob_fool"])
        grid = run(grid, make_summary, store_key=None)("{*}")

        for g in grid:
            print("Attack %s n_queries %d N_pgd %d eps %.3f eps_iter %.3f norm %s \t\t success_rate %.4f median_elapsed_time %.4f \
                median_L0 %.4f" % (g["attack"],g["median_n_queries"],g["N_pgd"],g["eps"],g["eps_iter"],g["norm"],g["success_rate"],g["median_elapsed_time"],g["median_L0"]))
