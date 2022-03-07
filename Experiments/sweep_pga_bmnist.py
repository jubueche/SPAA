from architectures import BMNIST
from datajuicer import configure, run, split, reduce_keys
from experiment_utils import *
import numpy as np
from copy import deepcopy
from datajuicer.visualizers import *


def split_attack_grid(grid, attacks):
    grid_tmp = []
    for g in grid:
        for attack in attacks:
            g_tmp = deepcopy(g)
            g_tmp["attack"] = attack
            g_tmp["attack_result"] = g[attack]
            for attack in attacks:
                g_tmp.pop(attack)
            grid_tmp.append(g_tmp)
    return grid_tmp

def torch2np(ar):
    if isinstance(ar[0], torch.Tensor):
        for idx,el in enumerate(ar):
            ar[idx] = float(el.cpu().float())
    return ar

def make_summary(d):
    d["attack_result"]["predicted"] = torch2np(d["attack_result"]["predicted"])
    d["attack_result"]["targets"] = torch2np(d["attack_result"]["targets"])
    d["attack_result"]["success"] = torch2np(d["attack_result"]["success"])
    network_correct = d["attack_result"]["predicted"] == d["attack_result"]["targets"]
    d["success_rate"] = 100 * np.mean(d["attack_result"]["success"][network_correct])
    d["attack_result"]["L0"][~np.array(d["attack_result"]["success"]).astype(bool)] = np.iinfo(int).max  # max it could possibly be

    d["median_elapsed_time"] = np.median(d["attack_result"]["elapsed_time"][network_correct])
    d["median_n_queries"] = np.median(d["attack_result"]["n_queries"][network_correct])
    d["mean_L0"] = np.nan
    d["median_L0"] = np.median(d["attack_result"]["L0"][network_correct])
    return d


label_dict = {
    "scar": "SCAR",
    "prob_fool": "Prob. PGA",
    "non_prob_fool": "PGA",
    "sparse_fool": "Sparse Fool",
    "frame_based_sparse_fool": "Frame Based Sparse Fool",
    "success_rate": "Success Rate",
    "median_elapsed_time": "Median Elapsed Time",
    "median_n_queries": "Median No. Queries",
    "mean_L0": "Mean L0 distance",
    "median_L0": "Median L0 distance",
    "attack": "Attack",
}


class sweep_pga_bmnist:
    @staticmethod
    def train_grid():
        grid = [BMNIST.make()]
        return grid

    @staticmethod
    def visualize():
        grid = sweep_pga_bmnist.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        eps = [0.2,0.4,0.6,0.8,1.0,1.2,1.4]
        eps_iter = [0.05,0.1,0.15,0.2,0.3]

        grid = split(grid, "eps", eps)
        grid = split(grid, "eps_iter", eps_iter)
        grid = split(grid, "norm", ["np.inf","2"])
        grid = split(grid, "N_pgd", [10,50])

        grid_ = []
        for g in grid:
            if g["eps"] > g["eps_iter"]:
                grid_.append(g)
        grid = grid_

        N_MC = 5
        max_hamming_distance = int(1e6)
        early_stopping = True
        boost = False
        verbose = True
        limit = 1000
        rand_minmax = 0.01
        round_fn = "round"

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
                "rand_minmax": rand_minmax,
            },
        )

        grid = run(grid, non_prob_fool_on_test_set, n_threads=1, store_key="non_prob_fool")(
            "{*}",
            "{N_pgd}",
            "{round_fn}",
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

        grid = run(grid, prob_fool_on_test_set, n_threads=1, store_key="prob_fool")(
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

        grid = split_attack_grid(grid, attacks = ["non_prob_fool","prob_fool"])
        grid = run(grid, make_summary, store_key=None)("{*}")

        for g in grid:
            print("Attack %s N_pgd %d eps %.3f eps_iter %.3f norm %s \t\t success_rate %.4f median_elapsed_time %.4f \
                median_L0 %.4f" % (g["attack"],g["N_pgd"],g["eps"],g["eps_iter"],g["norm"],g["success_rate"],g["median_elapsed_time"],g["median_L0"]))