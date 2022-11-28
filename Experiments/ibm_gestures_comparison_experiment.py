from architectures import IBMGestures
from datajuicer import run, configure, reduce_keys
from experiment_utils import *
from datajuicer.visualizers import *
from Experiments.bmnist_comparison_experiment import split_attack_grid, make_summary, label_dict


class ibm_gestures_comparison_experiment:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = ibm_gestures_comparison_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        # net = grid[0]["snn"]
        # from experiment_utils import get_test_acc
        # data_loader = IBMGesturesDataLoader()
        # test_loader = data_loader.get_data_loader("test", shuffle=False)
        # test_acc = get_test_acc(test_loader, net)
        # print(f"Test acc {test_acc}")

        # state_dict = net.state_dict()
        # q_state_dict = {}
        # for n,v in state_dict.items():
        #     if "weight" in n:
        #         scale = (2 ** (8-1) - 1) / v.abs().max()
        #         v *= scale
        #         v = v.round() / scale
        #     q_state_dict[n] = v
        
        # net.load_state_dict(q_state_dict)

        # print(f"8 bit test acc is {get_test_acc(test_loader, net)}")

        max_hamming_distance = int(1e6)
        early_stopping = True
        boost = False
        verbose = True
        limit = 1000
        lambda_ = 3.0
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.1
        max_iter_deep_fool = 50
        n_attack_frames = 1
        N_pgd = 50
        N_MC = 5
        rand_minmax = 0.01
        round_fn = "stoch_round"
        eps = 1.5
        eps_iter = 0.3
        norm = 2

        # - Marchisio
        # - Hyperparams from https://github.com/albertomarchisio/DVS-Attacks/blob/main/DVS128Gesture/DVS128GestureAttacks.ipynb
        frame_sparsity = 479. / 1450.
        n_iter = 5
        lr = 1.

        # - Liang et al. https://arxiv.org/pdf/2001.01587.pdf
        n_iter_liang = 50
        prob_mult = 0.01

        grid = configure(
            grid,
            {
                "N_pgd": N_pgd,
                "N_MC": N_MC,
                "eps": eps,
                "eps_iter": eps_iter,
                "norm": norm,
                "rand_minmax": rand_minmax,
                "round_fn": round_fn,
                "max_hamming_distance": max_hamming_distance,
                "boost": boost,
                "early_stopping": early_stopping,
                "lambda_": lambda_,
                "verbose": verbose,
                "limit": limit,
                "max_iter":max_iter,
                "epsilon":epsilon,
                "overshoot":overshoot,
                "n_attack_frames":n_attack_frames,
                "step_size":step_size,
                "max_iter_deep_fool":max_iter_deep_fool,
                "frame_sparsity": frame_sparsity,
                "lr": lr,
                "n_iter": n_iter,
                "n_iter_liang": n_iter_liang,
                "prob_mult": prob_mult
            },
        )

        rep_grid_liang = [run(
            grid,
            liang_on_test_set_v2,
            n_threads=1,
            store_key="liang"
        )(
            "{*}",
            "{n_iter_liang}",
            "{prob_mult}",
            "{limit}",
            iter
        ) for iter in range(5)]

        rep_grid_march = [run(
            grid,
            marchisio_on_test_set_v2,
            n_threads=1,
            store_key="marchisio"
        )(
            "{*}",
            "{frame_sparsity}",
            "{lr}",
            "{n_iter}",
            "{limit}",
            iter
        ) for iter in range(5)]

        rep_grid_spike_fool_lambda_3 = [run(grid, sparse_fool_on_test_set_v2, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            3.0,
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            True,  # - Use SNN
            iter
        ) for iter in range(5)]

        rep_grid_spike_fool_lambda_2 = [run(grid, sparse_fool_on_test_set_v2, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            2.0,
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            True,  # - Use SNN
            iter
        ) for iter in range(5)]

        rep_grid_spike_fool_lambda_1 = [run(grid, sparse_fool_on_test_set_v2, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            1.0,
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            True,  # - Use SNN
            iter
        ) for iter in range(5)]

        rep_grid_prob_pgd = run(
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
            5000,
            "{boost}",
            "{early_stopping}",
            "{verbose}",
            "{limit}",
        )

        rep_grid_non_prob_pgd = run(grid, non_prob_fool_on_test_set, n_threads=1, store_key="non_prob_fool")(
            "{*}",
            "{N_pgd}",
            "{round_fn}",
            "{eps}",
            "{eps_iter}",
            "{rand_minmax}",
            "{norm}",
            5000,
            "{boost}",
            "{early_stopping}",
            "{verbose}",
            "{limit}",
            True,
        )

        def f(d):
            l0 = [float(el) for el in d["L0"]]
            d["L0"] = np.array(l0)
            predicted = [int(el) for el in d["predicted"]]
            targets = [int(el) for el in d["targets"]]
            queries = np.array([el for el in d["n_queries"]])
            network_correct = np.array(predicted) == np.array(targets)
            success_rate = 100 * np.mean(d["success"][network_correct])
            d["L0"][~np.array(d["success"]).astype(bool)] = np.iinfo(int).max  # max it could possibly be
            median_elapsed_time = np.median(d["elapsed_time"][network_correct])
            median_n_queries = np.median(queries[network_correct])
            median_L0 = np.median(d["L0"][network_correct])
            return success_rate, median_elapsed_time, median_n_queries, median_L0

        grid = [rep_grid_liang, rep_grid_march, rep_grid_spike_fool_lambda_1, [rep_grid_non_prob_pgd], [rep_grid_prob_pgd]]

        analysed_grid = [gg[0] for g in grid for gg in g]
        attacks = ["liang","marchisio","sparse_fool", "non prob", "prob"]
        results_dict = {
            a: {"success_rate":[], "median_elapsed_time":[], "median_n_queries":[], "median_L0":[]} for a in attacks
        }

        for attack in attacks:
            for g in analysed_grid:
                if attack in g:
                    success_rate, median_elapsed_time, median_n_queries, median_L0 = f(g[attack])
                    results_dict[attack]["success_rate"].append(success_rate)
                    results_dict[attack]["median_elapsed_time"].append(median_elapsed_time)
                    results_dict[attack]["median_n_queries"].append(median_n_queries)
                    results_dict[attack]["median_L0"].append(median_L0)

        for attack in attacks:
            for key in results_dict[attack]:
                m = np.mean(results_dict[attack][key])
                s = np.std(results_dict[attack][key])
                print(f"{attack} {key} {m} {s}")