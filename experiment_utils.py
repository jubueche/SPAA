import torch
import numpy as np
from dataloader_NMNIST import NMNISTDataLoader
from dataloader_BMNIST import BMNISTDataLoader
from datajuicer import cachable
from concurrent.futures import ThreadPoolExecutor, as_completed

from attacks import prob_attack_pgd, boosted_hamming_attack, hamming_attack, scar_attack
from utils import get_prediction

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


@cachable(dependencies=["model:{architecture}_session_id", "eps", "eps_iter", "N_pgd", "N_MC", "norm", "rand_minmax", "limit", "N_samples"])
def get_prob_attack_robustness(
    model,
    eps,
    eps_iter,
    N_pgd,
    N_MC,
    norm,
    rand_minmax,
    limit,
    N_samples
):
    if model['architecture'] == "NMNIST":
        nmnist_dataloader = NMNISTDataLoader()
        data_loader = nmnist_dataloader.get_data_loader(dset="test", mode="snn", shuffle=True, num_workers=4, batch_size=1)
    else:
        assert model['architecture'] in ["NMNIST"], "No other architecture added so far"

    defense_probabilities = []
    for idx, (batch, target) in enumerate(data_loader):
        if idx == limit:
            break

        batch = torch.clamp(batch, 0.0, 1.0)

        P_adv = prob_attack_pgd(
            model['prob_net'],
            batch[0],
            eps,
            eps_iter,
            N_pgd,
            N_MC,
            norm,
            rand_minmax
        )

        correct = []
        for _ in range(N_samples):
            model_pred = get_prediction(model['prob_net'], P_adv, "prob")
            if model_pred == target:
                correct.append(1.0)

        defense_probabilities.append(float(sum(correct) / N_samples))

    return np.mean(np.array(defense_probabilities))


def get_data_loader_from_model(model, batch_size=1, max_size=10000):
    if model['architecture'] == "NMNIST":
        if batch_size == -1:
            assert False, "The following needs to be tested:batch size -1"
            # batch_size = nmnist_dataloader.mnist_test_ds.__len__()
        nmnist_dataloader = NMNISTDataLoader()
        data_loader = nmnist_dataloader.get_data_loader(dset="test", mode="snn", shuffle=True, num_workers=4, batch_size=batch_size)
    elif model['architecture'] == "BMNIST":
        bmnist_dataloader = BMNISTDataLoader()
        if batch_size == -1:
            ds_len = bmnist_dataloader.mnist_test_ds.__len__()
            batch_size = ds_len
            if ds_len > max_size:
                batch_size = max_size
        data_loader = bmnist_dataloader.get_data_loader(dset="test", shuffle=True, num_workers=4, batch_size=batch_size)
    else:
        assert model['architecture'] in ["NMNIST", "BMNIST"], "No other architecture added so far"
    return data_loader


@cachable(dependencies=["model:{architecture}_session_id", "N_pgd", "N_MC", "eps", "eps_iter", "rand_minmax", "norm", "k", "limit"])
def prob_boost_attack_on_test_set(
    model,
    N_pgd,
    N_MC,
    eps,
    eps_iter,
    rand_minmax,
    norm,
    k,
    verbose,
    limit
):
    def attack_fn(X0):
        d = boosted_hamming_attack(
            k=k,
            prob_net=model["prob_net"],
            P0=X0,
            eps=eps,
            eps_iter=eps_iter,
            N_pgd=N_pgd,
            N_MC=N_MC,
            norm=norm,
            rand_minmax=rand_minmax,
            verbose=verbose)
        return d
    return evaluate_on_test_set(model, limit, attack_fn)


@cachable(dependencies=["model:{architecture}_session_id", "N_pgd", "N_MC", "eps", "eps_iter", "rand_minmax", "norm", "hamming_distance_eps", "early_stopping", "limit"])
def prob_attack_on_test_set(
    model,
    N_pgd,
    N_MC,
    eps,
    eps_iter,
    rand_minmax,
    norm,
    hamming_distance_eps,
    early_stopping,
    verbose,
    limit
):

    def attack_fn(X0):
        d = hamming_attack(
            hamming_distance_eps=hamming_distance_eps,
            prob_net=model["prob_net"],
            P0=X0,
            eps=eps,
            eps_iter=eps_iter,
            N_pgd=N_pgd,
            N_MC=N_MC,
            norm=norm,
            rand_minmax=rand_minmax,
            early_stopping=early_stopping,
            verbose=verbose)
        return d

    return evaluate_on_test_set(model, limit, attack_fn)


@cachable(dependencies=["model:{architecture}_session_id", "hamming_distance_eps", "thresh", "early_stopping", "limit"])
def scar_attack_on_test_set(
    model,
    hamming_distance_eps,
    thresh,
    early_stopping,
    verbose,
    limit
):

    def attack_fn(X0):
        d = scar_attack(
            hamming_distance_eps=hamming_distance_eps,
            net=model["ann"],
            X0=X0,
            thresh=thresh,
            early_stopping=early_stopping,
            verbose=verbose)
        return d

    return evaluate_on_test_set(model, limit, attack_fn)


def evaluate_on_test_set(model, limit, attack_fn):
    data_loader = get_data_loader_from_model(model, batch_size=limit, max_size=10000)
    N_count = 0
    split_size = 10

    ret = {}
    ret["success"] = []
    ret["elapsed_time"] = []
    ret["L0"] = []
    ret["n_queries"] = []
    ret["targets"] = []
    ret["predicted"] = []
    ret["predicted_attacked"] = []

    def f(X_batched, targets, attack_fn, idx):
        ret_f = {"success": [], "elapsed_time": [], "L0": [], "n_queries": [], "targets": [], "predicted": [], "predicted_attacked": []}
        for i in range(X_batched.shape[0]):
            print(f"{i}/{X_batched.shape[0]}")
            X0 = X_batched[i]
            X0 = X0.reshape((1,) + X0.shape)
            target = int(targets[i])
            d = attack_fn(X0)
            d.pop("X_adv")
            for key in d:
                ret_f[key].append(d[key])
            ret_f["targets"].append(target)
        return ret_f, idx

    for batch, target in data_loader:
        if N_count >= limit:
            break
        X = batch.to(device)
        N_count += X.shape[0]
        X_list = list(torch.split(X, split_size))
        target_list = list(torch.split(target, split_size))

        with ThreadPoolExecutor(max_workers=None) as executor:
            parallel_results = []
            futures = [executor.submit(f, el, t, attack_fn, idx) for idx, (el, t) in enumerate(zip(X_list, target_list))]
            for future in as_completed(futures):
                result = future.result()
                parallel_results.append(result)
            # - Sort the results
            parallel_results = sorted(parallel_results, key=lambda k: k[1])

            for ret_f, idx in parallel_results:
                for key in ret_f:
                    ret[key].append(ret_f[key])

    return ret
