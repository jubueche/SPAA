import torch
from dataloader_NMNIST import NMNISTDataLoader
from dataloader_BMNIST import BMNISTDataLoader
from datajuicer import cachable
from concurrent.futures import ThreadPoolExecutor, as_completed
from attacks import non_prob_fool, prob_fool, SCAR
from sparsefool import sparsefool
import numpy as np

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

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


@cachable(dependencies=["model:{architecture}_session_id", "N_pgd", "N_MC", "eps", "eps_iter", "rand_minmax", "norm", "max_hamming_distance", "boost", "early_stopping", "limit"])
def prob_fool_on_test_set(
    model,
    N_pgd,
    N_MC,
    eps,
    eps_iter,
    rand_minmax,
    norm,
    max_hamming_distance,
    boost,
    early_stopping,
    verbose,
    limit
):

    def attack_fn(X0):
        d = prob_fool(
            max_hamming_distance=max_hamming_distance,
            prob_net=model["prob_net"],
            P0=X0,
            eps=eps,
            eps_iter=eps_iter,
            N_pgd=N_pgd,
            N_MC=N_MC,
            norm=norm,
            rand_minmax=rand_minmax,
            boost=boost,
            early_stopping=early_stopping,
            verbose=verbose)
        return d

    return evaluate_on_test_set(model, limit, attack_fn)

def get_round_fn(round_fn):
    assert round_fn in ["round", "stoch_round"], "Unknown rounding function"
    if round_fn == "round":
        round_fn_evaluated = torch.round
    else:
        round_fn_evaluated = lambda x : (torch.rand(size=x.shape) < x).float()
    return round_fn_evaluated


@cachable(dependencies=["model:{architecture}_session_id","max_hamming_distance","lambda_","max_iter","epsilon","overshoot","max_iter_deep_fool","rand_minmax","early_stopping","boost","limit"])
def prob_sparse_fool_on_test_set(
    model,
    max_hamming_distance,
    lambda_,
    max_iter,
    epsilon,
    overshoot,
    max_iter_deep_fool,
    rand_minmax,
    early_stopping,
    boost,
    verbose,
    limit
):
    def attack_fn(X0):
        d = sparsefool(
            x_0=X0,
            net=model["prob_net"],
            max_hamming_distance=max_hamming_distance,
            lambda_=lambda_,
            max_iter=max_iter,
            epsilon=epsilon,
            overshoot=overshoot,
            max_iter_deep_fool=max_iter_deep_fool,
            device=device,
            round_fn=None,
            probabilistic=True,
            rand_minmax=rand_minmax,
            early_stopping=early_stopping,
            boost=boost,
            verbose=verbose
        )
        return d
    return evaluate_on_test_set(model, limit, attack_fn)

@cachable(dependencies=["model:{architecture}_session_id","max_hamming_distance","lambda_","max_iter","epsilon","overshoot","max_iter_deep_fool","round_fn","early_stopping","boost","limit"])
def sparse_fool_on_test_set(
    model,
    max_hamming_distance,
    lambda_,
    max_iter,
    epsilon,
    overshoot,
    max_iter_deep_fool,
    round_fn,
    early_stopping,
    boost,
    verbose,
    limit
):

    round_fn_evaluated = get_round_fn(round_fn)

    def attack_fn(X0):
        d = sparsefool(
            x_0=X0,
            net=model["ann"],
            max_hamming_distance=max_hamming_distance,
            lambda_=lambda_,
            max_iter=max_iter,
            epsilon=epsilon,
            overshoot=overshoot,
            max_iter_deep_fool=max_iter_deep_fool,
            device=device,
            round_fn=round_fn_evaluated,
            probabilistic=False,
            rand_minmax=None,
            early_stopping=early_stopping,
            boost=boost,
            verbose=verbose
        )
        return d
    return evaluate_on_test_set(model, limit, attack_fn)
    

@cachable(dependencies=["model:{architecture}_session_id", "N_pgd", "round_fn", "eps", "eps_iter", "rand_minmax", "norm", "max_hamming_distance", "boost", "early_stopping", "limit"])
def non_prob_fool_on_test_set(
    model,
    N_pgd,
    round_fn,
    eps,
    eps_iter,
    rand_minmax,
    norm,
    max_hamming_distance,
    boost,
    early_stopping,
    verbose,
    limit
):

    round_fn_evaluated = get_round_fn(round_fn)

    def attack_fn(X0):
        d = non_prob_fool(
            max_hamming_distance=max_hamming_distance,
            net=model["ann"],
            X0=X0,
            round_fn=round_fn_evaluated,
            eps=eps,
            eps_iter=eps_iter,
            N_pgd=N_pgd,
            norm=norm,
            rand_minmax=rand_minmax,
            boost=boost,
            early_stopping=early_stopping,
            verbose=verbose)
        return d

    return evaluate_on_test_set(model, limit, attack_fn)


@cachable(dependencies=["model:{architecture}_session_id", "max_hamming_distance", "thresh", "early_stopping", "limit"])
def scar_attack_on_test_set(
    model,
    max_hamming_distance,
    thresh,
    early_stopping,
    verbose,
    limit
):

    def attack_fn(X0):
        d = SCAR(
            max_hamming_distance=max_hamming_distance,
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
    split_size = 100

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

    # - Unravel
    for key in ret:
        ret[key] = np.array([a for b in ret[key] for a in b])

    return ret
