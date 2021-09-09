import torch
from dataloader_NMNIST import NMNISTDataLoader
from dataloader_BMNIST import BMNISTDataLoader
from dataloader_IBMGestures import IBMGesturesDataLoader
from datajuicer import cachable
from concurrent.futures import ThreadPoolExecutor, as_completed
from attacks import non_prob_fool, prob_fool, SCAR
from sparsefool import sparsefool, universal_attack, frame_based_sparsefool, Heatmap, RandomEviction, universal_heatmap_attack
from adversarial_patch import adversarial_patch
import numpy as np

# - Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
device = "cuda"

# - Set a global seed
torch.manual_seed(0)


def get_data_loader_from_model(model, batch_size=1, dset="test", shuffle=False, max_size=10000):
    if model['architecture'] == "NMNIST":
        if batch_size == -1:
            assert False, "The following needs to be tested:batch size -1"
            # batch_size = nmnist_dataloader.mnist_test_ds.__len__()
        nmnist_dataloader = NMNISTDataLoader()
        data_loader = nmnist_dataloader.get_data_loader(dset=dset, mode="snn", shuffle=shuffle, num_workers=4, batch_size=batch_size)
    elif model['architecture'] == "BMNIST":
        bmnist_dataloader = BMNISTDataLoader()
        if batch_size == -1:
            ds_len = bmnist_dataloader.mnist_test_ds.__len__()
            batch_size = ds_len
            if ds_len > max_size:
                batch_size = max_size
        data_loader = bmnist_dataloader.get_data_loader(dset=dset, shuffle=shuffle, num_workers=4, batch_size=batch_size)
    elif model['architecture'] == "IBMGestures":
        ibm_gestures_dataloader = IBMGesturesDataLoader()
        data_loader = ibm_gestures_dataloader.get_data_loader(dset, shuffle=shuffle, num_workers=4, batch_size=batch_size)
    else:
        assert model['architecture'] in ["NMNIST", "BMNIST"], "No other architecture added so far"
    return data_loader

def get_even_batch(data_loader, num_samples, num_classes=11):
    X = []; y = []
    for i in range(num_classes):
        c = 0
        for X0,y0 in data_loader:
            if c == num_samples: break
            X0 = X0.float()
            X0 = X0.to(device)
            X0 = torch.clamp(X0, 0.0, 1.0)
            y0 = y0.long().to(device)
            if y0 == i:
                c += 1
                X.append(X0); y.append(y0)
    X,y = torch.stack(X).squeeze(), torch.stack(y).squeeze()
    rp = torch.randperm(X.shape[0])
    return X[rp], y[rp]

def get_test_acc(data_loader, net, pert_total=None):
    correct = 0; num = 0
    device = next(net.parameters()).device
    for idx, (X0, target) in enumerate(data_loader):
        X0 = X0.float()
        X0 = X0.to(device)
        X0 = torch.clamp(X0, 0.0, 1.0)
        if not pert_total is None:
            X0[:,pert_total] = 1. - X0[:,pert_total]
        target = target.long().to(device)
        net.reset_states()
        out = net.forward(X0)
        _, predict = torch.max(out, 1)
        correct += torch.sum((predict == target).float())
        num += X0.shape[0]
    ta = float(correct / num)
    return ta

@cachable(dependencies=["model:{architecture}_session_id", "max_hamming_distance", "use_snn"])
def random_universal_test_acc(
    model,
    max_hamming_distance,
    use_snn
):
    if use_snn:
        net = model["snn"]
    else:
        net = model["ann"]

    data_loader_test = get_data_loader_from_model(model, batch_size=6, dset="test", max_size=10000)

    shape = data_loader_test.dataset.__getitem__(0)[0].shape
    indices = np.indices(shape).reshape((len(shape),-1))[:,torch.randperm(np.prod(shape))][:,:max_hamming_distance]
    random_pert_total = torch.zeros(shape).bool()
    random_pert_total[indices] = True

    return_dict = {
        "attacked_test_acc": get_test_acc(data_loader_test, net, random_pert_total),
        "test_acc": get_test_acc(data_loader_test, net),
        "pert_total": random_pert_total,
        "L0":max_hamming_distance}

    return return_dict

@cachable(dependencies=["model:{architecture}_session_id", "attack_fn_name", "num_samples", "max_hamming_distance", "max_iter", "eviction", "use_snn"])
def universal_attack_test_acc(
    model,
    attack_fn,
    attack_fn_name,
    num_samples,
    max_hamming_distance,
    max_iter,
    eviction,
    use_snn
):
    if use_snn:
        net = model["snn"]
    else:
        net = model["ann"]

    data_loader = get_data_loader_from_model(model, batch_size=1, dset="train", max_size=10000)
    X,y = get_even_batch(data_loader=data_loader, num_samples=num_samples, num_classes=11)

    return_dict_universal_attack = universal_attack(
            X=X,
            y=y,
            net=net,
            attack_fn=attack_fn,
            max_hamming_distance=max_hamming_distance,
            max_iter=max_iter,
            eviction=Heatmap if eviction == "Heatmap" else RandomEviction,
            device=device
        )

    data_loader_test = get_data_loader_from_model(model, batch_size=8, dset="test", max_size=10000)

    return_dict_universal_attack.pop("X_adv")

    return_dict = {
        "attacked_test_acc": get_test_acc(data_loader_test, net, return_dict_universal_attack["pert_total"]),
        "test_acc": get_test_acc(data_loader_test, net),
        **return_dict_universal_attack}

    return return_dict

@cachable(dependencies=["model:{architecture}_session_id", "attack_fn_name", "num_samples", "max_hamming_distance", "use_snn"])
def universal_heatmap_attack_test_acc(
    model,
    attack_fn,
    attack_fn_name,
    num_samples,
    max_hamming_distance,
    use_snn
):
    if use_snn:
        net = model["snn"]
    else:
        net = model["ann"]

    data_loader = get_data_loader_from_model(model, batch_size=1, dset="train", max_size=10000)
    X,y = get_even_batch(data_loader=data_loader, num_samples=num_samples, num_classes=11)

    return_dict_universal_attack = universal_heatmap_attack(
            X=X,
            y=y,
            net=net,
            attack_fn=attack_fn,
            max_hamming_distance=max_hamming_distance,
            device=device
        )

    data_loader_test = get_data_loader_from_model(model, batch_size=8, dset="test", max_size=10000)

    return_dict_universal_attack.pop("X_adv")

    return_dict = {
        "attacked_test_acc": get_test_acc(data_loader_test, net, return_dict_universal_attack["pert_total"]),
        "test_acc": get_test_acc(data_loader_test, net),
        **return_dict_universal_attack}

    return return_dict

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
        round_fn_evaluated = lambda x : (torch.rand(size=x.shape, device=x.device) < x).float()
    return round_fn_evaluated


@cachable(dependencies=["model:{architecture}_session_id","max_hamming_distance","lambda_","max_iter","epsilon","overshoot","step_size","max_iter_deep_fool","limit"])
def sparse_fool_on_test_set(
    model,
    max_hamming_distance,
    lambda_,
    max_iter,
    epsilon,
    overshoot,
    step_size,
    max_iter_deep_fool,
    verbose,
    limit,
    use_snn=False,
):

    if use_snn:
        net = model["snn"]
    else:
        net = model["ann"]

    def attack_fn(X0):
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
            verbose=verbose
        )
        return d
    return evaluate_on_test_set(model, limit, attack_fn)

@cachable(dependencies=["model:{architecture}_session_id","n_epochs","target_label","patch_type","input_shape","patch_size","max_iter","label_conf","max_count"])
def adversarial_patches_exp(
    model,
    n_epochs,
    target_label,
    patch_type,
    input_shape,
    patch_size,
    max_iter,
    eval_after,
    max_iter_test,
    label_conf,
    max_count,
    use_snn
):
    if use_snn:
        net = model["snn"]
    else:
        net = model["ann"]

    data_loader_train = get_data_loader_from_model(model, batch_size=1, dset="train", shuffle=True, max_size=10000)
    data_loader_test = get_data_loader_from_model(model, batch_size=1, dset="test", shuffle=True, max_size=10000)

    return_dict_adv_patch = adversarial_patch(
        net=net,
        train_data_loader=data_loader_train,
        test_data_loader=data_loader_test,
        patch_type=patch_type,
        patch_size=patch_size,
        input_shape=input_shape,
        n_epochs=n_epochs,
        target_label=target_label,
        max_iter=max_iter,
        max_iter_test=max_iter_test,
        label_conf=label_conf,
        max_count=max_count,
        eval_after=eval_after,
        device=device
    )

    return return_dict_adv_patch

@cachable(dependencies=["model:{architecture}_session_id","max_hamming_distance","lambda_","max_iter","epsilon","overshoot","n_attack_frames","step_size","max_iter_deep_fool","limit"])
def frame_based_sparse_fool_on_test_set(
    model,
    max_hamming_distance,
    lambda_,
    max_iter,
    epsilon,
    overshoot,
    n_attack_frames,
    step_size,
    max_iter_deep_fool,
    verbose,
    limit,
    use_snn=False,
):

    if use_snn:
        net = model["snn"]
    else:
        net = model["ann"]

    def attack_fn(X0):
        d = frame_based_sparsefool(
            x_0=X0,
            net=net,
            max_hamming_distance=max_hamming_distance,
            lambda_=lambda_,
            max_iter=max_iter,
            epsilon=epsilon,
            overshoot=overshoot,
            n_attack_frames=n_attack_frames,
            step_size=step_size,
            max_iter_deep_fool=max_iter_deep_fool,
            device=device,
            verbose=verbose,
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
    limit,
    use_snn=False,
):

    round_fn_evaluated = get_round_fn(round_fn)

    if use_snn:
        net = model["snn"]
    else:
        net = model["ann"]

    def attack_fn(X0):
        d = non_prob_fool(
            max_hamming_distance=max_hamming_distance,
            net=net,
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
    data_loader = get_data_loader_from_model(model, batch_size=1, max_size=50)
    N_count = 0

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
            # print(f"{i}/{X_batched.shape[0]}")
            X0 = X_batched[i]
            if X0.ndim < 4:
                X0 = X0.reshape((1,) + X0.shape)
            target = int(targets[i])
            d = attack_fn(X0)
            d.pop("X_adv")
            for key in d:
                ret_f[key].append(d[key])
            ret_f["targets"].append(target)
        return ret_f, idx

    for batch, target in data_loader:
        if N_count >= limit and limit != -1:
            break
        X = batch.to(device)
        X = X.float()
        X = torch.clamp(X, 0.0, 1.0)
        max_workers = 1
        split_size = int(np.ceil(X.shape[0] / max_workers))
        N_count += X.shape[0]
        X_list = list(torch.split(X, split_size))
        target_list = list(torch.split(target, split_size))

        with ThreadPoolExecutor(max_workers=1) as executor:
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
