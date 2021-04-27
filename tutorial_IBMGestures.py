# TODO import data loader gestures
import torch
from networks import load_gestures_snn
from sparsefool import sparsefool
from utils import get_prediction, plot_attacked_prob
from dataloader_IBMGestures import get_data_loader
from functools import partial
from copy import deepcopy
from torch.multiprocessing import Pool, set_start_method

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


def sparse_fool_wrapper(
    net,
    max_hamming_distance,
    lambda_,
    device,
    epsilon,
    round_fn,
    max_iter,
    early_stopping,
    boost,
    verbose,
    shared,
):
    if round_fn == "stoch_round":
        round_fn = lambda x: (torch.rand(size=x.shape, device=device) < x).float()
    elif round_fn == "round":
        round_fn = torch.round
    x_0, n = shared
    return_list = []
    for x in x_0:
        if x.ndim == 4:
            x = x.reshape((1,) + x.shape)
        return_list.append(
            sparsefool(
                x_0=x,
                net=net,
                max_hamming_distance=max_hamming_distance,
                lambda_=lambda_,
                device=device,
                epsilon=epsilon,
                round_fn=round_fn,
                max_iter=max_iter,
                early_stopping=early_stopping,
                boost=boost,
                verbose=verbose,
            )
        )
    return (return_list, n)


if __name__ == "__main__":

    # - Load the spiking CNN for IBM gestures dataset
    snn = load_gestures_snn()

    data_loader_test = get_data_loader(
        dset="test", shuffle=False, num_workers=4, batch_size=20
    )  # - Can vary

    # - Attack parameters
    lambda_ = 4.0
    max_hamming_distance = 10000
    round_fn = "stoch_round"

    for idx, (X0, target) in enumerate(data_loader_test):

        X0 = X0.float()
        X0 = X0[:, :10]
        X0 = X0.to(device)
        X0 = torch.clamp(X0, 0.0, 1.0)
        target = target.long().to(device)

        X_split = [
            x.detach() for x in list(torch.split(X0, split_size_or_sections=5, dim=0))
        ]

        partial_sparse_fool = partial(
            sparse_fool_wrapper,
            deepcopy(snn),
            max_hamming_distance,
            lambda_,
            device,
            0.0,
            round_fn,
            20,
            False,
            False,
            True,
        )

        with Pool(None) as p:
            results = p.map(partial_sparse_fool, zip(X_split, range(len(X_split))))
            results.sort(key=lambda x: x[1])
            results = [r[0] for r in results]
            results = [a for b in results for a in b]

        for batch_idx, return_dict_sparse_fool in enumerate(results):
            target_cur = target[batch_idx]
            X0_cur = X0[batch_idx]
            X_adv_sparse_fool = return_dict_sparse_fool["X_adv"]
            num_flips_sparse_fool = return_dict_sparse_fool["L0"]
            original_prediction = return_dict_sparse_fool["predicted"]
            model_pred_sparse_fool = get_prediction(snn, X_adv_sparse_fool, "non_prob")

        # if (target_cur == original_prediction) and model_pred_sparse_fool != 10 and original_prediction != 10 and return_dict_sparse_fool["success"]:
        # break

    # - Plotting
    plot_attacked_prob(
        X0_cur,
        int(target_cur),
        snn,
        N_rows=2,
        N_cols=2,
        data=[(torch.clamp(torch.sum(X0.cpu(), 2), 0.0, 1.0), original_prediction)
              for _ in range(2 * 2)],
        figname=1,
        block=False,
    )

    plot_attacked_prob(
        X0_cur,
        int(target_cur),
        snn,
        N_rows=2,
        N_cols=2,
        data=[(torch.clamp(torch.sum(X_adv_sparse_fool.cpu(), 2), 0.0, 1.0),
               model_pred_sparse_fool, ) for _ in range(2 * 2)],
        figname=2,
    )
