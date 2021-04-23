# TODO import data loader gestures
import torch
from networks import load_gestures_snn
from sparsefool import sparsefool
from utils import get_prediction, plot_attacked_prob
from dataloader_IBMGestures import get_data_loader

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# - Load the spiking CNN for IBM gestures dataset
snn = load_gestures_snn()

data_loader_test = get_data_loader(
    dset="test",
    shuffle=False,
    num_workers=4,
    batch_size=1)

# - Attack parameters
lambda_ = 2.0
max_hamming_distance = 10000
round_fn = lambda x : (torch.rand(size=x.shape) < x).float()

for X0, target in data_loader_test:

    X0 = X0.float()
    X0 = X0[:,:25]
    X0 = X0.to(device)
    X0 = torch.clamp(X0, 0.0, 1.0)
    target = target.long().to(device)

    return_dict_sparse_fool = sparsefool(
            x_0=X0,
            net=snn,
            max_hamming_distance=max_hamming_distance,
            lambda_=lambda_,
            device=device,
            epsilon=0.0,
            round_fn=round_fn,
            max_iter=20,
            early_stopping=False,
            boost=False,
            verbose=True
        )

    X_adv_sparse_fool = return_dict_sparse_fool["X_adv"]
    num_flips_sparse_fool = return_dict_sparse_fool["L0"]
    original_prediction = return_dict_sparse_fool["predicted"]
    model_pred_sparse_fool = get_prediction(snn, X_adv_sparse_fool, "non_prob")

    if return_dict_sparse_fool["success"]:
        break


plot_attacked_prob(X0, int(target), snn, N_rows=1, N_cols=1, block=False, figname=1)

plot_attacked_prob(
    X0,
    int(target),
    snn,
    N_rows=1,
    N_cols=1,
    data=[
        (torch.clamp(torch.sum(X_adv_sparse_fool.cpu(), 2), 0.0, 1.0), model_pred_sparse_fool)
        for _ in range(1)
    ],
    figname=2,
)