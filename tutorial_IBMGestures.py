import torch
from networks import load_gestures_snn
from sparsefool import sparsefool, universal_sparsefool
from utils import get_prediction, plot_attacked_prob
from dataloader_IBMGestures import IBMGesturesDataLoader
from functools import partial
from copy import deepcopy
from architectures import IBMGestures
from datajuicer import run

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    ibm_gesture_dataloader = IBMGesturesDataLoader()

    data_loader_test = ibm_gesture_dataloader.get_data_loader(
        dset="test",
        shuffle=False,
        num_workers=4,
        batch_size=1)  # - Can vary

    grid = [IBMGestures.make()]
    grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

    # - Load the spiking CNN for IBM gestures dataset
    snn = grid[0]['snn']

    # # - Calculate the test accuracy
    # correct = 0; num = 0
    # for idx, (X0, target) in enumerate(data_loader_test):
    #     X0 = X0.float()
    #     X0 = X0.to(device)
    #     # X0 = torch.clamp(X0, 0.0, 1.0)
    #     target = target.long().to(device)
    #     pred = get_prediction(snn, X0, mode="non_prob")
    #     correct += (pred == target).float()
    #     num += 1
    # ta = float(correct / num)
    # print(f"Test accuracy is {100*ta}")

    # - Attack parameters
    lambda_ = 2.0
    max_hamming_distance = 1000

    for idx, (X0, target) in enumerate(data_loader_test):

        X0 = X0.float()
        X0 = X0.to(device)
        X0 = torch.clamp(X0, 0.0, 1.0)
        target = target.long().to(device)

        return_dict_sparse_fool = sparsefool(
            x_0=X0,
            net=snn,
            max_hamming_distance=max_hamming_distance,
            lambda_=lambda_,
            epsilon=0.0,
            overshoot=0.02,
            device=device,
            early_stopping=False,
            boost=False,
            verbose=True,
        )

        X_adv = return_dict_sparse_fool["X_adv"]
        break

    # - Plotting
    plot_attacked_prob(
        X0[0],
        int(target),
        snn,
        N_rows=2,
        N_cols=2,
        data=[(torch.clamp(torch.sum(X0[0].cpu(), 1), 0.0, 1.0), return_dict_sparse_fool["predicted"])
              for _ in range(2 * 2)],
        figname=1,
        block=False,
    )

    plot_attacked_prob(
        X0[0],
        int(target),
        snn,
        N_rows=2,
        N_cols=2,
        data=[(torch.clamp(torch.sum(X_adv[0].cpu(), 1), 0.0, 1.0),
               return_dict_sparse_fool["predicted_attacked"], ) for _ in range(2 * 2)],
        figname=2,
    )
