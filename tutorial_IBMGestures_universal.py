import torch
from networks import load_gestures_snn
from sparsefool import sparsefool, frame_based_sparsefool, universal_attack
from utils import get_prediction, plot_attacked_prob
from dataloader_IBMGestures import IBMGesturesDataLoader
from functools import partial
from copy import deepcopy
from architectures import IBMGestures
from datajuicer import run
import numpy as np

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    ibm_gesture_dataloader = IBMGesturesDataLoader()

    data_loader_test = ibm_gesture_dataloader.get_data_loader(
        dset="test",
        shuffle=True,
        num_workers=4,
        batch_size=64)  # - Can vary

    grid = [IBMGestures.make()]
    grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

    # - Load the spiking CNN for IBM gestures dataset
    snn = grid[0]['snn']

    # - Attack parameters
    lambda_ = 1.0
    max_hamming_distance = np.inf

    def attack_fn(X,y):
        return frame_based_sparsefool(
            x_0=X,
            net=snn,
            max_hamming_distance=max_hamming_distance,
            lambda_=lambda_,
            epsilon=0.0,
            overshoot=0.02,
            n_attack_frames=3,
            step_size=0.05,
            device=device,
            early_stopping=False,
            boost=False,
            verbose=True,
        )

    for idx, (X0, target) in enumerate(data_loader_test):

        X0 = X0.float()
        X0 = X0.to(device)
        X0 = torch.clamp(X0, 0.0, 1.0)
        target = target.long().to(device)

        return_dict_universal_attack = universal_attack(
            X=X0,
            y=target,
            net=snn,
            attack_fn=attack_fn,
            max_hamming_distance=2000,
            max_iter=1,
            device=device
        )

        X_adv = return_dict_universal_attack["X_adv"]

        for i in range(X_adv.shape[0]):

            # - Plotting
            plot_attacked_prob(
                X0[i],
                int(target[i]),
                snn,
                N_rows=2,
                N_cols=2,
                data=[(torch.clamp(torch.sum(X0[i].cpu(), 1), 0.0, 1.0), return_dict_universal_attack["predicted"][i])
                    for _ in range(2 * 2)],
                figname=1,
                block=False,
            )

            plot_attacked_prob(
                X0[i],
                int(target[i]),
                snn,
                N_rows=2,
                N_cols=2,
                data=[(torch.clamp(torch.sum(X_adv[i].cpu(), 1), 0.0, 1.0),
                        return_dict_universal_attack["predicted_attacked"][i], ) for _ in range(2 * 2)],
                figname=2,
            )
