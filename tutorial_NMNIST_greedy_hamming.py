from videofig import videofig
from dataloader_NMNIST import NMNISTDataLoader
from dataloader_BMNIST import BMNISTDataLoader
import torch
from experiment_utils import *
import matplotlib.pyplot as plt
import sys

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# - CNN for Binary MNIST
bmnist_dataloader = BMNISTDataLoader()

# - Train the CNN
ann_binary_mnist = train_ann_binary_mnist()

# - Get the probabilistic network
prob_net = get_prob_net_continuous()

# - Attack parameters
N_pgd = 50
N_MC = 10
eps = 1.5
eps_iter = 0.3
rand_minmax = 0.01
norm = 2
hamming_distance_eps = 400 / 784
verbose = False

# - Test data set
data_loader_test = bmnist_dataloader.get_data_loader(dset="test", shuffle=True, num_workers=4, batch_size=1)
for idx, (data,target) in enumerate(data_loader_test):
    X0 = data.to(device)

    if idx < 5:
        continue

    # if not (target == 0):
    #     continue
    
    return_dict = hamming_attack(
        hamming_distance_eps=hamming_distance_eps,
        prob_net=prob_net,
        P0=X0,
        eps=eps,
        eps_iter=eps_iter,
        N_pgd=N_pgd,
        N_MC=N_MC,
        norm=norm,
        rand_minmax=rand_minmax,
        early_stopping=True,
        verbose=True
    )

    # - Attack using greedy attack from https://openreview.net/pdf?id=xCm8kiWRiBT
    return_dict_scar = scar_attack(
        hamming_distance_eps=hamming_distance_eps,
        net=ann_binary_mnist,
        X0=X0,
        thresh=0.1,
        early_stopping=True,
        verbose=True
    )

    return_dict_boosted = boosted_hamming_attack(
        k=25,
        prob_net=prob_net,
        P0=X0,
        eps=eps,
        eps_iter=eps_iter,
        N_pgd=N_pgd,
        N_MC=N_MC,
        norm=norm,
        rand_minmax=rand_minmax,
        verbose=verbose
    )

    X_adv_scar = return_dict_scar["X_adv"]
    num_flipped_scar = return_dict_scar["L0"]
    X_adv_boosted = return_dict_boosted["X_adv"]
    num_flipped_boosted = return_dict_boosted["L0"]
    X_adv = return_dict["X_adv"]
    num_flipped = return_dict["L0"]

    model_pred = get_prediction(ann_binary_mnist, X0, "non_prob")
    model_pred_scar = get_prediction(ann_binary_mnist, X_adv_scar, "non_prob")
    model_pred_prob_boosted = get_prediction(ann_binary_mnist, X_adv_boosted, "non_prob")
    model_pred_prob = get_prediction(ann_binary_mnist, X_adv, "non_prob")
    print(f"Model: {int(model_pred)} Scar: {int(model_pred_scar)} with L_0 = {num_flipped_scar}")
    print(f"Model: {int(model_pred)} Boosted Prob.: {int(model_pred_prob_boosted)} with L_0 = {num_flipped_boosted}")
    print(f"Model: {int(model_pred)} Prob.: {int(model_pred_prob)} with L_0 = {num_flipped}")

    break

plt.subplot(141)
plt.imshow(torch.squeeze(X0)); plt.title("Orig.")
plt.subplot(142)
plt.imshow(torch.squeeze(X_adv)); plt.title(f"Adv. Hamming Pred.: {int(model_pred_prob)}")
plt.subplot(143)
plt.imshow(torch.squeeze(X_adv_boosted)); plt.title(f"Adv. Boosted Hamming Pred.: {int(model_pred_prob_boosted)}")
plt.subplot(144)
plt.imshow(torch.squeeze(X_adv_scar)); plt.title(f"Adv. Scar.: {int(model_pred_scar)}")
plt.show()