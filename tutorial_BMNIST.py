from dataloader_BMNIST import BMNISTDataLoader
import torch
import matplotlib.pyplot as plt
import random

from networks import train_ann_binary_mnist, get_prob_net_continuous
from sparsefool import sparsefool
from attacks import non_prob_pgd, hamming_attack, boosted_hamming_attack, scar_attack
from utils import get_prediction, reparameterization_bernoulli

# - Seed
random.seed(42)

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# - CNN for Binary MNIST
bmnist_dataloader = BMNISTDataLoader()

# - Train the CNN
ann_binary_mnist = train_ann_binary_mnist().to(device)

# - Get the probabilistic network
prob_net = get_prob_net_continuous()

# - Attack parameters
N_pgd = 50
N_MC = 10
eps = 1.5
eps_iter = 0.3
rand_minmax = 0.01
rand_minmax_deepfool = 0.01
norm = 2
hamming_distance_eps = 400 / 784
verbose = False
round_fn = torch.round

# - Test data set
data_loader_test = bmnist_dataloader.get_data_loader(
    dset="test", shuffle=True, num_workers=4, batch_size=1
)
for idx, (data, target) in enumerate(data_loader_test):
    X0 = data.to(device)

    if idx < 7:
        continue

    # if not (target == 0):
    #     continue
    return_dict_sparse_fool = sparsefool(
        x_0=X0,
        net=ann_binary_mnist,
        lambda_=0.5,
        device=device,
        round_fn=round_fn,
        probabilistic=False,
        early_stopping=True,
    )

    return_dict_sparse_fool_prob = sparsefool(
        x_0=X0,
        net=prob_net,
        lambda_=0.5,
        device=device,
        probabilistic=True,
        early_stopping=True,
        rand_minmax=rand_minmax_deepfool
    )

    return_dict_non_prob = non_prob_pgd(
        hamming_distance_eps=hamming_distance_eps,
        net=ann_binary_mnist,
        X0=X0,
        round_fn=lambda x: torch.round(x),
        eps=eps,
        eps_iter=eps_iter,
        N_pgd=N_pgd,
        norm=norm,
        rand_minmax=rand_minmax,
        boost=False,
        early_stopping=True,
        verbose=True,
    )

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
        verbose=True,
    )

    # - Attack using greedy attack from https://openreview.net/pdf?id=xCm8kiWRiBT
    return_dict_scar = scar_attack(
        hamming_distance_eps=hamming_distance_eps,
        net=ann_binary_mnist,
        X0=X0,
        thresh=0.1,
        early_stopping=True,
        verbose=True,
    )

    return_dict_boosted = boosted_hamming_attack(
        k=50,
        prob_net=prob_net,
        P0=X0,
        eps=eps,
        eps_iter=eps_iter,
        N_pgd=N_pgd,
        N_MC=N_MC,
        norm=norm,
        rand_minmax=rand_minmax,
        verbose=verbose,
    )

    X_adv_sparse_fool = return_dict_sparse_fool["X_adv"].to(device)
    num_flipped_sparse_fool = return_dict_sparse_fool["L0"]
    X_adv_sparse_fool_prob = return_dict_sparse_fool_prob["X_adv"].to(device)
    num_flipped_sparse_fool_prob = return_dict_sparse_fool_prob["L0"]
    X_adv_non_prob = return_dict_non_prob["X_adv"].to(device)
    num_flipped_non_prob = return_dict_non_prob["L0"]
    X_adv_scar = return_dict_scar["X_adv"].to(device)
    num_flipped_scar = return_dict_scar["L0"]
    X_adv_boosted = return_dict_boosted["X_adv"].to(device)
    num_flipped_boosted = return_dict_boosted["L0"]
    X_adv = return_dict["X_adv"].to(device)
    num_flipped = return_dict["L0"]

    model_pred = get_prediction(ann_binary_mnist, X0, "non_prob")
    model_pred_sparsefool = get_prediction(ann_binary_mnist, round_fn(X_adv_sparse_fool), "non_prob")
    model_pred_sparsefool_prob = get_prediction(prob_net, X_adv_sparse_fool_prob, "prob")
    model_pred_non_prob = get_prediction(ann_binary_mnist, X_adv_non_prob, "non_prob")
    model_pred_scar = get_prediction(ann_binary_mnist, X_adv_scar, "non_prob")
    model_pred_prob_boosted = get_prediction(ann_binary_mnist, X_adv_boosted, "non_prob")
    model_pred_prob = get_prediction(ann_binary_mnist, X_adv, "non_prob")

    print(f"Model: {int(model_pred)} SparseFool: {int(model_pred_sparsefool)} with L_0 = {num_flipped_sparse_fool}")
    print(f"Model: {int(model_pred)} SparseFool Prob: {int(model_pred_sparsefool_prob)} with L_0 = {num_flipped_sparse_fool_prob}")
    print(f"Model: {int(model_pred)} Non_prob: {int(model_pred_non_prob)} with L_0 = {num_flipped_non_prob}")
    print(f"Model: {int(model_pred)} Scar: {int(model_pred_scar)} with L_0 = {num_flipped_scar}")
    print(f"Model: {int(model_pred)} Boosted Prob.: {int(model_pred_prob_boosted)} with L_0 = {num_flipped_boosted}")
    print(f"Model: {int(model_pred)} Prob.: {int(model_pred_prob)} with L_0 = {num_flipped}")

    break

plt.figure(figsize=(20, 7))
plt.subplot(181)
plt.imshow(torch.squeeze(X0.cpu()))
plt.title("Orig.")
plt.subplot(182)
plt.imshow(torch.squeeze(X_adv.cpu()))
plt.title(f"Adv. Hamming Pred.: {int(model_pred_prob)}")
plt.subplot(183)
plt.imshow(torch.squeeze(X_adv_boosted.cpu()))
plt.title(f"Adv. Boosted Hamming Pred.: {int(model_pred_prob_boosted)}")
plt.subplot(184)
plt.imshow(torch.squeeze(X_adv_scar.cpu()))
plt.title(f"Adv. Scar.: {int(model_pred_scar)}")
plt.subplot(185)
plt.imshow(torch.squeeze(X_adv_non_prob.cpu()))
plt.title(f"Adv. non prob.: {int(model_pred_non_prob)}")
plt.subplot(186)
plt.imshow(torch.squeeze(round_fn(X_adv_sparse_fool).cpu()))
plt.title(f"Adv. SparseFool: {int(model_pred_sparsefool)}")
plt.subplot(187)
plt.imshow(torch.squeeze(
    torch.round(reparameterization_bernoulli(X_adv_sparse_fool_prob, temperature=0.01)).cpu()
))
plt.title(f"Adv. SparseFool Prob: {int(model_pred_sparsefool_prob)}")
plt.show()
