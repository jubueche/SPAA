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

# - Test data set
data_loader_test = bmnist_dataloader.get_data_loader(dset="test", shuffle=True, num_workers=4, batch_size=1)
for data, target in data_loader_test:
    X0 = data
    if target == 0:
        break

# - Attack using greedy attack from https://openreview.net/pdf?id=xCm8kiWRiBT
hamming_distance_eps = 1.0
X_adv_scar, num_flipped = scar_attack(
    hamming_distance_eps=hamming_distance_eps,
    net=ann_binary_mnist,
    X0=X0,
    thresh=0.1,
    early_stopping=True,
    verbose=True
)
model_pred_scar = get_prediction(ann_binary_mnist, X_adv_scar, "non_prob")
model_pred = get_prediction(ann_binary_mnist, X0, "non_prob")
print(f"Model: {int(model_pred)} Scar: {int(model_pred_scar)} with L_0 = {num_flipped}")

plt.subplot(121)
plt.imshow(torch.squeeze(X0)); plt.title("Orig.")
plt.subplot(122)
plt.imshow(torch.squeeze(X_adv_scar)); plt.title(f"Adv. Pred.: {int(model_pred_scar)}")
plt.show()

# - Get the probabilistic network
prob_net = get_prob_net_continuous()

# - Attack parameters
N_pgd = 50
N_MC = 10
eps = 1.5
eps_iter = 0.3
rand_minmax = 0.01
norm = 2
hamming_distance_eps = 1.0

X_adv, num_flipped = hamming_attack(
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

model_pred_prob = get_prediction(ann_binary_mnist, X_adv, "non_prob")
model_pred = get_prediction(ann_binary_mnist, X0, "non_prob")
print(f"Model: {int(model_pred)} Prob.: {int(model_pred_prob)} with L_0 = {num_flipped}")

plt.subplot(121)
plt.imshow(torch.squeeze(X0)); plt.title("Orig.")
plt.subplot(122)
plt.imshow(torch.squeeze(X_adv)); plt.title(f"Adv. Pred.: {int(model_pred_prob)}")
plt.show()
sys.exit(0)

# # - Spiking net
# # - Create data loader
# nmnist_dataloader = NMNISTDataLoader()

# # - Train ANN MNIST model
# ann = train_ann_mnist()

# # - Turn into spiking deterministic net
# spiking_net = get_det_net().to(device)

# data_loader_test_spikes = nmnist_dataloader.get_data_loader(dset="test", mode="snn", shuffle=True, num_workers=4, batch_size=1, dt=3000)

# for idx, (data, target) in enumerate(data_loader_test_spikes):
#     X0 = data
#     if target == 1:
#         break

# X0 = X0[0].to(device)
# X0 = torch.clamp(X0, 0.0, 1.0)
# model_pred = get_prediction(spiking_net, X0, "non_prob")

# # - Attack using greedy attack
# hamming_distance_eps = 0.0025
# X_adv_scar, num_flipped = scar_attack(
#     hamming_distance_eps=hamming_distance_eps,
#     net=spiking_net,
#     X0=X0,
#     thresh=0.1,
#     verbose=True
# )

# model_pred_scar = get_prediction(spiking_net, X_adv_scar, "non_prob")
# print(f"Model: {int(model_pred)} Scar: {int(model_pred_scar)}")