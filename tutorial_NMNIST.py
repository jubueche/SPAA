from dataloader_NMNIST import NMNISTDataLoader
import torch
from networks import train_ann_mnist, get_prob_net
from attacks import boosted_hamming_attack, prob_attack_pgd
from utils import get_prediction, plot_attacked_prob

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# - Create data loader
nmnist_dataloader = NMNISTDataLoader()

# - Train ANN MNIST model
ann = train_ann_mnist()

# - Turn into spiking prob net
prob_net = get_prob_net().to(device)

data_loader_test_spikes = nmnist_dataloader.get_data_loader(
    dset="test", mode="snn", shuffle=True, num_workers=4, batch_size=1, dt=3000
)

for idx, (data, target) in enumerate(data_loader_test_spikes):
    P0 = data
    if target == 1:
        break

P0 = P0[0].to(device)
P0 = torch.clamp(P0, 0.0, 1.0)
model_pred = get_prediction(prob_net, P0)

# - Attack parameters
N_pgd = 50
N_MC = 10
eps = 1.5
eps_iter = 0.3
rand_minmax = 0.01
norm = 2
hamming_distance_eps = 0.0025
k = 50

return_dict = boosted_hamming_attack(
    k=k,
    prob_net=prob_net,
    P0=P0,
    eps=eps,
    eps_iter=eps_iter,
    N_pgd=N_pgd,
    N_MC=N_MC,
    norm=norm,
    rand_minmax=rand_minmax,
    verbose=True,
)

X_adv = return_dict["X_adv"]
num_flips = return_dict["L0"]

print("Original prediction", int(get_prediction(prob_net, P0)))
# - Evaluate on the attacked image
model_pred_attack_hamming = get_prediction(prob_net, X_adv, "non_prob")
print(
    f"Hamming attack prediction {int(model_pred_attack_hamming)} with L_0 = {num_flips}"
)

P_adv = prob_attack_pgd(
    prob_net=prob_net,
    P0=P0,
    eps=eps,
    eps_iter=eps_iter,
    N_pgd=N_pgd,
    N_MC=N_MC,
    norm=norm,
    rand_minmax=rand_minmax,
    verbose=True,
)

# - Evaluate the network X times
N_eval = 300
correct = []
for i in range(N_eval):
    model_pred_tmp = get_prediction(prob_net, P_adv, "prob")
    if model_pred_tmp == target:
        correct.append(1.0)
print("Evaluation accuracy", float(sum(correct) / N_eval))

plot_attacked_prob(P_adv, target, prob_net, N_rows=4, N_cols=4, block=False, figname=1)

plot_attacked_prob(P0, target, prob_net, N_rows=2, N_cols=2, block=False, figname=2)

plot_attacked_prob(
    P0,
    target,
    prob_net,
    N_rows=2,
    N_cols=2,
    data=[
        (torch.clamp(torch.sum(X_adv.cpu(), 1), 0.0, 1.0), model_pred_attack_hamming)
        for _ in range(2 * 2)
    ],
    figname=3,
)
