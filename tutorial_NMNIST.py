import numpy as np
from videofig import videofig
from dataloader_NMNIST import NMNISTDataLoader
from sinabs.from_torch import _from_model
from sinabs.utils import normalize_weights
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from experiment_utils import *

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# - Create data loader
nmnist_dataloader = NMNISTDataLoader()

# - Set the seed
torch.manual_seed(42)

# - Create sequential model
ann = nn.Sequential(
    nn.Conv2d(2, 20, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(20, 32, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(32, 128, 3, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Flatten(),
    nn.Linear(128, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 10, bias=False),
)
ann = ann.to(device)

if not (nmnist_dataloader.path / "N-MNIST/mnist_ann.pt").exists():
    data_loader_train = nmnist_dataloader.get_data_loader(dset="train", mode="ann", shuffle=True, num_workers=4, batch_size=64)
    optim = torch.optim.Adam(ann.parameters(), lr=1e-3)
    n_epochs = 1
    for n in range(n_epochs):
        for data, target in data_loader_train:
            data, target = data.to(device), target.to(device) # GPU
            output = ann(data)
            optim.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optim.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
    torch.save(ann.state_dict(), nmnist_dataloader.path / "N-MNIST/mnist_ann.pt")
else:
    ann.load_state_dict(torch.load(nmnist_dataloader.path / "N-MNIST/mnist_ann.pt"))

data_loader_test = nmnist_dataloader.get_data_loader(dset="test", mode="ann", batch_size=64)

# - Get single batch
for data, target in data_loader_test:
    break

# - Normalize weights
normalize_weights(
    ann.cpu(), 
    torch.tensor(data).float(),
    output_layers=['1','4','7','11'],
    param_layers=['0','3','6','10','12'])

# - Create spiking model
input_shape = (2, 34, 34)
spk_model = _from_model(ann, input_shape=input_shape, add_spiking_output=True)

# - Create probabilistic network
prob_net = ProbNetwork(
        ann,
        spk_model,
        input_shape=input_shape
    )
prob_net.spiking_model[0].weight.data *= 7

data_loader_test_spikes = nmnist_dataloader.get_data_loader(dset="test", mode="snn", shuffle=True, num_workers=4, batch_size=1, dt=3000)

for idx, (data, target) in enumerate(data_loader_test_spikes):
    P0 = data
    if target == 0:
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

P_adv = prob_attack_pgd(
    prob_net=prob_net,
    P0=P0,
    eps=eps,
    eps_iter=eps_iter,
    N_pgd=N_pgd,
    N_MC=N_MC,
    norm=norm,
    rand_minmax=rand_minmax,
    verbose=True
)

# - Evaluate the network X times
print("Original prediction",int(get_prediction(prob_net, P0)))
N_eval = 300
correct = []
for i in range(N_eval):
    model_pred_tmp = get_prediction(prob_net, P_adv, "prob")
    if model_pred_tmp == target:
        correct.append(1.0)
print("Evaluation accuracy",float(sum(correct)/N_eval))

plot_attacked_prob(
    P_adv,
    target,
    prob_net,
    N_rows=4,
    N_cols=4,
    block=False,
    figname=1
)

plot_attacked_prob(
    P0,
    target,
    prob_net,
    N_rows=2,
    N_cols=2,
    figname=2
)