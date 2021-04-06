from videofig import videofig
from dataloader_NMNIST import NMNISTDataLoader
import torch
from experiment_utils import *

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# - Create data loader
nmnist_dataloader = NMNISTDataLoader()

# - Train ANN MNIST model
ann = train_ann_mnist()

# - Turn into spiking deterministic net
spiking_net = get_det_net().to(device)

data_loader_test_spikes = nmnist_dataloader.get_data_loader(dset="test", mode="snn", shuffle=True, num_workers=4, batch_size=1, dt=3000)

for idx, (data, target) in enumerate(data_loader_test_spikes):
    X0 = data
    if target == 1:
        break

X0 = X0[0].to(device)
X0 = torch.clamp(X0, 0.0, 1.0)
model_pred = get_prediction(spiking_net, X0, "non_prob")

# - Attack using greedy attack from https://openreview.net/pdf?id=xCm8kiWRiBT
hamming_distance_eps = 0.0025
X_adv_scar = scar_attack(
    hamming_distance_eps=hamming_distance_eps,
    net=spiking_net,
    X0=X0,
    thresh=0.0001,
    verbose=True
)

model_pred_scar = get_prediction(spiking_net, X_adv_scar, "non_prob")
print(f"Model: {int(model_pred)} Scar: {int(model_pred_scar)}")