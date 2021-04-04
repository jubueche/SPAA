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
    P0 = data
    if target == 0:
        break

P0 = P0[0].to(device)
P0 = torch.clamp(P0, 0.0, 1.0)
model_pred = get_prediction(spiking_net, P0)

# - Attack using greedy attack from https://par.nsf.gov/servlets/purl/10094021
# TODO