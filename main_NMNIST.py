"""
Train non-spiking model for N-MNIST and convert to spiking
"""
from dataloader_NMNIST import NMNISTDataLoader
from sinabs.from_torch import from_model
from sinabs.utils import normalize_weights
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from experiment_utils import ProbNetwork

def train_ann_mnist():
    """
    Checks if model exists. If not, train and store. Return model.
    """
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

    return ann

def get_prob_net(ann = None):
    """
    Description here
    """
    if ann == None:
        ann = train_ann_mnist()

    # - Create data loader
    nmnist_dataloader = NMNISTDataLoader()

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
    model = from_model(ann, input_shape=input_shape, add_spiking_output=True)

    # - Create probabilistic network
    prob_net = ProbNetwork(
            ann,
            model.spiking_model,
            input_shape=input_shape
        )
    prob_net.spiking_model[0].weight.data *= 7

    return prob_net