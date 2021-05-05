import torch
from sinabs.network import Network as SinabsNetwork
from sinabs.layers.iaf_bptt import SpikingLayer
from sinabs.from_torch import from_model
from sinabs.utils import normalize_weights
from torch import nn
import pathlib
from dataloader_NMNIST import NMNISTDataLoader
from dataloader_BMNIST import BMNISTDataLoader
from utils import reparameterization_bernoulli
import torch.nn.functional as F
import os

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


class ProbNetwork(SinabsNetwork):
    """
    Probabilistic Network
    A sinabs model that is evaluated using probabilities for Bernoulli random variables.
    Methods:
        forward: Received probabilities, draws from the Bernoulli random variables and evaluates the network
        forward_np: Non-probabilistic forward method. Input: Spikes
    """
    def __init__(
        self,
        model,
        spk_model,
        input_shape,
        synops=False,
        temperature=0.01
    ):
        self.temperature = temperature
        super().__init__(model, spk_model, input_shape, synops)

    def forward(self, P):
        X = reparameterization_bernoulli(P, self.temperature)
        return super().forward(X)

    def forward_np(self, X):
        return super().forward(X)


class ProbNetworkContinuous(torch.nn.Module):
    """
    Probabilistic Network
    Continuous torch model that is evaluated using probabilities for Bernoulli random variables.
    Methods:
        forward: Received probabilities, draws from the Bernoulli random variables and evaluates the network
        forward_np: Non-probabilistic forward method. Input: Spikes
    """
    def __init__(
        self,
        model,
        temperature=0.01
    ):
        super(ProbNetworkContinuous, self).__init__()
        self.temperature = temperature
        self.model = model

    def forward(self, P):
        X = reparameterization_bernoulli(P, self.temperature)
        return self.model.forward(X)

    def forward_np(self, X):
        return self.model.forward(X)


class SummedSNN(SinabsNetwork):

    def __init__(
        self,
        model,
        spk_model,
        input_shape,
        n_classes,
        synops=False
    ):
        self.n_classes = n_classes
        super().__init__(model, spk_model, input_shape, synops)

    def forward(self, X):
        spike_out = super().forward(X)
        X = torch.reshape(torch.sum(spike_out, axis=0), (1, self.n_classes))
        return X

class IBMGesturesBPTT(nn.Module):

    def __init__(self):
        super().__init__()
        specknet_ann = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),  # 8, 64, 64
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 8,32,32
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 16, 32, 32
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
            nn.Dropout2d(0.5),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 16, 16
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 8x8x8
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 11, bias=False),
        )
        self.model = from_model(specknet_ann, threshold=1).spiking_model

    def forward(self, x):
        out = self.forward_raw(x)
        out = torch.sum(out, dim=1)
        return out

    def forward_raw(self, x):
        if x.ndim == 4:
            x = torch.reshape(x, (1,) + x.shape)
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.model(x)
        out = out.reshape(batch_size, t_len, 11)
        return out

    def reset_states(self):
        for lyr in self.model:
            if isinstance(lyr, SpikingLayer):
                lyr.reset_states(randomize=False)

def get_ann_arch():
    """
    Generate ann architecture and return
    """
    # - Create sequential model
    ann = nn.Sequential(
        nn.Conv2d(2, 20, 5, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(20, 32, 5, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(32, 128, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(128, 500, bias=False),
        nn.ReLU(),
        nn.Linear(500, 10, bias=False),
    )
    ann = ann.to(device)
    return ann


def get_mnist_ann_arch():
    """
    Generate cnn architecture for MNIST described in https://openreview.net/pdf?id=xCm8kiWRiBT
    """
    # - Create sequential model
    ann = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(p=0.25),
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(9216, 128),
        nn.Linear(128, 10)
    )
    ann = ann.to(device)
    return ann


def load_ann(path, ann=None):
    """
    Tries to load ann from path, returns None if not successful
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    if ann is None:
        ann = get_ann_arch()
    if not path.exists():
        return None
    else:
        ann.load_state_dict(torch.load(path, map_location=torch.device(device)))
        ann.eval()
        return ann

def get_summed_network(ann, n_classes):
    # - Get the deterministic spiking model
    model = get_det_net(ann)

    # - Create summed network
    s_net = SummedSNN(
        ann,
        model.spiking_model,
        input_shape=(2, 34, 34),
        n_classes=n_classes
    )
    return s_net.to(device)

def get_det_net(ann=None):
    """
    Transform the continuous network into spiking network using the sinabs framework.
    Generate the ann if the passed ann is None.
    Normalize the weights and increase initial weights by multiplicative factor.
    """
    if ann is None:
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
        output_layers=['1', '4', '7', '11'],
        param_layers=['0', '3', '6', '10', '12'])

    # - Create spiking model
    model = from_model(ann, input_shape=(2, 34, 34), add_spiking_output=True)

    # - Increase 1st layer weights by magnitude
    model.spiking_model[0].weight.data *= 12
    return model


def get_prob_net(ann=None, snn=None, input_shape=(2,34,34)):
    """
    Create probabilistic network from spiking network and return.
    """
    # - Get the deterministic spiking model
    if snn is None:
        model = get_det_net(ann)
        snn = model.spiking_model

    # - Create probabilistic network
    prob_net = ProbNetwork(
        ann,
        snn,
        input_shape=input_shape
    )
    return prob_net


def get_prob_net_continuous(ann=None):
    """
    Create probabilistic network from standard torch model.
    """
    if ann is None:
        ann = train_ann_binary_mnist()
    prob_net = ProbNetworkContinuous(ann)
    return prob_net


def train_ann_mnist():
    """
    Checks if model exists. If not, train and store. Return model.
    """
    # - Create data loader
    nmnist_dataloader = NMNISTDataLoader()

    # - Set the seed
    torch.manual_seed(42)

    # - Setup path
    path = nmnist_dataloader.path / "mnist_ann.pt"

    ann = load_ann(path)
    if ann is None:
        ann = get_ann_arch()
        data_loader_train = nmnist_dataloader.get_data_loader(
            dset="train", mode="ann", shuffle=True, num_workers=4, batch_size=64)
        optim = torch.optim.Adam(ann.parameters(), lr=1e-3)
        n_epochs = 1
        for n in range(n_epochs):
            for data, target in data_loader_train:
                data, target = data.to(device), target.to(device)  # GPU
                output = ann(data)
                optim.zero_grad()
                loss = F.cross_entropy(output, target)
                loss.backward()
                optim.step()
        torch.save(ann.state_dict(), path)
    ann.eval()
    return ann


def train_ann_binary_mnist():
    """
    Checks if binary MNIST model exists. If not, train and store. Return model.
    """
    # - Create data loader
    bmnist_dataloader = BMNISTDataLoader()

    # - Set the seed
    torch.manual_seed(42)

    # - Setup path
    path = bmnist_dataloader.path / "B-MNIST/mnist_ann.pt"

    ann = load_ann(path, ann=get_mnist_ann_arch())

    if ann is None:
        ann = get_mnist_ann_arch()
        data_loader_train = bmnist_dataloader.get_data_loader(
            dset="train", shuffle=True, num_workers=4, batch_size=64)
        optim = torch.optim.Adam(ann.parameters(), lr=1e-3)
        n_epochs = 50
        for n in range(n_epochs):
            print(f"Epoch {n} / {n_epochs}")
            for data, target in data_loader_train:
                data, target = data.to(device), target.to(device)  # GPU
                output = ann(data)
                optim.zero_grad()
                loss = F.cross_entropy(output, target)
                loss.backward()
                optim.step()
        torch.save(ann.state_dict(), path)
    ann.eval()  # - Set into eval mode for dropout layers
    return ann

def load_gestures_snn(load_path):
    """
    Load IBM gesture spiking CNN, turn into eval mode and return. If load_path None, load
    mdoel from data/Gestures/ibm_gestures_snn.model
    """

    # - Load the model
    model = IBMGesturesBPTT()

    stat_dic = torch.load(load_path, map_location=torch.device(device))
    stat_dic["model.2.state"] = model.state_dict()["model.2.state"][:]
    stat_dic["model.6.state"] = model.state_dict()["model.6.state"][:]
    stat_dic["model.11.state"] = model.state_dict()["model.11.state"][:]
    stat_dic["model.15.state"] = model.state_dict()["model.15.state"][:]

    stat_dic["model.2.activations"] = model.state_dict()["model.2.activations"][:]
    stat_dic["model.6.activations"] = model.state_dict()["model.6.activations"][:]
    stat_dic["model.11.activations"] = model.state_dict()["model.11.activations"][:]
    stat_dic["model.15.activations"] = model.state_dict()["model.15.activations"][:]

    model.load_state_dict(stat_dic)
    model.eval()
    model.to(device)
    return model
