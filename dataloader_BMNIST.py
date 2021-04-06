"""
Dataloader for B-MNIST
"""

import pathlib
import torch
import torchvision
from torch.utils.data import DataLoader
import os

class BMNISTDataLoader:

    def __init__(
        self,
    ):

        # - Create data directory if not exist
        self.path = pathlib.Path(__file__).parent.absolute() / "data/"
        if not self.path.exists():
            os.mkdir(self.path)
        if not (self.path / "B-MNIST/").exists():
            os.mkdir(self.path / "B-MNIST/")
        
        p = self.path / "B-MNIST/"
        if not (self.path / "B-MNIST/MNIST.tar.gz").exists():
            os.system(f"wget -P {str(p)}  www.di.ens.fr/~lelarge/MNIST.tar.gz")
        if not (self.path / "B-MNIST/MNIST/").exists():
            os.system(f"tar -zxvf {str(p)}/MNIST.tar.gz -C {str(p)}/")

        class RoundT:
            def __call__(self,X):
                return torch.round(X)
            def __repr__(self):
                return self.__class__.__name__ + '()'

        # - Download the data if not already exist
        self.mnist_train_ds = torchvision.datasets.MNIST(
            p,
            train=True, 
            download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                RoundT()])
        )

        self.mnist_test_ds = torchvision.datasets.MNIST(
            p,
            train=False,
            download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                RoundT()])
        )


    def get_data_loader(self, dset, shuffle=True, num_workers=4, batch_size=128):
        """
        Get the torch dataloader 
        dset: "train" or "test"
        return dataloader
        """
        if dset == "train":
            dataloader = DataLoader(self.mnist_train_ds, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        elif dset == "test":
            dataloader = DataLoader(self.mnist_test_ds, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        else:
            assert dset in ["train","test"], "Unknown dset"
        return dataloader