"""
Dataloader for N-MNIST
"""
import zipfile
import urllib.request
import pathlib
import os
import numpy as np
from torch.utils.data import DataLoader
from aermanager.parsers import parse_nmnist
from aermanager.datasets import FramesDataset, SpikeTrainDataset
from aermanager.dataset_generator import gen_dataset_from_folders

class NMNISTDataLoader:

    def __init__(
        self,
    ):
        # - Create data directory if not exist
        self.path = pathlib.Path(__file__).parent.absolute() / "data/"
        if not self.path.exists():
            os.mkdir(self.path)
            os.mkdir(self.path / "N-MNIST/")

        # - Download the data if not already exist
        def load_n_extract(lab, url):
            if not ((self.path / f"N-MNIST/Test/").exists() and (self.path / f"N-MNIST/Train/").exists()) :
                p = str(self.path / f"N-MNIST/{lab}.zip")
                os.system(f"wget {url} -O {p}")
                with zipfile.ZipFile(self.path / f"N-MNIST/{lab}.zip", 'r') as f:
                    f.extractall(self.path / "N-MNIST/")
        load_n_extract("test_Files", "https://www.dropbox.com/sh/tg2ljlbmtzygrag/AADSKgJ2CjaBWh75HnTNZyhca/Test.zip?dl=1")
        load_n_extract("train_Files", "https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABlMOuR15ugeOxMCX0Pvoxga/Train.zip?dl=1")

        def gen_ds(lab):
            if not (self.path / f"N-MNIST/{lab.lower()}_DS/").exists():
                gen_dataset_from_folders(
                    source_path=self.path / f"N-MNIST/{lab}", 
                    destination_path=self.path / f"N-MNIST/{lab.lower()}_DS/",
                    pattern="*.bin",
                    spike_count=300,
                    parser=parse_nmnist)
        gen_ds("Test")
        gen_ds("Train")

    def get_data_loader(self, dset, mode, shuffle=True, num_workers=4, batch_size=128, dt=1000):
        """
        Get the torch dataloader 
        dset: "train" or "test"
        mode: "ann" or "snn"
        return dataloader
        """
        if mode == "ann":
            dataset = FramesDataset(
                self.path / f"N-MNIST/{dset}_DS/",
                transform=np.float32,
                target_transform=int)
        elif mode == "snn":
            dataset = SpikeTrainDataset(
                self.path / f"N-MNIST/{dset}_DS/",
                transform=np.float32,
                target_transform=int,
                dt=dt)
        else:
            assert mode in ["ann","snn"], "Unknown mode"

        dataloader = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return dataloader