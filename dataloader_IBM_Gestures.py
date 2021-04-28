"""
Dataloader for IBM Gestures
"""
import zipfile
import os
import pathlib
import numpy as np
from torch.utils.data import DataLoader
from aermanager.datasets import SpikeTrainDataset


class IBMGesturesDataLoader:
    def __init__(
        self,
    ):
        # - Create data directory if not exist
        self.path = pathlib.Path(__file__).parent.absolute() / "data" / "IBMGestures"
        self.path.mkdir(parents=True, exist_ok=True)
        self.subfolder = "gestures_dataset_200ms"

        # - Download the data if not already exist
        if not (self.path / self.subfolder).exists() :
            p = str(self.path / f"{self.subfolder}.zip")
            os.system(f"wget {url} -O {p}")
            with zipfile.ZipFile(self.path / f"{self.subfolder}.zip", 'r') as f:
                f.extractall(self.path)

    def get_data_loader(self, dset, shuffle=True, num_workers=4, batch_size=128, dt=5000):
        """
        Get the torch dataloader
        dset: "train" or "test"
        return dataloader
        """
        dataset = SpikeTrainDataset(
            self.path / self.subfolder / dset,
            transform=np.float32,
            target_transform=int,
            dt=dt
        )
        dataloader = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers,
                                batch_size=batch_size, drop_last=True)
        return dataloader
