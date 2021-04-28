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

        # - Download the data if not already exist
        if not (self.path / "gestures_dataset_200ms").exists() :
            p = str(self.path / "gestures_dataset_200ms.zip")
            os.system(f"wget {url} -O {p}")
            with zipfile.ZipFile(self.path / "gestures_dataset_200ms.zip", 'r') as f:
                f.extractall(self.path)

    def get_data_loader(self, dset, shuffle=True, num_workers=4, batch_size=128, dt=5000):
        """
        Get the torch dataloader
        dset: "train" or "test"
        return dataloader
        """
        dataset = SpikeTrainDataset(
            self.path / f"{dset}_DS/",
            transform=np.float32,
            force_n_bins=10,
            target_transform=int,
            dt=dt
        )
        dataloader = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return dataloader
