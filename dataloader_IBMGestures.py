"""
Dataloader for IBM Gestures.
Note that this downloads some pre-framed data prepared with AERManager at 200 ms.
"""
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
        self.path = pathlib.Path(__file__).parent.absolute() / "data" / "Gestures"
        self.path.mkdir(parents=True, exist_ok=True)
        self.subfolder = "gesture_dataset_200ms"

        # - Download the data if not already exist
        if not (self.path / self.subfolder).exists() :
            os.system("gdown https://drive.google.com/uc?id=1BqBaqoPpr1YUx8s1boYtt46Tz4IOxpLy")
            os.system("mv gesture_dataset_200ms.zip data/Gestures/")
            os.system("unzip data/Gestures/gesture_dataset_200ms.zip -d data/Gestures/")

    def get_data_loader(self, dset, shuffle=True, num_workers=4, batch_size=128, dt=10000):
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
