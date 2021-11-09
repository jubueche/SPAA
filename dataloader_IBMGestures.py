"""
Dataloader for IBM Gestures.
Note that this downloads some pre-framed data prepared with AERManager at 200 ms.
"""
import os
import pathlib
import numpy as np
from torch.utils.data import DataLoader
from aermanager.datasets import SpikeTrainDataset
import tonic
from tonic import CachedDataset, SlicedDataset
from tonic.slicers import SliceByTime


class TonicIBMGesturesDataLoader:
    """Downloads the DVS gestures dataset, slices samples into smaller subsamples and bins 
    events in one subsample into frames. These frames are cached on disk to speed up training
    considerably. Parameters that deal with time are given in microseconds.
    """
    def __init__(self, 
                 save_to='./data', 
                 slicing_time_window=200000, 
                 slicing_overlap=150000, 
                 slice_metadata_path='./metadata',
                 caching_path='./cache',
                ):
        self.save_to = save_to
        self.slicing_time_window = slicing_time_window
        self.slicing_overlap = slicing_overlap
        self.slice_metadata_path = slice_metadata_path
        self.caching_path = caching_path
        # download data if not already on disk
        tonic.datasets.DVSGesture(save_to=save_to, train=True)
        tonic.datasets.DVSGesture(save_to=save_to, train=False)

    def get_data_loader(self, dset, shuffle=True, num_workers=4,
                        batch_size=128, dt=2000):
        slicer = SliceByTime(time_window=self.slicing_time_window, overlap=self.slicing_overlap)
        frame_transform = tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, 
                                                   time_window=dt, 
                                                   include_incomplete=True)
        assert dset in ["train", "test"]
        train_flag = True if dset=='train' else False
        dataset = tonic.datasets.DVSGesture(save_to=self.save_to, train=train_flag)
        metadata_path = f'{self.slice_metadata_path}/dvs_gesture/{self.slicing_time_window//1000}ms/{dset}'
        sliced_dataset = SlicedDataset(dataset, slicer=slicer, transform=frame_transform, metadata_path=metadata_path)
        cached_dataset = CachedDataset(sliced_dataset, cache_path=f'{self.caching_path}/{batch_size}batch_{dt}dt/{dset}')
        return DataLoader(cached_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


class IBMGesturesDataLoader:
    def __init__(
        self,
    ):
        # - Create data directory if not exist
        self.path = pathlib.Path(__file__).parent.absolute() / "data" / "Gestures"
        self.path.mkdir(parents=True, exist_ok=True)
        self.subfolder = "gesture_dataset_200ms"

        # - Download the data if not already exist
        if not (self.path / self.subfolder).exists():
            os.system("gdown https://drive.google.com/uc?id=1BqBaqoPpr1YUx8s1boYtt46Tz4IOxpLy")
            os.system("mv gesture_dataset_200ms.zip data/Gestures/")
            os.system("unzip data/Gestures/gesture_dataset_200ms.zip -d data/Gestures/")

    def get_data_loader(self, dset, shuffle=True, num_workers=4,
                        batch_size=128, dt=10000, force_n_bins=None):
        """
        Get the torch dataloader
        dset: "train" or "test"
        return dataloader
        """
        dataset = SpikeTrainDataset(
            self.path / self.subfolder / dset,
            transform=np.float32,
            target_transform=int,
            dt=dt,
            force_n_bins=force_n_bins
        )
        dataloader = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers,
                                batch_size=batch_size, drop_last=True)
        return dataloader

    def get_spiketrain_dataset(self, dset, shuffle=True, num_workers=4):
        """
        Get the torch dataloader
        dset: "train" or "test"
        return dataloader
        """
        dataset = SpikeTrainDataset(
            self.path / self.subfolder / dset,
            transform=None,
            target_transform=int,
            dt=None,
        )
        dataloader = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers,
                                batch_size=None, collate_fn=lambda x: x)
        return dataloader
