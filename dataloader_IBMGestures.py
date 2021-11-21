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
import torch
import torchvision


class IBMGesturesDataLoader:
    """Downloads the DVS gestures dataset, slices samples into smaller subsamples and bins 
    events in one subsample into frames. These frames are cached on disk to speed up training
    considerably. Parameters that deal with time are given in microseconds. Keep in mind that
    the cached files will persist when you change a parameter such as dt or the slicing 
    window, which means that it is the user's responsibility to delete the cache manually,
    or use reset_cache=True to always clear out the cache during initialisation. See
    https://tonic.readthedocs.io/en/latest/reference/data_classes.html#cacheddataset
    for more info.
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
                        batch_size=128, dt=2000, aug_deg=20, aug_shift=0.1):
        """
        Get the torch dataloader
        dset: "train" or "test"
        return dataloader
        """
        sensor_size = tonic.datasets.DVSGesture.sensor_size
        frame_transform = tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, time_window=dt)
        augmentation = torchvision.transforms.Compose([torch.from_numpy, 
                                                       torchvision.transforms.RandomAffine(degrees=aug_deg, 
                                                                                           translate=(aug_shift,aug_shift))])
        assert dset in ["train", "test"]
        train_flag = True if dset=='train' else False
        dataset = tonic.datasets.DVSGesture(save_to=self.save_to, train=train_flag)
        # metadata_path = f'{self.slice_metadata_path}/dvs_gesture/frames/{self.slicing_time_window//1000}ms/{dset}'
        cache_path = f'{self.caching_path}/frames/{batch_size}batch_{dt}dt/{dset}'

        # trainset slices with overlap if enabled and applies frame transform (before caching) and augmentations (post caching)
        if train_flag:
            slicer = SliceByTime(time_window=self.slicing_time_window, overlap=self.slicing_overlap)
            sliced_dataset = SlicedDataset(dataset, slicer=slicer, transform=frame_transform)#, metadata_path=metadata_path)
            cached_dataset = CachedDataset(sliced_dataset, transform=augmentation, cache_path=cache_path, reset_cache=False)
        # testset slices without overlap and only applies frame transform
        else:
            slicer = SliceByTime(time_window=self.slicing_time_window, overlap=0)
            sliced_dataset = SlicedDataset(dataset, slicer=slicer, transform=frame_transform)#, metadata_path=metadata_path)
            cached_dataset = CachedDataset(sliced_dataset, cache_path=cache_path, reset_cache=False)
        return DataLoader(cached_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=True))

    def get_spiketrain_dataset(self, dset, shuffle=True, num_workers=4):
        """
        Get the torch dataloader
        dset: "train" or "test"
        return dataloader
        """
        slicer = SliceByTime(time_window=self.slicing_time_window, overlap=self.slicing_overlap)
        assert dset in ["train", "test"]
        train_flag = True if dset=='train' else False
        dataset = tonic.datasets.DVSGesture(save_to=self.save_to, train=train_flag)
        metadata_path = f'{self.slice_metadata_path}/dvs_gesture/spikes/{self.slicing_time_window//1000}ms/{dset}'
        sliced_dataset = SlicedDataset(dataset, slicer=slicer, metadata_path=metadata_path)
        cached_dataset = CachedDataset(sliced_dataset, cache_path=f'{self.caching_path}/spikes/{dset}')
        return DataLoader(cached_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=None, collate_fn=lambda x: x)


class OldIBMGesturesDataLoader:
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
