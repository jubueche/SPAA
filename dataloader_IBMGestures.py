"""
Dataloader for IBM Gestures
"""
import numpy as np
import os
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
import torch

classes_labels = [
    "hand clap",
    "right hand wave",
    "left hand wave",
    "right arm clockwise",
    "right arm counterclockwise",
    "left arm clockwise",
    "left arm counterclockwise",
    "arm roll",
    "air drums",
    "air guitar",
    "other gestures",]


def make_raster(spiketrain, dt, bins_xy, keep_polarity=False):
    x, y, t, p = spiketrain["x"], spiketrain["y"], spiketrain["t"], spiketrain["p"]
    timebins = np.arange(t[0], t[-1] + dt + 1, dt)
    timebins[-1] -= 1
    if keep_polarity:
        raster, _ = np.histogramdd((t, p, x, y), (timebins, (-1, 0.5, 2), *bins_xy))
    else:
        raster, _ = np.histogramdd((t, x, y), (timebins, *bins_xy))
        raster = raster[:, np.newaxis]
    return raster.astype(np.float32)


def pad_zeros(raster, n_bins=None):
    if len(raster) == n_bins:
        return raster
    else:
        new = np.zeros((n_bins, *raster.shape[1:]), dtype=raster.dtype)
        new[: len(raster)] = raster
        return new


class GestureRecognitionDataset(IterableDataset):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        from_spiketrain=False,
        dt=1000,
        twindow=50000,
        keep_polarity=False,
        csv_dir="./",
        random_shuffle=True,
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.keep_polarity = keep_polarity
        self.from_spiketrain = from_spiketrain
        self.twindow = twindow
        if from_spiketrain:
            self.dt = dt
            self.loader = self._ms_raster_loader
        else:
            self.loader = self._frame_loader

        framelist = np.loadtxt(
            os.path.join(root, "frames.csv"), delimiter=", ", dtype=str
        )
        self.samples = framelist[:, :2]
        self.csv_dir = csv_dir
        self.csv_dict = {}
        self._current_index = 0
        self.random_shuffle = random_shuffle

    def get_csv(self, filename):
        filename = os.path.basename(filename)
        csv_file = filename.split(".")[0] + "_labels.csv"
        if csv_file not in self.csv_dict.keys():
            self.csv_dict[csv_file] = np.loadtxt(
                os.path.join(self.csv_dir, csv_file), delimiter=",", skiprows=1
            ).T
        return self.csv_dict[csv_file]

    def label_from_csv(self, file):
        # should return the label, or None if it's ambiguous
        start, end = file["spiketrain"]["t"][[0, 1]]
        labels, start_times, end_times = self.get_csv(file["original_filename"].item())
        idx = np.searchsorted(start_times, start)
        if idx > 0 and end <= end_times[idx - 1]:
            # print("Frame between", start, end, "returned label", labels[idx-1])
            return labels[idx - 1] - 1  # CAREFUL, we subtract 1, labels are 0...10 now
        # print("Frame between", start, end, "returned label None")
        return None

    def _frame_loader(self, file):
        with np.load(os.path.join(self.root, file)) as f:
            label = self.label_from_csv(f)
            if label is not None:
                frames = f["frame"].astype(np.float32)
                return frames, label
        return None

    def _ms_raster_loader(self, file):
        with np.load(os.path.join(self.root, file)) as f:
            label = self.label_from_csv(f)
            if label is not None:
                bins_xy = f["bins_xy"]
                raster = make_raster(
                    f["spiketrain"], self.dt, bins_xy, self.keep_polarity
                )
                if self.from_spiketrain:
                    raster = pad_zeros(raster, n_bins=int(self.twindow / self.dt))
                return raster, label
        return None

    def __iter__(self):
        sample_idx = np.arange((len(self.samples)))
        if self.random_shuffle:
            np.random.shuffle(sample_idx)
        for i in sample_idx:
            path, _ = self.samples[i]  # '_' ignores the fake '0' label
            loaded_item = self.loader(path)

            if loaded_item is not None:  # if it's none, ignore it and do nothing
                sample, target = loaded_item

                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                yield sample, target, path


class BPTTIBMDataLoader(nn.Module):
    def __init__(self, datadir, listfile, mode = None):
        download()
        self.path = datadir
        self.samples_file_name = np.loadtxt(listfile, usecols=0, dtype=str)
        self.samples_label = np.loadtxt(listfile, usecols=1)

    def __getitem__(self, index):
       samples = torch.load(self.path + self.samples_file_name[index])
       labels  = torch.tensor(self.samples_label[index]-1)
       return samples, labels

    def __len__(self):
        return len(self.samples_file_name)


TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/Gestures/train_data/")
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/Gestures/test_data/")
TRAIN_FILE = os.path.join(TRAIN_DIR, "train_file.txt")
TEST_FILE = os.path.join(TEST_DIR, "test_file.txt")

def download():
    """
    Download the files and store in data/Gestures
    """
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base,"data/Gestures.zip")
    if not os.path.isfile(p):
        # pip install gdown
        os.system("gdown https://drive.google.com/uc?id=1diAsgKLxcRdj-P18WckNbHJ7s7x6jeA8")
        os.system("mv Gestures.zip data/")
        os.system("unzip data/Gestures.zip -d data/")

def get_data_loader(dset, shuffle=True, num_workers=4, batch_size=128):
    """
    Get the torch dataloader 
    dset: "train" or "test"
    return dataloader
    """
    download()
    if dset == "train":
        dataset = BPTTIBMDataLoader(datadir=os.path.join(TRAIN_DIR,"data/"), listfile=TRAIN_FILE, mode='test')
    elif dset == "test":
        dataset = BPTTIBMDataLoader(datadir=os.path.join(TEST_DIR,"data/"), listfile=TEST_FILE, mode='test')
    else:
        assert dset in ["train","test"], "Unknown dset"
    if batch_size == -1:
        batch_size = dataset.__len__()
    dataloader = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=True)
    return dataloader
