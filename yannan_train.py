import torch
import numpy as np
from torch.utils.data import DataLoader
import time
import torch.nn as nn
from sinabs.from_torch import from_model
import sinabs.layers as sl
# import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import confusion_matrix
# import seaborn as sn
from aermanager.datasets import SpikeTrainDataset
from aermanager.transform import apply_by_channel
from tqdm import tqdm

"""
This is an example training file to implement IAF bptt training on IBM-gesture dataset
Enssential files:

train.py (This file) --- to run the main simulation
dataloader.py        --- to load the generated train/test samples
DATALOAD.py          --- to generate the sample data for simulation

"""

torch.manual_seed(0)
train_dir = "data/Gestures/gesture_dataset_200ms/train"
test_dir = "data/Gestures/gesture_dataset_200ms/test"

save_filename = "bptt"

batch_size = 32

epochs = 10

train_transform = apply_by_channel(
    transforms.RandomAffine(degrees=5,
                            translate=(0.1, 0.1),
                            scale=(0.9, 1.1))
)

train = SpikeTrainDataset(
    train_dir,
    transform=train_transform,
    target_transform=int,
    dt=10000  # microseconds
)

test = SpikeTrainDataset(
    test_dir,
    transform=np.float32,
    target_transform=int,
    dt=10000  # microseconds
)

trainset = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=12, drop_last=True)
testset = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=12, drop_last=True)

device = torch.device("cuda")


class bptt2(nn.Module):
    def __init__(self):
        super().__init__()
        specknet_ann = nn.Sequential(
            # Core 0
            # nn.AvgPool2d(kernel_size=(2,2)), # 2 ,32 , 32
            nn.Conv2d(2, 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),  # 8, 64, 64
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 8,32,32
            # """Core 1"""
            # nn.Dropout2d(0.5),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 16, 32, 32
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
            # """Core 2"""
            nn.Dropout2d(0.5),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 16, 16
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 8x8x8
            nn.Flatten(),
            # nn.Dropout2d(0.5),
            nn.Linear(8 * 8 * 8, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 11, bias=False),
            # nn.ReLU(),
        )
        self.model = from_model(
            specknet_ann, batch_size=batch_size, threshold=1
        ).spiking_model

    def forward(self, x):
        # print(x.shape)
        (batch_size, t_len, channel, height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.model(x)
        out = out.reshape(batch_size, t_len, 11)

        return out

    def direct_forward(self, x):
        return self.model(x)

    def reset_states(self):
        # Reset states of all spiking layers/neurons
        for lyr in self.model:
            # For each spiking layer
            if isinstance(lyr, sl.SpikingLayer):
                lyr.reset_states(randomize=True)


device = torch.device("cuda")
model = bptt2().to(device)

# fanout = [16 * 3 * 3, 8 * 3 * 3, 11, 0]

# count = SpikingSynopCounter(model.model, fanout=fanout)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)

C = nn.CrossEntropyLoss()

start = time.time()

# stat_dic = torch.load(save_filename + ".pth")
# #
# # #
# # #
# model.state_dict()["model.1.weight"][:]  = torch.nn.Parameter(stat_dic["model.1.weight"])
# model.state_dict()["model.4.weight"][:]  = torch.nn.Parameter(stat_dic["model.4.weight"])
# model.state_dict()["model.9.weight"][:]  = torch.nn.Parameter(stat_dic["model.9.weight"])
# model.state_dict()["model.10.weight"][:] = torch.nn.Parameter(stat_dic["model.10.weight"])
# model.state_dict()["model.14.weight"][:] = torch.nn.Parameter(stat_dic["model.14.weight"])


train_acc = []
train_loss = []
test_acc = []
best_test = 0

for epoch in range(epochs):
    num_samples = 0
    correct_samples = 0
    model.train()
    for sample, target in tqdm(trainset):
        model.reset_states()
        sample = sample.float()

        # print(sample.shape)
        # for i  in range(proplength):
        #     plt.imshow(sample[0, i, 0, ...])
        #     plt.show()
        #     print(target[0])

        sample = sample.to(device)
        target = target.long().to(device)

        out = model.forward(sample)
        # print(out.shape)
        out = torch.sum(out, 1)
        _, predict = torch.max(out, 1)
        loss = C(out, target)

        # if counter() < 100e4:
        #
        #     loss = C(out, target)
        # else:
        #     loss = C(out,target) + 1e-5*counter()

        correct_samples += (predict == target).sum().item()

        num_samples += batch_size

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # if num_samples % 1024 == 0:
        # print("synaptic operation:", counter())
    print(out[0])
    torch.save(model.state_dict(), save_filename + ".pth")
    # print("synop", count().sum())
    print("--------------------------------------------------------------------------------------------")
    print(
        "current epoch num:",
        epoch,
        "     |     ",
        "current loss:",
        loss.item(),
        "  |  ",
        "Time elapsed:",
        time.time() - start,
    )
    print(
        "current training accuracy",
        100 * correct_samples / num_samples,
        "    |      ",
        "current progress:",
        num_samples,
    )
    print("--------------------------------------------------------------------------------------------")
    train_acc.append(100 * correct_samples / num_samples)
    train_loss.append(loss)
    #
    #
    num_samples = 0
    correct_samples = 0
    model.eval()

    pred = []
    labelss = []
    for sample, target in testset:
        model.reset_states()
        sample = sample.float()

        # print(sample.shape)
        # for i  in range(proplength):
        #     plt.imshow(sample[0, i, 0, ...])
        #     plt.show()
        #     print(target[0])

        sample = sample.to(device)
        target = target.long().to(device)

        out = model.forward(sample)
        # print(out.shape)
        out = torch.sum(out, 1)
        _, predict = torch.max(out, 1)

        correct_samples += (predict == target).sum().item()

        num_samples += batch_size

        labelss.append(target)
        pred.append(predict)

        # print(out)
    if 100 * correct_samples / num_samples > best_test:
        best_test = 100 * correct_samples / num_samples
        torch.save(model.state_dict(), "bestacc.pth")
    print("--------------------------------------------------------------------------------------------")
    print(
        "current testing accuracy",
        100 * correct_samples / num_samples,
        "    |      ",
        "current progress:",
        num_samples,
    )
    print("--------------------------------------------------------------------------------------------")
    print(f"best test{best_test}")

    test_acc.append(100 * correct_samples / num_samples)


pre = np.zeros((batch_size * len(labelss), 1))
tar = np.zeros((batch_size * len(labelss), 1))
print(pred)
print(pred[0][0])
cc = 0
for i in range(len(pred)):
    for batch in range(pred[i].shape[0]):

        pre[cc] = pred[i][batch].cpu().numpy()
        tar[cc] = labelss[i][batch].cpu().numpy()
        cc += 1

cm = confusion_matrix(tar, pre)
print(cm)
# sn.heatmap(cm, annot=True, annot_kws={"size": 11})
# plt.show()
#
# x_axis = np.arange(epochs)
# plt.figure(1)
# plt.plot(x_axis, train_acc, "r")
# plt.plot(x_axis, test_acc, "b")
# plt.show()
#
# plt.figure()
# plt.plot(x_axis, train_loss, "g")
# plt.show()
