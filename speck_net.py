import torch.nn as nn
import torch
from sinabs.layers import NeuromorphicReLU


class SpeckNet(nn.Module):
    def __init__(self):
        super().__init__()
        quantize = False
        final_relu = True

        seq = [
            # core 0
            nn.Conv2d(2, 16, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=144),
            # core 1
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=288),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 2
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=288),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 7
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=576),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 4
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=576),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 5
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=1024),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 6
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 256, kernel_size=(2, 2), padding=(0, 0), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=128),
            # core 3
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 128, kernel_size=(1, 1), padding=(0, 0), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=11),
            # core 8
            nn.Conv2d(128, 11, kernel_size=(1, 1), padding=(0, 0), bias=False),
        ]
        if final_relu: seq.append(NeuromorphicReLU(quantize=quantize, fanout=0))
        seq.append(nn.Flatten())
        self.main = nn.Sequential(*seq)
        # weight init is really important to speckNet!!!!!!
        self._init_weight()

    def forward(self, x):
        return self.main(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)