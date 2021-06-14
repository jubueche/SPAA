import torch
from torch import nn
from sinabs.from_torch import from_model
import numpy as np

# software to interact with dynapcnn
from sinabs.backend.dynapcnn import io
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork
from aermanager.preprocess import create_raster_from_xytp
from sinabs.layers import SpikingLayer

# data and networks from this library
from dataloader_IBMGestures import IBMGesturesDataLoader
from sparsefool import frame_based_sparsefool


def spiketrain_forward(spk):
    input_events = io.xytp_to_events(spiketrain, layer=0, device="dynapcnndevkit:0")
    evs_out = hardware_compatible_model(input_events)
    evs_out = io.events_to_xytp(evs_out, layer=8)
    print("N. spikes from chip:", len(evs_out))

    if len(evs_out) == 0:
        return 0  # wrong but actually imitates the behaviour of torch.
    labels, counts = np.unique(evs_out["channel"], return_counts=True)
    most_active_neuron = labels[np.argmax(counts)]
    return most_active_neuron


def attack_on_spiketrain(spk):
    # first, we need to rasterize the spiketrain
    raster = create_raster_from_xytp(
        spiketrain, dt=10000, bins_x=np.arange(129), bins_y=np.arange(129))
    # note that to do this we are forced to suppress many spikes
    print("Spikes before binarization:", raster.sum())
    raster = torch.clamp(torch.tensor(raster), 0., 1.)
    print("Spikes after binarization:", raster.sum())

    # performing the actual attack
    return_dict_sparse_fool = frame_based_sparsefool(
        x_0=raster,
        # y=target,
        net=snn,
        max_hamming_distance=np.inf,
        lambda_=1.0,
        epsilon=0.0,
        overshoot=0.02,
        n_attack_frames=3,
        step_size=0.05,
        device="cpu",
        early_stopping=False,
        boost=False,
        verbose=True,
    )
    attacked_raster = return_dict_sparse_fool["X_adv"]

    # now we only look at where spikes were ADDED (heuristically!)
    added_to_raster = attacked_raster > raster
    breakpoint()
    return spk


class GestureClassifier(nn.Module):
    def __init__(self, file):
        super().__init__()

        self.seq = nn.Sequential(
            # core 0
            nn.Conv2d(2, 16, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),
            nn.ReLU(),
            # core 1
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 2
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 7
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 4
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 5
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 6
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 256, kernel_size=(2, 2), padding=(0, 0), bias=False),
            nn.ReLU(),
            # core 3
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 128, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(),
            # core 8
            nn.Conv2d(128, 11, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(),
            # nn.Flatten(),  # otherwise torch complains
        )
        self.load_state_dict(torch.load(file))
        self.model = from_model(self.seq).spiking_model

    def forward(self, x):
        out = self.forward_raw(x)
        out = torch.sum(out, dim=1)
        return out

    def forward_raw(self, x):
        if x.ndim == 4:
            x = torch.reshape(x, (1,) + x.shape)
        (batch_size, t_len, channel, height, width) = x.shape

        # - Set the batch size in the spiking layer
        self.set_batch_size(batch_size)

        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.model(x)
        out = out.reshape(batch_size, t_len, 11)
        return out

    def set_batch_size(self, batch_size):
        for lyr in self.model:
            if isinstance(lyr, SpikingLayer):
                lyr.batch_size = batch_size

    def reset_states(self):
        for lyr in self.model:
            if isinstance(lyr, SpikingLayer):
                lyr.reset_states(randomize=False)


# - Dataloader of spiketrains (not rasters!)
data_loader_test = IBMGesturesDataLoader().get_spiketrain_dataset(
    dset="test",
    shuffle=True,
    num_workers=4,
)  # - Can vary

# - Preparing the model
snn = GestureClassifier("data/Gestures/Gestures_SpeckNetA_framebased.pth")
input_shape = (2, 128, 128)
hardware_compatible_model = DynapcnnCompatibleNetwork(
    snn.model,
    discretize=True,
    input_shape=input_shape,
)

# - Apply model to device
config = hardware_compatible_model.make_config(
    [0, 1, 2, 7, 4, 5, 6, 3, 8], monitor_layers=[8])
hardware_compatible_model.to(
    device="dynapcnndevkit:0",
    chip_layers_ordering=[0, 1, 2, 7, 4, 5, 6, 3, 8],
    monitor_layers=[8],
)


correct = 0
correct_sinabs = 0
for i, (spiketrain, label) in enumerate(data_loader_test):
    # attack_on_spiketrain(spiketrain)
    # resetting states
    hardware_compatible_model.samna_device.get_model().apply_configuration(config)
    # forward pass on the chip
    out_label = spiketrain_forward(spiketrain)

    # raster data for sinabs
    raster = create_raster_from_xytp(
        spiketrain, dt=1000, bins_x=np.arange(129), bins_y=np.arange(129))
    snn.reset_states()
    out_sinabs = snn.model(torch.tensor(raster)).squeeze().sum(0)
    out_label_sinabs = torch.argmax(out_sinabs).item()
    print("N. spikes from sinabs:", out_sinabs.sum())

    print("Ground truth:", label, "chip:", out_label, "sinabs:", out_label_sinabs)
    if out_label == label:
        correct += 1
    if out_label_sinabs == label:
        correct_sinabs += 1

    if i > 100:
        break

print("Accuracy on chip:", correct / (i + 1))
print("Accuracy in simulation:", correct_sinabs / (i + 1))


io.close_device("dynapcnndevkit")
