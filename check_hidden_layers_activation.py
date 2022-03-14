from networks import GestureClassifierSmall
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork
from sinabs.layers import SpikingLayer
from aermanager.preprocess import create_raster_from_xytp

import os
import torch
import h5py
import numpy as np

N_EPOCH = 9
TARGET_LABEL = 8
MAX = 50
PATCH_SIZE = 0.025

OUTPUTS_OF_SPIKING_LAYERS = []
DEVICE = "cpu"


def reset_states(net):
    for m in net.modules():
        if hasattr(m, "reset_states"):
            m.reset_states()


def hook_function(module, input_ts, output_ts):
    """forward hook function for nn.Modules that could save the sum of the outputs of hidden layers"""
    OUTPUTS_OF_SPIKING_LAYERS.append(output_ts.sum().item())


if __name__ == "__main__":
    # load weights and convert to quantized snn
    gesture_classifier = GestureClassifierSmall("BPTT_small_trained_200ms_2ms.pth", device=DEVICE)
    gesture_classifier = gesture_classifier.to(DEVICE)
    snn = gesture_classifier.model
    quantized_model = DynapcnnCompatibleNetwork(
        snn,
        discretize=True,
        input_shape=(2, 128, 128),
    )

    # get spiking layers of the quantized model, we need to register forward hook functions on for them
    spiking_layers = [each for each in quantized_model.sequence.modules() if isinstance(each, SpikingLayer)]

    # register forward hook functions
    for lyr in spiking_layers:
        lyr.register_forward_hook(hook_function)

    # loading the input data from h5 files
    attack_file_name = f"attacks_patches_ep{N_EPOCH}_lb{TARGET_LABEL}_num{MAX}_patchsize{str(PATCH_SIZE).split('.')[1]}.h5"
    attack_file = os.path.join("./attack_patches", attack_file_name)
    # read data from file
    data = h5py.File(attack_file, "r")
    successful_attacks = np.where(data["attack_successful"])[0]

    # start to inspect the activation of the hidden layers
    for idx in successful_attacks:
        print("=" * 50)
        origin_spk = data["original_spiketrains"][str(idx)]
        attacked_spk = data["attacked_spiketrains"][str(idx)]
        ground_truth = data["ground_truth"][idx]
        assert ground_truth == data["sinabs_label"][idx]
        print(f"Target: {TARGET_LABEL}, ground truth: {ground_truth}")
        # execute the forward-pass and the hook function will collect the sum of the spikes of each layer
        reset_states(quantized_model.sequence)
        raster_attacked = create_raster_from_xytp(attacked_spk, dt=500, bins_x=np.arange(129), bins_y=np.arange(129))
        output_ts = quantized_model(torch.tensor(raster_attacked).to(DEVICE)).squeeze().sum(0)
        print(f"N. spikes outputs by last layer: {output_ts.sum()}, activations for attacked output: {output_ts.detach().numpy().astype(int)}")
        print(f"N. spikes outputs by hidden layer from attacked input: {OUTPUTS_OF_SPIKING_LAYERS}")
        # empty this list after every iteration
        OUTPUTS_OF_SPIKING_LAYERS.clear()

