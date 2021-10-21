import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import h5py
import sys
import os

# software to interact with dynapcnn and data
from sinabs.backend.dynapcnn import io
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork
from sinabs.synopcounter import SNNSynOpCounter
from aermanager.preprocess import create_raster_from_xytp
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.utils import normalize_weights
from sinabs.from_torch import from_model

# data and networks from this library
from networks import GestureClassifierSmall

# hyper-param
DEVICE = "cpu"
DYNAPCNN_HARDWARE = "speck2b" # could be dynapcnndevkit or speck2b for example
TARGET_LABEL = int(sys.argv[1])
N_EPOCH = sys.argv[2]
PATCH_SIZE = sys.argv[3]
MAX = 50

torch.random.manual_seed(1)

events_struct = [("x", np.uint16), ("y", np.uint16), ("t", np.uint64), ("p", bool)]


def reset_states(net):
    for m in net.modules():
        if hasattr(m, "reset_states"):
            m.reset_states()


def spiketrain_forward(spiketrain, factory):
    input_events = factory.xytp_to_events(
        spiketrain, layer=layers_ordering[0])
    evs_out = hardware_compatible_model(input_events)
    output_neuron_index = [ev.feature for ev in evs_out]
    times = [ev.timestamp for ev in evs_out]
    activations = np.bincount(output_neuron_index)
#     import ipdb; ipdb.set_trace()
    print(f"N. spikes from chip: {len(output_neuron_index)}, activations: {activations}")
    if len(output_neuron_index) == 0:
        # wrong prediction if there is no spike
        return -1
    return activations.argmax()


if __name__ == "__main__":
    # - Preparing the model
    gesture_classifier = GestureClassifierSmall("BPTT_small_trained_200ms_2ms.pth", device=DEVICE)
    gesture_classifier = gesture_classifier.to(DEVICE)
    ann = gesture_classifier.seq
    ann.eval()
    snn = gesture_classifier.model
    snn.eval()


    gesture_classifier.eval()
    # convert to chip-compatible spiking network (discretized etc.)
    input_shape = (2, 128, 128)

    # Get file
    attack_file_name = f"attacks_patches_ep{N_EPOCH}_lb{TARGET_LABEL}_num{MAX}_patchsize{str(PATCH_SIZE).split('.')[1]}.h5"
    attack_file = os.path.join("./attack_patches", attack_file_name)
    print(f"getting data from file {attack_file_name}")

    # Read data from file
    data = h5py.File(attack_file, "r")
    successful_attacks = np.where(data["attack_successful"])[0]

    # normalise weights
#     spike_layers = [name for name, child in ann.cpu().named_children() if isinstance(child, (nn.ReLU))]
#     param_layers = [name for name, child in ann.cpu().named_children() if isinstance(child, (nn.Conv2d, nn.Linear))]
#     input_shape=(2, 128, 128)
#     batch_size = 10
#     frames = np.zeros((batch_size, *input_shape))
#     for i in range(batch_size):
#         spikes = data["original_spiketrains"][str(i)]
#         raster = create_raster_from_xytp(spikes, dt=500, bins_x=np.arange(129), bins_y=np.arange(129))
#         frames[i] = raster.sum(0)
#     normalize_weights(ann, torch.FloatTensor(frames), output_layers=spike_layers, param_layers=param_layers)
#     normalised_model = from_model(ann, input_shape=input_shape, add_spiking_output=False, synops=False)
#     snn = normalised_model.spiking_model
#     snn.eval()

#     scale_factor = 0.4
#     with torch.no_grad():
#         snn[0].weight *= scale_factor
#         snn[3].weight *= scale_factor
#         snn[7].weight *= scale_factor

    hardware_compatible_model = DynapcnnCompatibleNetwork(
        snn,
        discretize=True,
        input_shape=input_shape,
    )

    # Apply model to device
    layers_ordering = [0, 1, 2, 3]  # [0, 1, 2, 7, 4, 5, 6, 3, 8]

    hardware_compatible_model.to(
        device=DYNAPCNN_HARDWARE,
        chip_layers_ordering=layers_ordering,
        monitor_layers=[layers_ordering[-1]],
        )

    # Report file
    report_save_dir = "./attack_result_csv"
    if not os.path.exists(report_save_dir): os.makedirs(report_save_dir)
    report_save_name = f"report_ep{N_EPOCH}_lb{TARGET_LABEL}_num{MAX}_patchsize{PATCH_SIZE}.csv"
    report = open(os.path.join(report_save_dir, report_save_name), "w")
    success_rate_targeted = round(data["targeted_patch_successful_rate"][()], 3)
    success_rate_random = round(data["random_patch_successful_rate"][()], 3)
    report.write(f"ID,ground_truth,"
                 f"chip_out,sim_out,"
                 f"chip_out_attacked_targeted,chip_out_attacked_random,"  # attack result on-chip
                 f"sim_out_attacked_targeted,sim_out_attacked_random\n")  # attack result simulated

    # - Start testing
    counter = SNNSynOpCounter(snn)
    for i in tqdm(successful_attacks):
        print("=" * 50)
        spiketrain = data["original_spiketrains"][str(i)]
        attacked_spk = data["attacked_spiketrains"][str(i)]
        attacked_spk_random = data["attacked_spiketrains_random"][str(i)]
        ground_truth = data["ground_truth"][i]
        assert ground_truth == data["sinabs_label"][i]
        print(f"Target: {TARGET_LABEL}, ground truth: {ground_truth}")

        """On-chip Test"""
        factory = ChipFactory(DYNAPCNN_HARDWARE)
        first_layer_idx = hardware_compatible_model.chip_layers_ordering[0]
        # forward pass on the chip
        hardware_compatible_model.reset_states()
        out_label_chip = spiketrain_forward(spiketrain, factory)
        # attack (targeted) and get the attacked result for on-chip
        hardware_compatible_model.reset_states()
        out_label_attacked_targeted_chip = spiketrain_forward(attacked_spk, factory)
        # attack (randomly) and get the attacked result for on-chip
        hardware_compatible_model.reset_states()
        out_label_attacked_random_chip = spiketrain_forward(attacked_spk_random, factory)

        """Simulation Test"""
        dt = 500
        reset_states(snn)
        # raster data for sinabs
        raster = create_raster_from_xytp(spiketrain, dt=dt, bins_x=np.arange(129), bins_y=np.arange(129))
        out_sinabs = snn(torch.tensor(raster).to(DEVICE)).squeeze().sum(0)
        out_label_sim = torch.argmax(out_sinabs).item()
        # attack (targeted) and get the attacked result for simulation
        reset_states(snn)
        raster_attacked = create_raster_from_xytp(attacked_spk, dt=dt, bins_x=np.arange(129), bins_y=np.arange(129))
        out_sinabs_targeted = snn(torch.tensor(raster_attacked).to(DEVICE)).squeeze().sum(0)
        print(f"N. spikes from Sinabs: {out_sinabs_targeted.sum()}, activations for targeted attack: {out_sinabs_targeted.detach().numpy().astype(int)}")
        out_label_attacked_targeted_sim = torch.argmax(out_sinabs_targeted).item()
        # attack (randomly) and get the attacked result for simulation
        reset_states(snn)
        raster_random = create_raster_from_xytp(attacked_spk_random, dt=dt, bins_x=np.arange(129), bins_y=np.arange(129))
        out_sinabs_random = snn(torch.tensor(raster_random).to(DEVICE)).squeeze().sum(0)
        out_label_attacked_random_sim = torch.argmax(out_sinabs_random).item()

        """write results into report file"""
        report.write(f"{i},{ground_truth},"
                     f"{out_label_chip},{out_label_sim},"
                     f"{out_label_attacked_targeted_chip},{out_label_attacked_random_chip},"
                     f"{out_label_attacked_targeted_sim},{out_label_attacked_random_sim}\n")

        # command-line printing
        if out_label_sim != out_label_chip:
            print("Discrepancy between simulation and on-chip for prediction without attack.")

        # on-chip targeted attack
        if TARGET_LABEL == out_label_attacked_targeted_chip:
            print("Successful for on-chip, targeted attack!")
        else:
            print(f"Unsuccessful for on-chip, targeted attack, "
                  f"predicted {out_label_attacked_targeted_chip}")

        # on-chip randomly attack
        if TARGET_LABEL == out_label_attacked_random_chip:
            print("Successful for on-chip and randomly attack!")
        else:
            print(f"Unsuccessful for on-chip, randomly attack, "
                  f"predicted {out_label_attacked_random_chip}")

        # simulated targeted attack
        if TARGET_LABEL == out_label_attacked_targeted_sim:
            print("Successful for simulation, targeted attack!")
        else:
            print(f"Unsuccessful for simulation, targeted attack, "
                  f"predicted {out_label_attacked_targeted_sim}")

        # simulated randomly attack
        if TARGET_LABEL == out_label_attacked_random_sim:
            print("Successful for simulation, randomly attack!")
        else:
            print(f"Unsuccessful for simulation, randomly attack, "
                  f"predicted {out_label_attacked_random_sim}")

    # close handle
    io.close_device(DYNAPCNN_HARDWARE)
    data.close()
    report.close()
