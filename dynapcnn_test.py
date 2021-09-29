import torch
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

# data and networks from this library
from networks import GestureClassifierSmall

CHIP_AVAILABLE = False
DEVICE = "cpu"
DYNAPCNN_HARDWARE = "speck2b" # could be dynapcnndevkit or speck2b for example

torch.random.manual_seed(1)

events_struct = [("x", np.uint16), ("y", np.uint16), ("t", np.uint64), ("p", bool)]

USE_PATCHES = True
target_label = sys.argv[1]
n_epoch = sys.argv[2]
patch_size = sys.argv[3]
MAX = 50


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
    print("N. spikes from chip:", len(output_neuron_index))
    if len(output_neuron_index) == 0:
        # wrong prediction if there is no spike
        return -1
    return activations.argmax()


if __name__ == "__main__":
    # - Preparing the model
    gesture_classifier = GestureClassifierSmall("BPTT_small_trained_martino_200ms_2ms.pth", device=DEVICE)
    snn = gesture_classifier.model
    snn.eval()
    gesture_classifier.eval()
    # convert to chip-compatible spiking network (discretized etc.)
    input_shape = (2, 128, 128)
    hardware_compatible_model = DynapcnnCompatibleNetwork(
        snn,
        discretize=True,
        input_shape=input_shape,
    )
    gesture_classifier = gesture_classifier.to(DEVICE)

    # - Apply model to device
    if CHIP_AVAILABLE:
#         layers_ordering = [0, 1, 2, 7, 4, 5, 6, 3, 8]
        layers_ordering = [0, 1, 2, 3]
#         config = hardware_compatible_model.make_config(
#             layers_ordering, monitor_layers=[layers_ordering[-1]])
        hardware_compatible_model.to(
            device=DYNAPCNN_HARDWARE,
            chip_layers_ordering=layers_ordering,
            monitor_layers=[layers_ordering[-1]],
        )

    # Get file
    attack_file = f"./attack_patches/attacks_patches_ep{n_epoch}_lb{target_label}_num{MAX}_patchsize{patch_size}.h5" if USE_PATCHES else "attacks.h5"
    data = h5py.File(attack_file, "r")
    successful_attacks = np.where(data["attack_successful"])[0]

    # Report file
    if USE_PATCHES:
        if not os.path.exists("./attack_result_csv"): os.makedirs("./attack_result_csv")
        report = open(f"./attack_result_csv/report_ep{n_epoch}_lb{target_label}_num{MAX}_patchsize{patch_size}.csv", "w")
        success_rate_targeted = round(data["targeted_patch_successful_rate"][()], 3)
        success_rate_random = round(data["random_patch_successful_rate"][()], 3)
        report.write(f"ID,ground_truth,chip_out,chip_out_attacked,chip_out_attacked_random,"
                     f"targeted_patch_success_rate_simulation: {success_rate_targeted}, random_patch_success_rate_simulation: {success_rate_random}\n")
    else:
        report = open(f"report_0.3.csv", "w")
        report.write("ID,ground_truth,chip_out,chip_out_attacked\n")

    # - Start testing
    counter = SNNSynOpCounter(snn)
    for i in tqdm(successful_attacks):
        spiketrain = data["original_spiketrains"][str(i)]
        print(spiketrain)
        attacked_spk = data["attacked_spiketrains"][str(i)]
        if USE_PATCHES:
            attacked_spk_random = data["attacked_spiketrains_random"][str(i)]

        ground_truth = data["ground_truth"][i]
        assert ground_truth == data["sinabs_label"][i]

        if CHIP_AVAILABLE:
            # Normal spiketrain
            # resetting states
            factory = ChipFactory(DYNAPCNN_HARDWARE)
            first_layer_idx = hardware_compatible_model.chip_layers_ordering[0] 
            # forward pass on the chip
            hardware_compatible_model.reset_states()
            out_label = spiketrain_forward(spiketrain, factory)
            # Attack
            # resetting states
            hardware_compatible_model.reset_states()
            # forward pass on the chip
            out_label_attacked = spiketrain_forward(attacked_spk, factory)
            if USE_PATCHES:
                hardware_compatible_model.reset_states()
                out_label_attacked_random = spiketrain_forward(attacked_spk_random, factory)
        else:
            reset_states(snn)
            # raster data for sinabs
            raster = create_raster_from_xytp(
                spiketrain, dt=1000, bins_x=np.arange(129), bins_y=np.arange(129))
            out_sinabs = snn(torch.tensor(raster).to(DEVICE)).squeeze().sum(0)
            out_label = torch.argmax(out_sinabs).item()
            # print("N. spikes from sinabs:", out_sinabs.sum().item())
            # print("Power consumption:", counter.get_total_power_use())
            # Attack
            if USE_PATCHES:
                reset_states(snn)
                raster_random = create_raster_from_xytp(attacked_spk_random, dt=1000, bins_x=np.arange(129), bins_y=np.arange(129))
                out_sinabs_random = snn(torch.tensor(raster_random).to(DEVICE)).squeeze().sum(0)
                out_label_attacked_random = torch.argmax(out_sinabs_random).item()

            reset_states(snn)
            raster_attacked = create_raster_from_xytp(
                attacked_spk, dt=1000, bins_x=np.arange(129), bins_y=np.arange(129))
            out_sinabs_attacked = snn(torch.tensor(raster_attacked).to(DEVICE)).squeeze().sum(0)
            out_label_attacked = torch.argmax(out_sinabs_attacked).item()



        # print("Ground truth:", data["ground_truth"][i],
        #       "-- chip (if available):", out_label,
        #       "-- chip under attack:", out_label_attacked)
        if USE_PATCHES:
            report.write(f"{i},{ground_truth},{out_label},{out_label_attacked},{out_label_attacked_random}\n")
        else:
            report.write(f"{i},{ground_truth},{out_label},{out_label_attacked}\n")

        if ground_truth != out_label:
            print("Discrepancy between sinabs and chip.")
        if not USE_PATCHES:
            if out_label != out_label_attacked:
                print("Successful attack!")
            else:
                print("Attack converged but unsuccessful on chip.")
        else:
            if out_label_attacked == target_label:
                print("Successful attack!")
            else:
                print("Unsuccessful, predicted %s and was %s" % (str(out_label_attacked), str(out_label)))

    if CHIP_AVAILABLE:
        io.close_device(DYNAPCNN_HARDWARE)
    data.close()
    report.close()
