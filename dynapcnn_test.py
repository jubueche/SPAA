import torch
import numpy as np
from tqdm import tqdm
import h5py

# software to interact with dynapcnn and data
from sinabs.backend.dynapcnn import io
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork
from sinabs.synopcounter import SNNSynOpCounter
from aermanager.preprocess import create_raster_from_xytp

# data and networks from this library
from networks import GestureClassifierSmall

CHIP_AVAILABLE = False
MAX = 50
# DEVICE = torch.device("cuda")
DEVICE = torch.device("cpu")
torch.random.manual_seed(1)

events_struct = [("x", np.uint16), ("y", np.uint16), ("t", np.uint64), ("p", bool)]

USE_PATCHES = True
target_label = 8

def reset_states(net):
    for m in net.modules():
        if hasattr(m, "reset_states"):
            m.reset_states()


def spiketrain_forward(spiketrain):
    input_events = io.xytp_to_events(
        spiketrain, layer=layers_ordering[0], device="dynapcnndevkit:0")
    evs_out = hardware_compatible_model(input_events)
    evs_out = io.events_to_xytp(evs_out, layer=layers_ordering[-1])
    print("N. spikes from chip:", len(evs_out))

    if len(evs_out) == 0:
        return 0  # wrong but actually imitates the behaviour of torch.
    labels, counts = np.unique(evs_out["channel"], return_counts=True)
    most_active_neuron = labels[np.argmax(counts)]
    return most_active_neuron


if __name__ == "__main__":
    # - Preparing the model
    gesture_classifier = GestureClassifierSmall("BPTT_small_trained_martino_200ms_2ms.pth")
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
        layers_ordering = [0, 1, 2, 7, 4, 5, 6, 3, 8]
        # layers_ordering = [0, 1, 2, 3]
        config = hardware_compatible_model.make_config(
            layers_ordering, monitor_layers=[layers_ordering[-1]])
        hardware_compatible_model.to(
            device="dynapcnndevkit:0",
            chip_layers_ordering=layers_ordering,
            monitor_layers=[layers_ordering[-1]],
        )

    # Get file
    attack_file = "attacks_patches.h5" if USE_PATCHES else "attacks.h5"
    data = h5py.File(attack_file, "r")
    successful_attacks = np.where(data["attack_successful"])[0]

    # Report file
    report = open("report.csv", "w")
    report.write("ID,ground_truth,chip_out,chip_out_attacked\n")

    # - Start testing
    counter = SNNSynOpCounter(snn)
    # for i in tqdm(successful_attacks):
    for i in successful_attacks:
        if i >= MAX:
            break
        spiketrain = data["original_spiketrains"][str(i)]
        attacked_spk = data["attacked_spiketrains"][str(i)]
        ground_truth = data["ground_truth"][i]
        assert ground_truth == data["sinabs_label"][i]

        if CHIP_AVAILABLE:
            # Normal spiketrain
            # resetting states
            hardware_compatible_model.samna_device.get_model().apply_configuration(config)
            # forward pass on the chip
            out_label = spiketrain_forward(spiketrain)
            # Attack
            # resetting states
            hardware_compatible_model.samna_device.get_model().apply_configuration(config)
            # forward pass on the chip
            out_label_attacked = spiketrain_forward(attacked_spk)
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
            reset_states(snn)
            raster_attacked = create_raster_from_xytp(
                attacked_spk, dt=1000, bins_x=np.arange(129), bins_y=np.arange(129))
            out_sinabs_attacked = snn(torch.tensor(raster_attacked).to(DEVICE)).squeeze().sum(0)
            out_label_attacked = torch.argmax(out_sinabs_attacked).item()

        # print("Ground truth:", data["ground_truth"][i],
        #       "-- chip (if available):", out_label,
        #       "-- chip under attack:", out_label_attacked)
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
                print("Unsuccessful, predicted %s and was %s" % (str(out_label_attacked),str(out_label)))

    if CHIP_AVAILABLE:
        io.close_device("dynapcnndevkit")
    data.close()
    report.close()
