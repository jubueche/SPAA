import torch
import numpy as np

# software to interact with dynapcnn and data
from sinabs.backend.dynapcnn import io
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork
from sinabs.synopcounter import SNNSynOpCounter
from aermanager.preprocess import create_raster_from_xytp

# data and networks from this library
from dataloader_IBMGestures import IBMGesturesDataLoader
from sparsefool import sparsefool
from networks import GestureClassifierSmall

CHIP_AVAILABLE = False
DEVICE = torch.device("cuda")
torch.random.manual_seed(1)

events_struct = [("x", np.uint16), ("y", np.uint16), ("t", np.uint64), ("p", bool)]


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


def attack_on_spiketrain(net, spiketrain):
    dt = 2000
    # first, we need to rasterize the spiketrain
    raster = create_raster_from_xytp(
        spiketrain, dt=dt, bins_x=np.arange(129), bins_y=np.arange(129))
    # note that to do this we are forced to suppress many spikes
    print("Spikes before binarization:", raster.sum())
    raster = torch.clamp(torch.tensor(raster), 0., 1.).to(DEVICE)
    print("Spikes after binarization:", raster.sum())

    # performing the actual attack
    return_dict_sparse_fool = sparsefool(
        x_0=raster,
        net=net,
        max_hamming_distance=1e6,
        lb=0.0,
        ub=1.0,
        lambda_=2.,
        max_iter=10,
        epsilon=0.02,
        overshoot=0.02,
        step_size=1.0,
        max_iter_deep_fool=50,
        device=DEVICE,
        verbose=True,
    )
    if not return_dict_sparse_fool["success"]:
        return None

    attacked_raster = return_dict_sparse_fool["X_adv"]
    # now we only look at where spikes were ADDED (heuristically!)
    added_to_raster = attacked_raster > raster
    print("Spikes removed:", (attacked_raster < raster).sum().item())
    t, p, x, y = torch.where(added_to_raster)

    # we adapt and add them to the spiketrain
    t = t.cpu().numpy()
    t_microsec = spiketrain['t'][0] + (dt * (t + 0.5)).astype(int)
    added_spikes = np.empty(len(t_microsec), dtype=events_struct)
    added_spikes["t"] = t_microsec
    added_spikes["x"] = x.cpu()
    added_spikes["y"] = y.cpu()
    added_spikes["p"] = p.cpu()
    # add to spiketrain structured array
    new_spiketrain = np.concatenate((spiketrain, added_spikes))
    # re-sort and return
    new_spiketrain.sort(order="t")
    assert np.all(new_spiketrain["t"][1:] >= new_spiketrain["t"][:-1])
    return new_spiketrain


# - Dataloader of spiketrains (not rasters!)
data_loader_test = IBMGesturesDataLoader().get_spiketrain_dataset(
    dset="test",
    shuffle=True,
    num_workers=4,
)  # - Can vary

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


# - Start testing
correct = 0
attack_reports_success = 0
success = 0
counter = SNNSynOpCounter(snn)
for i, (spiketrain, label) in enumerate(data_loader_test):
    if CHIP_AVAILABLE:
        # resetting states
        hardware_compatible_model.samna_device.get_model().apply_configuration(config)
        # forward pass on the chip
        out_label = spiketrain_forward(spiketrain)
    else:
        reset_states(snn)
        # raster data for sinabs
        raster = create_raster_from_xytp(
            spiketrain, dt=1000, bins_x=np.arange(129), bins_y=np.arange(129))
        out_sinabs = snn(torch.tensor(raster).to(DEVICE)).squeeze().sum(0)
        out_label = torch.argmax(out_sinabs).item()
        print("N. spikes from sinabs:", out_sinabs.sum().item())
        print("Power consumption:", counter.get_total_power_use())

    print("Ground truth:", label, "model:", out_label)

    if out_label == label:
        correct += 1
        attacked_spk = attack_on_spiketrain(gesture_classifier, spiketrain)

        if attacked_spk is None:
            continue
        else:
            attack_reports_success += 1

        if CHIP_AVAILABLE:
            # resetting states
            hardware_compatible_model.samna_device.get_model().apply_configuration(config)
            # forward pass on the chip
            out_label_attacked = spiketrain_forward(attacked_spk)
        else:
            reset_states(snn)
            raster_attacked = create_raster_from_xytp(
                attacked_spk, dt=1000, bins_x=np.arange(129), bins_y=np.arange(129))
            out_sinabs_attacked = snn(torch.tensor(raster_attacked).to(DEVICE)).squeeze().sum(0)
            out_label_attacked = torch.argmax(out_sinabs_attacked).item()

        print(f"Attack; successful? {out_label_attacked != out_label} "
              f"(new label: {out_label_attacked})")
        if out_label_attacked != out_label:
            success += 1

    if i > 100:
        break


if CHIP_AVAILABLE:
    print("Accuracy on chip:", correct / (i + 1))
    io.close_device("dynapcnndevkit")
else:
    print("Accuracy in simulation:", correct / (i + 1))

print("Rate of attacks that converged:", attack_reports_success / correct)
print("Success rate attacks (verified on chip or simulation):", success / correct)
print("\n")
