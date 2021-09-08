import torch
import numpy as np
import h5py

# software to interact with dynapcnn and data
from aermanager.preprocess import create_raster_from_xytp

# data and networks from this library
from dataloader_IBMGestures import IBMGesturesDataLoader
from sparsefool import sparsefool
from networks import GestureClassifierSmall

DEVICE = torch.device("cuda")
torch.random.manual_seed(1)

events_struct = [("x", np.uint16), ("y", np.uint16), ("t", np.uint64), ("p", bool)]


def reset_states(net):
    for m in net.modules():
        if hasattr(m, "reset_states"):
            m.reset_states()


def raster_to_spiketrain(added_to_raster, dt, start_t):
    # we loop over the values in a bin, and add as many spikes as necessary
    spikelist = torch.ones((4, int(added_to_raster.sum().item())), dtype=int)
    idx = 0
    for n in range(1, int(added_to_raster.max().item()) + 1):
        where = torch.where(added_to_raster == n)
        n_added = len(where[0])
        if n_added > 0:
            spikelist[:, idx:idx+n_added*n] = torch.stack(where).reshape((4, -1)).tile((1, n))
            idx += n_added * n
    assert idx == len(spikelist[0])

    # we adapt and add them to the spiketrain
    t, p, y, x = spikelist
    t = t.cpu().numpy()
    t_microsec = start_t + (dt * (t + 0.5)).astype(int)
    added_spikes = np.empty(len(t_microsec), dtype=events_struct)
    added_spikes["t"] = t_microsec
    added_spikes["x"] = x.cpu()
    added_spikes["y"] = y.cpu()
    added_spikes["p"] = p.cpu()
    return added_spikes


def attack_on_spiketrain(net, spiketrain):
    dt = 2000
    # first, we need to rasterize the spiketrain
    raster = create_raster_from_xytp(
        spiketrain, dt=dt, bins_x=np.arange(129), bins_y=np.arange(129))
    # note that to do this we are forced to suppress many spikes
    print("Spikes:", raster.sum())
    raster = torch.tensor(raster).to(DEVICE)
    # raster = torch.clamp(torch.tensor(raster), 0., 1.).to(DEVICE)
    # print("Spikes after binarization:", raster.sum())

    # performing the actual attack
    print("Max per bin:", raster.max())
    return_dict_sparse_fool = sparsefool(
        x_0=raster,
        net=net,
        max_hamming_distance=1e6,
        lb=0.0,
        ub=raster.max(),
        lambda_=2.,
        max_iter=15,
        epsilon=0.02,
        overshoot=0.02,
        step_size=0.3,
        max_iter_deep_fool=50,
        device=DEVICE,
        verbose=True,
    )
    if not return_dict_sparse_fool["success"]:
        return None

    attacked_raster = return_dict_sparse_fool["X_adv"]
    # now we only look at where spikes were ADDED (heuristically!)
    diff = attacked_raster - raster
    added_to_raster = torch.clamp(diff, min=0)
    removed_from_raster = -torch.clamp(diff, max=0)
    print("Spikes removed:", removed_from_raster.sum().item())
    print("Spikes added:", added_to_raster.sum().item())
    maxval = int(attacked_raster.max().item())
    print("Max per bin after attack:", maxval)

    added_spikes = raster_to_spiketrain(added_to_raster, dt=dt, start_t=spiketrain['t'][0])
    # add to spiketrain structured array
    new_spiketrain = np.concatenate((spiketrain, added_spikes))
    # re-sort and return
    new_spiketrain.sort(order="t")
    return new_spiketrain


if __name__ == "__main__":
    MAX = 1000
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
    gesture_classifier = gesture_classifier.to(DEVICE)

    # Prepare file for saving
    savef = h5py.File("./attacks_0.3.h5", "w")
    saved_orig = savef.create_group("original_spiketrains")
    saved_attk = savef.create_group("attacked_spiketrains")
    ground_truth = savef.create_dataset("ground_truth", (MAX, ), dtype="<u1")
    sinabs_label = savef.create_dataset("sinabs_label", (MAX, ), dtype="<u1")
    attack_successful = savef.create_dataset("attack_successful", (MAX, ), dtype="?")
    n_spikes_orig = savef.create_dataset("n_spikes_orig", (MAX,), dtype="<i4")
    n_spikes_attk = savef.create_dataset("n_spikes_attk", (MAX,), dtype="<i4")

    # - Start testing
    correct = 0
    attack_reports_success = 0

    for i, (spiketrain, label) in enumerate(data_loader_test):
        if i >= MAX:
            break

        saved_orig.create_dataset(str(i), data=spiketrain, compression="lzf")
        ground_truth[i] = label
        n_spikes_orig[i] = len(spiketrain)

        reset_states(snn)
        # raster data for sinabs
        raster = create_raster_from_xytp(
            spiketrain, dt=1000, bins_x=np.arange(129), bins_y=np.arange(129))
        out_sinabs = snn(torch.tensor(raster).to(DEVICE)).squeeze().sum(0)
        out_label = torch.argmax(out_sinabs).item()

        sinabs_label[i] = out_label
        attack_successful[i] = False  # default, will change this if successful
        n_spikes_attk[i] = -1  # same

        if out_label == label:
            correct += 1
            attacked_spk = attack_on_spiketrain(gesture_classifier, spiketrain)

            if attacked_spk is None:
                continue
            else:
                attack_reports_success += 1
                attack_successful[i] = True
                saved_attk.create_dataset(str(i), data=attacked_spk, compression="lzf")
                n_spikes_attk[i] = len(attacked_spk)

    print("Accuracy in simulation:", correct / MAX)
    print("Rate of attacks that converged:", attack_reports_success / correct)
    print("\n")
    savef.close()
