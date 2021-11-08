import torch
import numpy as np
import sys
import h5py
import os
from sys import platform


# software to interact with dynapcnn and data
from aermanager.preprocess import create_raster_from_xytp

# data and networks from this library
from dataloader_IBMGestures import IBMGesturesDataLoader
from adversarial_patch import adversarial_patch, transform_circle
from networks import AbstractGestureClassifier, GestureClassifierSmall
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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


def attack_on_spiketrain(patch, spiketrain, dt):
    # first, we need to rasterize the spiketrain
    raster = create_raster_from_xytp(
        spiketrain, dt=dt, bins_x=np.arange(129), bins_y=np.arange(129))
    # note that to do this we are forced to suppress many spikes
    print("Spikes:", raster.sum())
    raster = torch.tensor(raster).to(DEVICE)
    max_num_spikes = raster.max()
    # Apply the patch
    # Transform patch randomly
    patch = transform_circle(patch, target_label, device=DEVICE)

    # - Create adversarial example
    if patch['patch_values'].shape != raster.shape:
        raster = torch.cat((raster, torch.zeros((patch['patch_values'].shape[0] - raster.shape[0], *raster.shape[1:]), device=raster.device)))      
    attacked_raster = (1. - patch['patch_mask']) * raster + patch['patch_values']
#     attacked_raster = torch.round(torch.clamp(attacked_raster, 0., max_num_spikes))
    attacked_raster = torch.round(torch.clamp(attacked_raster, 0., 1))
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
    MAX = 50
    dt = 500
    # - Dataloader of spiketrains (not rasters!)
    data_loader_test = IBMGesturesDataLoader().get_spiketrain_dataset(
        dset="test",
        shuffle=True,
        num_workers= 4 if platform == "linux" else 0, # - Needs to be set to 0 on MacOS
    )  # - Can vary

    # - Get the dataloader that we need for the patches
    patches_data_loader_train = IBMGesturesDataLoader().get_data_loader(
        dset="train",
        batch_size=1,
        dt=dt,
        num_workers= 4 if platform == "linux" else 0
    )
    # - Get the rasterized dataloader also for the patches
    patches_data_loader_test = IBMGesturesDataLoader().get_data_loader(
        dset="test",
        batch_size=1,
        dt=dt,
        num_workers= 4 if platform == "linux" else 0
    )

    # - Preparing the model
    gesture_classifier = GestureClassifierSmall("BPTT_small_trained_200ms_2ms.pth")
    snn = gesture_classifier.model
    snn.eval()
    gesture_classifier.eval()
    gesture_classifier = gesture_classifier.to(DEVICE)

    # - Hyperparameter for the patches
    n_epochs = int(sys.argv[1])
    patch_type = 'circle'
    input_shape = (200000 // dt, 2, 128, 128)
    patch_size = float(sys.argv[3])
    target_label = int(sys.argv[2])
    max_iter = 20 # - Number of samples per epoch
    eval_after = -1 # - Evaluate after X samples, -1 means never
    max_iter_test = 100
    label_conf = 0.75
    max_count = 100

    # - Generate quantized model
    snn = snn.to("cpu")
    q_model = DynapcnnCompatibleNetwork(
        snn,
        discretize=True,
        input_shape=input_shape[1:]
    )
    q_model = q_model.to(DEVICE)
    snn = snn.to(DEVICE)
    
    weights_orig = [v.data for n,v in gesture_classifier.model.named_parameters()]
    scales = []
    for w_o in weights_orig:
        bit_precision = 8
        min_val_disc = -(2 ** (bit_precision - 1))
        max_val_disc = 2 ** (bit_precision - 1) - 1

        # Range in which values lie
        min_val_obj = torch.min(w_o)
        max_val_obj = torch.max(w_o)

        # Determine if negative or positive values are to be considered for scaling
        # Take into account that range for diescrete negative values is slightly larger than for positive
        min_max_ratio_disc = abs(min_val_disc / max_val_disc)
        if abs(min_val_obj) <= abs(max_val_obj) * min_max_ratio_disc:
            scaling = abs(max_val_disc / max_val_obj)
        else:
            scaling = abs(min_val_disc / min_val_obj)
        scales.append(scaling)
    
    # import matplotlib.pyplot as plt
    for i,(v_snn,v_q) in enumerate(zip(gesture_classifier.model.parameters(),q_model.parameters())):
        new_w = torch.reshape(v_q / scales[i], v_snn.shape)
        # plt.clf()
        # plt.plot(v_snn.detach().reshape(-1).numpy())
        # plt.plot(new_w.detach().reshape(-1).numpy())
        # plt.show()
        v_snn.data = new_w

    # - Get the adversarial patches
    return_dict_patches = adversarial_patch(
        net=gesture_classifier,
        train_data_loader=patches_data_loader_train,
        test_data_loader=patches_data_loader_test,
        patch_type=patch_type,
        patch_size=patch_size,
        input_shape=input_shape,
        n_epochs=n_epochs,
        target_label=target_label,
        max_iter=max_iter,
        max_iter_test=max_iter_test,
        label_conf=label_conf,
        max_count=max_count,
        eval_after=eval_after,
        device=DEVICE,
        lambda_=0.0005,
        eta=1.0
    )

    # Prepare file for saving
    save_dir = "./attack_patches"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    savef = h5py.File(f"{save_dir}/attacks_patches_ep{n_epochs}_lb{target_label}_num{MAX}_patchsize{str(patch_size).split('.')[1]}.h5", "w")
    saved_orig = savef.create_group("original_spiketrains")
    saved_attk = savef.create_group("attacked_spiketrains")
    saved_attk_random = savef.create_group("attacked_spiketrains_random")
    ground_truth = savef.create_dataset("ground_truth", (MAX, ), dtype="<u1")
    sinabs_label = savef.create_dataset("sinabs_label", (MAX, ), dtype="<u1")
    attack_successful = savef.create_dataset("attack_successful", (MAX, ), dtype="?")
    n_spikes_orig = savef.create_dataset("n_spikes_orig", (MAX,), dtype="<i4")
    n_spikes_attk = savef.create_dataset("n_spikes_attk", (MAX,), dtype="<i4")

    savef.create_dataset("random_patch_successful_rate", data=return_dict_patches["success_rate_random"])
    savef.create_dataset("targeted_patch_successful_rate", data=return_dict_patches["success_rate_targeted"])

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
            spiketrain, dt=dt, bins_x=np.arange(129), bins_y=np.arange(129))
        out_sinabs = snn(torch.tensor(raster).to(DEVICE)).squeeze().sum(0)
        out_label = torch.argmax(out_sinabs).item()

        sinabs_label[i] = out_label
        attack_successful[i] = False  # default, will change this if successful
        n_spikes_attk[i] = -1  # same

        if out_label == label:
            correct += 1
            print("target patch: ")
            attacked_spk = attack_on_spiketrain(return_dict_patches["patch"], spiketrain, dt)
            print("random patch: ")
            attacked_spk_random = attack_on_spiketrain(return_dict_patches["patch_random"], spiketrain, dt)

            if attacked_spk is None:
                continue
            else:
                attack_reports_success += 1
                attack_successful[i] = True
                saved_attk.create_dataset(str(i), data=attacked_spk, compression="lzf")
                saved_attk_random.create_dataset(str(i), data=attacked_spk_random, compression="lzf")
                n_spikes_attk[i] = len(attacked_spk)

    print("Accuracy in simulation:", correct / MAX)
    print("Rate of attacks that converged:", attack_reports_success / correct)
    print("\n")
    savef.close()
