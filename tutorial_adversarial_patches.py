"""
Tutorial for showcasing generation of adversarial patches
"""
import torch
from dataloader_IBMGestures import IBMGesturesDataLoader
from adversarial_patch import adversarial_patch
from architectures import IBMGestures
from datajuicer import run
import numpy as np

# class_labels = [
#     "Hand Clap",
#     "RH Wave",
#     "LH Wave",
#     "RH Clockwise",
#     "RH Counter Clockw.",
#     "LH Clockwise",
#     "LH Counter Clockw.",
#     "Arm Roll",
#     "Air Drums",
#     "Air Guitar",
#     "Other",
# ]

# - Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":

    ibm_gesture_dataloader = IBMGesturesDataLoader()

    data_loader_test = ibm_gesture_dataloader.get_data_loader(
        dset="test",
        shuffle=True,
        num_workers=4,
        batch_size=1)  # - Can vary
    
    data_loader_train = ibm_gesture_dataloader.get_data_loader(
        dset="train",
        shuffle=True,
        num_workers=4,
        batch_size=1)

    grid = [IBMGestures.make()]
    grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

    # - Load the spiking CNN for IBM gestures dataset
    snn = grid[0]['snn']

    # - Hyperparams for adversarial patch
    n_epochs = 5
    patch_type = 'circle'
    input_shape = (20,2,128,128)
    patch_size = 0.05
    target_label = 10
    max_iter = 20 # - Number of samples per epoch
    eval_after = -1 # - Evaluate after X samples
    max_iter_test = 100
    label_conf = 0.75
    max_count = 300

    return_dict_adv_patch = adversarial_patch(
        net=snn,
        train_data_loader=data_loader_train,
        test_data_loader=data_loader_test,
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
        device=device
    )
    pass