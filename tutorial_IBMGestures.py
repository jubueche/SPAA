# TODO import data loader gestures
import torch
from networks import load_gestures_snn
from sparsefool import sparsefool
from utils import get_prediction, plot_attacked_prob
from dataloader_IBMGestures import get_data_loader

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# - Load the spiking CNN for IBM gestures dataset
snn = load_gestures_snn()

data_loader_test = get_data_loader(
    dset="test",
    shuffle=False,
    num_workers=4,
    batch_size=1)


num_samples = 0
correct_samples = 0
c = 0
batch_size = 1
for sample, target in data_loader_test:

    snn.reset_states()
    sample = sample.float()

    sample = sample.to(device)
    target = target.long().to(device)

    out = snn.forward(sample)
    _, predict = torch.max(out, 1)


    correct_samples += (predict == target).sum().item()

    num_samples += batch_size

    print("--------------------------------------------------------------------------------------------")
    print("current testing accuracy", 100 * correct_samples / num_samples, '    |      ', "current progress:",
          num_samples)
    print("--------------------------------------------------------------------------------------------")
    print(c)
    c +=1