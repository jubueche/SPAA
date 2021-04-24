"""
Train non-spiking model for BMNIST
"""
from experiment_utils import *
import os.path as path
from architectures import BMNIST as arch
from networks import train_ann_binary_mnist

if __name__ == "__main__":
    FLAGS = arch.get_flags()
    base_path = path.dirname(path.abspath(__file__))
    model_save_path = path.join(base_path, f"Resources/Models/{FLAGS.session_id}_model.pt")
    ann = train_ann_binary_mnist()
    torch.save(ann.state_dict(), model_save_path)
    print("Saved model in", model_save_path)