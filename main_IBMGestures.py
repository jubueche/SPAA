"""
Train spiking model for IBMGestures
"""
from experiment_utils import *
import os.path as path
from architectures import IBMGestures as arch
from networks import load_gestures_snn

if __name__ == "__main__":
    FLAGS = arch.get_flags()
    base_path = path.dirname(path.abspath(__file__))
    model_save_path = path.join(base_path, f"Resources/Models/{FLAGS.session_id}_model.pt")
    snn = load_gestures_snn()
    torch.save(snn.state_dict(), model_save_path)
    print("Saved model in", model_save_path)