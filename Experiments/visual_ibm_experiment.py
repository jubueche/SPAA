from architectures import IBMGestures
from dataloader_IBMGestures import IBMGesturesDataLoader
from sparsefool import sparsefool, frame_based_sparsefool
from datajuicer import run, split, configure, query, run, reduce_keys
from experiment_utils import *
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['xtick.bottom'] = False
mpl.rcParams['ytick.left'] = False
mpl.rcParams['xtick.labelbottom'] = False
mpl.rcParams['ytick.labelleft'] = False
import matplotlib.pyplot as plt

classes_labels = [
    "hand clap",
    "right hand wave",
    "left hand wave",
    "right arm clockwise",
    "right arm counterclockwise",
    "left arm clockwise",
    "left arm counterclockwise",
    "arm roll",
    "air drums",
    "air guitar",
    "other gestures",
]

label_dic = {
    "hand clap":"Hand Clap",
    "right hand wave":"RH Wave",
    "left hand wave": "LH Wave",
    "right arm clockwise": "RH Clockwise",
    "right arm counterclockwise": "RH Counter Clockw.",
    "left arm clockwise": "LH Clockwise",
    "left arm counterclockwise": "LH Counter Clockw.",
    "arm roll": "Arm Roll",
    "air drums": "Air Drums",
    "air guitar": "Air Guitar",
    "other gestures": "Other",
}

def generate_sample(attack_fn, data_loader, source_label, target_label, num):
    results = []
    got_set = set()
    if isinstance(source_label,str):
        source_label = [source_label]
    if isinstance(target_label,str):
        target_label = [target_label]
    if source_label is None:
        source_label = classes_labels
    if target_label is None:
        target_label = classes_labels

    for idx, (X0, target) in enumerate(data_loader):
        if len(got_set) == num:
            break
        X0 = X0.float()
        X0 = X0.to(device)
        X0 = torch.clamp(X0, 0.0, 1.0)
        target = target.long().to(device)
        if classes_labels[target] in source_label and not (classes_labels[target] in got_set):
            return_dict = attack_fn(X0)
            return_dict["X0"] = X0
            if classes_labels[return_dict["predicted_attacked"]] in target_label and return_dict["predicted"]==target:
                results.append(return_dict)
                got_set.add(classes_labels[target])
    
    return results

class visual_ibm_experiment:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        return grid

    @staticmethod
    def visualize():
        grid = visual_ibm_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        net = grid[0]["snn"]

        ibm_gesture_dataloader = IBMGesturesDataLoader()

        data_loader_test = ibm_gesture_dataloader.get_data_loader(dset="test",
                                                                shuffle=False,
                                                                num_workers=4,
                                                                batch_size=1)

        max_hamming_distance = 1000
        lambda_ = 3.0
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.1
        max_iter_deep_fool = 50
        n_attack_frames = 1

        def attack_fn(X0):
            d = sparsefool(
                x_0=X0,
                net=net,
                max_hamming_distance=max_hamming_distance,
                lambda_=lambda_,
                max_iter=max_iter,
                epsilon=epsilon,
                overshoot=overshoot,
                step_size=step_size,
                max_iter_deep_fool=max_iter_deep_fool,
                device=device,
                early_stopping=True,
                boost=False,
                verbose=True
            )
            return d

        source_labels = ["right hand wave","air guitar","hand clap"]
        target_labels = None

        samples = generate_sample(
            attack_fn=attack_fn,
            data_loader=data_loader_test,
            source_label=source_labels,
            target_label=target_labels,
            num=len(source_labels))

        def plot(args):
            axes, sample, idx = args
            dt = 200. / len(axes)
            time_labels = ["%.1f ms" % (dt*i) for i in range(len(axes))]
            X0 = sample["X0"].squeeze().sum(dim=1)
            X_adv = sample["X_adv"].squeeze().sum(dim=1)
            X_diff = torch.abs(X0-X_adv) 
            num_frames_available = len(axes)
            num_frames = X0.shape[0]
            t = int(num_frames / num_frames_available)
            frames_X0 = [X0[i*t:(i+1)*t].sum(dim=0).cpu().numpy()[::-1] for i in range(len(axes))]
            frames_X_diff = [X_diff[i*t:(i+1)*t].sum(dim=0).cpu().numpy()[::-1] for i in range(len(axes))]
            for ax_idx,(frame,frame_diff) in enumerate(zip(frames_X0,frames_X_diff)):
                axes[ax_idx].pcolormesh(frame, vmin=0, vmax=2, cmap=plt.cm.Blues)
                axes[ax_idx].pcolormesh(np.ma.masked_array(frame_diff,frame_diff==0.), vmin=0, vmax=2, cmap=plt.cm.Reds)
                if idx==0:
                    axes[ax_idx].text(0,115,time_labels[ax_idx])
                if ax_idx==0:
                    axes[ax_idx].set_ylabel(label_dic[classes_labels[sample["predicted"]]] + r"$\rightarrow$" +
                                                "\n" + label_dic[classes_labels[sample["predicted_attacked"]]])

        # - Create gridspec
        N_rows = 3
        N_cols = 5
        num_per_sample = int(N_rows*N_cols / len(samples))
        fig = plt.figure(constrained_layout=True, figsize=(10,6))
        spec = mpl.gridspec.GridSpec(ncols=N_cols, nrows=N_rows, figure=fig)
        axes = [fig.add_subplot(spec[i,j]) for i in range(N_rows) for j in range(N_cols)]
        sub_axes_samples = [(axes[i*num_per_sample:(i+1)*num_per_sample],samples[i],i) for i in range(len(samples))]
        list(map(plot, sub_axes_samples))

        plt.savefig("Resources/Figures/samples_ibm_gestures.pdf")
        plt.show()

        