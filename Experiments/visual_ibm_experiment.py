from architectures import IBMGestures
from dataloader_IBMGestures import IBMGesturesDataLoader
from sparsefool import sparsefool
from datajuicer import run
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from experiment_utils import device


class_labels = [
    "Hand Clap",
    "RH Wave",
    "LH Wave",
    "RH Clockwise",
    "RH Counter Clockw.",
    "LH Clockwise",
    "LH Counter Clockw.",
    "Arm Roll",
    "Air Drums",
    "Air Guitar",
    "Other",
]


def generate_sample(attack_fn, data_loader, source_label, target_label, num, class_labels):
    results = []
    got_set = set()
    if isinstance(source_label, str):
        source_label = [source_label]
    if isinstance(target_label, str):
        target_label = [target_label]
    if source_label is None:
        source_label = class_labels
    if target_label is None:
        target_label = class_labels

    for idx, (X0, target) in enumerate(data_loader):
        if len(got_set) == num:
            break  # we already have enough samples, end and return
        if not (class_labels[target] in source_label) or class_labels[target] in got_set:
            continue  # the sample true class is not the one we want or we have it already

        X0 = X0.float()
        X0 = X0.to(device)
        X0 = torch.clamp(X0, 0.0, 1.0)
        target = target.long().to(device)

        return_dict = attack_fn(X0)
        # return_dict = {}
        # return_dict["X_adv"] = X0
        # return_dict["predicted"] = target
        # return_dict["predicted_attacked"] = target

        return_dict["X0"] = X0
        # if class_labels[return_dict["predicted_attacked"]] in target_label: # del
        if (class_labels[return_dict["predicted_attacked"]] in target_label
                and return_dict["predicted"] == target
                and return_dict["predicted"] != return_dict["predicted_attacked"]):
            # we accept the result only if: it's the target we want,
            # and the net prediction was originally correct,
            # and the attack was successful.
            results.append(return_dict)
            got_set.add(class_labels[target])

    return results


colors = [(1., 0., 0., 0.0), (1., 0., 0., 1.0)]
nodes = [0, 1]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "transparent_red", list(zip(nodes, colors)))


def plot(args):
    axes, sample, idx, class_labels, sample_len_ms, is_last = args
    dt = sample_len_ms / len(axes)
    time_labels = ["%.1f ms" % (dt*i) for i in range(len(axes))]
    X0 = sample["X0"].squeeze().sum(dim=1)
    X_adv = sample["X_adv"].squeeze().sum(dim=1)
    X_diff = torch.abs(X0-X_adv)
    num_frames_available = len(axes)
    num_frames = X0.shape[0]
    t = int(num_frames / num_frames_available)
    frames_X0 = [X0[i*t:(i+1)*t].sum(dim=0).cpu().numpy()[::-1] for i in range(len(axes))]
    frames_X_diff = [X_diff[i*t:(i+1)*t].sum(dim=0).cpu().numpy()[::-1] for i in range(len(axes))]
    vmax = max([frame.max() for frame in frames_X0])  # sorry
    for ax_idx, (frame, frame_diff) in enumerate(zip(frames_X0, frames_X_diff)):
        axes[ax_idx].pcolormesh(frame, vmin=0, vmax=2, cmap=plt.cm.gray_r)
        axes[ax_idx].pcolormesh(frame_diff, vmin=0, vmax=1, cmap=cmap)
        if is_last:
            axes[ax_idx].text(1, 1, time_labels[ax_idx])
        if ax_idx == 0:
            axes[ax_idx].set_ylabel(class_labels[sample["predicted"]] + r"$\rightarrow$" +
                                    "\n" + class_labels[sample["predicted_attacked"]])


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

        max_hamming_distance = int(1e6)
        lambda_ = 1.0
        max_iter = 5
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.1
        max_iter_deep_fool = 50

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
                verbose=True
            )
            return d

        source_labels = ["RH Wave", "Air Guitar"]
        target_labels = None

        samples = generate_sample(
            attack_fn=attack_fn,
            data_loader=data_loader_test,
            source_label=source_labels,
            target_label=target_labels,
            num=len(source_labels),
            class_labels=class_labels
        )

        # - Create gridspec
        N_rows = 2
        N_cols = 5
        sample_len_ms = 200.
        num_per_sample = int(N_rows*N_cols / len(samples))
        fig = plt.figure(constrained_layout=True, figsize=(12, 4.7))
        spec = mpl.gridspec.GridSpec(ncols=N_cols, nrows=N_rows, figure=fig)
        axes = [fig.add_subplot(spec[i, j]) for i in range(N_rows) for j in range(N_cols)]

        for ax in axes:
            ax.set_aspect("equal")
            # ax.spines['right'].set_visible(False)
            # ax.spines['top'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           top=False,
                           labelbottom=False,
                           right=False,
                           left=False,
                           labelleft=False)

        sub_axes_samples = [(
            axes[i*num_per_sample:(i+1)*num_per_sample],
            samples[i],
            i,
            class_labels,
            sample_len_ms,
            i == N_rows - 1
        ) for i in range(len(samples))]
        list(map(plot, sub_axes_samples))

        plt.savefig("Resources/Figures/samples_ibm_gestures.pdf", bbox_inches='tight')
        plt.savefig("Resources/Figures/samples_ibm_gestures.png", bbox_inches='tight')

        # plt.show(block=False)
