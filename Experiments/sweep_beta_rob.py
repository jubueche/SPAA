from architectures import IBMGestures
from datajuicer import run, configure
from datajuicer.utils import query, split
from experiment_utils import *
from datajuicer.visualizers import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from Experiments.visual_ibm_experiment import generate_sample, class_labels, plot

beta_robustness = [0.0,0.01,0.05,0.1,0.2]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_test_acc(data_loader, model, limit):
    model.eval()
    correct = 0
    num = 0
    for idx, (X0, target) in enumerate(data_loader):
        print("%d/%d" % (idx,len(data_loader)))
        if limit != -1 and num > limit:
            break
        X0 = X0.float()
        X0 = X0.to(device)
        X0 = torch.clamp(X0, 0.0, 1.0)
        target = target.long().to(device)
        model.reset_states()
        out = model.forward(X0)
        _, predict = torch.max(out, 1)
        correct += torch.sum((predict == target).float())
        num += X0.shape[0]
    ta = float(correct / num)
    model.train()
    print("Test acc is %.4f" % ta)
    return ta

class sweep_beta_rob:
    @staticmethod
    def train_grid():
        grid = [IBMGestures.make()]
        grid = configure(grid, {"batch_size": 16, "boundary_loss": "trades", "epochs":2})
        grid = split(grid, "beta_robustness", beta_robustness)
        return grid

    @staticmethod
    def visualize():
        grid = sweep_beta_rob.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        max_hamming_distance = int(1e6)
        verbose = True
        limit = 1000
        lambda_ = 3.0
        max_iter = 20
        epsilon = 0.0
        overshoot = 0.02
        step_size = 0.1
        max_iter_deep_fool = 50

        grid = configure(
            grid,
            {
                "max_hamming_distance": max_hamming_distance,
                "lambda_": lambda_,
                "verbose": verbose,
                "limit": limit,
                "max_iter":max_iter,
                "epsilon":epsilon,
                "overshoot":overshoot,
                "step_size":step_size,
                "max_iter_deep_fool":max_iter_deep_fool
            },
        )

        def calc_data(grid):
            for g  in grid:
                sf = g["sparse_fool"]
                test_acc = np.mean(np.array(sf["predicted"] == sf["targets"], int))
                consider_index = np.array(sf["predicted"] == sf["targets"], bool)
                success_rate = np.sum(sf["success"][consider_index]) / np.sum(np.array(consider_index,int))
                median_L0 = np.median(sf["L0"][consider_index])
                g["test_acc"] = test_acc
                g["success_rate"] = success_rate
                g["median_L0"] = median_L0
            return grid

        def get_dataloader_test(sess_id, dt, batch_size=1):
            ibm_gesture_dataloader = IBMGesturesDataLoader(slicing_overlap=40000, caching_path='/home/jbu/SPAA/cache/'+str(sess_id))
            data_loader_test = ibm_gesture_dataloader.get_data_loader(
                "test", shuffle=False, num_workers=4, batch_size=batch_size, dt=dt)
            return data_loader_test

        def calc_test_acc(grid, limit):
            sess_id = grid[0]["IBMGestures_session_id"]
            data_loader_test = get_dataloader_test(sess_id=sess_id, dt=grid[0]["dt"], batch_size=10)
            for g in grid:
                test_acc = get_test_acc(data_loader_test, model=g["snn"], limit=limit)
                g["test_acc"] = test_acc
            return grid

        grid = run(grid, sparse_fool_on_test_set, n_threads=1, run_mode="normal", store_key="sparse_fool")(
            "{*}",
            "{max_hamming_distance}",
            "{lambda_}",
            "{max_iter}",
            "{epsilon}",
            "{overshoot}",
            "{step_size}",
            "{max_iter_deep_fool}",
            "{verbose}",
            "{limit}",
            True,  # - Use SNN
        )

        grid = calc_data(grid)
        grid = calc_test_acc(grid, limit=-1)

        for g in grid:
            print("beta rob %.3f success rate %.4f test_acc %.4f median L0 %.4f"
                % (g["beta_robustness"],g["success_rate"],g["test_acc"],g["median_L0"]))

        # - Generate samples
        grid_vis = [g for g in grid if g["beta_robustness"] in [0.0,0.05]]
        data_loader_test = get_dataloader_test(sess_id=grid[0]["IBMGestures_session_id"], dt=grid[0]["dt"])

        attack_fns = [lambda X0 : sparsefool(
            x_0=X0,
            net=g["snn"],
            max_hamming_distance=max_hamming_distance,
            lambda_=lambda_,
            max_iter=max_iter,
            epsilon=epsilon,
            overshoot=overshoot,
            step_size=step_size,
            max_iter_deep_fool=max_iter_deep_fool,
            device=device,
            verbose=True
        ) for g in grid_vis]

        source_labels = ["RH Wave"]
        target_labels = None

        samples = [generate_sample(
            attack_fn=attack_fns[i],
            data_loader=data_loader_test,
            source_label=source_labels,
            target_label=target_labels,
            num=len(source_labels),
            class_labels=class_labels
        ) for i in range(len(grid_vis))]

        # - Create gridspec
        N_rows = 2
        N_cols = 5
        sample_len_ms = 200.
        num_per_sample = int((N_rows * N_cols) / len(samples))
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
            samples[i][0],
            i,
            class_labels,
            sample_len_ms,
            i == N_rows - 1
        ) for i in range(len(samples))]
        list(map(plot, sub_axes_samples))

        axes[2].set_title("Standard training")
        axes[7].set_title("Adversarial training")    

        plt.savefig("Resources/Figures/samples_ibm_gestures_beta_rob.pdf", bbox_inches='tight')