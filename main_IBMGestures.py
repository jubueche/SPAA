"""
Train spiking model for IBMGestures
"""
from adversarial_patch import attack
from experiment_utils import *
import os.path as path
from architectures import IBMGestures as arch
from architectures import log
from dataloader_IBMGestures import IBMGesturesDataLoader
from networks import GestureClassifierSmall, IBMGesturesBPTT, load_gestures_snn
from loss import robust_loss
import torch
import time
import tonic

# - Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_test_acc(data_loader, model):
    model.eval()
    correct = 0
    num = 0
    for idx, (X0, target) in enumerate(data_loader):
        if num > 1000:
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
    return ta


if __name__ == "__main__":
    t0 = time.time()
    FLAGS = arch.get_flags()
    base_path = path.dirname(path.abspath(__file__))
    model_save_path = path.join(base_path, "Resources/Models/%d_model.pth" % FLAGS.session_id)

    batch_size = FLAGS.batch_size
    dt = FLAGS.dt
    torch.manual_seed(FLAGS.seed)
    epochs = FLAGS.epochs

    ibm_gesture_dataloader = IBMGesturesDataLoader()

    data_loader_train = ibm_gesture_dataloader.get_data_loader(
        "train", shuffle=True, num_workers=4, batch_size=batch_size, dt=dt)
    data_loader_test = ibm_gesture_dataloader.get_data_loader(
        "test", shuffle=True, num_workers=4, batch_size=32, dt=dt)
    data_loader_test_robustness_test = ibm_gesture_dataloader.get_data_loader(
        "test", shuffle=True, num_workers=4, batch_size=1, dt=dt)

    # - Generate transform for injecting random events
    add_noise_transform = tonic.transforms.UniformNoise(
        sensor_size=tonic.datasets.DVSGesture.sensor_size,
        n_noise_events = FLAGS.noise_n_samples
    )

    # - Generate the model
    model = IBMGesturesBPTT().to(device)

    # - Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Current test acc. is %.4f" % get_test_acc(data_loader_test, model=model))

    def get_sparsefool_robustness(model, N_samples, data_loader):
        model.eval()
        success = []; L0s = []
        for batch_idx, (X, y) in enumerate(data_loader):
            if batch_idx == N_samples:
                break
            X, y = X.to(device).float(), y.to(device).float()
            X = torch.clamp(X, min=0.0, max=1.0)
            pred_label = torch.argmax(model.forward(X).data).item()
            if pred_label != y:
                continue
            attack_dict = sparsefool(
                x_0=X,
                net=model,
                max_hamming_distance=int(1e6),
                lb=0.0,
                ub=1.0,
                lambda_=3.,
                max_iter=15,
                epsilon=0.02,
                overshoot=0.02,
                step_size=0.5,
                max_iter_deep_fool=50,
                device=device,
                verbose=True
            )
            success.append(attack_dict["success"])
            if not attack_dict["success"]:
                L0s.append(int(1e6))
            else:
                L0s.append(attack_dict["L0"])
        
        model.train()
        if success == []:
            return 0.0,0.0
        else:
            return np.mean(success), np.median(L0s)

    # - Begin the training
    for epoch in range(epochs):
        model.train()
        for batch_idx, (sample, target) in enumerate(data_loader_train):
            model.reset_states()
            sample = sample.float().to(device)
            noisy_sample = add_noise_transform(sample)
            target = target.long().to(device)
            loss = robust_loss(
                model=model,
                x_natural=noisy_sample,
                y=target,
                optimizer=optimizer,
                FLAGS=FLAGS,
                is_warmup=epoch < FLAGS.warmup
            )
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                model.reset_states()
                out = model.forward(sample)
                _, predict = torch.max(out, 1)
                t_passed = (time.time() - t0) / 3600
                b_acc = torch.mean((predict == target).float())
                print("Epoch %d Batch %d/%d Time %.3f h Loss %.3f Batch acc. %.3f" % (
                    epoch, batch_idx, data_loader_train.__len__(),
                    t_passed, float(loss), float(100 * b_acc)))
                log(FLAGS.session_id, "training_accuracy", float(b_acc))
                log(FLAGS.session_id, "loss", float(loss))

        # - End epoch
        # - Evaluate robustness
        mean_success_rate, median_L0 = get_sparsefool_robustness(
            model=model,
            N_samples=10,
            data_loader=data_loader_test_robustness_test
        )
        log(FLAGS.session_id, "mean_success_rate", mean_success_rate)
        log(FLAGS.session_id, "median_L0", median_L0)
        test_acc = get_test_acc(data_loader_test, model)
        print("Test acc. %.4f Robustness: Mean success rate: %.4f Median L0: %.4f" %\
            (test_acc,mean_success_rate,median_L0))

    # - End training
    # - Evaluate on test set and print accuracy
    print("Evaluating on test set...")
    test_acc = get_test_acc(data_loader_test, model)
    print("Test acc. is %.4f" % (float(100*test_acc)))
    log(FLAGS.session_id, "test_acc", float(test_acc))
    log(FLAGS.session_id, "done", True)

    # - Save the network
    torch.save(model.state_dict(), model_save_path)
    print(f"Done. Saved model under {model_save_path}")
