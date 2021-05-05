"""
Train spiking model for IBMGestures
"""
from experiment_utils import *
import os.path as path
from architectures import IBMGestures as arch
from architectures import log
from dataloader_IBMGestures import IBMGesturesDataLoader
from networks import IBMGesturesBPTT
from loss import robust_loss
import torch
import time
from networks import load_gestures_snn

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_test_acc(data_loader, model):
    model.eval()
    correct = 0; num = 0
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
    model_save_path = path.join(base_path, f"Resources/Models/{FLAGS.session_id}_model.pt")
    
    batch_size = FLAGS.batch_size
    dt = FLAGS.dt
    torch.manual_seed(FLAGS.seed)
    epochs = FLAGS.epochs

    if not FLAGS.boundary_loss == "None":
        assert FLAGS.boundary_loss in ["trades","madry"], "Unknown boundary loss"

    ibm_gesture_dataloader = IBMGesturesDataLoader()

    data_loader_train = ibm_gesture_dataloader.get_data_loader("train", shuffle=True, num_workers=4, batch_size=batch_size, dt=dt)
    data_loader_test = ibm_gesture_dataloader.get_data_loader("test", shuffle=True, num_workers=4, batch_size=32, dt=dt)

    if FLAGS.boundary_loss != "None":
        print("Loading pre-trained model...")
        model = load_gestures_snn("data/Gestures/pretrain.model")
    else:
        # - Create model
        model = IBMGesturesBPTT().to(device)

    # - Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    # - Begin the training
    for epoch in range(epochs):
        model.train()
        for batch_idx, (sample,target) in enumerate(data_loader_train):
            model.reset_states()
            sample = sample.float()
            sample = torch.clamp(sample, 0.0, 1.0)
            sample = sample.to(device)
            target = target.long().to(device)
            loss = robust_loss(
                model=model,
                x_natural=sample,
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
                print("Epoch %d Batch %d/%d Time %.3f h Loss %.3f Batch acc. %.3f" % (epoch,batch_idx,data_loader_train.__len__(),t_passed,float(loss),float(100*b_acc)))
                log(FLAGS.session_id,"training_accuracy",float(b_acc))
                log(FLAGS.session_id,"loss",float(loss))

        # - Evaluate on test set and print accuracy
        print("Evaluating on test set...")
        test_acc = get_test_acc(data_loader_test, model)
        print("Test acc. is %.4f" % (float(100*test_acc)))
        log(FLAGS.session_id,"test_acc",float(test_acc))

    # - Save the network
    torch.save(model.state_dict(), model_save_path)
    print(f"Done. Saved model under {model_save_path}")