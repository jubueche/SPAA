from dataloader_NMNIST import NMNISTDataLoader
import torch
from networks import train_ann_mnist, get_prob_net, get_det_net, get_summed_network
from attacks import prob_fool, prob_attack_pgd
from sparsefool import sparsefool
from utils import get_prediction, plot_attacked_prob

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# - Create data loader
nmnist_dataloader = NMNISTDataLoader()

# - Train ANN MNIST model
ann = train_ann_mnist()

# - Turn that into network that sums over time dimension
snn = get_summed_network(ann, n_classes=10)

# - Turn into spiking prob net
prob_net = get_prob_net().to(device)

data_loader_test_spikes = nmnist_dataloader.get_data_loader(
    dset="test", mode="snn", shuffle=True, num_workers=4, batch_size=1, dt=3000
)

# - Attack parameters
lambda_ = 3.0
max_hamming_distance = 200
round_fn = lambda x : (torch.rand(size=x.shape) < x).float()

for idx, (data, target) in enumerate(data_loader_test_spikes):
    P0 = data
    P0 = P0[0].to(device)
    P0 = torch.clamp(P0, 0.0, 1.0)
    
    return_dict_sparse_fool = sparsefool(
            x_0=P0,
            net=snn,
            max_hamming_distance=max_hamming_distance,
            lambda_=lambda_,
            device=device,
            round_fn=round_fn,
            max_iter=5,
            probabilistic=False,
            early_stopping=True,
            boost=False,
            verbose=True
        )

    X_adv_sparse_fool = return_dict_sparse_fool["X_adv"]
    num_flips_sparse_fool = return_dict_sparse_fool["L0"]
    original_prediction = return_dict_sparse_fool["predicted"]

    print("Original prediction %d Sparse fool orig. %d" % (int(get_prediction(snn, P0, "non_prob")),original_prediction))
    # - Evaluate on the attacked image
    model_pred_sparse_fool = get_prediction(snn, X_adv_sparse_fool, "non_prob")
    print(
        f"Sparse fool prediction {int(model_pred_sparse_fool)} with L_0 = {num_flips_sparse_fool}"
    )

    if idx >= 4:
        break

plot_attacked_prob(P0, target, prob_net, N_rows=2, N_cols=2, block=False, figname=1)

plot_attacked_prob(
    P0,
    target,
    ann,
    N_rows=2,
    N_cols=2,
    data=[
        (torch.clamp(torch.sum(X_adv_sparse_fool.cpu(), 1), 0.0, 1.0), model_pred_sparse_fool)
        for _ in range(2 * 2)
    ],
    figname=2,
)
