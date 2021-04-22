from dataloader_NMNIST import NMNISTDataLoader
import torch
from networks import train_ann_mnist, get_det_net, get_summed_network
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


data_loader_test_spikes = nmnist_dataloader.get_data_loader(
    dset="test", mode="snn", shuffle=True, num_workers=4, batch_size=1, dt=3000
)

# - Attack parameters
lambda_ = 2.0
max_hamming_distance = 500
round_fn = lambda x : (torch.rand(size=x.shape) < x).float()
# round_fn = lambda x : torch.clamp(x,0.0,1.0)

for idx, (data, target) in enumerate(data_loader_test_spikes):
    X0 = data
    X0 = X0[0].to(device)
    X0 = torch.clamp(X0, 0.0, 1.0)
    
    return_dict_sparse_fool = sparsefool(
            x_0=X0,
            net=snn,
            max_hamming_distance=max_hamming_distance,
            lambda_=lambda_,
            device=device,
            epsilon=0.0,
            round_fn=round_fn,
            max_iter=20,
            probabilistic=False,
            early_stopping=True,
            boost=False,
            verbose=True
        )

    X_adv_sparse_fool = return_dict_sparse_fool["X_adv"]
    num_flips_sparse_fool = return_dict_sparse_fool["L0"]
    original_prediction = return_dict_sparse_fool["predicted"]

    print("Original prediction %d Sparse fool orig. %d" % (int(get_prediction(snn, X0, "non_prob")),original_prediction))
    # - Evaluate on the attacked image
    model_pred_sparse_fool = get_prediction(snn, X_adv_sparse_fool, "non_prob")
    print(
        f"Sparse fool prediction {int(model_pred_sparse_fool)} with L_0 = {num_flips_sparse_fool}"
    )

plot_attacked_prob(X0, target, snn, N_rows=1, N_cols=1, block=False, figname=1)

plot_attacked_prob(
    X0,
    target,
    snn,
    N_rows=1,
    N_cols=1,
    data=[
        (torch.clamp(torch.sum(X_adv_sparse_fool.cpu(), 1), 0.0, 1.0), model_pred_sparse_fool)
        for _ in range(1)
    ],
    figname=2,
)
