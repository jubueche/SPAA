import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pathlib
from dataloader_NMNIST import NMNISTDataLoader
from sinabs.from_torch import from_model
from sinabs.utils import normalize_weights
from videofig import videofig
from sinabs.network import Network as SinabsNetwork
from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import optimize_linear
from cleverhans.torch.attacks.fast_gradient_method import _fast_gradient_method_grad
from datajuicer import cachable

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

class ProbNetwork(SinabsNetwork):
    """
    Probabilistic Network
    A sinabs model that is evaluated using probabilities for Bernoulli random variables.
    Methods:
        forward: Received probabilities, draws from the Bernoulli random variables and evaluates the network
        forward_np: Non-probabilistic forward method. Input: Spikes
    """
    def __init__(
        self,
        model,
        spk_model,
        input_shape,
        synops = False,
        temperature = 0.01
    ):
        self.temperature = temperature
        super().__init__(model, spk_model, input_shape, synops)

    def forward(self, P):
        X = reparameterization_bernoulli(P, self.temperature)
        return super().forward(X)

    def forward_np(self, X):
        return super().forward(X)

def reparameterization_bernoulli(
    P,
    temperature
):
    """
    Reparameterization of Bernoulli random sampling
    P: Matrix carrying probabilities
    temperature: The smaller, the closer is the output to zero or one. Typically 0.01.
    """
    # - Avoid -inf
    eps = 1e-20
    rand_unif = torch.rand(P.size())
    X = torch.sigmoid((torch.log(rand_unif+eps)-torch.log(1-rand_unif+eps)+torch.log(P+eps)-torch.log(1-P+eps))/temperature)
    return X

def loss_fn(
    spike_out,
    target
):
    """
    Loss function used to find the adversarial probabilities
    """
    outputs = torch.reshape(torch.sum(spike_out,axis=0), (1,10))
    target = torch.tensor([target])
    return F.cross_entropy(outputs, target)

def get_grad(
    prob_net,
    P_adv,
    eps_iter,
    norm,
    model_pred,
    loss_fn
):
    """
    Use fast_gradient_method_grad from cleverhans to get the gradients w.r.t. the input
    probabilities. Return the gradients.
    """
    prob_net.reset_states()
    g = _fast_gradient_method_grad(
            model_fn=prob_net,
            x=P_adv,
            eps=eps_iter,
            norm=norm,
            y=model_pred,
            loss_fn=loss_fn)
    return g

def get_mc_P_adv(
    prob_net,
    P_adv,
    model_pred,
    eps_iter,
    norm,
    loss_fn,
    N_MC
):
    """
    Sample the input probabilities N_MC times and approximate the mean gradient using the empirical mean.
    Find vector that best aligns with gradient vector in epsilon-constrained l-norm space (l-inf, l-2).
    Return P_adv.
    """
    g = 0.0
    for j in range(N_MC):
        g += 1 / N_MC * get_grad(prob_net, P_adv, eps_iter, norm, model_pred, loss_fn)
    eta = optimize_linear(g, eps_iter, norm)    
    P_adv = P_adv + eta    
    return P_adv

def prob_attack_pgd(
    prob_net,
    P0,
    eps,
    eps_iter,
    N_pgd,
    N_MC,
    norm,
    rand_minmax,
    verbose = False
):
    """
    Description here
    """
    if norm == "2":
        norm = 2
    if norm == "np.inf":
        norm = np.inf
    assert norm in [np.inf, 2], "Norm not supported"
    assert eps > eps_iter, "Eps must be bigger than eps_iter"
    assert eps >= rand_minmax, "rand_minmax should be smaller than or equal to eps"
    assert ((P0 >= 0.0) & (P0 <= 1.0)).all(), "P0 has entries outside of [0,1]"

    # - Make a prediction on the data
    model_pred = get_prediction(prob_net, P0, mode="prob")

    # - Calculate initial perturbation
    eta = torch.zeros_like(P0).uniform_(-rand_minmax, rand_minmax)
    # - Clip initial perturbation
    eta = clip_eta(eta, norm, eps) 
    P_adv = P0 + eta
    # - Clip for probabilities
    P_adv = torch.clamp(P_adv, 0.0, 1.0) 

    # - PGD steps
    for i in range(N_pgd):
        if verbose:
            print(f"Attack {i}/{N_pgd}")

        # - Update adversarial probabilities
        P_adv = get_mc_P_adv(prob_net, P_adv, model_pred, eps_iter, norm, loss_fn, N_MC)
        # - Compute perturbation
        eta = P_adv - P0
        # - Projection
        eta = clip_eta(eta, norm, eps)
        # - Add back
        P_adv = P0 + eta
        # - Clamp again to probabilities
        P_adv = torch.clamp(P_adv, 0.0, 1.0)
    
    return P_adv

def get_prediction(
    net,
    data,
    mode="prob"
):
    """
    Make prediction on data either probabilistically or deterministically. Returns class labels.
    """
    net.reset_states()
    if mode == "prob":
        output = net(data)
    elif mode == "non_prob":
        output = net.forward_np(data)
    else:
        assert mode in ["prob","non_prob"], "Unknown mode"
    output = output.sum(axis=0)
    pred = output.argmax()
    return pred

def get_test_acc(
    net,
    dataloader,
    limit=-1
):
    """
    Calculate test accuracy for data in dataloader. Limit -1 equals all data. 
    """
    acc = []
    for data, target in dataloader:
        data = data[0].to(device)
        data[data > 1] = 1.0
        pred = get_prediction(net, data)
        correct = pred.item() == target.item()
        acc.append(correct)
        if len(acc) > limit:
            break
    return sum(acc)/len(acc)*100

class Redraw(object):
    """
    Class for visualization using videofig
    """
    def __init__(
        self,
        data,
        pred,
        target
    ):
        self.initialized = False
        self.data = data
        self.pred = pred
        self.target = target
        self.f0 = 0
        self.max = self.data.size(0)
        self.color = 'green'
        if not self.pred == self.target:
            self.color = 'red'

    def draw(self, f, ax):
        X = self.data[int(self.f0 % self.max)]
        if not self.initialized:
            ax.set_ylabel(f"Pred {str(float(self.pred))}")
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2.5)
                ax.spines[axis].set_color(self.color)    
            ax.set_yticks([])
            ax.set_xticks([])
            self.im = ax.imshow(self.data[0])
            self.initialized = True
        else:
            self.im.set_data(X)
        self.f0 += 1


def plot_attacked_prob(
    P_adv,
    target,
    prob_net,
    N_rows=4,
    N_cols=4,
    block=True,
    figname=1
):
    """
    Sample adversarial images from the adversarial probabilities and plot frame-by-frame
    """
    def redraw_fn(f, axes):
        for i in range(len(redraw_fn.sub)):
            redraw_fn.sub[i].draw(f, axes[i])
    
    data = []
    for i in range(N_rows * N_cols):
        image = torch.round(reparameterization_bernoulli(P_adv, temperature=prob_net.temperature))
        assert ((image >= 0.0) & (image <= 1.0)).all()
        pred = get_prediction(prob_net, image, "non_prob")
        store_image = torch.clamp(torch.sum(image, 1), 0.0, 1.0)
        assert ((store_image == 0.0) | (store_image == 1.0)).all()
        data.append((store_image,pred))

    redraw_fn.sub = [Redraw(el[0],el[1],target) for el in data]

    videofig(
        num_frames=100,
        play_fps=50,
        redraw_func=redraw_fn, 
        grid_specs={'nrows': N_rows, 'ncols': N_cols},
        block=block,
        figname=figname)

def get_ann_arch():
    """
    Generate ann architecture and return
    """
    # - Create sequential model
    ann = nn.Sequential(
        nn.Conv2d(2, 20, 5, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2,2),
        nn.Conv2d(20, 32, 5, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2,2),
        nn.Conv2d(32, 128, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2,2),
        nn.Flatten(),
        nn.Linear(128, 500, bias=False),
        nn.ReLU(),
        nn.Linear(500, 10, bias=False),
    )
    ann = ann.to(device)
    return ann

def load_ann(path):
    """
    Tries to load ann from path, returns None if not successful
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    ann = get_ann_arch()
    if not path.exists():
        return None
    else:
        ann.load_state_dict(torch.load(path))
        return ann

def get_prob_net(ann = None):
    """
    Transform the continuous network into spiking network using the sinabs framework.
    Generate the ann if the passed ann is None.
    Normalize the weights and increase initial weights by mult. factor. Create prob. network and return.
    """
    if ann == None:
        ann = train_ann_mnist() 

    # - Create data loader
    nmnist_dataloader = NMNISTDataLoader()

    data_loader_test = nmnist_dataloader.get_data_loader(dset="test", mode="ann", batch_size=64)

    # - Get single batch
    for data, target in data_loader_test:
        break

    # - Normalize weights
    normalize_weights(
        ann.cpu(), 
        torch.tensor(data).float(),
        output_layers=['1','4','7','11'],
        param_layers=['0','3','6','10','12'])

    # - Create spiking model
    input_shape = (2, 34, 34)
    model = from_model(ann, input_shape=input_shape, add_spiking_output=True)

    # - Create probabilistic network
    prob_net = ProbNetwork(
            ann,
            model.spiking_model,
            input_shape=input_shape
        )
    prob_net.spiking_model[0].weight.data *= 7
    return prob_net

def train_ann_mnist():
    """
    Checks if model exists. If not, train and store. Return model.
    """
    # - Create data loader
    nmnist_dataloader = NMNISTDataLoader()

    # - Set the seed
    torch.manual_seed(42)

    # - Setup path
    path = nmnist_dataloader.path / "N-MNIST/mnist_ann.pt"

    ann = load_ann(path)
    if ann == None:
        ann = get_ann_arch()
        data_loader_train = nmnist_dataloader.get_data_loader(dset="train", mode="ann", shuffle=True, num_workers=4, batch_size=64)
        optim = torch.optim.Adam(ann.parameters(), lr=1e-3)
        n_epochs = 1
        for n in range(n_epochs):
            for data, target in data_loader_train:
                data, target = data.to(device), target.to(device) # GPU
                output = ann(data)
                optim.zero_grad()
                loss = F.cross_entropy(output, target)
                loss.backward()
                optim.step()
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
        torch.save(ann.state_dict(), path)
    return ann

@cachable(dependencies= ["model:{architecture}_session_id","eps","eps_iter","N_pgd","N_MC","norm","rand_minmax","limit","N_samples"])
def get_prob_attack_robustness(
    model,
    eps,
    eps_iter,
    N_pgd,
    N_MC,
    norm,
    rand_minmax,
    limit,
    N_samples
):
    if model['architecture'] == "NMNIST":
        nmnist_dataloader = NMNISTDataLoader()
        data_loader = nmnist_dataloader.get_data_loader(dset="test", mode="snn", shuffle=True, num_workers=4, batch_size=1)
    else:
        assert model['architecture'] in ["NMNIST"], "No other architecture added so far"
    
    defense_probabilities = []

    # TODO Split up the dataloader, evaluate probabilities in the threads, and join using the mean and the number samples
    for idx, (batch, target) in enumerate(data_loader):
        if idx == limit:
            break

        batch = torch.clamp(batch, 0.0, 1.0)

        P_adv = prob_attack_pgd(
            model['prob_net'],
            batch[0],
            eps,
            eps_iter,
            N_pgd,
            N_MC,
            norm,
            rand_minmax
        )

        correct = []
        for _ in range(N_samples):
            model_pred = get_prediction(model['prob_net'], P_adv, "prob")
            if model_pred == target:
                correct.append(1.0)
        
        defense_probabilities.append(float(sum(correct) / N_samples))

    return np.mean(np.array(defense_probabilities))




    
        
