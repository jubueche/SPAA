import torch
import torch.nn as nn
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
from cleverhans_additions import _fast_gradient_method_grad
from datajuicer import cachable
import itertools
import functools
import time

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
    rand_unif = torch.rand(P.size(), device=P.device)
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
    target = torch.tensor([target], device=spike_out.device)
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

def hamming_attack(
    hamming_distance_eps,
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
    Perform probabilistic attack using the parameters after "hamming_distance". Sort by largest deviation
    from initial probability (i.e. largest deviation from 0 or 1). Flip "hamming_distance"-many spikes with
    highest deviation. Return attacking input with hamming_distance.
    """
    P_adv = prob_attack_pgd(
        prob_net,
        P0,
        eps,
        eps_iter,
        N_pgd,
        N_MC,
        norm,
        rand_minmax,
        verbose=verbose
    )
    
    assert hamming_distance_eps <= 1.0, "Hamming distance eps must be smaller than or equal to 1"
    assert ((P0 == 1.0) | (P0 == 0.0)).all(), "Entries must be 0.0 or 1.0"
    hamming_distance = int(np.prod(P0.shape) * hamming_distance_eps)
    if verbose:
        print(f"Used Hamming distance is {hamming_distance}")
    X_adv = P0.clone()
    deviations = torch.abs(P_adv - P0).numpy()
    tupled = [(aa,) + el for (aa,el) in zip(deviations.flatten(),get_index_list(list(deviations.shape)))]
    tupled.sort(key = lambda a : a[0], reverse=True)
    flip_indices = list(map(lambda a : a[1:], tupled[:hamming_distance]))
    for idx in flip_indices:
        X_adv[idx] = 1.0 if X_adv[idx] == 0.0 else 0.0
    assert torch.sum(torch.abs(P0 - X_adv)) == hamming_distance, "Actual hamming distance does not equal the target hamming distance"
    return X_adv 

def scar_attack(
    hamming_distance_eps,
    net,
    X0,
    thresh,
    verbose=False
):
    """
    Implementation of the SCAR algorithm (https://openreview.net/pdf?id=xCm8kiWRiBT)
    """
    assert hamming_distance_eps <= 1.0, "Hamming distance eps must be smaller than or equal to 1"
    assert ((X0 == 1.0) | (X0 == 0.0)).all(), "Entries must be 0.0 or 1.0"
    t0 = time.time()
    hamming_distance = int(np.prod(X0.shape) * hamming_distance_eps)
    X_adv = X0.clone()
    y = get_prediction(net, X0, mode="non_prob") # - Use the model prediction, not the target label
    # - Find initial point p', points are stored as (g_p, idx1, idx2, ... , idxN)
    # - Pick the middle point TODO What is better here?
    points = {tuple([int(np.floor(el / 2)) for el in list(X_adv.shape)]) : 0.0} # - Dicts are hash maps and tuples are hashable
    flipped = {} # - Keep track of the bits that have been flipped
    crossed_threshold = {} # - Keep track of points that crossed threshold, every point in here must be in points as well
    max_point = list(points.keys())[0] + (0,) # - Keep track of the point that has the current biggest grad
    n_queries = 0
    def N(p, shape):
        surround = [[x for x in [el-1,el,el+1] if 0 <= x < shape[idx]] for idx,el in enumerate(list(p))]
        n = list(itertools.product(*surround))
        n.remove(p)
        return n
    def get_neighbor_dict(points, flipped, p, shape):
        neighbors = N(p, shape)
        r = {}
        for n in neighbors:
            if not n in flipped:
                if n in points:
                    r[n] = points[n]
                else:
                    r[n] = 0.0 # - This point was never visited
        return r
    def get_g_p(model_fn, pred, X_adv, p, F_X_adv):
        X_tmp = X_adv.clone()
        X_tmp[p] = 1.0 if X_tmp[p] == 0.0 else 0.0 # - Flip the bit
        g_p = F_X_adv - F.softmax(get_prediction_raw(model_fn, X_tmp, mode="non_prob"),dim=0)[int(pred)]
        return g_p
    def is_boundary(X_adv, p):
        # - Get list of surrounding pixels along spatial dimension
        surround = [p[:-2] + (i,j) for i in [p[-2]-1,p[-2],p[-2]+1] for j in [p[-1]-1,p[-1],p[-1]+1] if 0 <= i < X_adv.shape[-2] and 0 <= j < X_adv.shape[-1]]
        return functools.reduce(lambda a,b:a or b, [X_adv[p] != X_adv[q] for q in surround], False)
    def get_boundary_dict(X_adv):
        boundary_set = {}
        idx_list = get_index_list(X_adv.shape)
        for p in idx_list:
            if is_boundary(X_adv,p):
                boundary_set[p] = 0.0
        return boundary_set
    def update_boundary_dict(current_boundary_dict, flipped_point, X_updated, points):
        p = flipped_point
        surround = [p[:-2] + (i,j) for i in [p[-2]-1,p[-2],p[-2]+1] for j in [p[-1]-1,p[-1],p[-1]+1] if 0 <= i < X_updated.shape[-2] and 0 <= j < X_updated.shape[-1]]
        for q in surround:
            if is_boundary(X_updated, q):
                g_p = points.get(q)
                if g_p == None:
                    g_p = 0.0
                current_boundary_dict[q] = g_p
            else:
                current_boundary_dict.pop(q, None) # - Delete from boundary set if it's not in boundary set anymore or never was in it
        return current_boundary_dict

    current_neighbor_dict = get_neighbor_dict(points, flipped, list(points.keys())[0], X0.shape) # - Get neighbors dict of starting point
    current_boundary_dict = get_boundary_dict(X0) 
    num_flipped = 0
    while num_flipped < hamming_distance :
        if verbose:
            print(f"Queries: {n_queries}")
        # - Cache the confidence of the current adv. image
        F_X_adv = F.softmax(get_prediction_raw(net, X_adv, mode="non_prob"),dim=0)[int(y)]
        if verbose:
            print(f"Confidence: {float(F_X_adv)}")
        n_queries += 1
        max_point = max_point[:-1] + (0,) # - Reset the g_p value of max_point
        #! - ASSERTS, remove me 
        for p in crossed_threshold:
            assert p in points, "crossed threshold points must be in points"
            assert not p in flipped, "crossed threshold point can not be in flipped dict"
            assert crossed_threshold[p] >= thresh, "point in crossed_threshold is smaller than thresh"
        
        for p in crossed_threshold:
            g_p = get_g_p(net, y, X_adv, p, F_X_adv)
            n_queries += 1
            if g_p >= max_point[-1]:
                max_point = p + (g_p,)
            if g_p >= thresh:
                crossed_threshold[p] = g_p
            else:
                crossed_threshold.pop(p, None) # - Remove the point from crossed threshold dict
            points[p] = g_p
        for p in current_neighbor_dict:
            g_p = get_g_p(net, y, X_adv, p, F_X_adv)
            n_queries += 1
            if g_p >= max_point[-1]:
                max_point = p + (g_p,)
            if g_p >= thresh:
                crossed_threshold[p] = g_p
            points[p] = g_p # - Add or update the point's value

        if max_point[-1] < thresh:
            # - Find the best point in set of points that are on the boundary
            for p in current_boundary_dict:
                g_p = get_g_p(net, y, X_adv, p, F_X_adv)
                n_queries += 1
                if g_p >= max_point[-1]:
                    max_point = p + (g_p,)
        else:
            if verbose:
                print(f"Threshold crossed: {max_point[-1]}")
        
        # - Finally use the max point to update X_adv
        p_flip = max_point[:-1] 
        X_adv[p_flip] = 1.0 if X_adv[p_flip] == 0.0 else 0.0

        flipped[p_flip] = max_point[-1] # - Add to flipped dict
        crossed_threshold.pop(p_flip, None) # - Remove from crossed_threshold dict
        points.pop(p_flip, None) # - Remove from points dict
        current_neighbor_dict = get_neighbor_dict(points, flipped, p_flip, X0.shape) # - Update the neighbors for the current flipped point
        current_boundary_dict = update_boundary_dict(current_boundary_dict, p_flip, X_adv, points) # - Update the boundary set for the new image
        num_flipped += 1
        if verbose:
            print(f"Flipped {num_flipped} of {hamming_distance}")

    t1 = time.time()
    if verbose:
        print(f"Elapsed time {t1-t0}s")
    return X_adv.cpu()


def get_index_list(dims):
    if len(dims) == 2:
        return [(i,j) for i in range(dims[0]) for j in range(dims[1])]
    else:
        return [((i,) + el) for i in range(dims[0]) for el in get_index_list(dims[1:])]

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
    Probabilistic projected gradient descent attack. Evaluate network N_MC times by sampling from
    current attacking probabilities, calculate the gradient of the attacking loss, update the probabilities
    using reparameterization trick, repeat. Finally return the attacking probabilities.
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
    eta = torch.zeros_like(P0).uniform_(0, rand_minmax)
    # - Clip initial perturbation
    eta = clip_eta(eta, norm, eps)
    # - Eta is sampled between 0 and 1. Add it to entries with 0 and subtract from entries with value 1
    # - Note that sampling eta from uniform and adding would only add +eta to the zero values and -eta to the 1 values,
    # - and everything else would be clamped to 0 or 1 again.
    P_adv = P0 * (1 - eta) + (1 - P0) * eta
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
    output = get_prediction_raw(net, data, mode)
    pred = output.argmax()
    return pred.cpu()

def get_prediction_raw(
    net,
    data,
    mode="prob"
):
    """
    Make prediction on data either probabilistically or deterministically. Returns raw output.
    """
    net.reset_states()
    if mode == "prob":
        output = net(data)
    elif mode == "non_prob":
        try:
            output = net.forward_np(data)
        except:
            output = net.forward(data)
    else:
        assert mode in ["prob","non_prob"], "Unknown mode"
    output = output.sum(axis=0)
    return output.cpu()

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
    data=None,
    block=True,
    figname=1
):
    """
    Sample adversarial images from the adversarial probabilities and plot frame-by-frame
    """
    def redraw_fn(f, axes):
        for i in range(len(redraw_fn.sub)):
            redraw_fn.sub[i].draw(f, axes[i])

    if data == None:
        data = []
        for i in range(N_rows * N_cols):
            image = torch.round(reparameterization_bernoulli(P_adv, temperature=prob_net.temperature))
            assert ((image >= 0.0) & (image <= 1.0)).all()
            pred = get_prediction(prob_net, image, "non_prob")
            store_image = torch.clamp(torch.sum(image, 1), 0.0, 1.0)
            assert ((store_image == 0.0) | (store_image == 1.0)).all()
            data.append((store_image.cpu(), pred))

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

def get_det_net(ann = None):
    """
    Transform the continuous network into spiking network using the sinabs framework.
    Generate the ann if the passed ann is None.
    Normalize the weights and increase initial weights by multiplicative factor. 
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
    model = from_model(ann, input_shape=(2, 34, 34), add_spiking_output=True)
    
    # - Increase 1st layer weights by magnitude
    model.spiking_model[0].weight.data *= 7
    return model

def get_prob_net(ann = None):
    """
    Create probabilistic network from spiking network and return.
    """
    # - Get the deterministic spikign model
    model = get_det_net(ann)

    # - Create probabilistic network
    prob_net = ProbNetwork(
            ann,
            model.spiking_model,
            input_shape=(2, 34, 34)
        )
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
