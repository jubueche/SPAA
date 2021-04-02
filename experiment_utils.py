import torch
import torch.nn.functional as F
import numpy as np
from videofig import videofig
from sinabs.network import Network as SinabsNetwork
from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import optimize_linear
from cleverhans.torch.attacks.fast_gradient_method import _fast_gradient_method_grad

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