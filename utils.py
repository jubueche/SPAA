from videofig import videofig
import torch
import torch.nn.functional as F
import numpy as np

# - Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_index_list(dims):
    if len(dims) == 2:
        return [(i, j) for i in range(dims[0]) for j in range(dims[1])]
    else:
        return [((i,) + el) for i in range(dims[0]) for el in get_index_list(dims[1:])]

def get_X_adv_post_attack(flip_indices, max_hamming_distance, boost, verbose, X_adv, y, net, early_stopping):
    assert len(flip_indices) <= max_hamming_distance, "You cannot pass more than max_hamming_distance many indices to this method"
    n_queries = 0
    index_dict = {}
    for i in flip_indices:
        index_dict[i] = 0.0  # - Dummy value
    for idx in range(len(flip_indices)):
        if early_stopping:
            n_queries += 1
        if early_stopping and (not get_prediction(net, X_adv, mode="non_prob") == y):
            if verbose:
                print(f"Used Hamming distance {idx+1}")
            break
        if boost:
            n_queries += len(list(index_dict.keys()))
        flip_index = get_next_index(X_adv, index_dict, net, y, boost, verbose)
        X_adv[flip_index] = 1.0 if X_adv[flip_index] == 0.0 else 0.0
    return X_adv, n_queries, idx+1

def confidence(X, net, y):
    return F.softmax(get_prediction_raw(net, X, mode="non_prob"), dim=0)[y]

def get_next_index(X_adv, index_dict, net, y, boost, verbose):
    if not boost:
        best_point = list(index_dict.keys())[0]  # - Just return the first key/index
    else:
        F_X_adv = confidence(X_adv, net, y)
        if verbose:
            print(f"Confidence: {F_X_adv}")
        X_tmp = X_adv.clone()
        best_delta_conf = -np.inf
        best_point = None
        for p in index_dict:
            X_tmp[p] = 1.0 if X_tmp[p] == 0.0 else 0.0
            d_conf = F_X_adv - confidence(X_tmp, net, y)
            X_tmp[p] = 1.0 if X_tmp[p] == 0.0 else 0.0  # - Flip back
            if d_conf >= best_delta_conf:
                best_delta_conf = d_conf
                best_point = p
    index_dict.pop(best_point, None)
    return best_point

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
    X = torch.sigmoid(
        (torch.log(rand_unif + eps) - torch.log(1 - rand_unif + eps) + torch.log(P + eps) - torch.log(1 - P + eps)) / temperature
    )
    return X


def get_prediction(
    net,
    data,
    mode="prob",
    batch=False,
):
    """
    Make prediction on data either probabilistically or deterministically. Returns class labels.
    """
    output = get_prediction_raw(net, data, mode, batch)
    if batch:
        pred = output.argmax(dim=1)
    else:
        pred = output.argmax()
    return pred


def get_prediction_raw(
    net,
    data,
    mode="prob",
    batch=False
):
    """
    Make prediction on data either probabilistically or deterministically. Returns raw output.
    """
    try:
        net.reset_states()
    except:
        pass
    if mode == "prob":
        output = net(data)
    elif mode == "non_prob":
        try:
            output = net.forward_np(data)
        except:
            output = net.forward(data)
    else:
        assert mode in ["prob", "non_prob"], "Unknown mode"
    assert len(output.size()) == 2, "Summing over 1d"
    if batch:
        return output
    else:
        output = output.sum(axis=0)
        return output


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
    return sum(acc) / len(acc) * 100


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
        if data.ndim == 4:
            self.data = data[0]
        else:
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
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.5)
                ax.spines[axis].set_color(self.color)
            ax.set_yticks([])
            ax.set_xticks([])
            self.im = ax.imshow(X)
            self.initialized = True
        else:
            self.im.set_data(X)
        self.f0 += 1


def plot_attacked_prob(
    P_adv,
    target,
    net,
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
            if isinstance(axes, list):
                redraw_fn.sub[i].draw(f, axes[i])
            else:
                redraw_fn.sub[i].draw(f, axes)

    if data is None:
        data = []
        for i in range(N_rows * N_cols):
            pred = get_prediction(net, P_adv, "non_prob")
            if P_adv.ndim == 5:
                store_image = torch.clamp(torch.sum(P_adv, 2), 0.0, 1.0)
            else:
                store_image = torch.clamp(torch.sum(P_adv, 1), 0.0, 1.0)
            assert ((store_image == 0.0) | (store_image == 1.0)).all()
            data.append((store_image.cpu(), pred))

    redraw_fn.sub = [Redraw(el[0], el[1], target) for el in data]

    videofig(
        num_frames=100,
        play_fps=50,
        redraw_func=redraw_fn,
        grid_specs={'nrows': N_rows, 'ncols': N_cols},
        block=block,
        figname=figname)
