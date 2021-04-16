from videofig import videofig
import torch

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


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

    if data is None:
        data = []
        for i in range(N_rows * N_cols):
            image = torch.round(reparameterization_bernoulli(P_adv, temperature=prob_net.temperature))
            assert ((image >= 0.0) & (image <= 1.0)).all()
            pred = get_prediction(prob_net, image, "non_prob")
            store_image = torch.clamp(torch.sum(image, 1), 0.0, 1.0)
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
