"""
Implementation of normal CCE and TRADES loss, see TRADES https://github.com/yaodongyu/TRADES
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from functools import partial
from torch.multiprocessing import Pool, set_start_method
from sparsefool import sparsefool
# from attacks import prob_fool, non_prob_fool

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def sparse_fool_wrapper(
    net,
    max_hamming_distance,
    lambda_,
    device,
    epsilon,
    round_fn,
    max_iter,
    early_stopping,
    boost,
    verbose,
    shared,
):
    if round_fn == "stoch_round":
        round_fn = lambda x: (torch.rand(size=x.shape, device=device) < x).float()
    elif round_fn == "round":
        round_fn = torch.round
    x_0, n = shared
    return_list = []
    for x in x_0:
        if x.ndim == 4:
            x = x.reshape((1,) + x.shape)
        return_list.append(
            sparsefool(
                x_0=x,
                net=net,
                max_hamming_distance=max_hamming_distance,
                lambda_=lambda_,
                device=device,
                epsilon=epsilon,
                round_fn=round_fn,
                max_iter=max_iter,
                early_stopping=early_stopping,
                boost=boost,
                verbose=verbose,
            )
        )
    return (return_list, n)

def robust_loss(model,
                x_natural,
                y,
                optimizer,
                FLAGS,
                is_warmup):
    # Define KL-loss
    batch_size = x_natural.shape[0]
    criterion_kl = nn.KLDivLoss(size_average=False)
    x_adv = x_natural.detach()

    if FLAGS.beta_robustness != 0.0 and not is_warmup:
        model_copy = deepcopy(model)
        model_copy.eval()

        partial_sparse_fool = partial(
            sparse_fool_wrapper,
            model_copy,
            FLAGS.max_hamming_distance,
            FLAGS.lambda_,
            device,
            0.0,
            FLAGS.round_fn,
            FLAGS.max_iter_sparse_fool,
            False,
            False,
            True,
        )

        X_split = list(torch.split(x_adv, split_size_or_sections=10, dim=0))

        with Pool(processes=1) as p:
            results = p.map(partial_sparse_fool, zip(X_split, range(len(X_split))))
            results.sort(key=lambda x: x[1])
            results = [r[0] for r in results]
            results = [a for b in results for a in b]

        x_adv = [r["X_adv"] for r in results if r["success"]]
        if len(x_adv) > 0:
            x_adv = torch.stack(x_adv)
            x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        model_copy.train()

    optimizer.zero_grad()
    model.reset_states()
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    if x_adv == [] or FLAGS.beta_robustness == 0.0:
        loss = loss_natural
    else:
        model.reset_states()
        logits_model_x_adv = model(x_adv)
        model.reset_states()
        logits_model_x_natural = model(x_natural)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_model_x_adv, dim=1),
                                                        F.softmax(logits_model_x_natural, dim=1))
        loss = loss_natural + FLAGS.beta_robustness * loss_robust
    return loss