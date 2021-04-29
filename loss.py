"""
Implementation of normal CCE and TRADES loss, see TRADES https://github.com/yaodongyu/TRADES
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from functools import partial
from batched_sparsefool import sparsefool
import numpy as np

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

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

    if FLAGS.beta_robustness != 0.0 and not is_warmup and not FLAGS.boundary_loss == "None":
        model_copy = deepcopy(model)
        model_copy.eval()

        if FLAGS.round_fn == "round":
            round_fn = torch.round
        else:
            round_fn = lambda x: (torch.rand(size=x.shape, device=device) < x).float()

        return_dict_sparse_fool = sparsefool(
            x_0=x_adv,
            net=model_copy,
            max_hamming_distance=FLAGS.max_hamming_distance,
            lambda_=FLAGS.lambda_,
            epsilon=0.0,
            device=device,
            round_fn=round_fn,
            early_stopping=False,
            boost=False,
            verbose=False,
        )

        x_adv = return_dict_sparse_fool["X_adv"]
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