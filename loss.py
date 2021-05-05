"""
Implementation of normal CCE and TRADES loss, see TRADES https://github.com/yaodongyu/TRADES
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from functools import partial
from batched_sparsefool import sparsefool, universal_sparsefool
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
    x_adv = x_natural.clone()

    if FLAGS.beta_robustness != 0.0 and not is_warmup and not FLAGS.boundary_loss == "None":
        model_copy = deepcopy(model)
        model_copy.eval()

        return_dict_sparse_fool = universal_sparsefool(
            x_0=x_adv,
            net=model_copy,
            max_hamming_distance=FLAGS.max_hamming_distance,
            lambda_=FLAGS.lambda_,
            epsilon=0.0,
            overshoot=0.2,
            device=device,
            early_stopping=False,
            boost=False,
            verbose=True,
        )

        x_adv = return_dict_sparse_fool["X_adv"]
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        y_adv = y[return_dict_sparse_fool["success"].bool()]
        x_adv = x_adv[return_dict_sparse_fool["success"].bool()]
        model_copy.train()

    optimizer.zero_grad()
    model.reset_states()
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    if x_adv == [] or FLAGS.beta_robustness == 0.0:
        loss = loss_natural
    elif FLAGS.boundary_loss == "madry":
        model.reset_states()
        logits_model_x_adv = model(x_adv)
        loss_robust = F.cross_entropy(logits_model_x_adv, y_adv)
        loss = loss_natural + FLAGS.beta_robustness * loss_robust
    elif FLAGS.boundary_loss == "trades":
        model.reset_states()
        logits_model_x_adv = model(x_adv)
        model.reset_states()
        logits_model_x_natural = model(x_natural)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_model_x_adv, dim=1),
                                                        F.softmax(logits_model_x_natural, dim=1))
        loss = loss_natural + FLAGS.beta_robustness * loss_robust
    else:
        assert FLAGS.boundary_loss in ["trades","madry"], "Unknown boundary loss"

    return loss