"""
Implementation of normal CCE and TRADES loss, see TRADES https://github.com/yaodongyu/TRADES
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from attacks import non_prob_fool
import numpy as np

# - Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
        x_0 = x_natural.clone().float()
        # x_0 = torch.clamp(torch.round(x_0), 0.0, 1.0)

        # with torch.no_grad():
        x_adv_dict = non_prob_fool(
            max_hamming_distance=1000,
            net=model_copy,
            X0=x_0,
            round_fn=torch.round,
            eps=0.5,
            eps_iter=0.1,
            N_pgd=5,
            norm=np.inf,
            rand_minmax=0.01,
            boost=False,
            early_stopping=False,
            verbose=False,
            batch=True,
            clamp=False
        )

        x_adv = x_adv_dict["X_adv"]
        x_adv = Variable(x_adv, requires_grad=False)
        model_copy.train()

    optimizer.zero_grad()
    model.reset_states()
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    if x_adv == [] or FLAGS.beta_robustness == 0.0:
        loss = loss_natural
    elif FLAGS.boundary_loss == "madry":
        raise NotImplementedError
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