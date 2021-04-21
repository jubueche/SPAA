from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from copy import deepcopy
import torch
import numpy as np
import time

from utils import get_prediction, reparameterization_bernoulli, get_X_adv_post_attack, get_index_list


def deepfool(
    im,
    net,
    lambda_fac=3.0,
    overshoot=0.02,
    max_iter=50,
    device="cuda",
    round_fn=torch.round,
    probabilistic=False,
    rand_minmax=0.1,
):
    n_queries = 0
    X0 = deepcopy(im) # - Keep continuous version

    if probabilistic:
        eta = torch.zeros_like(X0).uniform_(0, rand_minmax)
        X0 = X0 * (1 - eta) + (1 - X0) * eta
        X0 = torch.clamp(X0, 0.0, 1.0)

    n_queries += 1
    f_image = net.forward(Variable(torch.round(X0), requires_grad=True)).data.cpu().numpy().flatten()
    try:
        net.reset_states()
    except: pass
    num_classes = len(f_image)
    input_shape = X0.size()

    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    X_adv = deepcopy(X0)
    r_tot = torch.zeros(input_shape).to(device)

    k_i = label
    loop_i = 0

    while k_i == label and loop_i < max_iter:

        if not probabilistic:
            x = Variable(round_fn(X_adv), requires_grad=True)
        else:
            x = Variable(X_adv, requires_grad=True)

        fs = net.forward(x)
        try:
            net.reset_states()
        except: pass

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = deepcopy(x.grad.data)

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_k.norm()
            assert not torch.isnan(pert_k).any(), "Found NaN"
            assert not torch.isinf(pert_k).any(), "Found Inf"

            if pert_k < pert:
                pert = pert_k + 0.
                w = w_k + 0.

        r_i = torch.clamp(pert, min=1e-4) * w / w.norm()
        assert not torch.isnan(r_i).any(), "Found NaN"
        assert not torch.isinf(r_i).any(), "Found Inf"
        r_tot = r_tot + r_i

        X_adv = X_adv + r_i

        if not probabilistic:
            check_fool = round_fn(X0 + (1 + overshoot) * r_tot) # torch.round results in NaNs in the gradient
            assert ((check_fool == 1.0) | (check_fool == 0.0)).all(), "Input must be binary"
        else:
            check_fool = X0 + (1 + overshoot) * r_tot

        n_queries += 1
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()
        try:
            net.reset_states()
        except: pass

        loop_i += 1

    if not probabilistic:
        x = Variable(round_fn(X_adv), requires_grad=True)
    else:
        x = Variable(X_adv, requires_grad=True)
    n_queries += 1
    fs = net.forward(x)
    try:
        net.reset_states()
    except: pass
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)

    grad = deepcopy(x.grad.data)
    grad = grad / grad.norm()

    torch.nan_to_num(grad, 0.0)
    # assert not torch.isnan(grad).any(), "Found NaN"
    # assert not torch.isinf(grad).any(), "Found Inf"

    r_tot = lambda_fac * r_tot
    X_adv = X0 + r_tot

    return grad, X_adv, n_queries


def sparsefool(
    x_0,
    net,
    max_hamming_distance,
    lb=0.0,
    ub=1.0,
    lambda_=2.,
    max_iter=20,
    epsilon=0.02,
    overshoot=0.02,
    max_iter_deep_fool=50,
    device="cuda",
    round_fn=torch.round,
    probabilistic=False,
    rand_minmax=0.1,
    early_stopping=False,
    boost=False,
    verbose=False,
):
    t0 = time.time()
    n_queries = 1
    try:
        net.reset_states()
    except: pass
    pred_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()

    x_i = deepcopy(x_0)
    fool_im = deepcopy(x_i)

    fool_label = pred_label
    loops = 0

    while fool_label == pred_label and loops < max_iter:

        if verbose:
            print(f"{loops}/{max_iter}")

        normal, x_adv, n_queries_deepfool = deepfool(
            im=x_i,
            net=net,
            lambda_fac=lambda_,
            overshoot=overshoot,
            max_iter=max_iter_deep_fool,
            device=device,
            round_fn=round_fn,
            probabilistic=probabilistic,
            rand_minmax=rand_minmax
        )

        x_i = linear_solver(x_i, normal, x_adv, lb, ub)

        fool_im = x_0 + (1 + epsilon) * (x_i - x_0)
        fool_im = clip_image_values(fool_im, lb, ub)
        if not probabilistic:
            fool_im_tmp = torch.round(fool_im) # TODO what to use here, round_fn or round?
        else:
            fool_im_tmp = torch.round(reparameterization_bernoulli(fool_im, net.temperature))

        fool_label = get_prediction(net, fool_im_tmp, mode="non_prob")

        n_queries += n_queries_deepfool + 1
        loops += 1

    # - Get the indices that are most likely to be flipped
    deviations = torch.abs(fool_im - x_0).cpu().numpy()
    tupled = [(aa,) + el for (aa, el) in zip(deviations.flatten(), get_index_list(list(deviations.shape)))]
    tupled.sort(key=lambda a : a[0], reverse=True)
    flip_indices = list(map(lambda a : a[1:], tupled))[:max_hamming_distance]

    X_adv, n_queries_extra, L0 = get_X_adv_post_attack(
        flip_indices=flip_indices,
        max_hamming_distance=max_hamming_distance,
        boost=boost,
        verbose=verbose,
        X_adv=deepcopy(x_0),
        y=pred_label,
        net=net,
        early_stopping=early_stopping
    )

    t1 = time.time()
    return_dict = {}
    return_dict["success"] = 1 if not (pred_label == get_prediction(net, X_adv, mode="non_prob")) else 0
    return_dict["elapsed_time"] = t1-t0
    return_dict["X_adv"] = X_adv
    return_dict["L0"] = L0
    return_dict["n_queries"] = n_queries
    return_dict["predicted"] = pred_label
    return_dict["predicted_attacked"] = get_prediction(net, X_adv, mode="non_prob")
    return return_dict


def linear_solver(x_0, normal, boundary_point, lb, ub):
    input_shape = x_0.size()

    coord_vec = deepcopy(normal)
    plane_normal = deepcopy(coord_vec).view(-1)
    plane_point = deepcopy(boundary_point).view(-1)

    x_i = deepcopy(x_0)

    f_k = torch.dot(plane_normal, x_0.view(-1) - plane_point)
    sign_true = f_k.sign().item()

    beta = 0.001 * sign_true
    current_sign = sign_true

    while current_sign == sign_true and coord_vec.nonzero().size()[0] > 0:

        f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta

        pert = f_k.abs() / coord_vec.abs().max()

        mask = torch.zeros_like(coord_vec)
        mask[np.unravel_index(torch.argmax(coord_vec.abs()).item(), input_shape)] = 1.

        r_i = torch.clamp(pert, min=1e-4) * mask * coord_vec.sign()

        x_i = x_i + r_i
        #TODO  x_i = constrained_projection(im, x_i, lb, ub, k=40)
        x_i = clip_image_values(x_i, lb, ub)

        f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
        current_sign = f_k.sign().item()

        coord_vec[r_i != 0] = 0

    return x_i

# def constrained_projection(x_0, x_i, lb, ub, k):
#     x_i = clip_image_values(x_i, lb, ub)

#     # - Find the bits that were touched when you round them
#     touched = (torch.round(x_0) != torch.round(x_i)).float()
#     if torch.sum(touched) > k:
#         indices = torch.nonzero(touched, as_tuple=False)[torch.randperm(int(torch.sum(touched)))][:k]
#         x_i[tuple(indices.T)] = x_0[tuple(indices.T)]

#     return x_i

def clip_image_values(x, minv, maxv):
    return torch.clamp(x, minv, maxv)
