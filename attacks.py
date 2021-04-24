import itertools
import functools
import time
from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import optimize_linear
from cleverhans_additions import _fast_gradient_method_grad
import torch.nn.functional as F
import torch
from utils import get_prediction, get_prediction_raw, get_X_adv_post_attack, confidence, get_next_index, get_index_list
import numpy as np
import random


def loss_fn(
    spike_out,
    target
):
    """
    Loss function used to find the adversarial probabilities
    """
    outputs = torch.reshape(torch.sum(spike_out, axis=0), (1, np.prod(spike_out.shape)))
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
    try:
        prob_net.reset_states()
    except:
        pass
    g = _fast_gradient_method_grad(
        model_fn=prob_net,
        x=P_adv,
        eps=eps_iter,
        norm=norm,
        y=model_pred,
        loss_fn=loss_fn
    )
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


def hamming_attack_get_indices(
    prob_net,
    P0,
    eps,
    eps_iter,
    N_pgd,
    N_MC,
    norm,
    rand_minmax,
    verbose=False
):
    """
    Perform probabilistic attack. Sort indices by largest deviation and return
    """
    assert ((P0 == 1.0) | (P0 == 0.0)).all(), "Entries must be 0.0 or 1.0"
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
    # X_adv = P0.clone()
    deviations = torch.abs(P_adv - P0).cpu().numpy()
    tupled = [(aa,) + el for (aa, el) in zip(deviations.flatten(), get_index_list(list(deviations.shape)))]
    tupled.sort(key=lambda a : a[0], reverse=True)
    flip_indices = list(map(lambda a : a[1:], tupled))
    return flip_indices


def non_prob_fool(
    max_hamming_distance,
    net,
    X0,
    round_fn,
    eps,
    eps_iter,
    N_pgd,
    norm,
    rand_minmax,
    boost=False,
    early_stopping=False,
    verbose=False
):
    """
    Perform non-probabilistic PGD on binary input.
    In each iteration of PGD, the gradient is calculated w.r.t. a binarized image. The gradients are then used
    to update a continuous version of the image, clipped between 0 and 1. In the next iteration, the new
    image is sampled from the continuous version using the round_fn. round_fn should be a function taking
    continuous values between 0 and 1 and outputting a binarized version of the input.
    E.g. round_fn = lambda X : torch.round(X)
    """
    if norm == "2":
        norm = 2
    if norm == "np.inf":
        norm = np.inf
    assert norm in [np.inf, 2], "Norm not supported"
    assert eps > eps_iter, "Eps must be bigger than eps_iter"
    assert eps >= rand_minmax, "rand_minmax should be smaller than or equal to eps"
    assert ((X0 == 0.0) | (X0 == 1.0)).all(), "X0 must be 0 or 1"
    t0 = time.time()
    X_adv = X0.clone()

    y = get_prediction(net, X0, mode="non_prob")
    eta = torch.zeros_like(X0).uniform_(0, rand_minmax)
    eta = clip_eta(eta, norm, eps)
    X_adv_cont = X0 * (1 - eta) + (1 - X0) * eta
    X_adv_cont = torch.clamp(X_adv_cont, 0.0, 1.0)

    for i in range(N_pgd):
        # if verbose:
        #     print(f"Attack {i}/{N_pgd}")

        X_adv_tmp = get_mc_P_adv(net, round_fn(X_adv_cont), y, eps_iter, norm, loss_fn, 1)
        eta_tmp = X_adv_tmp - round_fn(X_adv_cont)  # - Extract what was added to the rounded input
        X_adv_cont += eta_tmp
        eta = X_adv_cont - X0
        eta = clip_eta(eta, norm, eps)
        X_adv_cont = X0 + eta
        X_adv_cont = torch.clamp(X_adv_cont, 0.0, 1.0)

    deviations = torch.abs(X_adv_cont - X0).cpu().numpy()
    tupled = [(aa,) + el for (aa, el) in zip(deviations.flatten(), get_index_list(list(deviations.shape)))]
    tupled.sort(key=lambda a : a[0], reverse=True)
    flip_indices = list(map(lambda a : a[1:], tupled))[:max_hamming_distance]
    n_queries = 2 + N_pgd

    X_adv, n_queries_extra, L0 = get_X_adv_post_attack(flip_indices, max_hamming_distance, boost, verbose, X_adv, y, net, early_stopping)
    n_queries += n_queries_extra

    t1 = time.time()
    return_dict = {}
    return_dict["success"] = 1 if not (y == get_prediction(net, X_adv, mode="non_prob")) else 0
    return_dict["elapsed_time"] = t1 - t0
    return_dict["X_adv"] = X_adv
    return_dict["L0"] = L0
    return_dict["n_queries"] = n_queries
    return_dict["predicted"] = y
    return_dict["predicted_attacked"] = get_prediction(net, X_adv, mode="non_prob")
    return return_dict

def prob_fool(
    max_hamming_distance,
    prob_net,
    P0,
    eps,
    eps_iter,
    N_pgd,
    N_MC,
    norm,
    rand_minmax,
    boost=False,
    early_stopping=False,
    verbose=False
):
    """
    Perform probabilistic attack. Sort by largest deviation
    from initial probability (i.e. largest deviation from 0 or 1). Flip "max_hamming_distance"-many spikes with
    highest deviation. Return attacking input with at most max_hamming_distance.
    """
    t0 = time.time()
    X_adv = P0.clone()
    flip_indices = hamming_attack_get_indices(
        prob_net=prob_net,
        P0=P0,
        eps=eps,
        eps_iter=eps_iter,
        N_pgd=N_pgd,
        N_MC=N_MC,
        norm=norm,
        rand_minmax=rand_minmax,
        verbose=verbose
    )[:max_hamming_distance]
    y = get_prediction(prob_net, P0, mode="non_prob")
    n_queries = 2 + N_pgd * N_MC

    X_adv, n_queries_extra, L0 = get_X_adv_post_attack(flip_indices, max_hamming_distance, boost, verbose, X_adv, y, prob_net, early_stopping)
    n_queries += n_queries_extra

    t1 = time.time()
    return_dict = {}
    return_dict["success"] = 1 if not (y == get_prediction(prob_net, X_adv, mode="non_prob")) else 0
    return_dict["elapsed_time"] = t1 - t0
    return_dict["X_adv"] = X_adv
    return_dict["L0"] = L0
    return_dict["n_queries"] = n_queries
    return_dict["predicted"] = y
    return_dict["predicted_attacked"] = get_prediction(prob_net, X_adv, mode="non_prob")
    return return_dict

def SCAR(
    max_hamming_distance,
    net,
    X0,
    thresh,
    early_stopping=False,
    verbose=False
):
    """
    Implementation of the SCAR algorithm (https://openreview.net/pdf?id=xCm8kiWRiBT)
    """
    assert ((X0 == 1.0) | (X0 == 0.0)).all(), "Entries must be 0.0 or 1.0"
    t0 = time.time()
    X_adv = X0.clone()
    y = get_prediction(net, X0, mode="non_prob")  # - Use the model prediction, not the target label
    # - Find initial point p', points are stored as (g_p, idx1, idx2, ... , idxN)
    # - Pick the middle point TODO What is better here?
    points = {}  # - Dicts are hash maps and tuples are hashable
    flipped = {}  # - Keep track of the bits that have been flipped
    crossed_threshold = {}  # - Keep track of points that crossed threshold, every point in here must be in points as well
    max_point = tuple([0 for _ in range(len(X_adv.shape) + 1)])  # - Keep track of the point that has the current biggest grad, init with (0,...,0)
    n_queries = 0

    def N(p, shape):
        surround = [[x for x in [el - 1, el, el + 1] if 0 <= x < shape[idx]] for idx, el in enumerate(list(p))]
        n = list(itertools.product(*surround))
        n.remove(p)
        return n

    def get_neighbor_dict(points, flipped, p, shape):
        neighbors = N(p, shape)
        r = {}
        for n in neighbors:
            if n not in flipped:
                if n in points:
                    r[n] = points[n]
                else:
                    r[n] = 0.0  # - This point was never visited
        return r

    def get_g_p(model_fn, pred, X_adv, p, F_X_adv):
        X_tmp = X_adv.clone()
        X_tmp[p] = 1.0 if X_tmp[p] == 0.0 else 0.0  # - Flip the bit
        g_p = F_X_adv - F.softmax(get_prediction_raw(model_fn, X_tmp, mode="non_prob"), dim=0)[int(pred)]
        return g_p

    def is_boundary(X_adv, p):
        # - Get list of surrounding pixels along spatial dimension
        surround = [p[:-2] + (i, j) for i in [p[-2] - 1, p[-2], p[-2] + 1] for j in [p[-1] - 1, p[-1], p[-1] + 1] if 0 <= i < X_adv.shape[-2] and 0 <= j < X_adv.shape[-1]]
        return functools.reduce(lambda a, b: a or b, [X_adv[p] != X_adv[q] for q in surround], False)

    def get_boundary_dict(X_adv):
        boundary_set = {}
        idx_list = get_index_list(X_adv.shape)
        for p in idx_list:
            if is_boundary(X_adv, p):
                boundary_set[p] = 0.0
        return boundary_set

    def update_boundary_dict(current_boundary_dict, flipped_point, X_updated, points, flipped):
        p = flipped_point
        surround = [p[:-2] + (i, j) for i in [p[-2] - 1, p[-2], p[-2] + 1] for j in [p[-1] - 1, p[-1], p[-1] + 1] if 0 <= i < X_updated.shape[-2] and 0 <= j < X_updated.shape[-1]]
        for q in surround:
            if (flipped.get(q, None) is None) and is_boundary(X_updated, q):
                g_p = points.get(q)
                if g_p is None:
                    g_p = 0.0
                current_boundary_dict[q] = g_p
            else:
                current_boundary_dict.pop(q, None)  # - Delete from boundary set if it's not in boundary set anymore or never was in it
        return current_boundary_dict

    current_neighbor_dict = {}  # get_neighbor_dict(points, flipped, list(points.keys())[0], X0.shape) # - Get neighbors dict of starting point
    current_boundary_dict = get_boundary_dict(X0)
    num_flipped = 0
    while num_flipped < max_hamming_distance :
        if verbose:
            print(f"Queries: {n_queries}")
        # - Cache the confidence of the current adv. image
        output_raw = get_prediction_raw(net, X_adv, mode="non_prob")
        F_X_adv = F.softmax(output_raw, dim=0)[int(y)]
        if verbose:
            print(f"Confidence: {float(F_X_adv)}")

        if early_stopping and (not torch.argmax(output_raw) == y):
            break

        n_queries += 1
        max_point = max_point[:-1] + (0,)  # - Reset the g_p value of max_point

        to_pop = []
        for p in crossed_threshold:
            g_p = get_g_p(net, y, X_adv, p, F_X_adv)
            n_queries += 1
            if g_p >= max_point[-1]:
                max_point = p + (g_p,)
            if g_p >= thresh:
                crossed_threshold[p] = g_p
            else:
                to_pop.append(p)
            points[p] = g_p
        for p in to_pop:
            crossed_threshold.pop(p, None)  # - Remove the point from crossed threshold dict
        for p in current_neighbor_dict:
            g_p = get_g_p(net, y, X_adv, p, F_X_adv)
            n_queries += 1
            if g_p >= max_point[-1]:
                max_point = p + (g_p,)
            if g_p >= thresh:
                crossed_threshold[p] = g_p
            points[p] = g_p  # - Add or update the point's value

        if max_point[-1] < thresh:
            # - Find the best point in set of points that are on the boundary
            b_points = list(current_boundary_dict.keys())
            random.shuffle(b_points)
            for p in b_points:
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
        flipped[p_flip] = max_point[-1]  # - Add to flipped dict
        crossed_threshold.pop(p_flip, None)  # - Remove from crossed_threshold dict
        points.pop(p_flip, None)  # - Remove from points dict
        current_neighbor_dict = get_neighbor_dict(points, flipped, p_flip, X0.shape)  # - Update the neighbors for the current flipped point
        current_boundary_dict = update_boundary_dict(current_boundary_dict, p_flip, X_adv, points, flipped)  # - Update the boundary set for the new image
        num_flipped += 1
        if verbose:
            print(f"Flipped {num_flipped} of {max_hamming_distance}")

    t1 = time.time()
    if verbose:
        print(f"Elapsed time {t1-t0}s")
    return_dict = {}
    return_dict["success"] = 1 if not (y == get_prediction(net, X_adv, mode="non_prob")) else 0
    return_dict["X_adv"] = X_adv.cpu()
    return_dict["L0"] = num_flipped
    return_dict["elapsed_time"] = t1 - t0
    return_dict["n_queries"] = n_queries
    return_dict["predicted"] = y
    return_dict["predicted_attacked"] = get_prediction(net, X_adv, mode="non_prob")
    return return_dict

def prob_attack_pgd(
    prob_net,
    P0,
    eps,
    eps_iter,
    N_pgd,
    N_MC,
    norm,
    rand_minmax,
    verbose=False
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
