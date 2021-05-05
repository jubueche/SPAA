from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from copy import deepcopy
import torch
import numpy as np
import time

from utils import get_prediction, reparameterization_bernoulli, get_X_adv_post_attack, get_index_list

def reset(net):
    try:
        net.reset_states()
    except: pass

def universal_sparsefool(
    x_0,
    net,
    max_hamming_distance,
    lb=0.0,
    ub=1.0,
    lambda_=2.,
    max_iter=4,
    max_iter_sparse_fool=20,
    epsilon=0.02,
    overshoot=0.2,
    max_iter_deep_fool=50,
    device="cuda",
    early_stopping=False,
    boost=False,
    verbose=False,
):
    """
    Evaluate network on each frame. Sort frames by strongest prediction. Find perturbation for
    each sorted frame and apply universally. If misclas. return, else move to next frame.
    """
    t0 = time.time()
    B = x_0.shape[0]
    T = x_0.shape[1]
    reset(net)
    pred_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data, axis=1)

    def get_next_attack_frame(X):
        net.reset_states()
        maximum, index = torch.max(net.forward_raw(X),dim=2)
        frame_index_to_attack = torch.zeros(B)
        for b in range(B):
            correct_indices = torch.arange(0,T,1)[(index[b] == pred_label[b]).flatten()]
            correct_tuples = list(zip(maximum[b].flatten()[correct_indices],correct_indices))
            sorted_tuples = sorted(correct_tuples, key=lambda x : x[0])[::-1]
            frame_index_to_attack[b] = sorted_tuples[0][1]
        return frame_index_to_attack

    label = pred_label
    it = torch.zeros(B)
    n_queries = 0

    X_adv = x_0

    while (label == pred_label).any() and (it < max_iter).any():

        not_done = label == pred_label

        if (~not_done).all():
            break

        attack_frame = get_next_attack_frame(X_adv[not_done])
        n_queries += B

        x_tmp = torch.zeros(size=(torch.sum(not_done.int()),1) + x_0.shape[2:])
        for b in range(B):
            x_tmp[b] = X_adv[b,attack_frame[b].int()]

        return_dict_sparse_fool = sparsefool(
            x_tmp,
            net,
            max_hamming_distance,
            lb,
            ub,
            lambda_,
            max_iter_sparse_fool,
            epsilon,
            overshoot,
            max_iter_deep_fool,
            device,
            early_stopping,
            boost,
            True
        )

        n_queries += return_dict_sparse_fool["n_queries"]

        # - Get the perturbation
        pert = return_dict_sparse_fool["X_adv"] - x_tmp

        # - Apply universally
        X_adv = X_adv + pert
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        net.reset_states()
        label = torch.argmax(net.forward(Variable(X_adv, requires_grad=True)).data, axis=1)
        n_queries += B

        it[not_done] += 1

    L0 = torch.sum(torch.abs(X_adv - x_0), dim=(1,2,3,4)).int()            

    t1 = time.time()
    return_dict = {}
    return_dict["success"] = (~(pred_label == label) & (L0 <= max_hamming_distance)).int().cpu().numpy()
    return_dict["elapsed_time"] = t1-t0
    return_dict["X_adv"] = X_adv
    return_dict["L0"] = L0.cpu().numpy()
    return_dict["n_queries"] = n_queries
    return_dict["predicted"] = pred_label.cpu().numpy()
    return_dict["predicted_attacked"] = label.cpu().numpy()

    if verbose:
        print("Success rate %.4f" % np.mean(return_dict["success"]))
        print("L0 rate %.4f" % np.mean(return_dict["L0"]))
        print("Elapsed time %.4f" % return_dict["elapsed_time"])

    return return_dict

def deepfool(
    im,
    net,
    lambda_fac=3.0,
    overshoot=0.02,
    max_iter=50,
    device="cuda",
    verbose=False
):
    assert im.ndim == 5, "Expected dimension (batch,time,polarity,H,W)"
    n_queries = 0
    X0 = deepcopy(im) # - Keep continuous version
    batch_size = X0.shape[0]
    assert ((X0 == 0.0) | (X0 == 1.0)).all(), "Non binary input deepfool"

    n_queries += 1
    reset(net)
    f_image = net.forward(Variable(X0, requires_grad=True)).data.cpu().numpy()
    num_classes = f_image.shape[1]
    input_shape = X0.size()

    I = (np.array(f_image)).argsort(axis=1)[:,::-1]
    labels = torch.tensor(I[:,0]).to(device)

    X_adv = deepcopy(X0)
    check_fool = torch.zeros_like(X_adv, requires_grad=False)
    r_tot = torch.zeros(input_shape).to(device)

    k_i = labels
    loop_i = torch.zeros(batch_size, device=device)

    not_done = torch.ones(batch_size, requires_grad=False).bool()

    while (k_i == labels).any() and not_done.any():

        not_done = k_i == labels

        not_done[loop_i == max_iter] = False

        print(not_done)

        x = Variable(X_adv, requires_grad=True)

        reset(net)
        fs = net.forward(x)

        pert = torch.Tensor(batch_size * [np.inf]).to(device)
        w = torch.zeros(input_shape).to(device)

        for b in range(batch_size):
            if not_done[b]:
                fs[b,I[b,0]].backward(retain_graph=True)
        
        grad_orig = torch.zeros_like(x)
        for b in range(batch_size):
            if not_done[b]:
                grad_orig[b] = deepcopy(x.grad.data[b]) 
        
        # grad_orig = deepcopy(x.grad.data)
        
        for k in range(1, num_classes):
            zero_gradients(x)

            for b in range(batch_size):
                if not_done[b]:
                    fs[b,I[b,k]].backward(retain_graph=True)

            cur_grad = torch.zeros_like(x)
            for b in range(batch_size):
                if not_done[b]:
                    cur_grad[b] = deepcopy(x.grad.data[b]) 
            
            # cur_grad = deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = torch.zeros(batch_size, device=device)
            for b in range(batch_size):
                if not_done[b]:
                    f_k[b] = (fs[b,I[b,k]] - fs[b,I[b,0]]).data

            pert_k = torch.abs(f_k) / w_k.norm(p=2,dim=[1,2,3,4])
            if verbose:
                assert not torch.isnan(pert_k[not_done]).any(), "Found NaN"
                assert not torch.isinf(pert_k[not_done]).any(), "Found Inf"

            pk_sm_p = (pert_k < pert) & not_done
            pert[pk_sm_p] = pert_k[pk_sm_p] + 0.
            w[pk_sm_p] = w_k[pk_sm_p] + 0.

        r_i = torch.zeros_like(w)
        w_norm = w.norm(p=2,dim=[1,2,3,4])
        for b in range(batch_size):
            if not_done[b]:
                r_i[b] = torch.clamp(pert[b], min=1e-4) * w[b] / w_norm[b]

        if verbose:
            assert not torch.isnan(r_i[not_done]).any(), "Found NaN"
            assert not torch.isinf(r_i[not_done]).any(), "Found Inf"
            assert not_done.all() or (r_i[~not_done] == 0.0).all(), "non zero although done"
        
        r_tot[not_done] = r_tot[not_done] + r_i[not_done]

        X_adv[not_done] = X_adv[not_done] + r_i[not_done]

        check_fool[not_done] = X0[not_done] + (1 + overshoot) * r_tot[not_done]

        n_queries += batch_size
        reset(net)
        k_i = torch.zeros(batch_size, dtype=int)
        for b in range(batch_size):
            net.reset_states()
            k_i[b] = torch.argmax(net.forward(Variable(torch.reshape(check_fool[b], (1,)+check_fool[b].shape), requires_grad=True)).data, dim=1)
        
        # net.reset_states()
        # k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data, dim=1)

        loop_i += 1

    x = Variable(X_adv, requires_grad=True)
    
    n_queries += batch_size
    reset(net)
    fs = net.forward(x)
    for b in range(batch_size):
        (fs[b, k_i[b]] - fs[b, labels[b]]).backward(retain_graph=True)

    grad = deepcopy(x.grad.data)
    gnorm = grad.norm(p=2,dim=[1,2,3,4]) 
    if verbose:
        assert not torch.isnan(grad).any() and not torch.isinf(grad).any(), "found inf or nan"
    if (gnorm == 0.0).any():
        print("WARNING Grad norm is zero")
    
    for b in range(b):
        grad[b] = grad[b] / (gnorm[b] + 1e-10)
    if verbose:
        assert not torch.isnan(grad).any() and not torch.isinf(grad).any(), "found inf or nan"

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
    overshoot=0.2,
    max_iter_deep_fool=50,
    device="cuda",
    early_stopping=False,  # not for this version
    boost=False,
    verbose=False,
):
    t0 = time.time()
    assert x_0.ndim == 5, "Dimension must be (batch size,time,polarity,H,W)"
    batch_size = x_0.shape[0]
    n_queries = batch_size
    reset(net)
    pred_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data, axis=1)

    x_i = deepcopy(x_0)
    fool_im = deepcopy(x_i)

    fool_label = pred_label
    loops = torch.zeros(batch_size, requires_grad=False)
    not_done = torch.ones(batch_size, requires_grad=False).bool()

    while (fool_label == pred_label).any() and not_done.any():

        not_done = fool_label == pred_label
        not_done[loops == max_iter] = False

        if (~not_done).all():
            break

        if verbose:
            for l in loops:
                print(f"{int(l)}/{max_iter}")

        to_pass = x_i[not_done]
        if to_pass.ndim == 0 or to_pass.shape[0] == 0:
            assert False

        normal, x_adv, n_queries_deepfool = deepfool(
            im=x_i[not_done],
            net=net,
            lambda_fac=lambda_,
            overshoot=overshoot,
            max_iter=max_iter_deep_fool,
            device=device,
            verbose=verbose
        )
        if verbose:
            assert not torch.isnan(normal).any() and not torch.isnan(x_adv).any(), "Found NaN"
        x_i[not_done] = linear_solver(x_i[not_done], normal, x_adv, lb, ub, device, verbose)

        fool_im = deepcopy(x_i)
        fool_label = get_prediction(net, fool_im, mode="non_prob", batch=True)

        n_queries += n_queries_deepfool + batch_size
        loops[not_done] += 1

    X_adv = fool_im
    L0 = torch.sum(torch.abs(fool_im - x_0), dim=(1,2,3,4)).int()            

    t1 = time.time()
    return_dict = {}
    return_dict["success"] = (~(pred_label == get_prediction(net, X_adv, mode="non_prob", batch=True)) & (L0 <= max_hamming_distance)).int().cpu().numpy()
    return_dict["elapsed_time"] = t1-t0
    return_dict["X_adv"] = X_adv
    return_dict["L0"] = L0.cpu().numpy()
    return_dict["n_queries"] = n_queries
    return_dict["predicted"] = pred_label.cpu().numpy()
    return_dict["predicted_attacked"] = get_prediction(net, X_adv, mode="non_prob", batch=True).cpu().numpy()

    print("Success rate %.4f" % np.mean(return_dict["success"]))
    print("L0 rate %.4f" % np.mean(return_dict["L0"]))
    print("Elapsed time %.4f" % return_dict["elapsed_time"])

    return return_dict


def linear_solver(x_0, normal, boundary_point, lb, ub, device, verbose):
    input_shape = x_0.size()
    batch_size = x_0.shape[0]
    coord_vec = deepcopy(normal)
    last_sign = coord_vec.sign()
    mask = torch.zeros_like(coord_vec)
    plane_normal = deepcopy(coord_vec).view(batch_size,-1)
    plane_point = deepcopy(boundary_point).view(batch_size,-1)

    x_i = deepcopy(x_0)

    f_k = torch.squeeze(torch.bmm(plane_normal.view(batch_size,1,-1), (x_0.view(batch_size,-1) - plane_point).view(batch_size,-1,1)))
    if f_k.ndim == 0:
        f_k = f_k.view(-1)

    sign_true = f_k.sign()

    pert = torch.zeros_like(f_k, requires_grad=False)

    beta = 0.001 * sign_true
    current_sign = sign_true

    nonzeros = torch.nonzero(coord_vec)
    sizes = torch.tensor([len(nonzeros[nonzeros[:,0] == b]) for b in range(batch_size)], device=device)

    not_done = (current_sign == sign_true) & (sizes > 0)

    while not_done.any():

        print(sizes)
        f_k = torch.squeeze(torch.bmm(plane_normal.view(batch_size,1,-1), (x_i.view(batch_size,-1) - plane_point).view(batch_size,-1,1))) + beta
        if f_k.ndim == 0:
            f_k = f_k.view(-1)

        not_done = (current_sign == sign_true) & (sizes > 0)

        for b in range(batch_size):
            if not_done[b]:
                pert[b] = f_k[b].abs() / coord_vec[b].abs().max()

        mask = torch.zeros_like(coord_vec)
        for b in range(batch_size):
            if not_done[b]:
                m_index = (b,) + np.unravel_index(torch.argmax(coord_vec[b].abs()).cpu().numpy(),input_shape[1:])
                mask[m_index] = 1.

        r_i = torch.zeros_like(x_i, requires_grad=False, device=device)
        for b in range(batch_size):
            if not_done[b]:
                r_i[b] = torch.clamp(pert[b], min=1e-4) * mask[b] * coord_vec[b].sign()
    
        if verbose:
            assert not_done.all() or (r_i[~not_done] == 0.0).all()
        
        x_i = x_i + r_i
        x_i = torch.clamp(x_i, lb, ub)

        f_k = torch.squeeze(torch.bmm(plane_normal.view(batch_size,1,-1), (x_i.view(batch_size,-1) - plane_point).view(batch_size,-1,1)))
        if f_k.ndim == 0:
            f_k = f_k.view(-1)
        current_sign = f_k.sign()

        last_sign = deepcopy(coord_vec.sign())
        for b in range(batch_size):
            if not_done[b]:
                coord_vec[b,r_i[b] != 0] = 0
        
        nonzeros = torch.nonzero(coord_vec)
        sizes = torch.tensor([len(nonzeros[nonzeros[:,0] == b]) for b in range(batch_size)], device=device)

    if not (mask == 0.0).all():
        x_i[mask.bool()] =  (last_sign[mask.bool()]+1.) / 2.
        
    x_i[(x_i != 0.0) & (x_i != 1.0)] = -(x_0[(x_i != 0.0) & (x_i != 1.0)]-1.)
    assert ((x_i == 0.0) | (x_i == 1.0)).all(), "Not all binary"
    return x_i
