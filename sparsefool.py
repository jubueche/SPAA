from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from copy import deepcopy
import torch
import numpy as np
import time
from utils import get_prediction, get_prediction_raw, plot_attacked_prob
import matplotlib.pyplot as plt

def reset(net):
    try:
        net.reset_states()
    except: pass

def deepfool(
    im,
    net,
    lambda_fac=3.0,
    overshoot=0.02,
    step_size=0.01,
    max_iter=50,
    device="cuda",
):
    n_queries = 0
    X0 = deepcopy(im) # - Keep continuous version
    assert ((X0 == 0.0) | (X0 == 1.0)).all(), "Non binary input deepfool"

    n_queries += 1
    reset(net) #! reset
    f_image = net.forward(Variable(X0, requires_grad=True)).data.cpu().numpy().flatten()
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

        x = Variable(X_adv, requires_grad=True) #! Rounding

        reset(net) #! reset
        fs = net.forward(x)

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
            # assert not (cur_grad == 0.0).all(), "Zero cur grad"

            if pert_k < pert:
                pert = pert_k + 0.
                w = w_k + 0.

        r_i = torch.clamp(pert, min=step_size) * w / w.norm() #! change here maybe
        assert not torch.isnan(r_i).any(), "Found NaN"
        assert not torch.isinf(r_i).any(), "Found Inf"
        r_tot = r_tot + r_i

        X_adv = X_adv + r_i

        check_fool = X0 + (1 + overshoot) * r_tot

        n_queries += 1
        reset(net)
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()

        loop_i += 1

    x = Variable(X_adv, requires_grad=True)
    
    n_queries += 1
    reset(net)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)

    grad = deepcopy(x.grad.data)
    assert not torch.isnan(grad).any() and not torch.isinf(grad).any(), "found inf or nan"
    if grad.norm() == 0.0:
        print("WARNING Grad norm is zero")
    grad = grad / (grad.norm() + 1e-10)
    assert not torch.isnan(grad).any() and not torch.isinf(grad).any(), "found inf or nan"

    r_tot = lambda_fac * r_tot
    X_adv = X0 + r_tot

    return grad, X_adv, n_queries

def heatmap_pruning(
    pert,
    heatmap,
    max_hamming_distance
):
    assert heatmap.shape == pert.shape, "Shapes must be the same"
    flip_set = []
    indices = torch.nonzero(torch.ones_like(heatmap), as_tuple=False)
    indices = sorted([(heatmap[tuple(xx.numpy())],) + tuple(xx.numpy()) for xx in indices], key= lambda x : x[0])[::-1]
    added = 0
    for i,idx in enumerate(indices):
        true_idx = idx[1:]
        if pert[true_idx]:
            flip_set.append(true_idx)
            added += 1
        if added == max_hamming_distance:
            break

    pert_projected = torch.zeros_like(pert)
    for flip_idx in flip_set:
        pert_projected[flip_idx] = True
    
    return pert_projected


def universal_attack(
    X,
    y,
    net,
    attack_fn,
    max_hamming_distance,
    target_success_rate,
    max_iter,
    device="gpu"
):
    """
    Find universal adversarial perturbation that produces success rate at least target_success_rate
    that runs for a maximum of max_iter iterations. The perturbation will have at most max_hamming_distance
    L0 norm.
    attack_fn takes as input X_i and y_i and returns a dictionary.
    """
    success_rate = 0.0
    count = 0
    input_shape = X.shape
    t0 = time.time()
    pert_total = torch.zeros((1,) + X.shape[1:]).bool().to(device)

    # - Create a heat map
    pert_aggregated = torch.zeros(X.shape[1:]).float()
    for idx, (x_i,y_i) in enumerate(zip(X,y)):
        # - Apply current universal perturbation
        x_i_p = x_i.clone()
        reset(net)
        pred_label = torch.argmax(net.forward(Variable(x_i_p, requires_grad=False)).data).item()
        if pred_label == y_i:
            return_dict_adv = attack_fn(x_i_p,y_i)
            pert = (return_dict_adv["X_adv"] != x_i_p)
            pert_aggregated[pert] += 1.

        if idx > 5: break

    plt.imshow(torch.sum(pert_aggregated, axis=[0,1]))
    plt.show()

    # plot_attacked_prob(pert_aggregated,0,net,N_rows=2,N_cols=2,data=4*[(torch.sum(pert_aggregated, axis=1),0)])

    while count < max_iter and success_rate < target_success_rate:

        for idx, (x_i,y_i) in enumerate(zip(X,y)):
            # - Apply current universal perturbation
            x_i_p = x_i.clone()
            x_i_p[torch.squeeze(pert_total)] = 1. - x_i[torch.squeeze(pert_total)]
            reset(net)
            pred_label = torch.argmax(net.forward(Variable(x_i_p, requires_grad=False)).data).item()

            if pred_label == y_i:

                return_dict_adv = attack_fn(x_i_p,y_i)
                pert = (return_dict_adv["X_adv"] != x_i_p)
                
                # - Update the current universal attack
                pert_total = pert_total | torch.tensor(pert).to(device)

        pert_total = heatmap_pruning(torch.squeeze(pert_total),heatmap=pert_aggregated, max_hamming_distance=max_hamming_distance)
        X_pert = X.clone()
        X_pert[:,pert_total] = 1. - X[:,pert_total]
        reset(net)
        pred_X_pert = torch.argmax(net.forward(X_pert), dim=1)
        success_rate = torch.mean((pred_X_pert != y).float())
        print(f"Success rate {success_rate}")
        count += 1

    t1 = time.time()
    X_adv = X.clone()
    X_adv[:,pert_total] = 1. - X[:,pert_total]
    return_dict = {}
    return_dict["predicted"] = torch.argmax(net.forward(X), dim=1)
    return_dict["predicted_attacked"] = torch.argmax(net.forward(X_adv), dim=1)
    return_dict["success_rate"] = success_rate
    return_dict["elapsed_time"] = t1-t0
    return_dict["X_adv"] = torch.reshape(X_adv, input_shape)
    return return_dict


def universal_sparsefool(
    x_0,
    y,
    net,
    max_hamming_distance,
    lb=0.0,
    ub=1.0,
    lambda_=2.,
    max_iter=10,
    max_iter_sparse_fool=20,
    epsilon=0.02,
    overshoot=0.02,
    n_attack_frames=1,
    step_size=1.0,
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
    input_shape = x_0.shape
    if x_0.ndim == 5:
        x_0 = x_0[0]
        
    t0 = time.time()
    T = x_0.shape[0]
    reset(net)
    pred_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()

    def get_next_attack_frame(X, n_attack_frames):
        net.reset_states()
        maximum, index = torch.max(net.forward_raw(X),dim=2)
        correct_indices = torch.arange(0,T,1)[(index == pred_label).flatten()]
        correct_tuples = list(zip(maximum.flatten()[correct_indices],correct_indices))
        sorted_tuples = sorted(correct_tuples, key=lambda x : x[0])[::-1]
        frame_index_to_attack = [t[1] for t in sorted_tuples[:n_attack_frames]]
        return torch.tensor(frame_index_to_attack)

    label = pred_label
    it = 0
    n_queries = 0

    X_adv = x_0.clone()
    pert_total = torch.zeros((1,) + X_adv.shape[1:]).bool().to(device)

    while label == pred_label and it < max_iter and pred_label == y:

        attack_frame = get_next_attack_frame(X_adv, n_attack_frames)
        n_queries += 1

        x_tmp = X_adv[attack_frame] 

        return_dict_sparse_fool = sparsefool(
            x_tmp,
            net=net,
            max_hamming_distance=max_hamming_distance,
            lb=lb,
            ub=ub,
            lambda_=lambda_,
            max_iter=5,
            epsilon=epsilon,
            overshoot=overshoot,
            step_size=step_size,
            max_iter_deep_fool=max_iter_deep_fool,
            device=device,
            early_stopping=early_stopping,
            boost=boost,
            verbose=False
        )

        n_queries += return_dict_sparse_fool["n_queries"]

        # - Get the perturbation
        pert = (return_dict_sparse_fool["X_adv"] != x_tmp)
        pert = np.logical_or.reduce(pert.cpu().numpy(), axis=0, keepdims=True)

        pert_total = pert_total | torch.tensor(pert).to(device)

        # - Apply universally
        X_adv[:,torch.squeeze(pert_total)] = 1. - x_0[:,torch.squeeze(pert_total)]
        assert ((X_adv == 0.0) | (X_adv ==1.0)).all(), "Non binary X_adv"

        # plot_attacked_prob(
        #     X_adv,
        #     0,
        #     net,
        #     N_rows=2,
        #     N_cols=2,
        #     data=[(torch.clamp(torch.sum(X_adv.cpu(), 1), 0.0, 1.0),
        #         return_dict_sparse_fool["predicted_attacked"], ) for _ in range(2 * 2)],
        #     figname=2,
        # )

        net.reset_states()
        label = torch.argmax(net.forward(Variable(X_adv, requires_grad=False)).data).item()
        n_queries += 1

        it += 1

    L0 = int(torch.sum(torch.abs(X_adv - x_0)))

    t1 = time.time()
    return_dict = {}
    return_dict["success"] = 1 if not (pred_label == label) or (pred_label != y) and L0 <= max_hamming_distance else 0
    return_dict["elapsed_time"] = t1-t0
    return_dict["X_adv"] = torch.reshape(X_adv, input_shape)
    return_dict["L0"] = L0
    return_dict["n_queries"] = n_queries
    return_dict["predicted"] = int(pred_label)
    return_dict["predicted_attacked"] = int(label)

    if verbose:
        if return_dict["success"]:
            print("UNIVERSAL Succes L0",L0)
        else:
            print("UNIVERSAL No success")

    return return_dict

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
    step_size=0.01,
    max_iter_deep_fool=50,
    device="cuda",
    early_stopping=False,
    boost=False,
    verbose=False,
):
    t0 = time.time()
    n_queries = 1
    reset(net)
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
            step_size=step_size,
            max_iter=max_iter_deep_fool,
            device=device,
        )

        assert not torch.isnan(normal).any() and not torch.isnan(x_adv).any(), "Found NaN"
        x_i = linear_solver(x_i, normal, x_adv, lb, ub)

        fool_im = deepcopy(x_i)
        fool_label = get_prediction(net, fool_im, mode="non_prob")

        n_queries += n_queries_deepfool + 1
        loops += 1

    X_adv = fool_im
    L0 = int(torch.sum(torch.abs(fool_im - x_0)))

    if early_stopping:
        flip_indices = torch.nonzero(torch.abs(x_0-fool_im))
        X_adv_tmp = deepcopy(x_0)
        for k,flip_index in enumerate(flip_indices):
            X_adv_tmp[tuple(flip_index)] = fool_im[tuple(flip_index)]
            if not (pred_label == get_prediction(net, X_adv_tmp, mode="non_prob")):
                L0 = k+1
                X_adv = X_adv_tmp
            

    t1 = time.time()
    return_dict = {}
    return_dict["success"] = 1 if not (pred_label == get_prediction(net, X_adv, mode="non_prob")) and L0 <= max_hamming_distance else 0
    return_dict["elapsed_time"] = t1-t0
    return_dict["X_adv"] = X_adv
    return_dict["L0"] = L0
    return_dict["n_queries"] = n_queries
    return_dict["predicted"] = pred_label
    return_dict["predicted_attacked"] = int(get_prediction(net, X_adv, mode="non_prob"))

    if return_dict["success"]:
        assert pred_label != get_prediction(net, X_adv, mode="non_prob"), "Success but the same label"

    if verbose:
        if return_dict["success"]:
            print("Succes L0",L0)
        else:
            print("No success")

    return return_dict


def linear_solver(x_0, normal, boundary_point, lb, ub):
    input_shape = x_0.size()

    coord_vec = deepcopy(normal)
    last_sign = coord_vec.sign()
    mask = torch.zeros_like(coord_vec)
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
        x_i = torch.clamp(x_i, lb, ub)

        f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
        current_sign = f_k.sign().item()

        last_sign = deepcopy(coord_vec.sign())
        coord_vec[r_i != 0] = 0

    if not (mask == 0.0).all():
        x_i[mask.bool()] =  (last_sign[mask.bool()]+1.) / 2.
        
    x_i[(x_i != 0.0) & (x_i != 1.0)] = -(x_0[(x_i != 0.0) & (x_i != 1.0)]-1.) #! questionable
    assert ((x_i == 0.0) | (x_i == 1.0)).all(), "Not all binary"
    return x_i
