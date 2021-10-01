"""
Script containing functions implementing adversarial patches
"""
import numpy as np
import math
import time
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import plot_attacked_prob

from scipy.ndimage.interpolation import rotate
from sparsefool import reset
from torch.autograd import Variable
import torch.nn.functional as F

MIDDLE = ((70,80),(50,60))
TOP_LEFT = ((10,50),(20,40)) 
TOP_RIGHT = ((10,50),(60,90))
GEN = ((10,90),(10,90))
receptive_field = {
    0: MIDDLE, # y,x
    1: TOP_LEFT,
    2: TOP_RIGHT,
    3: TOP_LEFT,
    4: TOP_LEFT,
    5: TOP_RIGHT,
    6: TOP_RIGHT,
    7: MIDDLE,
    8: MIDDLE,
    9: MIDDLE,
    10: TOP_LEFT
}

def init_patch_circle(
    input_shape,
    patch_size,
    init,
    device
):
    """
    Return patch of shape patch_shape. Patch is a boolean mask.
    """
    patch_mask = torch.zeros(input_shape).float().to(device)
    image_size = np.prod(input_shape[-2:])
    noise_size = int(image_size*patch_size) # - patch_size is in %
    radius = int(math.sqrt(noise_size/math.pi))
    y, x = np.ogrid[-radius: radius, -radius: radius]
    patch_mask[:,:,:2*radius,:2*radius] = torch.tensor(x**2 + y**2 <= radius**2).float().to(device)
    if init == 'zeros':
        patch_values = torch.zeros(size=patch_mask.shape, device=device) * patch_mask
    else:
        patch_values = (2 * torch.rand(size=patch_mask.shape, device=device) - 1) * patch_mask
    return {
        'angle':0,
        'cx':0,
        'cy':0,
        'w':2*radius,
        'h':2*radius,
        'patch_mask':patch_mask,
        'patch_values':patch_values
    }

def translate_patch(
    d_cx,
    d_cy,
    p_dic,
    device
):
    patch_mask = torch.zeros_like(p_dic['patch_mask']).float().to(device)
    patch_mask[:,:,d_cx:d_cx+p_dic['w'],d_cy:d_cy+p_dic['h']] = p_dic['patch_mask'][:,:,p_dic['cx']:p_dic['cx']+p_dic['w'],p_dic['cy']:p_dic['cy']+p_dic['h']]
    patch_values = torch.zeros_like(p_dic['patch_values']).float().to(device)
    patch_values[:,:,d_cx:d_cx+p_dic['w'],d_cy:d_cy+p_dic['h']] = p_dic['patch_values'][:,:,p_dic['cx']:p_dic['cx']+p_dic['w'],p_dic['cy']:p_dic['cy']+p_dic['h']]
    return {
        'angle':0,
        'cx':d_cx,
        'cy':d_cy,
        'w':p_dic['w'],
        'h':p_dic['h'],
        'patch_mask':patch_mask,
        'patch_values':patch_values
    }

def init_patch(
    patch_type,
    patch_size,
    input_shape,
    init,
    device
):
    if patch_type == 'circle':
        patch = init_patch_circle(input_shape, patch_size, init, device)
    elif patch_type == 'square':
        assert False, "Invalid patch shape"
    else:
        assert False, "Invalid patch shape"
    return patch

def transform_circle(
    patch,
    target_label,
    device
):
    x, y = receptive_field[target_label]
    dx = np.random.randint(x[0],x[1])
    dy = np.random.randint(y[0],y[1])
    patch = translate_patch(d_cx=dx,d_cy=dy,p_dic=patch,device=device)
    return patch

def test(
    patch,
    net,
    test_data_loader,
    max_iter_test,
    target_label,
    device
):
    success = 0; N = 0
    for idx, (X, target) in enumerate(test_data_loader):
        if idx >= max_iter_test:
            break

        X = X.float()
        X = X.to(device)
        X = torch.clamp(X, 0.0, 1.0)
        target = target.long().to(device)

        # - Get the prediction
        reset(net)
        pred_label = torch.argmax(net.forward(Variable(X, requires_grad=False)).data).item()

        if pred_label != target or pred_label == target_label:
            continue

        # - Transform the patch randomly
        patch = transform_circle(patch, target_label, device=device)

        # - Create adversarial example
        X = X.squeeze()
        if patch['patch_values'].shape != X.shape:
            X = torch.cat((X, torch.zeros((patch['patch_values'].shape[0] - X.shape[0], *X.shape[1:]), device=X.device)))  
        X_adv = (1. - patch['patch_mask']) * X + patch['patch_values']
        X_adv = torch.round(torch.clamp(X_adv, 0., 1.))
        reset(net)
        adv_label = torch.argmax(net.forward(Variable(X_adv, requires_grad=False)).data).item()

        if adv_label == target_label:
            success += 1
        N += 1

        # if idx == 0:
        #     plot_attacked_prob(
        #         X.squeeze(),
        #         int(target[0]),
        #         net,
        #         N_rows=2,
        #         N_cols=2,
        #         data=[(torch.clamp(torch.sum(X_adv[0].cpu(), 1), 0.0, 1.0),
        #                 adv_label, ) for _ in range(2 * 2)],
        #         figname=2,
        #     )

    print("\033[92mSuccess rate is ", success / N, "\033[0m")
    return success / N

def train(
    patch,
    net,
    train_data_loader,
    test_data_loader,
    target_label,
    max_iter,
    label_conf,
    max_count,
    eval_after,
    device
):
    for idx, (X, target) in enumerate(train_data_loader):
        if idx >= max_iter:
            break

        X = X.float()
        X = X.to(device)
        X = torch.clamp(X, 0.0, 1.0)
        target = target.long().to(device)

        # - Get the prediction
        reset(net)
        pred_label = torch.argmax(net.forward(Variable(X, requires_grad=False)).data).item()

        if pred_label != target:
            continue

        patch = transform_circle(patch, target_label, device=device)

        X_adv, patch = attack(
            X,
            net,
            patch,
            target_label,
            label_conf,
            max_count
        )

        if idx % eval_after == 1:
            test(
                patch,
                net,
                test_data_loader,
                100,
                target_label,
                device
            )

    return patch

def attack(
    X,
    net,
    patch,
    target_label,
    label_conf,
    max_count
):
    reset(net)
    f_image = F.log_softmax(net.forward(Variable(X, requires_grad=False)))
    target_conf = f_image[0][target_label]

    X_adv = (1. - patch['patch_mask']) * X + patch['patch_mask'] * patch['patch_values']
    X_adv = torch.round(torch.clamp(X_adv, 0., 1.))
    count = 0
    while target_conf < label_conf:
        count += 1
        X_adv = Variable(X_adv.data, requires_grad=True)
        reset(net)
        adv_out = F.log_softmax(net.forward(X_adv))
        Loss = -adv_out[0][target_label]
        Loss.backward()
        adv_grad = X_adv.grad.clone()
        X_adv.grad.data.zero_()
        patch['patch_values'] = patch['patch_mask'] * (patch['patch_values'] - adv_grad.squeeze())
        X_adv = (1. - patch['patch_mask']) * X + patch['patch_values']
        X_adv = torch.round(torch.clamp(X_adv, 0., 1.))
        reset(net)
        out = F.softmax(net.forward(X_adv))
        target_conf = out.data[0][target_label]

        print("Count ",count," Target conf ",
                float(target_conf)," label_conf ",label_conf,
                " Max(Abs(.)) ",float(torch.max(torch.abs(patch['patch_values']))))

        if count >= max_count:
            break

    return X_adv, patch

def adversarial_patch(
    net,
    train_data_loader,
    test_data_loader,
    patch_type,
    patch_size,
    input_shape,
    n_epochs,
    target_label,
    max_iter,
    max_iter_test,
    label_conf,
    max_count,
    eval_after,
    device  
):
    t0 = time.time()

    # - Initialize patch
    patch = init_patch(patch_type, patch_size, input_shape, 'zeros', device)

    for _ in range(n_epochs):

        patch = train(
                    patch,
                    net,
                    train_data_loader,
                    test_data_loader,
                    target_label,
                    max_iter,
                    label_conf,
                    max_count,
                    eval_after,
                    device
                )

    patch_random = init_patch(patch_type, patch_size, input_shape, 'random', device)
    success_rate_random = test(
        patch_random,
        net,
        test_data_loader,
        max_iter_test,
        target_label,
        device
    )

    success_rate_targeted = test(
            patch,
            net,
            test_data_loader,
            max_iter_test,
            target_label,
            device
        )

    return_dict = {
        "L0": torch.nonzero(patch["patch_values"] * patch["patch_mask"]).shape[0],
        "pert_total": patch["patch_values"],
        "patch_mask": patch["patch_mask"],
        "patch":patch,
        "patch_random": patch_random,
        "elapsed_time": time.time()-t0,
        "success_rate_targeted": success_rate_targeted,
        "success_rate_random": success_rate_random
    }
    return return_dict