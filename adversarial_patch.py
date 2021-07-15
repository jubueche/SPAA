"""
Script containing functions implementing adversarial patches
"""
import numpy as np
import math
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
from Experiments.visual_ibm_experiment import plot, class_labels
from utils import plot_attacked_prob

from scipy.ndimage.interpolation import rotate
from sparsefool import reset
from torch.autograd import Variable
import torch.nn.functional as F

def init_patch_circle(
    input_shape,
    patch_size,
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
    patch_values = torch.rand(size=patch_mask.shape) * patch_mask
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

def rotate_patch(
    p_dic,
    angle,
    device
):
    inp_shape = p_dic['patch_values'].shape
    assert len(inp_shape) == 4, "Wrong number of dimensions"
    patch_values = torch.zeros_like(p_dic['patch_values']).float().to(device)
    for t in range(inp_shape[0]):
        for p in range(inp_shape[1]):
            rotated = rotate(p_dic['patch_values'][t,p,p_dic['cx']:p_dic['cx']+p_dic['w'],p_dic['cy']:p_dic['cy']+p_dic['h']], angle=angle, reshape=False)
            patch_values[t,p,p_dic['cx']:p_dic['cx']+p_dic['w'],p_dic['cy']:p_dic['cy']+p_dic['h']] = torch.tensor(rotated)
    
    return {
        'angle': (p_dic['angle'] + angle) % 360,
        'cx':p_dic['cx'],
        'cy':p_dic['cy'],
        'w':p_dic['w'],
        'h':p_dic['h'],
        'patch_mask':p_dic['patch_mask'],
        'patch_values':patch_values
    }

def init_patch(
    patch_type,
    patch_size,
    input_shape,
    device
):
    if patch_type == 'circle':
        patch = init_patch_circle(input_shape, patch_size, device)
    elif patch_type == 'square':
        assert False, "Invalid patch shape"
    else:
        assert False, "Invalid patch shape"
    return patch

def vis_patch(patch):
    fig = plt.figure(constrained_layout=True, figsize=(10,6))
    spec = mpl.gridspec.GridSpec(ncols=4, nrows=5, figure=fig)
    axes = [fig.add_subplot(spec[i,j]) for i in range(5) for j in range(4)]
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        top=False,
                        labelbottom=False,
                        right=False,
                        left=False,
                        labelleft=False)
    plot((axes,{'X0':patch['patch_values'],'X_adv':patch['patch_values'],'predicted':0,'predicted_attacked':0},0,class_labels))
    plt.show()

def transform_circle(
    patch,
    device
):
    # patch = rotate_patch(patch, angle=np.random.choice(360), device=device)
    # dx = np.random.randint(0,patch['patch_mask'].shape[-2]-patch['w'])
    # dy = np.random.randint(0,patch['patch_mask'].shape[-1]-patch['h'])
    # patch = translate_patch(d_cx=dx,d_cy=dy,p_dic=patch,device=device)
    patch = translate_patch(d_cx=30,d_cy=80,p_dic=patch,device=device)
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
        patch = transform_circle(patch, device=device)

        # - Create adversarial example
        X_adv = (1. - patch['patch_mask']) * X + patch['patch_values']
        X_adv = torch.round(torch.clamp(X_adv, 0., 1.))
        reset(net)
        adv_label = torch.argmax(net.forward(Variable(X_adv, requires_grad=False)).data).item()

        if adv_label == target_label:
            success += 1
        N += 1

        if idx == 0:
            plot_attacked_prob(
                X.squeeze(),
                int(target[0]),
                net,
                N_rows=2,
                N_cols=2,
                data=[(torch.clamp(torch.sum(X_adv[0].cpu(), 1), 0.0, 1.0),
                        adv_label, ) for _ in range(2 * 2)],
                figname=2,
            )

    print("\033[92mSuccess rate is ", success / N, "\033[0m")

def train(
    patch,
    net,
    train_data_loader,
    target_label,
    max_iter,
    label_conf,
    max_count,
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

        patch = transform_circle(patch, device=device)

        X_adv, patch = attack(
            X,
            net,
            pred_label,
            patch,
            target_label,
            label_conf,
            max_count
        )

    return patch

def attack(
    X,
    net,
    pred,
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
        adv_out_probs, adv_out_labels = adv_out.max(1)
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
    device  
):

    # - Initialize patch
    patch = init_patch(patch_type, patch_size, input_shape, device)

    for epoch in range(n_epochs):

        # - Evaluate
        test(
            patch,
            net,
            test_data_loader,
            max_iter_test,
            target_label,
            device
        )

        patch = train(
                    patch,
                    net,
                    train_data_loader,
                    target_label,
                    max_iter,
                    label_conf,
                    max_count,
                    device
                )