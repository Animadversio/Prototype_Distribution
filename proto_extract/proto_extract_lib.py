import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from torchvision.models import resnet18
import torch
import torch.nn as nn
from os.path import join
from collections import OrderedDict
from circuit_toolkit import upconvGAN, get_module_names
from circuit_toolkit.layer_hook_utils import get_module_name_shapes
from circuit_toolkit.plot_utils import show_imgrid, save_imgrid, to_imgrid
from circuit_toolkit.GAN_utils import upconvGAN
from circuit_toolkit.grad_RF_estim import grad_RF_estimate, fit_2dgauss, grad_RF_estimate_torch_naming
from tqdm import trange, tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, GaussianBlur, RandomAffine
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.utils import save_image
#%%
# model type
# ckpt_path = "simclr_stl10-epoch=19-val_loss_ssl=0.00.ckpt"
def load_reformate_ckpt(ckpt_path):
    """takes a ckpt path and return a model with loaded weights.

    :return
        new_model: a new model with the same architecture as resnet18, without renaming.
            only the final fc layer is substituted with an identity layer.
        surrogate: a surrogate model with the same architecture as resnet18,
            but the layers are put in a sequential container, so no more layer1, layer2, etc.
    """
    state_dict = torch.load(ckpt_path)
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    state_dict_bb = OrderedDict((k.replace("backbone.", ""), v) for k,v in
                                state_dict.items() if k.startswith("backbone"))
    surrogate = nn.Sequential(*list(resnet18().children())[:-1])
    surrogate.load_state_dict(state_dict_bb, )
    new_model = resnet18()
    for i, module in enumerate(list(new_model.children())[:-1]):
        module.load_state_dict(surrogate[i].state_dict())
    new_model.fc = nn.Identity()
    return new_model, surrogate


def get_layer_shape_torch_naming(model, layerkey, input_shape=(3, 96, 96)):
    model_feat = create_feature_extractor(model, return_nodes=[layerkey])
    with torch.no_grad():
        out = model_feat(torch.zeros(input_shape).cuda()[None])
        feattsr = out[layerkey]

    _, C, H, W = feattsr.shape
    return C, H, W


def pix_prototype_extract(model, layerkey, channel_rng=None, cnt_pos=None, imgpix=96,
                          optcfg={"lr": 0.01, "steps": 100, "pert_std": 0.001, "init_std": 0.02, },
                          optim_transform=None, show=True, seed=42):
    """ extract prototypes from a model, using pixel space optimization.

    :param model:   model to extract prototypes from
    :param layerkey:   layer to extract prototypes from
    :param channel_rng:  range of channels to extract prototypes from, if None, all channels are used
                         expect a tuple of (start, end)
    :param cnt_pos:  position of the prototype center, if None, the center of the feature map is used
    :param imgpix:   size of the image to generate
    :param optcfg:
    :param show: whether to show the prototypes, Bool, default True
    :param seed: random seed for initializing latent code / pixel. default 42
    :return:
    """
    if optim_transform is None:
        optim_transform = Compose([GaussianBlur(5, sigma=(2, 2)),
                               RandomAffine(degrees=3, translate=(0.02, 0.02), ),
                               Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    vis_transform = Compose([Resize((imgpix, imgpix)), ])
    model_feat = create_feature_extractor(model, return_nodes=[layerkey])
    with torch.no_grad():
        out = model_feat(torch.randn(1, 3, imgpix, imgpix).cuda())
        feattsr = out[layerkey]

    _, C, H, W = feattsr.shape
    if channel_rng is None:
        channel_rng = (0, C)
    else:
        channel_rng = (max(0, channel_rng[0]), min(C, channel_rng[1]))
    batch_size = channel_rng[1] - channel_rng[0]
    cnt_pos = ((H - 1) // 2, (W - 1) // 2) if cnt_pos is None else cnt_pos

    grad_input = 0.5 + torch.randn(batch_size, 3, imgpix, imgpix,
                                   generator=torch.Generator().manual_seed(seed)).cuda() * optcfg["init_std"]
    grad_input.requires_grad_(True)
    optim = torch.optim.Adam([grad_input], lr=optcfg["lr"])
    pert_std = optcfg["pert_std"]
    score_traj = []
    for i in trange(optcfg["steps"]):
        imgs = torch.clamp(grad_input, 0, 1)
        out = model_feat(optim_transform(imgs))
        act_mat = out[layerkey][:, channel_rng[0]:channel_rng[1], cnt_pos[0], cnt_pos[1]]
        diag_resp = act_mat[torch.arange(batch_size), torch.arange(batch_size)]
        loss = -torch.sum(diag_resp)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # for zero activations add perturbation to the zs
        failure_mask = diag_resp == 0
        grad_input[failure_mask].data = grad_input[failure_mask, :].data + \
                                        torch.randn_like(grad_input[failure_mask, :]) * pert_std
        print(f"{diag_resp.mean():.3f}+-{diag_resp.std():.3f} , (eq zero {torch.sum(diag_resp == 0).item()})")
        score_traj.append(diag_resp.detach().cpu())
        # imgmean = grad_input.data.mean(dim=(2, 3), keepdims=True)
        # imgstd = grad_input.data.std(dim=(2, 3), keepdims=True)
        # grad_input.data = imgmean + (grad_input.data - imgmean) / imgstd * 0.22
    imgs = imgs.detach().cpu()
    torch.cuda.empty_cache()
    score_traj = torch.stack(score_traj, dim=0)
    if show:
        show_imgrid(vis_transform(imgs), nrow=int(np.sqrt(batch_size)), )
    mtg = to_imgrid(vis_transform(imgs), nrow=int(np.sqrt(batch_size)))
    return imgs, mtg, score_traj, grad_input


def GAN_prototype_extract(G, model, layerkey, channel_rng=None, zs_init=None, cnt_pos=None, imgpix=96,
                          lr=0.01, steps=100, pert_std=0.1, optim_transform=None, show=True, seed=42, ):
    if optim_transform is None:
        optim_transform = Compose([Resize((imgpix, imgpix)),
                               GaussianBlur(5, sigma=(2, 2)),
                               RandomAffine(degrees=3, translate=(0.02, 0.02), ),
                               Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    vis_transform = Compose([Resize((imgpix, imgpix)), ])
    model_feat = create_feature_extractor(model, return_nodes=[layerkey])
    with torch.no_grad():
        out = model_feat(torch.randn(1, 3, imgpix, imgpix).cuda())
        feattsr = out[layerkey]

    _, C, H, W = feattsr.shape
    if channel_rng is None:
        channel_rng = (0, C)
    else:
        channel_rng = (max(0, channel_rng[0]), min(C, channel_rng[1]))
    batch_size = channel_rng[1] - channel_rng[0]
    cnt_pos = ((H - 1) // 2, (W - 1) // 2) if cnt_pos is None else cnt_pos
    if zs_init is None:
        zs = torch.randn(batch_size, 4096, generator=torch.Generator().manual_seed(seed)).cuda()
    else:
        zs = zs_init.clone().cuda().float()
    zs.requires_grad_(True)
    optim = torch.optim.Adam([zs], lr=lr)
    pert_std = pert_std
    score_traj = []
    for i in trange(steps):
        imgs = G.visualize(zs)
        out = model_feat(optim_transform(imgs))
        act_mat = out[layerkey][:, channel_rng[0]:channel_rng[1], cnt_pos[0], cnt_pos[1]]
        diag_resp = act_mat[torch.arange(batch_size), torch.arange(batch_size)]
        loss = -torch.sum(diag_resp)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # for zero activations add perturbation to the zs
        failure_mask = diag_resp == 0
        zs[failure_mask].data = zs[failure_mask, :].data + \
                                torch.randn_like(zs[failure_mask, :]) * pert_std
        print(f"{diag_resp.mean():.3f}+-{diag_resp.std():.3f} , (eq zero {torch.sum(diag_resp == 0).item()})")
        score_traj.append(diag_resp.detach().cpu())
    imgs = imgs.detach().cpu()
    torch.cuda.empty_cache()
    score_traj = torch.stack(score_traj, dim=0)
    if show:
        show_imgrid(vis_transform(imgs), nrow=int(np.sqrt(batch_size)), )
    mtg = to_imgrid(vis_transform(imgs), nrow=int(np.sqrt(batch_size)))
    return imgs, mtg, score_traj, zs


def cmaes_weights(n):
    weights = np.log(n + 1) - np.log(np.arange(n) + 1)
    weights = weights / np.sum(weights)
    return weights


def cma_popmigrate_parallel_optimize(G, model, layerkey, channel_rng=None, imgpix=96, optim_transform=None,
                              cnt_pos=None, zs_init=None, topK=8, cma_std=2, cma_steps=20, multiplier=4, seed=42):
    """The independent parallel version of the CMA-ES algorithm below is preferred over this."""
    if optim_transform is None:
        optim_transform = Compose([Resize((imgpix, imgpix)),
                               Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    # batch_size = 256
    model_feat = create_feature_extractor(model, return_nodes=[layerkey])
    with torch.no_grad():
        out = model_feat(torch.randn(1, 3, imgpix, imgpix).cuda())
        feattsr = out[layerkey]

    _, C, H, W = feattsr.shape
    if channel_rng is None:
        channel_rng = (0, C)
    else:
        channel_rng = (max(0, channel_rng[0]), min(C, channel_rng[1]))
    batch_size = channel_rng[1] - channel_rng[0]
    cnt_pos = ((H - 1) // 2, (W - 1) // 2) if cnt_pos is None else cnt_pos
    # initialize zs
    if zs_init is None:
        zs = torch.randn(batch_size, 4096, generator=torch.Generator().manual_seed(seed)).cuda()
    else:
        zs = zs_init.clone().detach().float().cuda()
    # no gradient needed for zs
    # zs.requires_grad_(True)
    score_trajs = []
    zmean_score_traj = []
    weights = torch.from_numpy(cmaes_weights(topK)).float().cuda()
    for i in trange(cma_steps):
        act_mat_tsr = []
        diag_resp_tsr = []
        zs_all = []
        for k in range(multiplier):
            # mutation, add CMA noise to zs
            zs_batch = zs + torch.randn(batch_size, 4096, ).cuda() * cma_std
            with torch.no_grad():
                imgs = G.visualize(zs_batch)
                out = model_feat(optim_transform(imgs))[layerkey]

            act_mat = out[:, channel_rng[0]:channel_rng[1], cnt_pos[0], cnt_pos[1]]  # (batch_size, channel_range)
            diag_resp = act_mat[torch.arange(batch_size), torch.arange(batch_size)]  # (batch_size,)
            act_mat_tsr.append(act_mat)
            diag_resp_tsr.append(diag_resp)
            zs_all.append(zs_batch)
        act_mat_tsr = torch.cat(act_mat_tsr, dim=0)  # (batch_size * multiplier, channel_range)
        diag_resp_tsr = torch.cat(diag_resp_tsr, dim=0)  # (batch_size * multiplier,)
        zs_all = torch.cat(zs_all, dim=0)  # (batch_size * multiplier, 4096)
        # select topK zs, considering all zs
        sort_idx = act_mat_tsr.argsort(dim=0, descending=True)
        topk_idx = sort_idx[:topK, :batch_size]  # (topK, batch_size), the topK idx for each channel
        zs = torch.einsum("i,ijk->jk", weights, zs_all[topk_idx, :])
        print(f"{diag_resp_tsr.mean():.3f}+-{diag_resp_tsr.std():.3f} , "
              f"(eq zero {torch.sum(diag_resp_tsr==0).item()} / {batch_size * multiplier})")
        score_trajs.append(diag_resp_tsr.detach().cpu())
        # validate the new zs are good
        with torch.no_grad():
            imgs = G.visualize(zs)
            out = model_feat(optim_transform(imgs))[layerkey]

        act_mat_final = out[:, channel_rng[0]:channel_rng[1], cnt_pos[0], cnt_pos[1]]  # (batch_size, channel_range)
        diag_resp_final = act_mat_final[torch.arange(batch_size), torch.arange(batch_size)]  # (batch_size,)
        print(f"zmean center eq zero {torch.sum(diag_resp_final == 0).item()} / {batch_size}")
        zmean_score_traj.append(diag_resp_final.detach().cpu())
    return zs, score_trajs, zmean_score_traj


def cma_independent_parallel_optimize(G, model, layerkey, channel_rng=None, imgpix=96, optim_transform=None,
                      cnt_pos=None, zs_init=None, topK=15, cma_std=2, cma_steps=20, multiplier=30, seed=42):
    if optim_transform is None:
        optim_transform = Compose([Resize((imgpix, imgpix)),
                               Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    # batch_size = 256
    model_feat = create_feature_extractor(model, return_nodes=[layerkey])
    with torch.no_grad():
        out = model_feat(torch.randn(1, 3, imgpix, imgpix).cuda())
        feattsr = out[layerkey]

    _, C, H, W = feattsr.shape
    if channel_rng is None:
        channel_rng = (0, C)
    else:
        channel_rng = (max(0, channel_rng[0]), min(C, channel_rng[1]))
    batch_size = channel_rng[1] - channel_rng[0]
    cnt_pos = ((H - 1) // 2, (W - 1) // 2) if cnt_pos is None else cnt_pos
    # initialize zs
    if zs_init is None:
        zs = torch.randn(batch_size, 4096, generator=torch.Generator().manual_seed(seed)).cuda()
    else:
        zs = zs_init.clone().detach().float().cuda()
    # no gradient needed for zs
    # zs.requires_grad_(True)
    score_trajs = []
    zmean_score_traj = []
    weights = torch.from_numpy(cmaes_weights(topK)).float().cuda()
    for i in trange(cma_steps):
        # act_mat_tsr = []
        diag_resp_tsr = []
        zs_all = []
        for k in range(multiplier):
            # mutation, add CMA noise to zs
            zs_batch = zs + torch.randn(batch_size, 4096, ).cuda() * cma_std
            with torch.no_grad():
                imgs = G.visualize(zs_batch)
                out = model_feat(optim_transform(imgs))[layerkey]

            act_mat = out[:, channel_rng[0]:channel_rng[1], cnt_pos[0], cnt_pos[1]]  # (batch_size, channel_range)
            diag_resp = act_mat[torch.arange(batch_size), torch.arange(batch_size)]  # (batch_size,)
            # act_mat_tsr.append(act_mat)
            diag_resp_tsr.append(diag_resp)
            zs_all.append(zs_batch)
        # act_mat_tsr = torch.stack(act_mat_tsr, dim=1)  # (batch_size, multiplier, channel_range)
        diag_resp_tsr = torch.stack(diag_resp_tsr, dim=1)  # (batch_size, multiplier,)
        zs_all = torch.stack(zs_all, dim=1)  # (batch_size, multiplier, 4096)
        # different from the previous version, here, each channel evole independently, only weighted averaging their own "images" and their top k.
        sort_idx = diag_resp_tsr.argsort(dim=1, descending=True)
        topk_idx = sort_idx[:batch_size, :topK]  # (batch_size, topK), the topK idx for each channel
        zs_topk = torch.gather(zs_all, dim=1,
                               index=topk_idx.unsqueeze(-1).expand(-1, -1, 4096))  # (batch_size, topK, 4096
        zs = torch.einsum("i,jik->jk", weights, zs_topk)
        print(f"{diag_resp_tsr.mean():.3f}+-{diag_resp_tsr.std():.3f} , "
              f"(eq zero {torch.sum(diag_resp_tsr == 0).item()} / {batch_size * multiplier})")
        score_trajs.append(diag_resp_tsr.detach().cpu())
        # validate the new zs are good
        with torch.no_grad():
            imgs = G.visualize(zs)
            out = model_feat(optim_transform(imgs))[layerkey]

        act_mat_final = out[:, channel_rng[0]:channel_rng[1], cnt_pos[0], cnt_pos[1]]  # (batch_size, channel_range)
        diag_resp_final = act_mat_final[torch.arange(batch_size), torch.arange(batch_size)]  # (batch_size,)
        print(f"zmean center {diag_resp_final.mean():.3f}+-{diag_resp_final.std():.3f} eq zero {torch.sum(diag_resp_final == 0).item()} / {batch_size}")
        zmean_score_traj.append(diag_resp_final.detach().cpu())

    score_trajs = torch.stack(score_trajs, dim=0)
    zmean_score_traj = torch.stack(zmean_score_traj, dim=0)
    return zs, score_trajs, zmean_score_traj