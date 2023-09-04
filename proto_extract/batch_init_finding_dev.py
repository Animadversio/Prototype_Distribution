""" Massive parallel CMAES (256 threads) for optimizing population """
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


def cmaes_weights(n):
    weights = np.log(n + 1) - np.log(np.arange(n) + 1)
    weights = weights / np.sum(weights)
    return weights


def cma_popmigrate_parallel_optimize(model, layerkey, channel_rng=None, imgpix=96, optim_transform=None,
                              cnt_pos=None, zs_init=None, topK=8, cma_std=2, cma_steps=20, multiplier=4, seed=42):
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


def cma_independent_parallel_optimize(model, layerkey, channel_rng=None, imgpix=96, optim_transform=None,
                      cnt_pos=None, zs_init=None, topK=15, cma_std=2, cma_steps=20, multiplier=30, seed=42):
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


def GAN_prototype_extract(G, model, layerkey, channel_rng=None, zs_init=None, cnt_pos=None, imgpix=96,
                          optcfg={"lr": 0.01, "steps": 100}, show=True, seed=42, optim_transform=None, ):
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
    optim = torch.optim.Adam([zs], lr=optcfg["lr"])
    pert_std = 0.0005
    score_traj = []
    for i in trange(optcfg["steps"]):
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
#%%
G = upconvGAN("fc6")
G.cuda().eval().requires_grad_(False)
#%%
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


expdir = r"D:\DL_Projects\SelfSupervise\ssl_train\stl10_rn18_RND1"
# new_model0, surrogate0 = load_reformate_ckpt(join(expdir, "checkpoints", "model_init.pth"))
# new_model1, surrogate1 = load_reformate_ckpt(join(expdir, "checkpoints", "model-epoch=00-train_loss_ssl=6.29.ckpt"))
# new_model2, surrogate2 = load_reformate_ckpt(join(expdir, "checkpoints", "model-epoch=02-train_loss_ssl=5.93.ckpt"))
# new_model3, surrogate3 = load_reformate_ckpt(join(expdir, "checkpoints", "model-epoch=03-train_loss_ssl=5.90.ckpt"))
new_model, surrogate = load_reformate_ckpt(join(expdir, "checkpoints", "model-epoch=03-train_loss_ssl=5.90.ckpt"))
#%%
new_model.cuda().eval().requires_grad_(False)
imgpix = 96
optim_transform = Compose([Resize((imgpix, imgpix)),
                           Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
vis_transform = Compose([Resize((imgpix, imgpix)), ])
#%%
zs_cma, score_trajs_cma, zmean_score_traj_cma = cma_independent_parallel_optimize(new_model, "layer2", channel_rng=None,
                                  imgpix=96, optim_transform=optim_transform, seed=42,
                                  topK=15, cma_std=2, cma_steps=20, multiplier=30)
imgs, mtg, score_traj_adam, zs_adam = GAN_prototype_extract(G, new_model, "layer2", channel_rng=None,
                                  zs_init=zs_cma, optim_transform=optim_transform,)
#%%
zs_cma, score_trajs_cma, zmean_score_traj_cma = cma_independent_parallel_optimize(new_model, "layer3", channel_rng=None,
                                  optim_transform=optim_transform, seed=42,
                                  topK=15, cma_std=2, cma_steps=20, multiplier=30)
imgs, mtg, score_traj_adam, zs_adam = GAN_prototype_extract(G, new_model, "layer3", channel_rng=(0, 256),
                                  zs_init=zs_cma, optim_transform=optim_transform, optcfg={"lr": 0.01, "steps": 50},)
#%%
zs_cma, score_trajs_cma, zmean_score_traj_cma = cma_independent_parallel_optimize(new_model, "layer4", channel_rng=(0, 256),
                      imgpix=96, optim_transform=optim_transform, seed=42,
                      topK=15, cma_std=2, cma_steps=20, multiplier=30)
imgs, mtg, score_traj_adam, zs_adam = GAN_prototype_extract(G, new_model, "layer4", channel_rng=(0, 256),
                      zs_init=zs_cma, optim_transform=optim_transform, optcfg={"lr": 0.01, "steps": 50},)
# %%
zs_cma, score_trajs_cma, zmean_score_traj_cma = cma_independent_parallel_optimize(new_model, "layer4",
                                  channel_rng=(0, 256), imgpix=96, optim_transform=optim_transform,
                                  seed=42, topK=20, cma_std=2, cma_steps=20, multiplier=40)
imgs, mtg, score_traj_adam, zs_adam = GAN_prototype_extract(G, new_model, "layer4", channel_rng=(0, 256),
                                                            zs_init=zs_cma, optim_transform=optim_transform,
                                                            optcfg={"lr": 0.01, "steps": 50}, )
#%%
model = new_model
layerkey = "layer3"
channel_rng = (0, 256)
cnt_pos = None
zs_init = None
cma_std = 2
cma_steps = 10
topK = 15
multiplier = 30
seed = 42


imgpix = 96
# optim_transform = Compose([Resize((imgpix, imgpix)),
#                                GaussianBlur(5, sigma=(2, 2)),
#                                RandomAffine(degrees=3, translate=(0.02, 0.02), ),
#                                Normalize([0.485, 0.456, 0.406],
#                                          [0.229, 0.224, 0.225])])
optim_transform = Compose([Resize((imgpix, imgpix)),
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
# initialize zs
if zs_init is None:
    zs = torch.randn(batch_size, 4096, generator=torch.Generator().manual_seed(seed)).cuda()
else:
    zs = zs_init.clone().detach().float().cuda()
# no gradient needed for zs
# zs.requires_grad_(True)
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
    sort_idx = act_mat_tsr.argsort(dim=0, descending=True)
    topk_idx = sort_idx[:topK, :batch_size]  # (topK, batch_size), the topK idx for each channel
    zs = torch.einsum("i,ijk->jk", weights, zs_all[topk_idx, :])
    print(f"{diag_resp_tsr.mean():.3f}+-{diag_resp_tsr.std():.3f} , "
          f"(eq zero {torch.sum(diag_resp_tsr==0).item()} / {batch_size * multiplier})")

with torch.no_grad():
    imgs = G.visualize(zs)
    out = model_feat(optim_transform(imgs))[layerkey]

act_mat_final = out[:, channel_rng[0]:channel_rng[1], cnt_pos[0], cnt_pos[1]]  # (batch_size, channel_range)
diag_resp_final = act_mat[torch.arange(batch_size), torch.arange(batch_size)]  # (batch_size,)
print(f"eq zero {torch.sum(diag_resp_final==0).item()} / {batch_size}")


#%%
model = new_model
layerkey = "layer3"
channel_rng = (0, 256)
cnt_pos = None
zs_init = None
cma_std = 2
cma_steps = 100
topK = 20
multiplier = 40
seed = 42


imgpix = 96
# optim_transform = Compose([Resize((imgpix, imgpix)),
#                                GaussianBlur(5, sigma=(2, 2)),
#                                RandomAffine(degrees=3, translate=(0.02, 0.02), ),
#                                Normalize([0.485, 0.456, 0.406],
#                                          [0.229, 0.224, 0.225])])
optim_transform = Compose([Resize((imgpix, imgpix)),
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
    act_mat_tsr = torch.stack(act_mat_tsr, dim=1)  # (batch_size, multiplier, channel_range)
    diag_resp_tsr = torch.stack(diag_resp_tsr, dim=1)  # (batch_size, multiplier,)
    zs_all = torch.stack(zs_all, dim=1)  # (batch_size, multiplier, 4096)
    # different from the previous version, here, each channel evole independently, only weighted averaging their own "images" and their top k.
    sort_idx = diag_resp_tsr.argsort(dim=1, descending=True)
    topk_idx = sort_idx[:batch_size, :topK]  # (batch_size, topK), the topK idx for each channel
    zs_topk = torch.gather(zs_all, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, 4096))  # (batch_size, topK, 4096
    zs = torch.einsum("i,jik->jk", weights, zs_topk)
    print(f"{diag_resp_tsr.mean():.3f}+-{diag_resp_tsr.std():.3f} , "
          f"(eq zero {torch.sum(diag_resp_tsr == 0).item()} / {batch_size * multiplier})")

    score_trajs.append(diag_resp_tsr.detach().cpu())
    with torch.no_grad():
        imgs = G.visualize(zs)
        out = model_feat(optim_transform(imgs))[layerkey]

    act_mat_final = out[:, channel_rng[0]:channel_rng[1], cnt_pos[0], cnt_pos[1]]  # (batch_size, channel_range)
    diag_resp_final = act_mat_final[torch.arange(batch_size), torch.arange(batch_size)]  # (batch_size,)
    print(f"zmean center eq zero {torch.sum(diag_resp_final==0).item()} / {batch_size}")
    zmean_score_traj.append(diag_resp_final.detach().cpu())

# topk=10, multiplier=20, cma_std=2, cma_steps=100  : 12.968+-3.293 , (eq zero 0 / 5120) 06:23<00:00,  3.83s/it
# topk=15, multiplier=30, cma_std=2, cma_steps=100  : 13.712+-3.233 , (eq zero 0 / 7680) [09:30<00:00,  5.70s/it]
# topk=20, multiplier=40, cma_std=2, cma_steps=100  : 12.202+-3.208 , (eq zero 0 / 10240) 50/100 [06:14<06:13,  7.47s/it]
# topk=20, multiplier=40, cma_std=2, cma_steps=100  : 14.162+-3.246 , (eq zero 0 / 10240) 100/100 [12:28<00:00,  7.49s/it]
score_trajs = torch.stack(score_trajs, dim=0)
zmean_score_traj = torch.stack(zmean_score_traj, dim=0)
#%% plot
plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.plot(score_trajs.mean(dim=-1).numpy())
plt.xlabel("CMA step")
plt.ylabel("mean activation")
plt.subplot(122)
plt.plot(zmean_score_traj.numpy())
plt.xlabel("CMA step")
plt.ylabel("mean activation")
plt.show()
#%%
mtg = to_imgrid(vis_transform(imgs), nrow=16, padding=2, )
mtg.show()
#%%


import torch

# Create some example data for demonstration
batch = 3
Nidx = 2
Ntotal = 5
Nlatent = 4

# Random tensor A of shape [batch, Nidx]
A = torch.randint(0, Ntotal, (batch, Nidx))

# Random tensor B of shape [batch, Ntotal, Nlatent]
B = torch.randn(batch, Ntotal, Nlatent)

# Expand A's dimensions for gather operation
A_expanded = A.unsqueeze(-1).expand(-1, -1, Nlatent)

# Use gather to get the desired tensor
C = torch.gather(B, 1, A_expanded)

# C should have shape [batch, Nidx, Nlatent]
print(C.shape)
print(C)
