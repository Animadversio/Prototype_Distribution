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


def pix_prototype_extract(model, layerkey, channel_rng=None, cnt_pos=None, imgpix=96, optcfg=None, show=True, seed=42):
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

    grad_input = 0.5 + torch.randn(batch_size, 3, imgpix, imgpix, generator=torch.Generator().manual_seed(seed)).cuda() * 0.02
    grad_input.requires_grad_(True)
    optim = torch.optim.Adam([grad_input], lr=0.1)
    pert_std = 0.001
    score_traj = []
    for i in trange(100):
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


def GAN_prototype_extract(model, layerkey, channel_rng=None, zs_init=None, cnt_pos=None, imgpix=96, optcfg=None, show=True, seed=42):
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
    optim = torch.optim.Adam([zs], lr=0.01)
    pert_std = 0.0005
    score_traj = []
    for i in trange(100):
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
# expdir = r"D:\DL_Projects\SelfSupervise\ssl_train\stl10_rn18_RND1"
# new_model0, surrogate0 = load_reformate_ckpt(join(expdir, "checkpoints", "model_init.pth"))
# new_model1, surrogate1 = load_reformate_ckpt(join(expdir, "checkpoints", "model-epoch=00-train_loss_ssl=6.29.ckpt"))
# new_model2, surrogate2 = load_reformate_ckpt(join(expdir, "checkpoints", "model-epoch=02-train_loss_ssl=5.93.ckpt"))
# new_model3, surrogate3 = load_reformate_ckpt(join(expdir, "checkpoints", "model-epoch=03-train_loss_ssl=5.90.ckpt"))
# #%%
# new_model3.cuda().eval().requires_grad_(False)
# model_df = get_module_name_shapes(new_model3, [torch.randn(1, 3, 96, 96).to("cuda")],
#                         deepest=2, show=False, show_input=True, return_df=True,)
#%%
# savedir = r"D:\DL_Projects\SelfSupervise\ssl_train\stl10_rn18_RND1_keepclr_protodist"
# expdir = r"D:\DL_Projects\SelfSupervise\ssl_train\stl10_rn18_RND1_keepclr"
savedir = r"D:\DL_Projects\SelfSupervise\ssl_train\stl10_rn18_RND1_clrjit_protodist"
expdir = r"D:\DL_Projects\SelfSupervise\ssl_train\stl10_rn18_RND1_clrjit"
os.makedirs(savedir, exist_ok=True)
vis_transform = Compose([Resize((96, 96)), ])
ckptlist = [*(Path(expdir)/"checkpoints").glob("*.pth")]+[*(Path(expdir)/"checkpoints").glob("*.ckpt")]
# layerkey = "layer4"
# epoch = 0
batch_size = 256
for epoch, ckptpath in tqdm(enumerate(ckptlist)):
    netname = f"rn18_ep{epoch-1:03d}"
    new_model, surrogate = load_reformate_ckpt(str(ckptpath))
    new_model.cuda().eval().requires_grad_(False)
    for layerkey in ["layer1", "layer2", "layer3", "layer4"]:
        C, H, W = get_layer_shape_torch_naming(new_model, layerkey, input_shape=(3, 96, 96))
        cnt_pos = ((H - 1) // 2, (W - 1) // 2)
        gradAmpmap = grad_RF_estimate_torch_naming(new_model, layerkey, (slice(None), *cnt_pos),
                                      input_size=(3, 96, 96), reps=10, batch=50, show=False)
        fitdict = fit_2dgauss(gradAmpmap, layerkey, outdir=None, plot=False)
        fitrfmap = fitdict["fitmap"]
        fitrfmap = fitrfmap / fitrfmap.max()
        np.savez(join(savedir, f"{netname}_{layerkey}_fitrfmap.npz"), **fitdict)
        for chan_beg in range(0, C, batch_size):
            RND = 42
            chan_end = min(chan_beg + batch_size, C)
            batch_str = f"{netname}_{layerkey}_{chan_beg:03d}-{chan_end:03d}"
            imgs_G, mtg_G, score_traj_G, zs = GAN_prototype_extract(new_model, layerkey,
                                    channel_rng=(chan_beg, chan_end), imgpix=96, show=False, seed=RND)
            imgs_pix, mtg_pix, score_traj_pix, _ = pix_prototype_extract(new_model, layerkey,
                                    channel_rng=(chan_beg, chan_end), imgpix=96, show=False, seed=RND)
            mtg_pix.save(join(savedir, f"{batch_str}_pix_RND{RND}.jpg") )
            mtg_G.save(join(savedir, f"{batch_str}_GAN_RND{RND}.jpg") )
            # show_imgrid(imgs_pix * fitrfmap[None,None], nrow=16)
            # show_imgrid(vis_transform(imgs_G) * fitrfmap[None,None], nrow=16)
            save_imgrid(imgs_pix * fitrfmap[None,None],
                        join(savedir, f"{batch_str}_pixRF_RND{RND}.jpg"), nrow=int(np.sqrt(len(imgs_pix))), )
            sucs_msk = ~(score_traj_pix[-1, :] == 0)
            imgs_pix_sucs = imgs_pix * sucs_msk[:, None, None, None].float()
            save_imgrid(imgs_pix_sucs * fitrfmap[None,None],
                        join(savedir, f"{batch_str}_pixRF_sucs_RND{RND}.jpg"), nrow=int(np.sqrt(len(imgs_pix))), )
            save_imgrid(vis_transform(imgs_G) * fitrfmap[None,None],
                        join(savedir, f"{batch_str}_GANRF_RND{RND}.jpg"), nrow=int(np.sqrt(len(imgs_G))), )
            sucs_msk = ~(score_traj_G[-1, :] == 0)
            imgs_G_sucs = imgs_G * sucs_msk[:, None, None, None].float()
            save_imgrid(vis_transform(imgs_G_sucs) * fitrfmap[None,None],
                        join(savedir, f"{batch_str}_GANRF_sucs_RND{RND}.jpg"), nrow=int(np.sqrt(len(imgs_G))), )
            torch.save(score_traj_G, join(savedir, f"{batch_str}_GANRF_RND{RND}_score.pth"))
            torch.save(score_traj_pix, join(savedir, f"{batch_str}_pixRF_RND{RND}_score.pth"))
#%%
# 101 models, evol all filters, 101it [4:19:14, 154.00s/it]
#%%
import imageio
# Set your image path here
for suffix in ["layer1_000-064_GANRF_RND42",
               "layer2_000-128_GANRF_RND42",
               "layer3_000-256_GANRF_RND42",
               "layer4_000-256_GANRF_RND42",
               "layer4_256-512_GANRF_RND42",]:
    # The images should be in format rn18_epXXX, where XXX is a three digit number
    imgfps = [join(savedir, f"rn18_ep{epoch:03d}_{suffix}.jpg") for epoch in range(-1, 100)]
    frames = [imageio.imread(imgfp) for imgfp in imgfps]
    imageio.mimsave(join(savedir, f"rn18_{suffix}.mp4"), frames, format='mp4', fps=2)
    imageio.mimsave(join(savedir, f"rn18_{suffix}.gif"), frames, format='gif', fps=2)

#%%
surrogate = surrogate3
surrogate.eval().cuda()
#%%
optim_transform = Compose([# Resize((96, 96)),
                            RandomAffine(degrees=2, translate=(0.02, 0.02), ),
                            GaussianBlur(5, sigma=(2, 2)),
                            Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])
vis_transform = Compose([Resize((96, 96)), ])

chan_cnt = 256
grad_input = 0.5 + torch.randn(chan_cnt, 3, 96, 96).cuda() * 0.02
grad_input.requires_grad_(True)
optim = torch.optim.Adam([grad_input], lr=0.1)
pert_std = 0.001
cnt_pos = (1, 1)
for i in trange(150):
    imgs = torch.clamp(grad_input, 0, 1)
    out = surrogate[:-1](optim_transform(imgs))
    act_mat = out[:, :, cnt_pos[0], cnt_pos[1]]
    diag_resp = act_mat[torch.arange(chan_cnt), torch.arange(chan_cnt)]
    loss = -torch.sum(diag_resp)
    optim.zero_grad()
    loss.backward()
    optim.step()
    failure_mask = diag_resp == 0
    grad_input[failure_mask].data = grad_input[failure_mask, :].data + \
                            torch.randn_like(grad_input[failure_mask, :]) * pert_std
    print(f"{diag_resp.mean():.3f}+-{diag_resp.std():.3f} , (eq zero {torch.sum(diag_resp==0).item()})")
    imgmean = grad_input.data.mean(dim=(2, 3), keepdims=True)
    imgstd = grad_input.data.std(dim=(2, 3), keepdims=True)
    grad_input.data = imgmean + (grad_input.data - imgmean) / imgstd * 0.22
show_imgrid(vis_transform(imgs), nrow=16, )
torch.cuda.empty_cache()
#%%
surrogate = surrogate0
surrogate.eval().cuda()
surrogate.requires_grad_(False)
#%%
optim_transform = Compose([Resize((96, 96)),
                           GaussianBlur(5, sigma=(2, 2)),
                           RandomAffine(degrees=3, translate=(0.02, 0.02), ),
                           Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
vis_transform = Compose([Resize((96, 96)),])
batch_size = 256
zs = torch.randn(batch_size, 4096, generator=torch.Generator().manual_seed(42)).cuda()
zs.requires_grad_(True)
optim = torch.optim.Adam([zs], lr=0.01)
cnt_pos = (1, 1)
pert_std = 0.0005
score_traj = []
for i in trange(100):
    imgs = G.visualize(zs)
    out = surrogate[:-1](optim_transform(imgs))
    act_mat = out[:, :, cnt_pos[0], cnt_pos[1]]
    diag_resp = act_mat[torch.arange(batch_size), torch.arange(batch_size)]
    loss = -torch.sum(diag_resp)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # for zero activations add perturbation to the zs
    failure_mask = diag_resp == 0
    zs[failure_mask].data = zs[failure_mask, :].data + \
                            torch.randn_like(zs[failure_mask, :]) * pert_std
    print(f"{diag_resp.mean():.3f}+-{diag_resp.std():.3f} , (eq zero {torch.sum(diag_resp==0).item()})")
    score_traj.append(diag_resp.detach().cpu())
# show_imgrid(torch.clamp((1 + grad_input.detach().cpu())/2, 0, 1), nrow=16, )
show_imgrid(vis_transform(imgs), nrow=16, )
torch.cuda.empty_cache()
#%%
show_imgrid(vis_transform(imgs[~failure_mask]), nrow=16, )
#%%
show_imgrid(vis_transform(imgs.detach().cpu())*fitrfmap[None,None], nrow=16, )
#%%
G = upconvGAN("fc6")
G.eval().requires_grad_(False)
#%%
#%% Screatch zone
model = resnet18(pretrained=True)
module_names, module_types, module_spec = get_module_name_shapes(model, [torch.randn(1,3,224,224)],
                                                                deepest=3, show=True, show_input=True, return_df=False);
module_df = pd.concat([
    pd.DataFrame(module_names, index=["module_names"]).T,
    pd.DataFrame(module_types, index=["module_types"]).T,
    pd.DataFrame(module_spec, ).T,], axis=1)

#%%
model = resnet18(pretrained=True)
module_df = get_module_name_shapes(model, [torch.randn(1,3,224,224)],
                        deepest=3, show=True, show_input=True, return_df=True)


#%%
get_graph_node_names(new_model3, )
layerkey = "layer3"
rn_feat = create_feature_extractor(new_model3, return_nodes=[layerkey])
# [*torch.load(join(expdir, "checkpoints", "model-epoch=00-train_loss_ssl=6.29.ckpt"))["state_dict"].keys()]
#%%
module_df2 = get_module_name_shapes(new_model0, [torch.randn(1,3,224,224)],
                        deepest=3, show=False, show_input=True, return_df=True)
print(module_df2)
#%%
module_df2 = get_module_name_shapes(surrogate0, [torch.randn(1,3,96,96)],
                        deepest=2, show=False, show_input=True, return_df=True)
print(module_df2)
