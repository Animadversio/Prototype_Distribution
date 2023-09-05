import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from torchvision.models import resnet18
import torch
import torch.nn as nn
from os.path import join
from tqdm import trange, tqdm
from collections import OrderedDict
# from circuit_toolkit import upconvGAN, get_module_names
# from circuit_toolkit.layer_hook_utils import get_module_name_shapes
from circuit_toolkit.plot_utils import show_imgrid, save_imgrid, to_imgrid
from circuit_toolkit.GAN_utils import upconvGAN
from circuit_toolkit.grad_RF_estim import grad_RF_estimate, fit_2dgauss, grad_RF_estimate_torch_naming
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, GaussianBlur, RandomAffine
# from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import sys
sys.path.append(r"/home/binxu/Github/Prototype_Distribution")
sys.path.append(r"/home/biw905/Github/Prototype_Distribution")
from proto_extract.proto_extract_lib import load_reformate_ckpt, get_layer_shape_torch_naming, \
    cma_independent_parallel_optimize, GAN_prototype_extract
import re


def parse_epoch_from_filename(filename):
    if filename.endswith("model_init.pth"):
        return -1

    try:
        match = re.search(r'epoch=(\d+)-', filename)
        if match:
            epoch_number = match.group(1)
            return int(epoch_number)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


import argparse
parser = argparse.ArgumentParser(description="Your script description here.")

# Add arguments
parser.add_argument("--expdir", type=str, default=r"ssl_train/stl10_rn18_RND4_clrjit", help="Experiment directory")
parser.add_argument("--suffix", type=str, default="_protodist", help="Suffix for the experiment")
parser.add_argument("--ckpt_beg", type=int, default=0, help="Checkpoint range start")
parser.add_argument("--ckpt_end", type=int, default=101, help="Checkpoint range end")
parser.add_argument("--init_RND", type=int, default=42, help="Random seed")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")

# Parse the arguments
args = parser.parse_args()
expdir = join("/n/scratch3/users/b/biw905/", args.expdir)
suffix = args.suffix
ckpt_rng = (args.ckpt_beg, args.ckpt_end)
RND = args.init_RND
batch_size = args.batch_size

# expdir = r"/home/biw905/ssl_train/stl10_rn18_RND3_clrjit"
# expdir = r"/n/scratch3/users/b/biw905/ssl_train/stl10_rn18_RND4_clrjit"
# suffix =  "_protodist"
# ckpt_rng = (0, 92)
# RND = 42
# batch_size = 256

savedir = expdir + suffix
os.makedirs(savedir, exist_ok=True)

G = upconvGAN("fc6")
G.cuda().eval().requires_grad_(False)
optim_transform = Compose([Resize((96, 96)),
                           Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
vis_transform = Compose([Resize((96, 96)), ])
ckptlist = [*(Path(expdir)/"checkpoints").glob("*.pth")]+sorted([*(Path(expdir)/"checkpoints").glob("*.ckpt")])
# note we'd better sort it or the order will be messed up by the OS, esp. on OSX or linux

for ckptpath in tqdm(ckptlist[ckpt_rng[0]:ckpt_rng[1]]):
    epoch = parse_epoch_from_filename(ckptpath.name)
    if epoch is None:
        print(f"Failed to parse epoch number from {ckptpath.name}")
        continue
    elif epoch == -1:
        netname = f"rn18_init"
    else:
        netname = f"rn18_ep{epoch:03d}"
    # prepare model, load in the checkpoint
    new_model, surrogate = load_reformate_ckpt(str(ckptpath))
    new_model.cuda().eval().requires_grad_(False)
    for layerkey in ["layer1", "layer2", "layer3", "layer4"]:
        C, H, W = get_layer_shape_torch_naming(new_model, layerkey, input_shape=(3, 96, 96))
        cnt_pos = ((H - 1) // 2, (W - 1) // 2)
        # Compute the gradient RF maps
        gradAmpmap = grad_RF_estimate_torch_naming(new_model, layerkey, (slice(None), *cnt_pos),
                                      input_size=(3, 96, 96), reps=10, batch=50, show=False)
        failcnt = 0
        while True:
            try:
                fitdict = fit_2dgauss(gradAmpmap, layerkey, outdir=None, plot=False)
                fitrfmap = fitdict["fitmap"]
                fitrfmap = fitrfmap / fitrfmap.max()
                np.savez(join(savedir, f"{netname}_{layerkey}_fitrfmap.npz"), **fitdict)
                break
            except np.linalg.LinAlgError:
                failcnt += 1
                if failcnt > 1E4:
                    fitrfmap = gradAmpmap
                    fitrfmap = fitrfmap / fitrfmap.max()
                    np.savez(join(savedir, f"{netname}_{layerkey}_fitrf_failed.npz"), gradAmpmap=gradAmpmap)
                    break
                else:
                    continue

        # except np.linalg.LinAlgError:
        #     print(f"Failed to fit 2d gaussian for {netname}_{layerkey}, using the gradient map instead.")
        #     fitrfmap = gradAmpmap
        #     fitrfmap = fitrfmap / fitrfmap.max()
        #     np.savez(join(savedir, f"{netname}_{layerkey}_fitrf_failed.npz"), gradAmpmap=gradAmpmap)
        # Batch evolving the prototypes for all channels in this layer
        for chan_beg in range(0, C, batch_size):
            chan_end = min(chan_beg + batch_size, C)
            batch_str = f"{netname}_{layerkey}_{chan_beg:03d}-{chan_end:03d}"

            # find a proper initialization that makes the gradient non-zero
            zs_cma, score_trajs_cma, zmean_score_traj_cma = cma_independent_parallel_optimize(G, new_model, layerkey,
                              channel_rng=(chan_beg, chan_end), imgpix=96,  optim_transform=optim_transform, seed=RND,
                              topK=15, multiplier=30, cma_std=2, cma_steps=20)
            # Adam optimization from the CMA initialization
            imgs_adam, mtg_adam, score_traj_adam, zs_adam = GAN_prototype_extract(G, new_model, layerkey,
                              channel_rng=(chan_beg, chan_end), zs_init=zs_cma,
                              optim_transform=optim_transform, lr=0.01, steps=50, pert_std=0.1, show=False)
            # saving the images in various formats
            mtg_adam.save(join(savedir, f"{batch_str}_GAN_RND{RND}.jpg") )
            save_imgrid(vis_transform(imgs_adam) * fitrfmap[None,None],
                        join(savedir, f"{batch_str}_GANRF_RND{RND}.jpg"), nrow=int(np.sqrt(len(imgs_adam))), )
            # masking out the failed ones 0 activation
            sucs_msk = ~(score_traj_adam[-1, :] == 0)
            imgs_G_sucs = imgs_adam * sucs_msk[:, None, None, None].float()
            save_imgrid(vis_transform(imgs_G_sucs) * fitrfmap[None,None],
                        join(savedir, f"{batch_str}_GANRF_sucs_RND{RND}.jpg"), nrow=int(np.sqrt(len(imgs_adam))), )
            # save the score trajectories
            torch.save({"cma_zmean_score_traj": zmean_score_traj_cma,
                        "cma_score_traj": score_trajs_cma,
                        "adam_score_traj": score_traj_adam,},
                       join(savedir, f"{batch_str}_GANRF_RND{RND}_score_cmaesgrad.pth"))

# #%%
# # 101 models, evol all filters, 101it [4:19:14, 154.00s/it]
# #%%
# import imageio
# # Set your image path here
# for suffix in ["layer1_000-064_GANRF_RND42",
#                "layer2_000-128_GANRF_RND42",
#                "layer3_000-256_GANRF_RND42",
#                "layer4_000-256_GANRF_RND42",
#                "layer4_256-512_GANRF_RND42",
#                "layer1_000-064_GANRF_sucs_RND42",
#                "layer2_000-128_GANRF_sucs_RND42",
#                "layer3_000-256_GANRF_sucs_RND42",
#                "layer4_000-256_GANRF_sucs_RND42",
#                "layer4_256-512_GANRF_sucs_RND42",
#                ]:
#     # The images should be in format rn18_epXXX, where XXX is a three digit number
#     imgfps = [join(savedir, f"rn18_ep{epoch:03d}_{suffix}.jpg") for epoch in range(-1, 100)]
#     frames = [imageio.imread(imgfp) for imgfp in imgfps]
#     imageio.mimsave(join(savedir, f"rn18_{suffix}.mp4"), frames, format='mp4', fps=2)
#     imageio.mimsave(join(savedir, f"rn18_{suffix}.gif"), frames, format='gif', fps=2)