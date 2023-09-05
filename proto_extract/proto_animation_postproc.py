
import imageio
from os.path import join
savedir = r"D:\DL_Projects\SelfSupervise\ssl_train\stl10_rn18_RND2_keepclr_protodist"
savedir = r"D:\DL_Projects\SelfSupervise\ssl_train\stl10_rn18_RND2_clrjit_protodist"

# Set your image path here
for suffix in ["layer1_000-064_GANRF_RND42",
               "layer2_000-128_GANRF_RND42",
               "layer3_000-256_GANRF_RND42",
               "layer4_000-256_GANRF_RND42",
               "layer4_256-512_GANRF_RND42",
               "layer1_000-064_GANRF_sucs_RND42",
               "layer2_000-128_GANRF_sucs_RND42",
               "layer3_000-256_GANRF_sucs_RND42",
               "layer4_000-256_GANRF_sucs_RND42",
               "layer4_256-512_GANRF_sucs_RND42",
               ]:
    # The images should be in format rn18_epXXX, where XXX is a three digit number
    imgfps = [join(savedir, f"rn18_init_{suffix}.jpg")] + \
             [join(savedir, f"rn18_ep{epoch:03d}_{suffix}.jpg") for epoch in range(0, 100)]
    frames = [imageio.imread(imgfp) for imgfp in imgfps]
    imageio.mimsave(join(savedir, f"rn18_{suffix}.mp4"), frames, format='mp4', fps=2)
    imageio.mimsave(join(savedir, f"rn18_{suffix}.gif"), frames, format='gif', fps=2)

#%%%
