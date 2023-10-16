# pip install lightly
# pip install pytorch_lightning
#%%
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets import STL10
import argparse

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
parser.add_argument('--input_size', type=int, default=96, help='Input size')
parser.add_argument('--num_ftrs', type=int, default=32, help='Number of features')
parser.add_argument('--max_epochs', type=int, default=50, help='Max epochs')
parser.add_argument('--seed', type=int, default=1, help='Seed')
parser.add_argument('--dataroot', type=str, default="/home/biw905/Datasets", help='Data root path')
parser.add_argument('--exproot', type=str, default="/home/biw905/ssl_train", help='exp root path')
parser.add_argument('--expname', type=str, default="stl10_rn18_RND1", help='exp dir')
parser.add_argument('--cj_prob', type=float, default=0.8, help='Color jitter probability')
parser.add_argument('--random_gray_scale', type=float, default=0.2, help='Random gray scale probability')

args = parser.parse_args()
# example usage: python simclr_STL10train_O2.py --num_workers 16 --batch_size 1024 --input_size 96 --num_ftrs 32 --max_epochs 50 --seed 1 --dataroot /home/biw905/Datasets --exproot /home/biw905/ssl_train --expname stl10_rn18_RND1

num_workers = args.num_workers
batch_size = args.batch_size
input_size = args.input_size
max_epochs = args.max_epochs
num_ftrs = args.num_ftrs
seed = args.seed
dataroot = args.dataroot
exproot = args.exproot
expname = args.expname

cj_prob = args.cj_prob
random_gray_scale = args.random_gray_scale
# num_workers = 16
# batch_size = 1024
# num_workers = 8
# batch_size = 512
# input_size = 96
# num_ftrs = 32
# seed = 1
pl.seed_everything(seed)
# dataroot = "/home/biw905/Datasets"
# dataroot = r"E:\Datasets"
# exproot = "/home/biw905/ssl_train"
# exproot = r"D:\DL_Projects\SelfSupervise\ssl_train"
# stl10_dir = join(dataroot, "stl10_binary")
expdir = join(exproot, expname)
#%%
transform = SimCLRTransform(input_size=input_size, vf_prob=0.5, rr_prob=0.5,
                            cj_prob=cj_prob, random_gray_scale=random_gray_scale)

# We create a torchvision transformation for embedding the dataset after
# training
test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

dataset_train_simclr = STL10(dataroot, split="unlabeled", transform=transform)
dataset_test = STL10(dataroot, split="test", transform=test_transform)

dataset_train_simclr = LightlyDataset.from_torch_dataset(dataset_train_simclr)
dataset_test = LightlyDataset.from_torch_dataset(dataset_test)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)


class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.hparam = vars(args)
        self.save_hyperparameters(self.hparam)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]
# Train the module using the PyTorch Lightning Trainer on a single GPU.
os.makedirs(join(expdir, "checkpoints"), exist_ok=True)
# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=join(expdir, "checkpoints"),
    filename='model-{epoch:02d}-{train_loss_ssl:.2f}',
    monitor='train_loss_ssl',
    save_weights_only=True,
    mode='min',
    every_n_epochs=1,
    save_top_k=-1  # Saves all models
    # save_top_k=3  # Saves the top 3 models
)

# Create the TensorBoard logger
logger = TensorBoardLogger(expdir, )  # name='RND1'

model = SimCLRModel()
torch.save(model.state_dict(), join(expdir, "checkpoints", "model_init.pth"))
trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu",
                     logger=logger, callbacks=[checkpoint_callback])
trainer.fit(model, dataloader_train_simclr)

#%%