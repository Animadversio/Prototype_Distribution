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
from torchvision.datasets import STL10
#%%
num_workers = 16
batch_size = 1024
input_size = 96
num_ftrs = 32
seed = 1
pl.seed_everything(seed)
dataroot = "/home/biw905/Datasets"
dataroot = r"E:\Datasets"
stl10_dir = join(dataroot, "stl10_binary")

#%%
transform = SimCLRTransform(input_size=input_size, vf_prob=0.5, rr_prob=0.5)

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
#%%
dataset_train_simclr = STL10(dataroot, split="unlabeled", transform=transform)
dataset_test = STL10(dataroot, split="test", transform=test_transform)
dataset_train_simclr = LightlyDataset.from_torch_dataset(dataset_train_simclr)
dataset_test = LightlyDataset.from_torch_dataset(dataset_test)
#%%
# dataset_train_simclr = LightlyDataset(input_dir=path_to_data, transform=transform)
#
# dataset_test = LightlyDataset(input_dir=path_to_data, transform=test_transform)

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

#%%
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

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


# max_epochs = 20
# model = SimCLRModel()
# trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu", )
# trainer.fit(model, dataloader_train_simclr)
#%%
exproot = "/home/biw905/ssl_train"
exproot = r"D:\DL_Projects\SelfSupervise\ssl_train"
expdir = join(exproot, "stl10_rn18_RND1")
os.makedirs(join(expdir, "checkpoints"), exist_ok=True)
max_epochs = 50
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
logger = TensorBoardLogger(expdir, name='RND1')

model = SimCLRModel()
torch.save(model.state_dict(), join(expdir, "checkpoints", "model_init.pth"))
trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu",
                     logger=logger, callbacks=[checkpoint_callback])
trainer.fit(model, dataloader_train_simclr)

#%%