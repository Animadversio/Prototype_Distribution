# Evaluate ssl trained models on STL10
from os.path import join
import os
# import click
import tqdm
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
# dataloader
from torch.utils.data import DataLoader, Dataset
#%%
from collections import OrderedDict
from torch import nn
from torchvision.models import resnet18
# Load pretrained model, and evaluate on STL10
def load_reformate_ckpt(ckpt_path, num_classes=10):
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
    new_model = resnet18(num_classes=num_classes)
    for i, module in enumerate(list(new_model.children())[:-1]):
        module.load_state_dict(surrogate[i].state_dict())
    # new_model.fc = nn.Identity()
    return new_model, surrogate

#%% load dataset
train_dataset = STL10(r'E:/Datasets', split='train', download=False)  # E:\Datasets\stl10_binary
test_dataset = STL10(r'E:/Datasets', split='test', download=False)  # E:\Datasets\stl10_binary
#%%
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, GaussianBlur, RandomAffine
eval_transform = Compose(
    [
        Resize((96, 96)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]
)
#%% load dataset
train_dataset = STL10(r'E:/Datasets', split='train', download=False, transform=eval_transform)  # E:\Datasets\stl10_binary
test_dataset = STL10(r'E:/Datasets', split='test', download=False, transform=eval_transform)  # E:\Datasets\stl10_binary
#%%
expdir = r"F:\insilico_exps\Prototype_distribution_ssl\stl10_rn18_RND1_keepclr"
model_fc, model_feat = load_reformate_ckpt(join(expdir, "checkpoints", "model-epoch=99-train_loss_ssl=5.85.ckpt"))

# %%
from pathlib import Path
def find_ckpt_for_epoch(expdir, epoch):
    match = list((Path(expdir)/"checkpoints").glob(f"*epoch={epoch:02d}*.ckpt"))
    if len(match) == 1:
        return str(match[0])
    else:
        return None

def get_image_features(model, loader):
    model.eval().cuda()
    feat_list = []
    label_list = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm.tqdm(loader)):
            feat = model(x.cuda()).detach().cpu()
            feat_list.append(feat)
            label_list.append(y)
    feat = torch.cat(feat_list, dim=0)
    label = torch.cat(label_list, dim=0)
    feat.squeeze_(dim=-1).squeeze_(dim=-1)
    return feat, label

expdir = r"F:\insilico_exps\Prototype_distribution_ssl\stl10_rn18_RND1_clrjit"
model_fc, model_feat = load_reformate_ckpt(find_ckpt_for_epoch(expdir, 99))
#%%
expdir = r"F:\insilico_exps\Prototype_distribution_ssl\stl10_rn18_RND1_keepclr"
model_fc, model_feat = load_reformate_ckpt(find_ckpt_for_epoch(expdir, 99))
#%%
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=5, drop_last=False, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         num_workers=5, drop_last=False, shuffle=True)


feat_train, label_train = get_image_features(model_feat, train_loader)
feat_test, label_test = get_image_features(model_feat, test_loader)
print(feat_train.shape, label_train.shape, "\n", feat_test.shape, label_test.shape)

#%%
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
def eval_features_sklearn(feat_train, label_train, feat_test, label_test, classifier_cls=LogisticRegression):
    print(f"using sklearn {classifier_cls.__name__}")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(feat_train)
    X_test = scaler.transform(feat_test)
    clf = classifier_cls(max_iter=1000, random_state=42)
    clf.fit(X_train, label_train)
    y_fit = clf.predict(X_train)
    accuracy_train = accuracy_score(label_train, y_fit)
    print(f"Accuracy on training set: {accuracy_train}")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(label_test, y_pred)
    # conf_matrix = confusion_matrix(label_test, y_pred)
    print(f"Accuracy: {accuracy}")
    # print(f"Confusion Matrix: \n{conf_matrix}")
    return clf, accuracy_train, accuracy


#%%

def eval_train(loader, model, criterion, optimizer):
    loss_epoch = 0
    correct_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item()
        correct_epoch += acc
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item() * y.size(0)

    return loss_epoch, correct_epoch


def eval_test(loader, model, criterion):
    loss_epoch = 0
    correct_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item()
        correct_epoch += acc
        loss_epoch += loss.item() * y.size(0)

    return loss_epoch, correct_epoch


def create_data_loaders_from_tensors(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        X_train, y_train
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )
    test = torch.utils.data.TensorDataset(
        X_test, y_test
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def eval_features_CE_adam(feature_train, label_train, feature_test, label_test, n_features=512, n_classes=10,
                 device="cuda", logistic_epochs=250, lr=1E-2, batch_size=1024, print_every_epoch=50):
    linearhead = nn.Linear(n_features, n_classes)
    linearhead = linearhead.to(device)
    optimizer = torch.optim.Adam(linearhead.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    feat_train_loader, feat_test_loader = create_data_loaders_from_tensors(
        feature_train, label_train, feature_test, label_test, batch_size=batch_size)

    for epoch in range(logistic_epochs):
        loss_epoch, accuracy_epoch = eval_train(
            feat_train_loader, linearhead, criterion, optimizer
        )
        if (1 + epoch) % print_every_epoch == 0 or epoch == 0:
            print( f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(feature_train)}\t Accuracy: {accuracy_epoch / len(feature_train)}")

    final_train_loss = loss_epoch / len(label_train)
    final_train_acc  = accuracy_epoch / len(label_train)
    # final testing
    loss_epoch, accuracy_epoch = eval_test(
        feat_test_loader, linearhead, criterion
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(label_test)}\t Accuracy: {accuracy_epoch / len(label_test)}"
    )
    final_test_loss = loss_epoch / len(label_test)
    final_test_acc  = accuracy_epoch / len(label_test)
    return linearhead, final_train_loss, final_train_acc, final_test_loss, final_test_acc

import tqdm
import json
# expdir = r"F:\insilico_exps\Prototype_distribution_ssl\stl10_rn18_RND1_keepclr"
exproot = r"F:\insilico_exps\Prototype_distribution_ssl"
tabdir = join(exproot, "stats_table")
os.makedirs(tabdir, exist_ok=True)
run_names = ['stl10_rn18_RND1_clrjit',
             'stl10_rn18_RND1_keepclr',
             'stl10_rn18_RND2_clrjit',
             'stl10_rn18_RND2_keepclr',
             'stl10_rn18_RND3_clrjit',
             'stl10_rn18_RND3_keepclr']
for run_name in tqdm.tqdm(run_names):
    expdir = join(exproot, run_name)
    for epoch in tqdm.tqdm([*range(0, 100, 20), 99]):
        ckpt_path = find_ckpt_for_epoch(expdir, epoch)
        model_fc, model_feat = load_reformate_ckpt(ckpt_path)
        #%
        cnn_batch_size = 256
        train_loader = DataLoader(train_dataset, batch_size=cnn_batch_size,
                                  num_workers=5, drop_last=False, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cnn_batch_size,
                                 num_workers=5, drop_last=False, shuffle=True)

        feat_train, label_train = get_image_features(model_feat, train_loader)
        feat_test, label_test = get_image_features(model_feat, test_loader)

        print(feat_train.shape, label_train.shape, "\n", feat_test.shape, label_test.shape)

        clf_LogReg, acc_train_LogReg, acc_test_LogReg = eval_features_sklearn(feat_train, label_train, feat_test, label_test,
                                                              classifier_cls=LogisticRegression)
        clf_LinSVC, acc_train_LinSVC, acc_test_LinSVC = eval_features_sklearn(feat_train, label_train, feat_test, label_test,
                                                              classifier_cls=LinearSVC)
        linearhead, train_loss_adamCE, train_acc_adamCE, test_loss_adamCE, test_acc_adamCE = \
            eval_features_CE_adam(feat_train, label_train, feat_test, label_test,
            logistic_epochs=250, print_every_epoch=50, lr=1E-2, batch_size=512
        )
        meta = {"ckpt_path": ckpt_path, "epoch": epoch, "expdir": expdir, "run_name": run_name}
        stats = {"acc_train_LogReg": acc_train_LogReg, "acc_test_LogReg": acc_test_LogReg,
                    "acc_train_LinSVC": acc_train_LinSVC, "acc_test_LinSVC": acc_test_LinSVC,
                    "train_loss_adamCE": train_loss_adamCE, "train_acc_adamCE": train_acc_adamCE,
                    "test_loss_adamCE": test_loss_adamCE, "test_acc_adamCE": test_acc_adamCE,
                    }
        meta.update(stats)
        json.dump(meta, open(join(tabdir, f"eval_results_{run_name}_ep{epoch:02d}.json"), "w"), indent=4)
#%%
# save results as json
import json
with open(join(expdir, "eval_results.json"), "w") as f:
    json.dump({"acc_train_LogReg": acc_train_LogReg, "acc_test_LogReg": acc_test_LogReg,
               "acc_train_LinSVC": acc_train_LinSVC, "acc_test_LinSVC": acc_test_LinSVC,
               "train_loss_adamCE": train_loss_adamCE, "train_acc_adamCE": train_acc_adamCE,
               "test_loss_adamCE": test_loss_adamCE, "test_acc_adamCE": test_acc_adamCE,
               }, f, indent=4)

#%% summary as table
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from circuit_toolkit.plot_utils import saveallforms
figdir = r"E:\OneDrive - Harvard University\NeurIPS2023_UniReps_ProtoDev\Figure"
#%%
tabdir = r"F:\insilico_exps\Prototype_distribution_ssl\stats_table"
tabdir = Path(tabdir)

df_col = []
for tabfile in tabdir.glob("*.json"):
    df_col.append(json.load(open(tabfile, "r")))
df = pd.DataFrame(df_col)
#%%
df["colorjit"] = df.run_name.str.contains("clrjit")
df["keepcolor"] = df.run_name.str.contains("keepclr")
#%%
df.to_csv(join(figdir, "eval_results.csv"))
#%%
df[df.epoch == 99].groupby(["colorjit", "keepcolor"]).mean().T
#%%
df[df.epoch == 99].plot(x="colorjit", y="test_acc_adamCE", kind="scatter")
plt.show()

#%%
plt.figure(figsize=[3., 5])
sns.stripplot(data=df[df.epoch == 99], x="colorjit", y="test_acc_adamCE",
              jitter=0.35, alpha=0.35)
sns.pointplot(data=df[df.epoch == 99], x="colorjit", y="test_acc_adamCE",
              join=False, color="black")
plt.axhline(0.1, ls="--", color="black", alpha=0.5)
plt.ylabel("Test Accuracy")
plt.xlabel("Color Jitter")
plt.title("STL10 Uncond. SSL\n100 epoch")
plt.ylim([0.08, 1.0])
plt.tight_layout()
saveallforms([figdir], "stl10_rn18_test_acc_strip_point")
plt.show()

#%%
def shorten_runname(run_name):
    shortname = run_name
    if shortname.startswith("stl10_rn18_RND"):
        shortname = shortname[len("stl10_rn18_"):]
    shortname = shortname.replace("_clrjit", " color jit")
    shortname = shortname.replace("_keepclr", " keep color")
    return shortname
#%%
# plot the results epoch vs accuracy for each runname
figh, axs = plt.subplots(1, 3, figsize=(9, 4)) # sharey=True,
for run_name in df.run_name.unique():
    run_name_label = shorten_runname(run_name)
    df_run = df[df.run_name == run_name]
    axs[0].plot(df_run.epoch, df_run.acc_test_LogReg, label=run_name_label, lw=2.5, alpha=0.7)
    axs[1].plot(df_run.epoch, df_run.acc_test_LinSVC, label=run_name_label, lw=2.5, alpha=0.7)
    axs[2].plot(df_run.epoch, df_run.test_acc_adamCE, label=run_name_label, lw=2.5, alpha=0.7)
    # plt.title(run_name)
axs[0].legend()
axs[0].set_title("LogReg")
axs[1].set_title("LinSVC")
axs[2].set_title("AdamCE")
figh.supxlabel("Epoch", fontsize=12)
figh.supylabel("Test Accuracy", fontsize=12)
figh.suptitle("Linear probe Test Acc. during SimCLR training", fontsize=14)
plt.tight_layout()
saveallforms([figdir], "stl10_rn18_test_acc_epoch_curve")
plt.show()





#%% Dev zone
# Step 3: Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(feat_train)
X_test = scaler.transform(feat_test)
# Step 4: Model Selection
# clf = LogisticRegression(max_iter=1000, random_state=42)
clf = LinearSVC(max_iter=1000, random_state=42)
# Step 5: Training
clf.fit(X_train, label_train)
y_fit = clf.predict(X_train)
accuracy_train = accuracy_score(label_train, y_fit)
print(f"Accuracy on training set: {accuracy_train}")
# Step 6: Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(label_test, y_pred)
conf_matrix = confusion_matrix(label_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: \n{conf_matrix}")
#%%
device = "cuda"
n_features = 512
logistic_epochs = 250
print_every_epoch = 50
n_classes = 10  # CIFAR-10 / STL-10
linearhead = nn.Linear(n_features, n_classes)
linearhead = linearhead.to(device)
optimizer = torch.optim.Adam(linearhead.parameters(), lr=1e-2)
criterion = torch.nn.CrossEntropyLoss()

feat_train_loader, feat_test_loader = create_data_loaders_from_tensors(
    feat_train, label_train, feat_test, label_test, batch_size=1024)

for epoch in range(logistic_epochs):
    loss_epoch, accuracy_epoch = eval_train(
        feat_train_loader, linearhead, criterion, optimizer
    )
    if (1 + epoch) % print_every_epoch == 0 or epoch == 0:
        print(
f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(label_train)}\t Accuracy: {accuracy_epoch / len(label_train)}"
        )

final_train_loss = loss_epoch / len(label_train)
final_train_acc  = accuracy_epoch / len(label_train)
# final testing
loss_epoch, accuracy_epoch = eval_test(
    feat_test_loader, linearhead, criterion
)
print(
    f"[FINAL]\t Loss: {loss_epoch / len(label_test)}\t Accuracy: {accuracy_epoch / len(label_test)}"
)
final_test_loss = loss_epoch / len(label_test)
final_test_acc  = accuracy_epoch / len(label_test)