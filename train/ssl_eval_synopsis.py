#%% summary as table
import json
from os.path import join
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
df[df.epoch == 99].groupby(["colorjit", "keepcolor"]).agg(["mean", "sem"]).T
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

