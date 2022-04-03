import sys
root = "."
sys.path.append(root)
import os
from omegaconf import OmegaConf as yaml
import copy
from pathlib import Path
from lib.data_loaders import load_data_tinyimagenet
import torch
from torch.utils.data import DataLoader
from model.Two_D_SNet import Two_D_SNet
import hdvw.ops.tests as tests
from hdvw.models import load_snapshot
import hdvw.ops.loss_landscapes as lls
config_path = './config/train_tinyimagenet.yaml'
with open(config_path) as f:
    args = yaml.load(f)
    print(args)
dataset_train, dataset_test= load_data_tinyimagenet(args['parameters']['batch_size'],
                                                           args['parameters']['batch_size'], args['data_url'])
model = Two_D_SNet(
    sample_data=torch.randn(args['parameters']['batch_size'], 3, 64, 64),
    dataoption="tinyimagenet",
    num_classes=args["num_classes"],
    decay_rate=args["decay_rate"],
    down_rate=args["down_rate"],
    dropout=args["parameters"]["dropout"],
    Tem=args["Tem"],
    channel_list=args["channel_list"],
    size_list=args["size_list"],
    bn_size_list=args["bn_size_list"],
    path_nums_list=args["path_nums_list"],
    nums_head_list=args["nums_head_list"],
    nums_layer_list=args["nums_layer_list"],
    breadth_threshold_list=args["breadth_threshold_list"],
    att_type_list=args["att_type_list"])
num_classes = args["num_classes"]
print("Train: %s, Test: %s, Classes: %s" % (
    len(dataset_train.dataset),
    len(dataset_test.dataset),
    num_classes
))

from timm.data import Mixup

def mixup_function(train_args):
    train_args = copy.deepcopy(train_args)
    smoothing = train_args.get("smoothing", 0.0)
    mixup_args = train_args.get("mixup", None)

    mixup_function = Mixup(
        **mixup_args,
        label_smoothing=smoothing,
    ) if mixup_args is not None else None
    return mixup_function


transform = mixup_function({"smoothing":0.1})

map_location = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("D:\\Product\\checkpoint\\tinyimagenet\\tinyimagenet_199\\tinyimagenet_199_2dsnet_5_6_7.pth.tar", map_location=map_location)
model.load_state_dict(checkpoint["state_dict"])

import copy
import timm
import torch
import torch.nn as nn

scale = 1e-0
n = 21
gpu = torch.cuda.is_available()

metrics_grid = lls.get_loss_landscape(
    model, 1, dataset_train, transform=transform,
    kws=["pos_embed", "relative_position"],
    x_min=-1.0 * scale, x_max=1.0 * scale, n_x=n, y_min=-1.0 * scale, y_max=1.0 * scale, n_y=n, gpu=gpu,
)
uid=1
leaderboard_path = os.path.join("leaderboard", "logs", 'cifar10', '2dsnet218')
Path(leaderboard_path).mkdir(parents=True, exist_ok=True)
metrics_dir = os.path.join(leaderboard_path, "%s_%s_%s_x%s_losslandscape.csv" % ('cifar10', model.name, uid, int(1 / scale)))
metrics_list = [[*grid, *metrics] for grid, metrics in metrics_grid.items()]
tests.save_metrics(metrics_dir, metrics_list)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
# load losslandscape raw data of ResNet-50 or ViT-Ti
names = ["x", "y", "l1", "l2", "NLL", "Cutoff1", "Cutoff2", "Acc", "Acc-90", "Unc", "Unc-90", "IoU", "IoU-90", "Freq", "Freq-90", "Top-5", "Brier", "ECE", "ECSE"]
# path = "%s/resources/results/cifar100_resnet_dnn_50_losslandscape.csv" % root  # for ResNet-50
path = "%s/resources/results/cifar100_vit_ti_losslandscape.csv" % root  # for ViT-Ti
data = pd.read_csv(path, names=names)
data["loss"] = data["NLL"] + args['optimizer'][args['optimizer']['optimizer_choice']]["weight_decay"] * data["l2"]  # NLL + l2

# prepare data
p = int(math.sqrt(len(data)))
shape = [p, p]
xs = data["x"].to_numpy().reshape(shape)
ys = data["y"].to_numpy().reshape(shape)
zs = data["loss"].to_numpy().reshape(shape)

zs = zs - zs[np.isfinite(zs)].min()
zs[zs > 42] = np.nan

norm = plt.Normalize(zs[np.isfinite(zs)].min(), zs[np.isfinite(zs)].max())  # normalize to [0,1]
colors = cm.plasma(norm(zs))
rcount, ccount, _ = colors.shape

fig = plt.figure(figsize=(4.2, 4), dpi=120)
ax = fig.gca(projection="3d")
ax.view_init(elev=15, azim=15)  # angle

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

surf = ax.plot_surface(
    xs, ys, zs,
    rcount=rcount, ccount=ccount,
    facecolors=colors, shade=False,
)
surf.set_facecolor((0,0,0,0))

# remove white spaces
adjust_lim = 0.8
ax.set_xlim(-1 * adjust_lim, 1 * adjust_lim)
ax.set_ylim(-1 * adjust_lim, 1 * adjust_lim)
ax.set_zlim(10, 32)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis('off')

plt.show()